# -*- coding: utf-8 -*-
"""
Created on Monday September, 2023

@author: Fizzy Huiying Fan
All rights reserved
"""


import os
import glob
import time
import tqdm
import argparse
#
import numpy as np
import pandas as pd
import math
#
import networkx as nx
#
import geopandas as gpd
import geopy
import geopy.distance


def filter_nodes_by_stop(df_stops, nodes, node_stop_thres): ##### edit!
  #
  print(' - Filtering down nodes...')
  # Format stops
  df_stops = gpd.GeoDataFrame(df_stops, 
                              geometry=gpd.points_from_xy(df_stops.stop_lon, df_stops.stop_lat),
                              crs = 'EPSG:4269')
  df_stops = df_stops.to_crs('EPSG:2163')
  # Format nodes
  nodes = nodes[['osmid','y','x']].rename(columns={'y':'lat','x':'lon'})
  nodes = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.lon, nodes.lat),
                           crs = 'EPSG:4269')
  nodes = nodes.to_crs('EPSG:2163')
  # Stop outline
  stop_outline = df_stops.dissolve()
  stop_outline = gpd.GeoDataFrame(stop_outline, geometry=stop_outline.buffer(1609.344*node_stop_thres))
  # Filter node
  nodes = gpd.sjoin(nodes, stop_outline, how='left', predicate='within')
  nodes = nodes[~nodes['stop_id'].isnull()]
  #
  print(' - Generating stop-node pairs...')
  # Format nodes
  nodes = gpd.GeoDataFrame(nodes, geometry=nodes.buffer(1609.344*node_stop_thres))
  nodes = nodes[['osmid','lat','lon','geometry']]
  # Join nodes to stops
  stop_node = gpd.sjoin(df_stops, nodes, how='left', predicate='within')
  stop_node = stop_node[~stop_node['osmid'].isnull()]
  stop_node = stop_node[['stop_id','stop_lat','stop_lon','osmid','lat','lon']]
  #
  print(' - Looking for stop-node pair with the shortest distance...')
  # Looking for stop-node pair with the shortest distance
  stop_node['dist'] = stop_node.apply(lambda x: geopy.distance.geodesic((x.stop_lat, x.stop_lon),
                                                                        (x.lat, x.lon)).miles, 
                                      axis=1)
  #
  stop_node.reset_index(drop = True, inplace = True)
  stop_node = stop_node.loc[stop_node.groupby('stop_id')['dist'].idxmin()]
  stop_node = stop_node[['stop_id', 'osmid']]
  return(stop_node)

def shortest_path_finder(samp_in, links):
    if samp_in.shape[0] == 0:
        return(None)
    # Build graphs
    DG_dist = nx.DiGraph()
    for ind, row in links.iterrows():
        DG_dist.add_weighted_edges_from([(row['from'], row['to'], row['length'])])
    # Find shortest paths
    samp_in['dist'] = 0
    samp_in['path'] = 'nan'
    #
    for ind, row in tqdm.tqdm(samp_in.iterrows(), total=samp_in.shape[0]):
        #
        source = row.node1
        target = row.node2
        #
        try:
            dist, path = nx.single_source_dijkstra(DG_dist, source, target, weight='weight')
            samp_in.loc[ind, 'dist'] = dist
            samp_in.loc[ind, 'path'] = '__'.join([str(x) for x in path])
        except:
            samp_in.loc[ind, 'dist'] = 99999
            samp_in.loc[ind, 'path'] = 'Error, node not in network'
    return(samp_in)


def snap_sample_network(RAW_NETWORK_PATH, SAMP_INT_PATH, PROC_GTFS_PATH, snap_dist=0.5):
    # import samples
    f_l_mile = pd.read_csv(os.path.join(SAMP_INT_PATH, 'sample_stops.csv'))
    # import stops
    stops = pd.read_csv(os.path.join(PROC_GTFS_PATH, 'stops.txt'))
    # import network nodes
    nodes = gpd.read_file(os.path.join(RAW_NETWORK_PATH, 'nodes.shp'))
    # find first and last stop coordinates
    f_l_mile = f_l_mile.merge(stops.rename(columns={'stop_lat':'o_stop_lat','stop_lon':'o_stop_lon','stop_id':'o_stop'}), how='left', on='o_stop')
    f_l_mile = f_l_mile.merge(stops.rename(columns={'stop_lat':'d_stop_lat','stop_lon':'d_stop_lon','stop_id':'d_stop'}), how='left', on='d_stop')
    # combine with stops information - for O and D 
    o_stop_node = filter_nodes_by_stop(f_l_mile.rename(columns={'trip_id':'stop_id', 'ori_lat':'stop_lat', 'ori_lon':'stop_lon'}), nodes, snap_dist)
    d_stop_node = filter_nodes_by_stop(f_l_mile.rename(columns={'trip_id':'stop_id', 'dest_lat':'stop_lat', 'dest_lon':'stop_lon'}), nodes, snap_dist)
    f_l_mile = f_l_mile.merge(o_stop_node.rename(columns={'stop_id':'trip_id', 'osmid':'o_osm'}))
    f_l_mile = f_l_mile.merge(d_stop_node.rename(columns={'stop_id':'trip_id', 'osmid':'d_osm'}))
    # combine with stops information - for O stop and D stop
    o_stop_node = filter_nodes_by_stop(f_l_mile.rename(columns={'trip_id':'stop_id', 'o_stop_lat':'stop_lat', 'o_stop_lon':'stop_lon'}), nodes, 0.5) # snap within 0.5 mile
    d_stop_node = filter_nodes_by_stop(f_l_mile.rename(columns={'trip_id':'stop_id', 'd_stop_lat':'stop_lat', 'd_stop_lon':'stop_lon'}), nodes, 0.5) # snap within 0.5 mile
    f_l_mile = f_l_mile.merge(o_stop_node.rename(columns={'stop_id':'trip_id', 'osmid':'o_stop_osm'}))
    f_l_mile = f_l_mile.merge(d_stop_node.rename(columns={'stop_id':'trip_id', 'osmid':'d_stop_osm'}))
    # save 
    f_l_mile.to_csv(os.path.join(SAMP_INT_PATH, 'sample_f_l_network.csv'), index=False)
    return(None)

def find_f_l_paths(RAW_NETWORK_PATH, SAMP_INT_PATH):
    # import f_l file
    f_l_mile = pd.read_csv(os.path.join(SAMP_INT_PATH, 'sample_f_l_network.csv'))
    # import network edges
    edges = gpd.read_file(os.path.join(RAW_NETWORK_PATH, 'edges.shp'))
    edges = edges[['from','to','length']]
    # length is in meters
    # Since it's walk links, assuming everything reversable
    edges_rev = edges.copy()
    edges_rev['from'] = edges['to']
    edges_rev['to'] = edges['from']
    edges = pd.concat([edges, edges_rev])
    edges = edges.groupby(['from','to']).mean().reset_index()
    edges = edges.astype({'from':'float', 'to':'float'})
    # Create ingress and egress with shortest path finder
    t0 = time.time()
    #
    t0_transfer = time.time()
    trans_ingress = shortest_path_finder(f_l_mile.rename(columns={'o_osm':'node1','o_stop_osm':'node2'})[['node1','node2']], edges)
    print(' - Ingress time used: ', round((time.time()-t0_transfer)/60, 2), ' minutes.')
    #
    t0_transfer = time.time()
    trans_egress = shortest_path_finder(f_l_mile.rename(columns={'d_stop_osm':'node1','d_osm':'node2'})[['node1','node2']], edges)
    print(' - Egress time used: ', round((time.time()-t0_transfer)/60, 2), ' minutes.')
    #
    print('Done! Time used for creating ingress and egress: ', round((time.time()-t0)/60, 2), ' minutes.')
    # Formatting ingress and egress
    trans_ingress['dist']=trans_ingress['dist']/1609.344
    trans_egress['dist']=trans_egress['dist']/1609.344
    #
    trans_ingress.columns=['o_osm','o_stop_osm','ingress_dist','ingress_path']
    trans_egress.columns=['d_stop_osm','d_osm','egress_dist','egress_path']
    #
    f_l_mile = f_l_mile.merge(trans_ingress, how='left', on=['o_osm','o_stop_osm'])
    f_l_mile = f_l_mile.merge(trans_egress, how='left', on=['d_stop_osm','d_osm'])
    f_l_mile = f_l_mile.drop_duplicates().reset_index(drop=True)
    # Save file
    f_l_mile.to_csv(os.path.join(SAMP_INT_PATH, 'sample_f_l_path.csv'), index=False)
    return(None)

def find_trajectory(links, df_paths, part):
    ## Find trajectory for each segment of the trip
    df_traj = pd.DataFrame()
    for ind, row in tqdm.tqdm(df_paths.iterrows(), total=df_paths.shape[0]):
        try:
            # From network, generate trajectory as a list
            trajectory = [float(x) for x in row['{}_path'.format(part)].split('__')]
            # Make a DF to start with
            df_traj_ent = pd.DataFrame({
                'trip_id': row.trip_id,
                'mode': row['{}_mode'.format(part)],
                'from': trajectory[:-1],
                'to': trajectory[1:],
                'tt_seconds': row['{}_time'.format(part)],
                'status': 'success',
                'part': part
            })
            # Join with links file to find weight in each segment, and use that to find TT in each segment 
            df_traj_ent = df_traj_ent.merge(links[['from','to','length']].rename(columns={'length':'weight'}), 
                                            how='left', on=['from','to'])
            sum_weight = df_traj_ent['weight'].sum()
            df_traj_ent['weight'] = df_traj_ent['weight']/sum_weight
            df_traj_ent['tt_seconds'] = df_traj_ent['tt_seconds']*df_traj_ent['weight']
            # Find time 1 and time 2 for each segment
            df_traj_ent['time_from_o'] = pd.to_timedelta(df_traj_ent['tt_seconds'].cumsum(), unit='s')
            df_traj_ent['time_to_d'] = np.max(df_traj_ent['time_from_o']) - df_traj_ent['time_from_o']
            # Format, & join to the df_traj file
            df_traj_ent = df_traj_ent[['trip_id','mode','from','to','tt_seconds','time_from_o','time_to_d','status']]
            df_traj = pd.concat([df_traj, df_traj_ent], ignore_index=True)
        except:
            df_traj_ent = pd.DataFrame({
                'trip_id': row.trip_id,
                'mode': row['{}_mode'.format(part)],
                'from': -99999999.99,
                'to': -99999999.99,
                'tt_seconds': row['{}_time'.format(part)],
                'time_from_o': pd.to_timedelta(0, unit='s'),
                'time_to_d': pd.to_timedelta(0, unit='s'),
                'status': ['error'],
                'part': part
            })
            df_traj = pd.concat([df_traj, df_traj_ent], ignore_index=True)
    return(df_traj)

def f_l_trajectories(RAW_NETWORK_PATH, SAMP_INT_PATH, TRAJ_PATH, F_L_PATH, dict_speed=None):
    # import f_l file
    f_l_mile = pd.read_csv(os.path.join(SAMP_INT_PATH, 'sample_f_l_path.csv'))
    # import network edges
    edges = gpd.read_file(os.path.join(RAW_NETWORK_PATH, 'edges.shp'))
    edges = edges[['from','to','length']]
    # length is in meters
    # Since it's walk links, assuming everything reversable
    edges_rev = edges.copy()
    edges_rev['from'] = edges['to']
    edges_rev['to'] = edges['from']
    edges = pd.concat([edges, edges_rev])
    edges = edges.groupby(['from','to']).mean().reset_index()
    edges = edges.astype({'from':'float', 'to':'float'})
    # Define speed for each mode
    if dict_speed != None:
        speed = dict_speed
    else:
        speed = {'auto': 40,
                'bike': 15,
                'bike_oneway': 15,
                'scooter_oneway': 6,
                'skateboard': 9,
                'walk': 2.5,
                'wheelchair': 1}
    # Get row-wise travel speed based on mode
    if 'ingress_mode' in f_l_mile.columns:
        f_l_mile['ingress_speed'] = [speed[x] for x in f_l_mile['ingress_mode']]
    else:
        f_l_mile['ingress_speed'] = 2.5
        f_l_mile['ingress_mode'] = 'walk_ingress'
    if 'egress_mode' in f_l_mile.columns:
        f_l_mile['egress_speed'] = [speed[x] for x in f_l_mile['egress_mode']]
    else:
        f_l_mile['egress_speed'] = 2.5
        f_l_mile['egress_mode'] = 'walk_egress'
    # Get row-wise travel time
    f_l_mile['ingress_time'] = f_l_mile['ingress_dist']/f_l_mile['ingress_speed']*60*60
    f_l_mile['egress_time'] = f_l_mile['egress_dist']/f_l_mile['egress_speed']*60*60
    # Get ingress/egress trajectories
    ingress_traj = find_trajectory(edges, f_l_mile, 'ingress')
    ingress_traj.to_csv(os.path.join(F_L_PATH, 'ingress_traj.csv'), index=False)
    egress_traj = find_trajectory(edges, f_l_mile, 'egress')
    egress_traj.to_csv(os.path.join(F_L_PATH, 'egress_traj.csv'), index=False)
    return(None)

def combine_f_l_traj(traj, traj_ingress, traj_egress):
    if traj_ingress.shape[0] + traj_egress.shape[0] == 0:
        return(traj, False)
    else:
        # backward calculating time 1 and time 2 for every row in traj_ingress
        try:
            traj_ingress.loc[:,'time_to_d'] = traj_ingress.apply(lambda x: pd.Timedelta(x.time_to_d), axis=1)
            traj_ingress.loc[:,'time2'] = pd.to_datetime(traj['time1'].min())
            traj_ingress.loc[:,'time2'] = traj_ingress['time2'] - traj_ingress['time_to_d']
            traj_ingress.loc[:,'time1'] = 0
            traj_ingress.iloc[1:, traj_ingress.columns.get_loc('time1')] = traj_ingress['time2'][:-1]
            traj_ingress.at[traj_ingress.index[0], 'time1'] = (traj_ingress.at[traj_ingress.index[0], 
                                                                                'time2'] - 
                                                                pd.Timedelta(traj_ingress.at[traj_ingress.index[0], 
                                                                                            'tt_seconds'], 
                                                                            unit = 's'))
            traj_ingress = traj_ingress.rename(columns={'from':'stop1','to':'stop2'})
            traj_ingress = traj_ingress[['trip_id','mode','stop1','stop2','tt_seconds','time1','time2']]
            traj_ingress.loc[:,'status'] = 'success'
        except: pass
        # forward calculating time 1 and time 2 for every row in traj_ingress
        try:
            traj_egress.loc[:,'time_from_o'] = traj_egress.apply(lambda x: pd.Timedelta(x.time_from_o), axis=1)
            traj_egress.loc[:,'time2'] = pd.to_datetime(traj['time2'].max())
            traj_egress.loc[:,'time2'] = traj_egress['time2'] + traj_egress['time_from_o']
            traj_egress.loc[:,'time1'] = 0
            traj_egress.iloc[1:, traj_egress.columns.get_loc('time1')] = traj_egress['time2'][:-1]
            traj_egress.at[traj_egress.index[0], 'time1'] = pd.to_datetime(traj['time2'].max())
            traj_egress = traj_egress.rename(columns={'from':'stop1','to':'stop2'})
            traj_egress = traj_egress[['trip_id','mode','stop1','stop2','tt_seconds','time1','time2']]
            traj_egress.loc[:,'status'] = 'success'
        except: pass
        # combine
        traj = pd.concat([traj_ingress, traj, traj_egress], ignore_index=True)
        traj = traj[['trip_id','mode','stop1','stop2','tt_seconds','time1','time2','status']]
        return(traj, True)

def integrate_all_traj(TRAJ_PATH, NEW_TRAJ_PATH, F_L_PATH):
    # create path if necessary
    if not os.path.exists(NEW_TRAJ_PATH):
        os.mkdir(NEW_TRAJ_PATH)
    # read all ingress and egress file
    all_ingress = pd.read_csv(os.path.join(F_L_PATH, 'ingress_traj.csv'))
    all_egress = pd.read_csv(os.path.join(F_L_PATH, 'egress_traj.csv'))
    # Follow OSM naming convention
    all_ingress['from'] = ['OSM_'+str(x) for x in all_ingress['from'].astype(np.int64)]
    all_ingress['to'] = ['OSM_'+str(x) for x in all_ingress['to'].astype(np.int64)]
    all_egress['from'] = ['OSM_'+str(x) for x in all_egress['from'].astype(np.int64)]
    all_egress['to'] = ['OSM_'+str(x) for x in all_egress['to'].astype(np.int64)]
    # iterate over trips
    ls_traj_dir = os.listdir(TRAJ_PATH)
    for p_traj in tqdm.tqdm(ls_traj_dir, total=len(ls_traj_dir)):
        # get trip ID:
        trip_id = p_traj.split('.')[0]
        if '_' not in trip_id: ##### rule to be modified 
            continue
        # read in the file, and formatting
        traj = pd.read_csv(os.path.join(TRAJ_PATH, p_traj))
        # get the corresponding ingress and egress
        traj_ingress = all_ingress[all_ingress['trip_id'] == trip_id]
        traj_egress = all_egress[all_egress['trip_id'] == trip_id]
        # run the integration
        new_traj, _ = combine_f_l_traj(traj, traj_ingress, traj_egress)
        # save new trajectory
        new_traj.to_csv(os.path.join(NEW_TRAJ_PATH, p_traj), index=False)
    return(None)