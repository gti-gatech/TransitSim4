# -*- coding: utf-8 -*-
"""
Created on Monday July, 2023

@author: Fizzy Huiying Fan
All rights reserved
"""

import os
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString

import geopandas as gpd
import geopy
from geopy.distance import geodesic 

import warnings
import time
import itertools
import argparse
warnings.filterwarnings('ignore')
import glob
from tqdm import tqdm
import argparse

import networkx as nx

from program import transit_links
from program import f_l_mile


def update_stop_id(gtfs_pre_dir, gtfs_post_dir, df_paths):
    # import reference stop files
    gtfs_pre = pd.read_csv(os.path.join(gtfs_pre_dir, 'stops.txt'))
    gtfs_post = pd.read_csv(os.path.join(gtfs_post_dir, 'stops.csv'))
    # formatting
    gtfs_pre = gtfs_pre[['stop_lat', 'stop_lon', 'stop_id']]
    gtfs_post = gtfs_post.rename(columns={'stop_id':'raptor_id'})
    # create a pair-up dictionary
    gtfs_pair = gtfs_pre.merge(gtfs_post, how='left', on=['stop_lat', 'stop_lon'])
    gtfs_pair = gtfs_pair[['raptor_id', 'stop_id']]
    #
    # Update df_paths
    df_paths['origin'] = df_paths['origin'].astype(np.float64)
    df_paths['destination'] = df_paths['destination'].astype(np.float64)
    # Update ID for origins
    df_paths = df_paths.rename(columns={'origin':'raptor_id'})
    df_paths = df_paths.merge(gtfs_pair, how='left', on='raptor_id')
    df_paths = df_paths.rename(columns={'stop_id':'origin'})
    df_paths = df_paths.drop(columns='raptor_id')
    # Update ID for destinations
    df_paths = df_paths.rename(columns={'destination':'raptor_id'})
    df_paths = df_paths.merge(gtfs_pair, how='left', on='raptor_id')
    df_paths = df_paths.rename(columns={'stop_id':'destination'})
    df_paths = df_paths.drop(columns='raptor_id')
    #
    return(df_paths)

def find_trajectory(DG, links, df_paths):
    ## Find trajectory for each segment of the trip
    df_traj = pd.DataFrame()
    for ind, row in df_paths.iterrows():
        if row.origin == row.destination:
            continue
        try:
            # From network, generate trajectory as a list
            trajectory = nx.shortest_path(DG, row.origin, row.destination, weight='weight')
            # Make a DF to start with
            df_traj_ent = pd.DataFrame({
                'trip_id': row.trip_id,
                'mode': row['mode'],
                'stop1': trajectory[:-1],
                'stop2': trajectory[1:],
                'tt_seconds': row.tt_seconds,
                'time1': row.time1,
                'time2': row.time2,
                'status': 'success'
            })
            # Join with links file to find weight in each segment, and use that to find TT in each segment 
            df_traj_ent = df_traj_ent.merge(links[['stop1','stop2','weight']], how='left', on=['stop1','stop2'])
            sum_weight = df_traj_ent['weight'].sum()
            df_traj_ent['weight'] = df_traj_ent['weight']/sum_weight
            df_traj_ent['tt_seconds'] = df_traj_ent['tt_seconds']*df_traj_ent['weight']
            # Find time 1 and time 2 for each segment
            df_traj_ent['time_elapsed'] = pd.to_timedelta(df_traj_ent['tt_seconds'].cumsum(), unit='s')
            df_traj_ent['time2'] = df_traj_ent['time1'] + df_traj_ent['time_elapsed']
            df_traj_ent['time1'][1:] = df_traj_ent['time2'][:-1]
            # Format, & join to the df_traj file
            df_traj_ent = df_traj_ent[['trip_id','mode','stop1','stop2','tt_seconds','time1','time2','status']]
            df_traj = pd.concat([df_traj, df_traj_ent], ignore_index=True)
        except:
            df_traj_ent = pd.DataFrame({
                'trip_id': row.trip_id,
                'mode': row['mode'],
                'stop1': row.origin,
                'stop2': row.destination,
                'tt_seconds': row.tt_seconds,
                'time1': row.time1,
                'time2': row.time2,
                'status': ['error']
            })
            df_traj = pd.concat([df_traj, df_traj_ent], ignore_index=True)
    return(df_traj)

def separate_paths_by_mode(df_paths, df_sum, trip_id):
    # Filter to retain only the first choice (shortest) route
    df_paths = df_paths[df_paths['choice'] == 1].reset_index()
    # include a status column
    df_paths['status'] = 'None'
    # Select paths for a single trip
    df_path = df_paths[df_paths['trip_id'] == trip_id]
    # Find departure time from the summary file
    dep_time = df_sum[df_sum['trip_id'] == trip_id].dep_time.tolist()
    # Obtain previous time from the former row
    df_path['prev_arr'] = dep_time + df_path['arrival_time'][:df_path.shape[0]-1].tolist()
    # Format columns
    df_path['prev_arr'] = [pd.to_datetime(x) for x in df_path['prev_arr']]
    df_path['arrival_time'] = [pd.to_datetime(x) for x in df_path['arrival_time']]
    # Calculate total time spent (wait & travel) in seconds
    df_path['total_seconds'] = (df_path['arrival_time'] - df_path['prev_arr'])/pd.to_timedelta(1, unit='s')
    # Wait time = total time - travel time 
    df_path['wait_seconds'] = df_path['total_seconds'] - df_path['tt_seconds']
    #
    ## Create all DF's
    # Create wait DF
    df_wait = df_path[df_path['mode'] == 'riding'].copy()
    df_wait['destination'] = df_wait['origin']
    df_wait['tt_seconds'] = df_wait['wait_seconds']
    df_wait['time1'] = df_wait['prev_arr']
    df_wait['time2'] = df_wait['prev_arr'] + pd.to_timedelta(df_wait['wait_seconds'], unit='s')
    df_wait['mode'] = 'waiting'
    df_wait = df_wait[['trip_id', 'mode', 'origin', 'destination', 'tt_seconds', 'time1', 'time2', 'status']]
    df_wait = df_wait.rename(columns={'origin':'stop1', 'destination':'stop2'})
    # Create walk DF
    df_walk = df_path[df_path['mode'] == 'walking'].copy()
    df_walk = df_walk[['trip_id', 'mode', 'origin', 'destination', 'tt_seconds', 'prev_arr', 'arrival_time', 'status']]
    df_walk = df_walk.rename(columns={'prev_arr': 'time1', 'arrival_time': 'time2'})
    # Create ride DF
    df_ride = df_path[df_path['mode'] == 'riding'].copy()
    df_ride['time2'] = df_ride['arrival_time']
    df_ride['time1'] = df_ride['arrival_time'] - pd.to_timedelta(df_ride['tt_seconds'], unit='s')
    df_ride = df_ride[['trip_id', 'mode', 'origin', 'destination', 'tt_seconds', 'time1', 'time2', 'status']]
    # return
    return(df_ride, df_walk, df_wait)

def find_samp_stops(samp_in, df_stops, walk_thres, which_smaller, silence=False):
    '''
    which_smaller: ['stops', 'sample']
    '''
    if samp_in.shape[0] == 0:
        return(samp_in)
    if not silence:
        print(' - Looking for stops for each sample...')
    ## First find stops that are within walkable distance (0.5 mile)
    # Make stations geospatial file
    df_stops = gpd.GeoDataFrame(df_stops, 
                                geometry=gpd.points_from_xy(df_stops.stop_lon, df_stops.stop_lat),
                                crs = 'EPSG:4269')
    o_samp = gpd.GeoDataFrame(samp_in, 
                              geometry=gpd.points_from_xy(samp_in.ori_lon, samp_in.ori_lat),
                              crs = 'EPSG:4269').copy()
    d_samp = gpd.GeoDataFrame(samp_in, 
                              geometry=gpd.points_from_xy(samp_in.dest_lon, samp_in.dest_lat),
                              crs = 'EPSG:4269').copy()
    df_stops = df_stops.to_crs('EPSG:2163')
    o_samp = o_samp.to_crs('EPSG:2163')
    d_samp = d_samp.to_crs('EPSG:2163')
    # Create 0.5 mile buffer for stops
    if not silence:
        print(' - Calculating walking distance and finding walkable stops near sample O & D...')
    if which_smaller == 'stops':
        df_stops_buff = df_stops.buffer(1609.344*walk_thres)
        df_stops_buff = gpd.GeoDataFrame(df_stops, geometry = df_stops_buff)
        # Find stops that are walkable to each sample O & D 
        o_samp = gpd.sjoin(o_samp, df_stops_buff, how='left', predicate='within')
        d_samp = gpd.sjoin(d_samp, df_stops_buff, how='left', predicate='within')
    else:
        o_samp_buff = o_samp.buffer(1609.344*walk_thres)
        o_samp_buff = gpd.GeoDataFrame(o_samp, geometry = o_samp_buff)
        d_samp_buff = d_samp.buffer(1609.344*walk_thres)
        d_samp_buff = gpd.GeoDataFrame(d_samp, geometry = d_samp_buff)
        # Find stops that are walkable to each sample O & D
        o_samp = gpd.sjoin(o_samp_buff, df_stops, how='left', predicate='contains')
        d_samp = gpd.sjoin(d_samp_buff, df_stops, how='left', predicate='contains')
    # Filter out nan and 0 values
    o_samp = o_samp[~((o_samp['stop_lat'].isnull()) | (o_samp['stop_lon'].isnull()))]
    o_samp = o_samp[~((o_samp['ori_lat'].isnull()) | (o_samp['ori_lon'].isnull()))]
    #
    d_samp = d_samp[~((d_samp['stop_lat'].isnull()) | (d_samp['stop_lon'].isnull()))]
    d_samp = d_samp[~((d_samp['ori_lat'].isnull()) | (d_samp['ori_lon'].isnull()))]
    # Calculate distance
    o_samp['dist'] = o_samp.apply(lambda x: geopy.distance.geodesic((x.stop_lat, x.stop_lon),
                                                                    (x.ori_lat, x.ori_lon)).miles, 
                                  axis=1)
    d_samp['dist'] = d_samp.apply(lambda x: geopy.distance.geodesic((x.stop_lat, x.stop_lon),
                                                                    (x.dest_lat, x.dest_lon)).miles, 
                                  axis=1)
    # Find the origin and destination stops, join to the original sample file
    if not silence:
        print(' - Select origin and destination stops...')
    o_samp.reset_index(drop = True, inplace = True)
    o_samp = o_samp.loc[o_samp.groupby('trip_id')['dist'].idxmin()]
    o_samp = o_samp[['trip_id', 'stop_id']].rename(columns = {'stop_id': 'o_stop'})
    #
    d_samp.reset_index(drop = True, inplace = True)
    d_samp = d_samp.loc[d_samp.groupby('trip_id')['dist'].idxmin()]
    d_samp = d_samp[['trip_id', 'stop_id']].rename(columns = {'stop_id': 'd_stop'})
    # 
    samp_in = samp_in.merge(o_samp, on = 'trip_id', how = 'left')
    samp_in = samp_in.merge(d_samp, on = 'trip_id', how = 'left')
    return(samp_in)

def revert_walk_to_osm(walk_nodes, df_walk, gtfsDir, DG_walk, walk_links, walk_thres):
    #
    ## 0. Some crazy formatting to make sure it aligns with the find_samp_stops function set-up
    # renaming walk_nodes to match the sample file format
    walk_nodes = walk_nodes[['osmid', 'x', 'y']].rename(columns={'osmid':'stop_id', 'x':'stop_lon', 'y':'stop_lat'})
    # get latitute and longtitude for walk file
    df_stops = pd.read_csv(os.path.join(gtfsDir, 'stops.txt'))[['stop_id', 'stop_lat', 'stop_lon']]
    df_walk = df_walk.merge(df_stops
                            .rename(columns={'stop_id':'origin','stop_lat':'ori_lat','stop_lon':'ori_lon'}), 
                            how='left', on='origin')
    df_walk = df_walk.merge(df_stops
                            .rename(columns={'stop_id':'destination','stop_lat':'dest_lat','stop_lon':'dest_lon'}), 
                            how='left', on='destination')
    # Make each record of output segment unique and name it "trip_id" (to match the function format)
    df_walk['trip_id_real'] = df_walk['trip_id']
    df_walk['trip_id'] = df_walk.apply(lambda x: str(x.origin) + '_' + str(x.destination), axis=1)
    #
    ## 1. Find corresponding OSM node to each GTFS stops in the output record
    df_walk = find_samp_stops(df_walk, walk_nodes, walk_thres, 'sample', silence=True)
    df_walk['trip_id'] = df_walk['trip_id_real']
    #
    ## 2. Create transit-walk network correspondence file
    df_corres1 = (df_walk[['trip_id', 'mode', 'origin', 'o_stop', 'tt_seconds', 'time1', 'time2', 'status']]
                  .rename(columns={'o_stop':'destination'}))
    df_corres1['destination'] = ['OSM_'+str(x) for x in df_corres1['destination'].astype(np.int64)]
    df_corres1['time2'] = df_corres1['time1']
    df_corres1['tt_seconds'] = 0
    #
    df_corres2 = (df_walk[['trip_id', 'mode', 'd_stop', 'destination', 'tt_seconds', 'time1', 'time2', 'status']]
                  .rename(columns={'d_stop':'origin'}))
    df_corres2['origin'] = ['OSM_'+str(x) for x in df_corres2['origin'].astype(np.int64)]
    df_corres2['time1'] = df_corres2['time2']
    df_corres2['tt_seconds'] = 0
    #
    df_corres = pd.concat([df_corres1, df_corres2], ignore_index=True)
    df_corres['mode'] = 'correspondence'
    df_corres = df_corres.rename(columns={'origin':'stop1', 'destination':'stop2'})
    #
    ## 3. Create final walk file
    df_walk = df_walk[['trip_id', 'mode', 'o_stop', 'd_stop', 'tt_seconds', 'time1', 'time2', 'status']]
    df_walk = df_walk.rename(columns={'o_stop':'origin', 'd_stop':'destination'})
    df_walk_traj = find_trajectory(DG_walk, walk_links, df_walk)
    # Format
    df_walk_traj['stop1'] = ['OSM_'+str(x) for x in df_walk_traj['stop1'].astype(np.int64)]
    df_walk_traj['stop2'] = ['OSM_'+str(x) for x in df_walk_traj['stop2'].astype(np.int64)]
    #
    return(df_walk_traj, df_corres)

def transit_links_shp(RAW_GTFS_PATH, PROC_GTFS_PATH, SHP_PATH,
                        route_map_db, 
                        buff_dist, space_count, min_true_screen, max_space_tol,
                        plot_links=True, spatial_filtered=False, activate_subsample=False):
    print('--------------- PREPARING INPUT FILES ---------------')
    # Import Samples
    df_stops, df_stop_times, df_trips, df_shapes = transit_links.import_data(RAW_GTFS_PATH, PROC_GTFS_PATH, route_map_db, spatial_filtered)
    # Subsample
    if activate_subsample:
        df_stop_times, df_trips = transit_links.sub_sample(df_stop_times, df_trips)
    print('--------------- REGISTER + SPACING ---------------')
    df_stop_times = transit_links.space_reg(buff_dist, space_count, min_true_screen, max_space_tol, df_stop_times, df_trips, df_shapes, df_stops)
    # --------------- Preparing Transit Links ---------------
    links_transit = transit_links.get_transit_links(df_stop_times)
    links_transit = transit_links.fill_links_dist_na(links_transit)
    links_transit['stop1'] = links_transit['stop1'].astype(np.float64).astype(np.int64)
    links_transit['stop2'] = links_transit['stop2'].astype(np.float64).astype(np.int64)
    if plot_links:
        links_transit = transit_links.fill_links_geo_na(links_transit, df_stops)
        if not os.path.exists(SHP_PATH):
            os.mkdir(SHP_PATH)
        transit_links.export_link_shp(links_transit, os.path.join(SHP_PATH, 'transit_links.shp'))
    return(links_transit)

def post_proc_traj(walk_nwDir, transit_nwDir, raptor_resDir, gtfs_pre_dir, gtfs_post_dir, output_traj_dir, walk_thres=0.21, link_version = 'processed'):
    print('====================== 1. Running ======================')
    # Read in walk links
    print(' --------------------------- Building transfer network --------------------------- ')
    walk_links = gpd.read_file(os.path.join(walk_nwDir, 'edges.shp'))
    walk_nodes = gpd.read_file(os.path.join(walk_nwDir, 'nodes.shp'))
    # Format
    walk_links = (walk_links[['from','to','length']]
                .rename(columns={'from':'stop1','to':'stop2', 'length':'weight'}))
    # Update walk_links to make it bi-directional
    walk_links_rev = walk_links.rename(columns={'stop1':'stop2', 'stop2':'stop1'})
    walk_links = pd.concat([walk_links,walk_links_rev], ignore_index=True).groupby(['stop1', 'stop2']).min().reset_index()
    # Build walking network
    DG_walk = nx.DiGraph()
    for ind, row in walk_links.iterrows():
        DG_walk.add_weighted_edges_from([(row['stop1'], row['stop2'], row['weight'])])

    # Read in transit links
    print(' --------------------------- Building transit network --------------------------- ')
    transit_links = gpd.read_file(os.path.join(transit_nwDir, 'transit_links.shp'))
    # Format
    transit_links = (transit_links[['stop1','stop2','t_time']]
                        .rename(columns={'t_time':'weight'}))
    # Build transit network
    DG_ride = nx.DiGraph()
    for ind, row in transit_links.iterrows():
        DG_ride.add_weighted_edges_from([(row['stop1'], row['stop2'], row['weight'])])
    # Import RAPTOR direct outputs
    df_paths = pd.read_csv(os.path.join(raptor_resDir, 'paths.csv'))
    df_sum = pd.read_csv(os.path.join(raptor_resDir, 'summ.csv'))
    # Fomrat df_paths to match the stop_id if needed
    if link_version == 'raw':
        df_paths = update_stop_id(gtfs_pre_dir, gtfs_post_dir, df_paths)
    #
    # iterate trip_id
    for trip_id in tqdm(np.unique(df_paths['trip_id']), total=len(np.unique(df_paths['trip_id']))):
        ## Generate trajectory files for each mode
        df_ride, df_walk, df_wait = separate_paths_by_mode(df_paths, df_sum, trip_id)
        df_ride_traj = find_trajectory(DG_ride, transit_links, df_ride)
        if df_walk.shape[0] == 0:
            df_walk_traj = pd.DataFrame()
            df_corres = pd.DataFrame()
        elif df_walk['tt_seconds'].sum() == 0:
            df_walk_traj = df_walk.rename(columns={'origin':'stop1', 'destination':'stop2'})
            df_corres = pd.DataFrame()
        else:
            try:
                df_walk_traj, df_corres = revert_walk_to_osm(walk_nodes, df_walk, gtfs_post_dir, DG_walk, walk_links, walk_thres)
            except: 
                df_walk_traj = df_walk.rename(columns={'origin':'stop1', 'destination':'stop2'})
                df_walk_traj['status'] = 'error: nan exists in DF'
                df_corres = pd.DataFrame()
        # Join all records together
        df_traj = pd.concat([df_ride_traj, df_walk_traj, df_wait, df_corres], ignore_index=True)
        # sort: first by time2, then by time1
        df_traj = df_traj.sort_values(by=['time2','time1'])
        # Saving
        if not os.path.exists(output_traj_dir):
            os.mkdir(output_traj_dir)
        df_traj.to_csv(os.path.join(output_traj_dir, trip_id+'.csv'))

def integrate_ingress_egress(RAW_NETWORK_PATH, PROC_GTFS_PATH, SAMP_INT_PATH, TRAJ_PATH, NEW_TRAJ_PATH, F_L_PATH,
                             snap_dist=0.5, dict_speed=None):
    if not os.path.exists(F_L_PATH):
        os.mkdir(F_L_PATH)
    f_l_mile.snap_sample_network(RAW_NETWORK_PATH, SAMP_INT_PATH, PROC_GTFS_PATH, snap_dist=0.5)
    f_l_mile.find_f_l_paths(RAW_NETWORK_PATH, SAMP_INT_PATH)
    f_l_mile.f_l_trajectories(RAW_NETWORK_PATH, SAMP_INT_PATH, TRAJ_PATH, F_L_PATH, dict_speed=None)
    f_l_mile.integrate_all_traj(TRAJ_PATH, NEW_TRAJ_PATH, F_L_PATH)
    return(None)