# -*- coding: utf-8 -*-
"""
Created on Monday July, 2023

@author: Fizzy Huiying Fan
All rights reserved
"""
import os
import glob
import time
from tqdm import tqdm
import argparse
import zipfile
#
import numpy as np
import pandas as pd
import math
from math import ceil
import pickle
#
import networkx as nx
#
import geopandas as gpd
import geopy
import geopy.distance


from program import gtfs_wrapper as gtfs_proc
from program import dict_builder_functions as gtfs_dict


def filter_trips_routes_ondays(valid_routes_set: set, calendar_dates, calendar, trips, day_type: int) -> tuple:
    """
    Filter the trips file based on calendar. Only One-days data is assumed here.

    Args:
        valid_routes_set (set): set containing valid route ids
        calendar_dates: GTFS Calendar_dates.txt file
        calendar: GTFS calendar.txt file
        trips: GTFS trips.txt file
        DATE_TOFILTER_ON (int): date on which GTFS set is filtered

    Returns:
        Filtered trips file and a set of valid trips and routes.

    Note:
        calendar_dates can be used in two sense. In the first case, it acts as a supplement to calendar.txt by defining listing the service id
        removed or added on a particular day (recommended usage).In the second case, it acts independently by listing all the service active
        on the particular day. See  GTFS reference for more details.
    """
    valid_service_id = set(calendar[calendar[f'{day_type}'] == 1].service_id)
    trips = trips[trips.service_id.isin(valid_service_id) & trips.route_id.isin(valid_routes_set)]
    valid_trips = set(trips.trip_id)
    valid_route = set(trips.route_id)
    print(f"After Filtering on day {day_type}")
    print(f"Valid trips:  {len(valid_trips)}")
    print(f"Valid routes:  {len(valid_route)}")
    return trips, valid_trips, valid_route

def filter_stoptimes(valid_trips: set, trips, DATE_TOFILTER_ON: int, stop_times) -> tuple:
    """
    Filter stoptimes file

    Args:
        valid_trips (set): GTFS set containing trips
        trips: GTFS trips.txt file
        DATE_TOFILTER_ON (int): date on which GTFS set is filtered
        stop_times: GTFS stoptimes.txt file

    Returns:
        Filtered stops mapping and stoptimes file
    """
    stop_times['arrival_time'] = [x.split(':')[0].zfill(2) + ':' + x.split(':')[1].zfill(2) + ':' + x.split(':')[2].zfill(2) for x in stop_times['arrival_time']]
    ################### MODIFIED
    print("Filtering stop_times.txt")
    stop_times.stop_sequence = stop_times.stop_sequence - 1
    stop_times.stop_id = stop_times.stop_id.astype(str)
    stop_times = stop_times[stop_times.trip_id.isin(valid_trips)]
    stop_times.loc[:, 'stop_sequence'] = stop_times.groupby("trip_id")["stop_sequence"].rank(method="first", ascending=True).astype(int) - 1

    stop_times = pd.merge(stop_times, trips, on='trip_id')
    stops_map = pd.DataFrame([t[::-1] for t in enumerate(set(stop_times.stop_id), 1)], columns=['stop_id', 'new_stop_id'])
    stop_times = pd.merge(stop_times, stops_map, on='stop_id').drop(columns=['stop_id']).rename(columns={'new_stop_id': 'stop_id'})
    print("Applying dates")
    DATE_TOFILTER_ON = f'{str(DATE_TOFILTER_ON)[:4]}-{str(DATE_TOFILTER_ON)[4:6]}-{str(DATE_TOFILTER_ON)[6:]}'
    last_stamp = stop_times.sort_values(by="arrival_time").arrival_time.iloc[-1]
    data_list = pd.date_range(DATE_TOFILTER_ON, periods=ceil(int(last_stamp.split(':')[0]) / 24))
    date_list = [data_list[int(int(x[:2]) / 24)] + pd.to_timedelta(str(int(x[:2]) - 24 * int(int(x[:2]) / 24)) + x[2:]) for x in tqdm(stop_times.arrival_time)]
    # nex_date = DATE_TOFILTER_ON[:-2] + str(int(DATE_TOFILTER_ON[-2:]) + 1)
    # date_list = [
    #     pd.to_datetime(DATE_TOFILTER_ON + ' ' + x) if int(x[:2]) < 24 else pd.to_datetime(nex_date + ' ' + str(int(x[:2]) - 24) + x[2:])
    #     for x in stop_times.arrival_time]
    stop_times['board_time'] = date_list ################### MODIFIED
    return stops_map, stop_times

def find_first_for_day(calendar, day_type):
    first_date = np.max(calendar['start_date'])
    last_date = np.min(calendar['end_date'])
    target_date = None
    for i in range(first_date, last_date+1):
        date_code = str(i)[:4]+'-'+str(i)[4:6]+'-'+str(i)[6:]
        day_name = pd.Timestamp(date_code).day_name()
        if day_name.lower() == day_type:
            target_date = i
            break
    return(target_date)

def filter_stops_ls(FILE_PATH, SAMP_PATH, stop_buff = 1):
    print('Filtering GTFS by spatial locations of the samples...')
    # import samples
    sample = pd.DataFrame()
    for p in glob.glob(os.path.join(SAMP_PATH, '*.csv')):
       entry = pd.read_csv(p)
       sample = pd.concat([sample, entry], axis=0)
    # process into unique points, set crs, buffer
    samp_o = sample[['ori_lat', 'ori_lon']].drop_duplicates()
    samp_o.columns = ['lat', 'lon']
    samp_d = sample[['dest_lat', 'dest_lon']].drop_duplicates()
    samp_d.columns = ['lat', 'lon']
    samp_locs = pd.concat([samp_o, samp_d])
    samp_locs = gpd.GeoDataFrame(samp_locs, 
                                 geometry=gpd.points_from_xy(samp_locs.lon, samp_locs.lat))
    samp_locs = samp_locs.set_crs('EPSG:4269')
    samp_locs = samp_locs.to_crs('EPSG:2163')
    samp_locs = gpd.GeoDataFrame(samp_locs, 
                                 geometry=samp_locs.buffer(1609.344*stop_buff))
    samp_locs = samp_locs.to_crs('EPSG:4269')
    # import stops, create geometry
    stops = pd.read_csv(os.path.join(FILE_PATH, 'stops.txt'))
    stops = stops[['stop_id', 'stop_lat', 'stop_lon']]
    stops = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat))
    stops = stops.set_crs('EPSG:4269')
    # intersect
    stops_buff = gpd.sjoin(stops, samp_locs, how='inner', op='within')
    ls_stops = np.unique(stops_buff['stop_id'])
    return(ls_stops)

def filter_GTFS_spatially(ls_stops, route, trips, stop_times, stops):
    n0 = trips.shape[0]
    # identify list of trips to keep
    stop_trip_ref = stop_times[['stop_id', 'trip_id']]
    ref_filtered = stop_trip_ref[stop_trip_ref['stop_id'].isin(ls_stops)]
    ls_trips = np.unique(ref_filtered['trip_id'])
    # Filter files 
    trips = trips[trips['trip_id'].isin(ls_trips)]
    route = route[route['route_id'].isin(np.unique(trips['route_id']))]
    stop_times = stop_times[stop_times['trip_id'].isin(ls_trips)]
    stops = stops[stops['stop_id'].isin(np.unique(stop_times['stop_id']))]
    # Done
    print('Filtered and retained ', round(trips.shape[0]/n0*100,1), '% of trips.')
    return(route, trips, stop_times, stops)

def gtfs_preprocessing(READ_PATH, SAVE_PATH, SAMP_PATH, VALID_ROUTE_TYPES, 
                        day_type='monday', DATE_TOFILTER_ON='', filter_spatial=True, stop_buff=1):
    calendar_dates, route, trips, stop_times, stops, calendar, _ = gtfs_proc.read_gtfs(READ_PATH)
    print('----------------------------------------')
    if filter_spatial:
        ls_stops = filter_stops_ls(READ_PATH, SAMP_PATH, stop_buff)
        route, trips, stop_times, stops = filter_GTFS_spatially(ls_stops, route, trips, stop_times, stops)
        print('----------------------------------------')
    valid_routes, route = gtfs_proc.remove_unwanted_route(VALID_ROUTE_TYPES, route)
    print('----------------------------------------')
    trips, valid_trips, valid_route = filter_trips_routes_ondays(valid_routes, calendar_dates, calendar, trips, day_type)
    print('----------------------------------------')
    if DATE_TOFILTER_ON == '':
        DATE_TOFILTER_ON = find_first_for_day(calendar, day_type)
    stops_map, stop_times = filter_stoptimes(valid_trips, trips, DATE_TOFILTER_ON, stop_times)
    print('----------------------------------------')
    stops = gtfs_proc.filter_stopsfile(stops_map, stops)
    print('----------------------------------------')
    route_map_db, stop_times, trips = gtfs_proc.rename_route(stop_times, trips)
    print('----------------------------------------')
    stop_times, trips = gtfs_proc.rename_trips(stop_times, trips)
    print('----------------------------------------')
    stop_times, trips = gtfs_proc.remove_overlapping_trips(stop_times, trips)
    print('----------------------------------------')
    gtfs_proc.check_trip_len(stop_times)
    print('----------------------------------------')
    stop_times = gtfs_proc.stoptimes_filter(stop_times)
    print('----------------------------------------')
    trips, stop_times, stops = gtfs_proc.filter_trips(trips, stop_times, stops)
    print('----------------------------------------')
    gtfs_proc.save_final(SAVE_PATH, trips, stop_times, stops)

    first_date = np.max(calendar['start_date'])
    last_date = np.min(calendar['end_date'])
    target_date = None
    for i in range(first_date, last_date+1):
        if int(str(i)[6:]) > 30:
            continue
        if int(str(i)[6:]) == 0:
            continue
        date_code = str(i)[:4]+'-'+str(i)[4:6]+'-'+str(i)[6:]
        day_name = pd.Timestamp(date_code).day_name()
        if day_name.lower() == day_type:
            target_date = i
            break
    return(target_date, route_map_db)


def obtain_bpoly(stops, node_stop_thres):
  # Format stops file
  stop_gis = stops.copy()
  stop_gis = gpd.GeoDataFrame(stop_gis, geometry = gpd.points_from_xy(stop_gis.stop_lon, stop_gis.stop_lat))
  # Create convex hull bounding for all stops
  stop_bound = pd.DataFrame({'geometry':[stop_gis.unary_union.convex_hull]})
  stop_bound = gpd.GeoDataFrame(stop_bound, geometry=stop_bound.geometry)
  stop_bound = stop_bound.set_crs('EPSG:4269')
  stop_bound = stop_bound.to_crs('EPSG:2163')
  stop_bound = gpd.GeoDataFrame(stop_bound, geometry=stop_bound.buffer(1609.344*node_stop_thres))
  stop_bound = stop_bound.to_crs('EPSG:4326')
  return(stop_bound)

def find_nearby_stops(df_stops, walk_thres):
  print(' - Looking for nearby stops...')
  ## First find stops that are within walkable distance (0.5 mile)
  # Make stations geospatial file
  df_stops = gpd.GeoDataFrame(df_stops, 
                              geometry=gpd.points_from_xy(df_stops.stop_lon, df_stops.stop_lat),
                              crs = 'EPSG:4269')
  df_stops = df_stops.to_crs('EPSG:2163')
  # Create 0.5 mile buffer for stops
  df_stops_buff = df_stops.buffer(1609.344*walk_thres)
  df_stops_buff = gpd.GeoDataFrame(df_stops, geometry = df_stops_buff)
  # Find stops that are walkable to each other 
  df_pairs = gpd.sjoin(df_stops, df_stops_buff, how='inner', predicate='within')
  # Filter out all those that are to itself
  df_pairs = df_pairs[df_pairs['stop_id_left'] != df_pairs['stop_id_right']]
  df_pairs = df_pairs.rename(columns={'stop_id_left':'stop1', 'stop_id_right':'stop2'})
  return(df_pairs)

def filter_same_route(df_trips, df_stop_times, df_pairs):
  print(' - Filtering out stops that are on the same routes...')
  ## Find stops that are on the same route (no transfer needed)
  # Find routes associated with each stop
  df_trips = df_trips[['route_id', 'trip_id']]
  df_trips = df_trips.merge(df_stop_times[['trip_id','stop_id']].drop_duplicates(), how='left', on='trip_id')
  df_trips = df_trips[['route_id', 'stop_id']].drop_duplicates()
  #
  route_ref = pd.DataFrame(df_trips.groupby('stop_id')['route_id'].unique())
  route_ref.reset_index(inplace=True)
  # Join this information to the stops pairs DF, and keep only those that are not on the same route in any circumstances
  route_ref['stop_id'] = route_ref['stop_id'].astype(type(df_pairs['stop1'].iloc[0]))
  df_pairs = (df_pairs
              .merge(route_ref
                      .rename(columns={'stop_id':'stop1','route_id':'route1'}), 
                      how='inner', on='stop1')
              .merge(route_ref
                      .rename(columns={'stop_id':'stop2','route_id':'route2'}), 
                      how='inner', on='stop2'))
  df_pairs['same_route'] = df_pairs.apply(lambda x: sum([y in x.route1 for y in x.route2]) > 0, axis=1)
  df_pairs = df_pairs[df_pairs['same_route'] == False]
  return(df_pairs)

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
    for ind, row in tqdm(samp_in.iterrows(), total=samp_in.shape[0]):
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

def get_transfer(FILE_PATH: str,
                 walk_threshold: float = 0.20833333,
                 network_prep: bool = False,
                 network_path: str = '',
                 breaker: str = '---',
                 node_stop_thres = 0.5,
                 data: set = None) -> pd.DataFrame:
    """
    Create transfer pairs with transitive closure feature

    Args:
    FILE_PATH (str): path to all the files, both for both import (stops, trips, stop_times, nodes, edges) and export (transfer).
    walk_threshold (float): maximum allowable walk transfer distance in mile. Default is 0.21 mile, or 5 minutes walk at 2.5 mph.
    network_prep (bool): whether the network has been prepared. If False, the function will prepare the network based on the stops location. Default is False.
    network_path (str): path to read and save the network files. If not provided, the default is the same as FILE_PATH.
    breaker (str): a string separating diffrent parts of the function. Default is '---'.
    node_stop_thres (float): maximum allowable buffer distance to relate a node to a stop in mile. Default is 0.5 mile.
    data (set): input data, if provided, the data will be used directly in the function instead of import from FILE_PATH. Default is None.

    Returns:
    out (pd.DataFrame): pandas dataframe of transfer distance in miles, columns: stop1, stop2, dist.

    Examples:
    >>> transfers = get_transfer(FILE_PATH, 0.1)
    >>> transfers = get_transfer(FILE_PATH, 0.1, False, data = (stops, trips, stop_times))
    >>> transfers = get_transfer(FILE_PATH, 0.1, True, data = (stops, trips, stop_times, nodes, edges))
    """
    t0 = time.time()

    ## Import files
    t0_import = time.time()
    if network_path == '':
        network_path = FILE_PATH
    if data == None:
        stops = pd.read_csv(os.path.join(FILE_PATH, 'stops.txt'))
        trips = pd.read_csv(os.path.join(FILE_PATH, 'trips.txt'))
        stop_times = pd.read_csv(os.path.join(FILE_PATH, 'stop_times.txt'))
    else:
        stops = data[0]
        trips = data[1]
        stop_times = data[2]
    print('Time used for importing files: ', round((time.time()-t0_import)/60, 2), ' minutes.')
    print(breaker)

    ## Prepare the network, if it hasn't been done already
    t0_network = time.time()
    if not network_prep:
        print(' - preparing network...')
        # Find the appropriate bounding polygon for the stops
        stop_bound = obtain_bpoly(stops, node_stop_thres)
        # Grab network from OSM
        network = ox.graph_from_polygon(stop_bound['geometry'][0], network_type='walk')
        # Save
        ox.save_graph_shapefile(network, filepath=network_path)
        nodes = gpd.read_file(os.path.join(network_path, 'nodes.shp'))
        edges = gpd.read_file(os.path.join(network_path, 'edges.shp'))
    elif data == None or len(data) == 3:
        nodes = gpd.read_file(os.path.join(network_path, 'nodes.shp'))
        edges = gpd.read_file(os.path.join(network_path, 'edges.shp'))
    else:
        nodes = data[3]
        edges = data[4]
    print('Time used for preparing network: ', round((time.time()-t0_network)/60, 2), ' minutes.')
    print(breaker)

    ## Obtain possible transfer pairs
    t0_pairs = time.time()
    pairs = find_nearby_stops(stops, walk_threshold)
    pairs = filter_same_route(trips, stop_times, pairs)
    print('Time used for finding possible transfer pairs: ', round((time.time()-t0_pairs)/60, 2), ' minutes.')
    print(breaker)

    ## Find corresponding stops and nodes
    t0_stop_node = time.time()
    stop_node = filter_nodes_by_stop(stops, nodes, node_stop_thres) ##### edit!
    print('Time used for stops and nodes correspondence: ', round((time.time()-t0_stop_node)/60, 2), ' minutes.')
    print(breaker)

    ## A bit of formatting
    t0_formatting = time.time()
    stop_node.reset_index(drop=True, inplace=True)
    #
    pairs['index'] = range(0,pairs.shape[0])
    pairs = pairs[['index','stop1','stop2']]
    pairs = pairs.merge(stop_node.rename(columns={'stop_id':'stop1', 'osmid':'node1'}), how='left', on='stop1')
    pairs = pairs.merge(stop_node.rename(columns={'stop_id':'stop2', 'osmid':'node2'}), how='left', on='stop2')
    #
    edges = edges[['from','to','length']]
    # length is in meters
    # Since it's walk links, assuming everything reversable
    edges_rev = edges.copy()
    edges_rev['from'] = edges['to']
    edges_rev['to'] = edges['from']
    edges = pd.concat([edges, edges_rev])
    edges = edges.groupby(['from','to']).mean().reset_index()
    print('Time used for formatting: ', round((time.time()-t0_formatting)/60, 2), ' minutes.')
    print(breaker)

    ## Create the transfer paths
    t0_transfer = time.time()
    transfers = shortest_path_finder(pairs, edges)
    print(transfers.shape[0]) ###############################################
    transfers = transfers[transfers['dist'] != 99999] ###############################################
    print(transfers.shape[0]) ###############################################
    print('Time used for creating transfer: ', round((time.time()-t0_transfer)/60, 2), ' minutes.')
    print(breaker)

    ## create network graph
    t0_transclos = time.time()
    time_start = time.time()
    DGo = nx.Graph()  # create directed graph
    for ind, row in transfers.iterrows(): #transfers_anls
        DGo.add_weighted_edges_from([(str(row['stop1']), str(row['stop2']), float(row['dist']))],weight='distance')  
    # get transitive closure of graph
    DGofinal = nx.transitive_closure(DGo,reflexive=None)
    # get edge list
    transfer_output = nx.to_pandas_edgelist(DGofinal)
    # rename columns
    transfer_output.columns = ['stop1','stop2','dist'] # dist in meters
    transfer_output['dist'] = transfer_output['dist']/1609.344 # convert to miles
    print('Time used for transitive closure: ', round((time.time()-t0_transclos)/60, 2), ' minutes.')
    print(breaker)

    ## Find shortest travel time for transfer_output
    t0_final_time = time.time()
    for ind, row in tqdm(transfer_output.iterrows(), total=transfer_output.shape[0]):
        if row.dist > 0:
            continue
        else:
            try:
                dist = nx.shortest_path_length(DGo, row.stop1, row.stop2, weight='distance')
                transfer_output.loc[ind, 'dist'] = dist
            except:
                continue
    print('Time used for finalizing transfer distances: ', round((time.time()-t0_final_time)/60, 2), ' minutes.')
    print(breaker)

    ##
    print(f'TOTAL time used for {walk_threshold} miles transfer: {round(((time.time() - t0)/60), 2)} minutes')
   
    return(transfer_output)


def build_dict(GTFS_PATH, TRANSFER_PATH, SAVE_PATH):
    _ = gtfs_dict.build_save_route_by_stop(GTFS_PATH, SAVE_PATH)
    print('----------------------------------------')
    _ = gtfs_dict.build_save_stops_dict(GTFS_PATH, SAVE_PATH)
    print('----------------------------------------')
    _ = gtfs_dict.build_save_stopstimes_dict(GTFS_PATH, SAVE_PATH)
    print('----------------------------------------')
    p_transfer = os.path.join(TRANSFER_PATH, 'transfer.csv')
    _ = gtfs_dict.build_save_footpath_dict(p_transfer, SAVE_PATH)
    # user can enter transfer walk speed after SAVE_PATH. If not provided, default is 2.5
    print('----------------------------------------')
    _ = gtfs_dict.stop_idx_in_route(GTFS_PATH, SAVE_PATH)
    return(None)