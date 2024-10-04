# -*- coding: utf-8 -*-
"""
Created on Monday May 30, 2022

@author: Fizzy Huiying Fan
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

import networkx as nx

from program import raptor_preproc

### ----------------------- SECTION 1 -----------------------

def import_data(RAW_GTFS_PATH, PROC_GTFS_PATH, route_map_db, spatial_filtered=False): 

    # Import Data
    df_stops = pd.read_csv(os.path.join(PROC_GTFS_PATH, 'stops.txt'))
    df_stop_times = pd.read_csv(os.path.join(PROC_GTFS_PATH, 'stop_times.txt'))
    df_trips = pd.read_csv(os.path.join(PROC_GTFS_PATH, 'trips.txt'))
    df_shapes = pd.read_csv(os.path.join(RAW_GTFS_PATH, 'shapes.txt'))
    # Join shape ID back to the trips file
    df_trips_raw = pd.read_csv(os.path.join(RAW_GTFS_PATH, 'trips.txt'))[['trip_id', 'shape_id']]
    route_shape = route_map_db.merge(df_trips_raw, how='left', on='trip_id')
    route_shape = route_shape[['new_route_id','shape_id']].drop_duplicates()
    route_shape = route_shape.rename(columns={'new_route_id':'route_id'})
    df_trips = df_trips.merge(route_shape, how='left', on='route_id')
    # Filter shapes file to retain only the ones in the scope
    df_shapes = df_shapes[df_shapes['shape_id'].isin(np.unique(df_trips['shape_id']))]
    
    return(df_stops, df_stop_times, df_trips, df_shapes)


def sub_sample(df_stop_times, df_trips):
    ## Sub-sampling when the file is too large
    if df_stop_times.shape[0] > 1000000:
        print(' - Yes, subsampled')
        try:
            df_trips = (df_trips
                        .groupby("route_id", group_keys=False)
                        .apply(lambda df: df.sample(n=10, random_state=233, replace=False)))
        except:
            df_trips = (df_trips
                        .groupby("route_id", group_keys=False)
                        .apply(lambda df: df.sample(n=10, random_state=233, replace=True)))
            df_trips = df_trips.drop_duplicates()
        df_stop_times = df_stop_times[df_stop_times['trip_id'].isin(df_trips['trip_id'].unique())]
    return(df_stop_times, df_trips)

def spacing(df_shape, n_nodes):
    # Time used for spacing function (compared to number of rows):
    # - 1k ~ 0.8s
    # - 5k ~ 4.2s
    # - 10k ~ 10.0s
    # - 20k ~ 29.2s
    # - 30k ~ 58.0s
    # - 50k ~ 140.5s
    # Set the max. procesing time to be about half a minute (limit n rows to 20k)
    df_shape_out = pd.DataFrame()
    df_shape.reset_index(inplace=True, drop=True)
    for ind, row in df_shape.iloc[0:len(df_shape) - 1, :].iterrows():
        # the next row
        row_shape_pos = df_shape.loc[ind + 1, :]
        # Create 10 additional shape points between every adjacent pair
        lats_pos = np.linspace(row['shape_pt_lat'], row_shape_pos['shape_pt_lat'], n_nodes+1)
        lons_pos = np.linspace(row['shape_pt_lon'], row_shape_pos['shape_pt_lon'], n_nodes+1)
        shape_pt_sequence_pos = np.linspace(row['shape_pt_sequence'], row_shape_pos['shape_pt_sequence'], n_nodes+1)
        df_shapes_toAppend = pd.DataFrame(
            {'shape_id': df_shape['shape_id'].unique()[0], 
             'shape_pt_lat': lats_pos, 'shape_pt_lon': lons_pos, 
             'shape_pt_sequence': shape_pt_sequence_pos})
        df_shape_out = df_shape_out.append(df_shapes_toAppend)
    df_shape_out = gpd.GeoDataFrame(df_shape_out, 
                                 geometry=gpd.points_from_xy(df_shape_out.shape_pt_lon, 
                                                             df_shape_out.shape_pt_lat),
                                 crs = 'EPSG:4269')
    df_shape_out = df_shape_out.to_crs('EPSG:2163')
    df_shape_out.reset_index(drop=True, inplace=True)
    return(df_shape_out)

def back_space(df_time, df_shape, max_space_tol, buff_dist):
    def buffer(df_time, dist):
        # Create new buffer zone by the maximum tolerance 
        df_time_buf = df_time.copy()
        df_time_buf = gpd.GeoDataFrame(df_time_buf,  
                                       geometry = gpd.points_from_xy(df_time_buf.stop_lon, 
                                                                     df_time_buf.stop_lat), 
                                       crs = 'EPSG:4269')
        df_time_buf = df_time_buf.to_crs('EPSG:2163')
        df_time_buf = gpd.GeoDataFrame(df_time_buf, 
                                       geometry=df_time_buf.buffer(dist))
        return(df_time_buf)
    #
    dist = 0
    uniq_ls = [0]
    df_shape_sel = pd.DataFrame()
    while (len(uniq_ls) == len(df_time['stop_sequence'].unique()) and min(uniq_ls)) == False:
        dist += 30
        if dist > 1609.344*max_space_tol/2 or df_shape_sel.shape[0] > 100000: 
            # Time needed to calculate distance for dataframe with different number of rows:
            # To control the processing time below 30s:
            # 10k ~ 1.6s
            # 20k ~ 3.2s
            # 50k ~ 7.7s
            # 100k ~ 15.4s
            # 200k ~ 31.4s
            return(' - Shape points too far away to be accurate.')
        df_time_buf = buffer(df_time, dist)
        df_shape_sel = gpd.sjoin(df_shape, df_time_buf, how='inner', predicate='within')
        uniq_ls = df_shape_sel.groupby('stop_sequence').size().tolist()
    #
    df_shape_sel['dist'] = df_shape_sel.apply(lambda x: geopy.distance.geodesic((x.stop_lat, x.stop_lon),
                                                                                (x.shape_pt_lat, x.shape_pt_lon)).meters, 
                                              axis=1)
    selected_ind = pd.DataFrame(df_shape_sel.groupby('stop_sequence')['dist'].nsmallest(3)).reset_index()
    try:
        df_shape_add = df_shape_sel.loc[selected_ind['level_1']].drop(columns='dist')
    except: 
        df_shape_add = df_shape_sel.loc[selected_ind['index']].drop(columns='dist')

    return(df_shape_add)

def filter_mismatched_shapes(df_shape_sel, min_true_screen):
    # Summarize stops that have miscoded shapes
    screen_shape = pd.DataFrame(df_shape_sel.groupby('stop_sequence')['shape_pt_sequence'].max())
    screen_shape = screen_shape.rename(columns={'shape_pt_sequence':'max_shape'})
    screen_shape['min_shape'] = df_shape_sel.groupby('stop_sequence')['shape_pt_sequence'].min().tolist()
    #
    screen_shape['max_shape_prev'] = screen_shape['max_shape'].shift(1)
    screen_shape['error'] = screen_shape.apply(lambda x: 0 if x['min_shape'] > x['max_shape_prev'] else 1, axis = 1)
    
    # If there is no error, return the original DF
    if sum(screen_shape['error']) == 1:
        return(df_shape_sel)
    
    # Otherwise, find the continuous min_true_screen stops that don't have any error, and use it as a benchmark point
    pos = 9999
    for ind, row in screen_shape.iterrows():
        if sum(screen_shape.loc[ind:(ind+min_true_screen-1),'error']) == 0:
            pos = ind
            break
            
    # If such group can't be found, return an error
    if pos == 9999:
        error = ' - Too much spatial information mismatched.'
        return(error)
    
    # Otherwise, process the entire dataframe from there
    # For the continuous error-free stops group, keep them as a starting point
    df_shape_match = df_shape_sel[df_shape_sel['stop_sequence'].isin(range(pos,(pos+min_true_screen)))]
    
    # For each of the stop before the group, filter out those larger than a latter stop's minimum
    for i in range(pos-1):
        try:
            df_shape_entry = df_shape_sel[df_shape_sel['stop_sequence'] == pos-i-1]
            min_shape_seq = min(df_shape_sel.loc[df_shape_sel['stop_sequence'] == pos-i, 'shape_pt_sequence'])
            df_shape_entry = df_shape_entry[df_shape_entry['shape_pt_sequence'] < min_shape_seq]
            df_shape_match = df_shape_match.append(df_shape_entry)
        except: continue # Some trips got messy stop sequence, ignore those
        
    # For each of the stop after the group, filter out those smaller than a former stop's maximum
    for i in range(pos+min_true_screen, df_shape_sel['stop_sequence'].max()+1):
        try:
            df_shape_entry = df_shape_sel[df_shape_sel['stop_sequence'] == i]
            max_shape_seq = max(df_shape_sel.loc[df_shape_sel['stop_sequence'] == i-1, 'shape_pt_sequence'])
            df_shape_entry = df_shape_entry[df_shape_entry['shape_pt_sequence'] > max_shape_seq]
            df_shape_match = df_shape_match.append(df_shape_entry)
        except: continue # Some trips got messy stop sequence, ignore those
        
    return(df_shape_match)

def find_distance(df_shape):
    
    # Delete the last couple NA's
    max_ind = max(df_shape[~df_shape['stop_sequence'].isna()].index)
    df_shape = df_shape.loc[0:max_ind]

    # Process all entries in df_shape to group them together
    df_shape['stop_sequence'] = df_shape['stop_sequence'].interpolate(method='pad')
    df_shape['stop_sequence'] = df_shape['stop_sequence'].shift(1)
    df_shape = df_shape[~df_shape['stop_sequence'].isna()]

    # Reformat shape file if necessary
    inc_ind = pd.DataFrame(df_shape.groupby('stop_sequence').size()).rename(columns={0:'times'})
    inc_ls = inc_ind[inc_ind['times'] < 2].index
    if len(inc_ls) > 0:
        for i in inc_ls:
            add_ind = df_shape[df_shape['stop_sequence'] == i].index[0]
            try:
                entry = df_shape.loc[[add_ind-1, add_ind+1]]
            except:
                entry = df_shape.loc[[add_ind-1]]
            entry['stop_sequence'] = i
            df_shape = pd.concat([df_shape, entry], axis=0)
            df_shape = df_shape.sort_values('shape_pt_sequence')
    # Transform stops to lines and calculate distance
    df_shape = df_shape.groupby('stop_sequence')['geometry'].apply(lambda x: LineString(x.tolist()))
    df_shape = gpd.GeoDataFrame(df_shape.length, geometry = df_shape).reset_index()

    # Reformat df_shape and combine to the time DF
    df_shape = df_shape.rename(columns={0:'dist'})
    df_shape['stop_sequence'] = df_shape['stop_sequence'] + 1
    df_shape['dist'] = df_shape['dist']*0.000621371192
    df_shape['dist'] = df_shape['dist'].cumsum()
    
    return(df_shape)


def space_reg(buff_dist, space_count, min_true_screen, max_space_tol, df_stop_times, df_trips, df_shapes, df_stops):

    ## All input files
    # Create the time sequence of stops
    df_stop_times = df_stop_times[['trip_id', 'arrival_time', 'board_time', 
                                'stop_id', 'stop_sequence']]
    # Create a unique list of shapes ID and stops features
    trip_shape_dict = df_trips[['trip_id','shape_id']].drop_duplicates()
    df_times = df_stop_times.merge(trip_shape_dict, how='left', on='trip_id')
    df_times = df_times.merge(df_stops[['stop_id', 'stop_lat', 'stop_lon']], how='left', on='stop_id')
    df_times = df_times[['shape_id', 'stop_id', 'stop_lat', 'stop_lon', 'stop_sequence']].drop_duplicates()
    df_times = df_times[~df_times['stop_lat'].isna()]
    #
    df_times = gpd.GeoDataFrame(df_times, 
                                geometry=gpd.points_from_xy(df_times.stop_lon, df_times.stop_lat),
                                crs = 'EPSG:4269')
    df_times = df_times.to_crs('EPSG:2163')
    df_times = gpd.GeoDataFrame(df_times, 
                                geometry=df_times.buffer(buff_dist))
    # Create the spatial (shape) sequence of stops
    df_shapes = gpd.GeoDataFrame(df_shapes, 
                                geometry=gpd.points_from_xy(df_shapes.shape_pt_lon, df_shapes.shape_pt_lat),
                                crs = 'EPSG:4269')
    df_shapes = df_shapes.to_crs('EPSG:2163')


    # Register for each trip
    df_time_all = pd.DataFrame()
    for shape_id in df_times['shape_id'].unique():
        
        # Subset time and space sequence for a single trip
        df_time = df_times[df_times['shape_id'] == shape_id]
        df_shape = df_shapes[df_shapes['shape_id'] == shape_id]
        
        # Return error only one stop present
        if df_time.shape[0] < 2:
            print(shape_id, ' - Less than two stops in the dataset, cannot form a link.')
            continue
        
        # Find corresponding shapes for each time stop, iterate until all are matched
        df_shape_sel = gpd.sjoin(df_shape, df_time, how='inner', predicate='within')
        # Try spacing 
        while len(np.unique(df_shape_sel['stop_sequence'])) < len(np.unique(df_time['stop_sequence'])):
            # If spacing will take over 30 seconds, 
            # try a less rigorous and more complicated algorithm (but will probably take less time than spacing)
            if df_shape.shape[0] >= 2000: ### used to be 20000
                df_time_missing = df_time[~df_time['stop_sequence'].isin(df_shape_sel['stop_sequence'].unique())]
                df_shape_add = back_space(df_time_missing, df_shape, max_space_tol, buff_dist)
                # Error
                if type(df_shape_add) == str:
                    df_shape_sel = df_shape_add
                else:
                    df_shape_sel = pd.concat([df_shape_sel, df_shape_add], axis=0)
                    df_shape_sel = df_shape_sel.sort_values('stop_sequence')
                break
            # Normal spacing procedure - decision brought before the main code chunk
            df_shape = spacing(df_shape, space_count)
            df_shape_sel = gpd.sjoin(df_shape, df_time, how='inner', predicate='within')
        # We can't tolerate what's beyond the maximum tolerance (e.g., 1-mile). 
        # If it still doesn't work at this level, return an error message
        if type(df_shape_sel) == str:
            print(shape_id, df_shape_sel)
            continue
        
        # Filter out those shape points that are misordered
        df_shape_sel = filter_mismatched_shapes(df_shape_sel, min_true_screen)
        if type(df_shape_sel) == str:
            print(shape_id, df_shape_sel)
            continue
            
        # Calculate distance between stops and shapes points, select the minimum for each stop and match back to df_shape
        df_shape_sel['dist'] = df_shape_sel.apply(lambda x: geopy.distance.geodesic((x.stop_lat, x.stop_lon),
                                                                                    (x.shape_pt_lat, x.shape_pt_lon)).meters, 
                                                axis=1)
        df_shape_sel = df_shape_sel.loc[df_shape_sel.groupby('stop_sequence').dist.idxmin()]
        df_shape = df_shape.merge(df_shape_sel[['shape_pt_sequence', 'stop_sequence']], how='left', on='shape_pt_sequence')
        if len(df_shape[~df_shape['stop_sequence'].isna()]['stop_sequence'].unique()) < 2:
            print(shape_id, ' - Less than two valid stops in the dataset, cannot form a link.')
            continue
        
        # Find travel distance and travel time
        df_shape = find_distance(df_shape)
        df_time = (df_time[['shape_id','stop_id','stop_lat','stop_lon','stop_sequence']]
                   .merge(df_shape, how='left', on='stop_sequence'))
        
        # Combine to the master DF
        df_time_all = df_time_all.append(df_time)
        # print(shape_id, ' - Success.')

    # df_time_all['dist'] = df_time_all['dist'].fillna(0)
    df_stop_times = df_stop_times.merge(trip_shape_dict, on='trip_id', how='left')
    df_stop_times = df_stop_times.merge(df_time_all[['shape_id', 'stop_id', 'stop_sequence', 'dist', 'geometry']], 
                                        on=['shape_id','stop_id', 'stop_sequence'], 
                                        how='left').drop_duplicates()
    
    return(df_stop_times)


### ----------------------- SECTION 2 -----------------------

def get_transit_links(df_stop_times):
    print('--------------- GENERATING TRANSIT LINKS ---------------')
    links = pd.DataFrame()
    for trip_id in df_stop_times['trip_id'].unique():
        # subset the stop time records
        df_trip = df_stop_times[df_stop_times['trip_id'] == trip_id]
        # format arrival time to number of seconds
        df_trip['arrival_time'] = (pd.to_timedelta(df_trip['arrival_time'])
                                   .astype('timedelta64[s]')
                                   .astype(float))
        # Put an 0 in the first place of a trip sequence
        df_trip = df_trip.sort_values('stop_sequence')
        df_trip.iloc[0, df_trip.columns.get_loc('dist')] = 0
        # if there's NA, interpolate linearly with nearby entries
        if sum(df_trip['arrival_time'].isna()):
            df_trip['arrival_time'] = df_trip['arrival_time'].interpolate()
        if sum(df_trip['dist'].isna()):
            df_trip['dist'] = df_trip['dist'].interpolate()
        # make stops into links of two stops
        df_trip = (df_trip[['trip_id', 'arrival_time', 'dist', 'stop_id', 'stop_sequence', 'geometry']].
                   rename(columns = {'arrival_time': 'time2', 'dist': 'dist2', 'stop_id': 'stop2'}))
        df_trip['stop1'] = df_trip['stop2'].shift(1)
        df_trip['time1'] = df_trip['time2'].shift(1)
        df_trip['dist1'] = df_trip['dist2'].shift(1)
        # df_trip = df_trip.dropna(axis=0) 
        df_trip['time'] = (df_trip['time2'] - df_trip['time1']) / 60
        df_trip['dist'] = (df_trip['dist2'] - df_trip['dist1']) 
        # format a little bit
        df_trip = df_trip[['trip_id','stop1','stop2','time', 'dist', 'geometry']]
        df_trip['stop1'] = df_trip['stop1'].astype(str)
        df_trip['stop2'] = df_trip['stop2'].astype(str)
        # Add that into the links df
        links = links.append(df_trip)
    # Format links
    links = links[links['stop1'] != 'nan']
    # Get unique stop paris
    links_time = pd.DataFrame(links.groupby(['stop1', 'stop2'])['time'].unique())
    links_dist = pd.DataFrame(links.groupby(['stop1', 'stop2'])['dist'].unique())
    links_geo = pd.DataFrame(links.groupby(['stop1', 'stop2'])['geometry'].unique())
    # Get the average travel time and average travel distance
    links_time['t_time'] = links_time.apply(lambda x: np.mean(x.time), axis=1)
    links_dist['t_dist'] = links_dist.apply(lambda x: np.mean(x.dist), axis=1)
    # Get the Coefficient of Variance of times, if CV > 1, the row may need a second examination
    links_time['time_cv'] = links_time.apply(lambda x: np.std(x.time)/x.t_time, axis=1)
    links_dist['dist_cv'] = links_dist.apply(lambda x: np.std(x.dist)/x.t_dist, axis=1)
    # Process geometry: get the number of unique geometry for trouble shooting, and keep the first one only
    links_geo['n_uniq_geo'] = links_geo.apply(lambda x: len(x.geometry), axis=1)
    links_geo['geometry'] = links_geo.apply(lambda x: x.geometry[0] if len(x.geometry)>0 else None, axis=1)
    # Formatting...
    links = links_time
    links[['t_dist', 'dist_cv']] = links_dist[['t_dist', 'dist_cv']]
    links['geometry'] = links_geo['geometry']
    links.reset_index(inplace = True)
    print("Warning: ", links[links['time_cv']>0.5].shape[0], " trips have high time variation.")
    print("Warning: ", links[links['dist_cv']>0.5].shape[0], " trips have high distance variation.")
    print("Warning: ", links_geo[links_geo['n_uniq_geo'] > 1].shape[0], " trips have different geometries.")
    links = links[['stop1','stop2','t_time', 't_dist', 'geometry']]
    links['type'] = 'ride'
    return(links)


def get_transfer_links(df_stop_times, df_stops, df_trips, walk_speed, walk_thres):
    #### Calculate Transfer Wait Time
    print('--------------- GENERATING TRANSFER LINKS ---------------')
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
        return(df_pairs)

    def calc_transfer_dist(df_pairs):
        print(' - Calculating distance between pairs of stops...')
        ## Then, calculate the distance
        # Calculate distance
        df_pairs['dist'] = df_pairs.apply(lambda x: geopy.distance.geodesic((x.stop_lat_left, x.stop_lon_left), 
                                                                            (x.stop_lat_right, x.stop_lon_right)).miles, axis=1)
        # Now get rid of unnecessary information
        df_pairs = (df_pairs[['stop_id_left', 'stop_id_right', 'dist']]
                    .rename(columns={'stop_id_left': 'stop1', 'stop_id_right': 'stop2'}))
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
        route_ref['stop_id'] = route_ref['stop_id'].astype(str)
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

    # First find stops that are within walkable distance (0.5 mile)
    df_pairs = find_nearby_stops(df_stops, walk_thres)
    # Then, calculate the distance
    df_pairs = calc_transfer_dist(df_pairs)
    # Find stops that are on the same route (no transfer needed)
    df_pairs = filter_same_route(df_trips, df_stop_times, df_pairs)
    # Calculate walk time and format the stops DF
    df_pairs['t_time'] = df_pairs['dist']/walk_speed*60 # walk_speed by default is 2.5 mph walk speed, time in minutes
    df_pairs = df_pairs[['stop1', 'stop2', 't_time', 'dist']].rename(columns = {'dist': 't_dist'})
    df_pairs['type'] = 'transfer'
    return(df_pairs)

def fill_links_dist_na(links):
    # Mark the errors as False
    links['Distance Computed'] = True
    links.loc[links['t_dist'] == 0, 'Distance Computed']= False
    links.loc[links['t_dist'] == 0, 'type']= 'ride_derived'
    # Calculate average travel speed by provider
    links['prov'] = [x.split('_')[0] for x in links['stop1']]
    links['speed'] = links['t_dist']/links['t_time']
    spd_ref = (pd.DataFrame(links[links['t_dist'] > 0]
                            .groupby('prov')['speed'].median())
               .reset_index()
               .rename(columns={'speed':'derived_dist'}))
    # For those with 0 distance, interpolate distance from speed
    links = links.merge(spd_ref, on='prov', how='left')
    links['t_dist'] = links.apply(lambda x: x.t_dist if x.t_dist > 0 else x.t_time*x.derived_dist, axis=1)
    links = links[['stop1', 'stop2', 't_time', 't_dist', 'geometry', 'type']]
    return(links)


def fill_links_geo_na(links, df_stops):
    # Links with geometry is fine... 
    geo_fill = links[links['geometry'].isna()]
    links_fine = links[~links['geometry'].isna()]
    
    # Format Stops:
    df_stops = df_stops[['stop_id', 'stop_lat', 'stop_lon']]
    df_stops = gpd.GeoDataFrame(df_stops, geometry = gpd.points_from_xy(df_stops.stop_lon, df_stops.stop_lat))
    df_stops = df_stops.set_crs('EPSG:4269')
    df_stops = df_stops.to_crs('EPSG:2163')
    
    # Add geometry to links that need filled
    geo_fill = geo_fill.merge(df_stops[['stop_id', 'geometry']]
                              .rename(columns={
                                  'stop_id': 'stop1',
                                  'geometry': 'o_point'
                              }), 
                              on='stop1', how='left')
    geo_fill = geo_fill.merge(df_stops[['stop_id', 'geometry']]
                              .rename(columns={
                                  'stop_id': 'stop2',
                                  'geometry': 'd_point'
                              }), 
                              on='stop2', how='left')
    
    
    # If point geometry is missing, the links are abandoned...
    links_geo_fail = geo_fill[geo_fill['o_point'].isna() | geo_fill['d_point'].isna()]
    links_geo_fail['geometry'] = None
    
    # Fill the rest of the links
    links_geo_fill = geo_fill[~(geo_fill['o_point'].isna() | geo_fill['d_point'].isna())]
    links_geo_fill['geometry'] = links_geo_fill.apply(lambda x: LineString([x.o_point, x.d_point]), axis=1)
    
    # Format links
    links_fine['geotype'] = 'Computed'
    #
    links_geo_fail['geotype'] = 'Missing'
    links_geo_fail = links_geo_fail[['stop1', 'stop2', 't_time', 't_dist', 'geometry', 'type', 'geotype']]
    #
    links_geo_fill['geotype'] = 'Straight-line'
    links_geo_fill = links_geo_fill[['stop1', 'stop2', 't_time', 't_dist', 'geometry', 'type', 'geotype']]
    #
    links = pd.concat([links_fine, links_geo_fail, links_geo_fill])
    return(links)


### ----------------------- SECTION 3 -----------------------

def import_sample(Period, Time):
    print('--------------- PROCESSING SAMPLES ---------------')
    samp_in = pd.read_csv(os.path.join(homeDir, 'Data/sample_in.csv'))
    if Time == '':
        samp_in = samp_in[samp_in['period'] == Period]
    else:
        samp_in['time'] = [x.replace(':00 ', '') for x in samp_in['time']]
        samp_in = samp_in[(samp_in['time'] == Time) & (samp_in['period'] == Period)]
    return(samp_in)

def find_samp_stops(samp_in, df_stops, walk_thres):
    if samp_in.shape[0] == 0:
        return(samp_in)
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
    print(' - Calculating walking distance and finding walkable stops near sample O & D...')
    df_stops_buff = df_stops.buffer(1609.344*walk_thres)
    df_stops_buff = gpd.GeoDataFrame(df_stops, geometry = df_stops_buff)
    # Find stops that are walkable to each sample O & D 
    o_samp = gpd.sjoin(o_samp, df_stops_buff, how='left', predicate='within')
    d_samp = gpd.sjoin(d_samp, df_stops_buff, how='left', predicate='within')
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

def shortest_path_finder(samp_in, links, Period, Time):
    if samp_in.shape[0] == 0:
        pd.DataFrame().to_csv(os.path.join(homeDir, 
                                           'Data/Paths/{}_{}_paths.csv'.format(str(Period), Time)))
        return(True)
    # Build graphs
    DG_time = nx.DiGraph()
    for ind, row in links.iterrows():
        DG_time.add_weighted_edges_from([(row['stop1'], row['stop2'], row['t_time'])], 
                                        ttype=row['type'])
    DG_dist = nx.DiGraph()
    for ind, row in links.iterrows():
        DG_dist.add_weighted_edges_from([(row['stop1'], row['stop2'], row['t_dist'])], 
                                        ttype=row['type'])
    # Find shortest paths
    samp_in['t_time'] = 0
    samp_in['t_dist'] = 0
    samp_in['p_time'] = 'nan'
    samp_in['p_dist'] = 'nan'
    #
    for ind, row in samp_in.iterrows():
        if pd.isna(row.o_stop) or pd.isna(row.d_stop):
            samp_in.loc[ind, 't_time'] = 99999
            samp_in.loc[ind, 't_dist'] = 99999
            samp_in.loc[ind, 'p_time'] = 'Error, node not matched to a stop'
            samp_in.loc[ind, 'p_dist'] = 'Error, node not matched to a stop'
        #
        source = row.o_stop
        target = row.d_stop
        #
        try:
            samp_in.loc[ind, 't_time'] = nx.shortest_path_length(DG_time, source, target, 
                                                                 weight='weight')
            samp_in.loc[ind, 't_dist'] = nx.shortest_path_length(DG_dist, source, target, 
                                                                 weight='weight')
            samp_in.loc[ind, 'p_time'] = '__'.join(nx.shortest_path(DG_time, source, target, 
                                                                    weight='weight'))
            samp_in.loc[ind, 'p_dist'] = '__'.join(nx.shortest_path(DG_dist, source, target, 
                                                                    weight='weight'))
        except:
            samp_in.loc[ind, 't_time'] = 99999
            samp_in.loc[ind, 't_dist'] = 99999
            samp_in.loc[ind, 'p_time'] = 'Error, node not in network'
            samp_in.loc[ind, 'p_dist'] = 'Error, node not in network'
    # Save Output 
    samp_in.drop(columns = 'geometry').to_csv(os.path.join(homeDir, 
                                                           'Data/Paths/{}_{}_paths3.csv'.format(str(Period), Time)))
    return(True)

### ----------------------- WAIT-TIME EXTENSION -----------------------

# Create a file with stop sequence 
# Sort stop_time file by departure time 
# (better than arrival time, cuz it's the last minute one can catch the bus)
def generate_stop_sequence(df_stop_times):
    df_stop_seq = df_stop_times[['stop_id', 'departure_time', 'arrival_time', 'trip_id']]
    df_stop_seq = df_stop_seq.sort_values(by=['departure_time'])
    df_stop_seq['stop_id'] = 'ride_' + df_stop_seq['stop_id']
    df_stop_seq.reset_index(inplace=True)
    return(df_stop_seq[['stop_id', 'departure_time']])


# The path list has all the stops that the path suggests
# This function first put the path list into a dataframe
# Combine with necessary information for travel time, type of the link, etc.
# Since we are concerned about transit schedule, all the links that are not ride link are removed
# The time for these links, though, are considered by combining all these links into a transfer link
# The second stop's departure time is compared against the cumulative sum of travel time in the next step
# so the second stop is renamed
def generate_transit_paths(path, links):
    # Generate link-like file from path, and combine with necesary information
    df_path = pd.DataFrame({'stop1': path[:-1], 'stop2': path[1:]})
    df_path =  df_path.merge(links[['stop1', 'stop2', 't_time', 'type']], how='left', on=['stop1', 'stop2'])
    # Filter to retain only the links that go by ride
    # The links that involve walking and other modes are added up to "transfer" time
    # This is done by iteratively searching for a non-ride link, 
    # and adding all time column until the next ride-link is met
    prev_type = 'ride'
    cum_time = 0
    o_stop = ''
    d_stop = ''
    transfer_paths = []
    df_transfer_ent = pd.DataFrame()
    end_node = max(df_path.index)
    for ind, row in df_path.iterrows():
        if row.type == 'ride':
            if prev_type == 'ride':
                continue
            else:
                df_path.loc[ind-1] = o_stop, d_stop, cum_time, 'transfer'
                transfer_paths.append(df_transfer_ent)
                df_transfer_ent = pd.DataFrame()
                prev_type = 'ride'
                cum_time = 0
        else:
            df_transfer_ent = df_transfer_ent.append(df_path.loc[ind])
            d_stop = row.stop2
            cum_time += row.t_time
            df_path = df_path.drop(ind)
            if prev_type == 'ride':
                o_stop = row.stop1
            elif ind == end_node: 
                # When at the last stop, collect information just as do when it is a ride
                df_path.loc[ind] = o_stop, d_stop, cum_time, 'transfer'
                transfer_paths.append(df_transfer_ent)
            prev_type = 'transfer'

    df_path = df_path.sort_index().reset_index(drop=True)
    # For later comparison: find cumulative travel time experienced at each stop
    df_path['time_pt'] = df_path['t_time'].cumsum()
    # For later comparison: rename stop 2 so that it can be combined with arrival time schedule
    df_path = df_path[['stop1','stop2', 't_time', 'time_pt', 'type']].rename(columns={'stop2':'stop_id'})
    # Get an index for iterative processing in the next step
    df_path['stop_seq'] = df_path.index
    return(df_path, transfer_paths)


# Get a list of transfer nodes, to be used to truncate the path in the later step
def get_transfer_nodes(df_path):
    transfer_node = df_path.index[df_path['type'] == 'transfer'].tolist()
    transfer_node.append(max(df_path['stop_seq'])+1)
    return(transfer_node)


def time_to_minute(time):
    hr = int(time.split(':')[0])
    mn = int(time.split(':')[1])
    sc = int(time.split(':')[2])
    return(hr*60 + mn + sc/60)


# Produce final path file based on the formatted path DF and transfer nodes
# Basically, the algorithm iterrate through every possible start of the path,
# and find the total travel time for each path.
# In the end, the path-starter with the shortest travel time (in case of equal, the first selected) is selected.
# For processing of path starting with each starter, the entire path is truncated into sections by transfer node
# For each transfer node, the total travel time change is collected for before the transfer
# Wait time is calculated as the difference between specific travel (e.g., walk) and actual time
# For the all after the transfer node, wait time is added to the cumulative travel time
def produce_final_path(df_path, transfer_node):
    to_jug = df_path[df_path['stop_seq'] != 0]
    if to_jug.shape[0] == 0:
        return(df_path.loc[[df_path.time_abs.idxmin()]])
    ls_seq = np.unique(to_jug['stop_seq'])
    if 0 in transfer_node:
        transfer_node.remove(0)

    prev_time = np.inf
    for ind, row in df_path[df_path['stop_seq'] == 0].iterrows():
        df_jug = to_jug.copy()
        df_jug['time_abs'] = df_jug['time_abs'] - row.time_abs + row.time_pt # allow for rounding diff.
        df_jug = df_jug[df_jug['time_abs']  + 0.01 >= df_jug['time_pt']]
        if len(np.unique(df_jug['stop_seq'])) < len(ls_seq):
            continue
        else:
            prev_node = 0
            df_complete_path = pd.DataFrame()
            for transfer in transfer_node:
                # Process the path before the transfer node (after previous transfer node)
                df_prev_trans = df_jug[df_jug['stop_seq'].isin(range(prev_node+1, transfer))]
                df_prev_trans = df_prev_trans.loc[df_prev_trans.groupby('stop_seq').time_abs.idxmin()]
                # Process the path after the transfer node
                df_jug = df_jug[df_jug['stop_seq'] >= transfer]
                if df_prev_trans.shape[0] == 0:
                    prev_addon = 0
                else:
                    prev_addon = sum(df_prev_trans.apply(lambda x: x.time_abs - x.time_pt, axis=1))
                #
                if prev_addon > 0.1:
                    df_jug['time_pt'] = df_jug['time_pt'] + prev_addon
                # 
                df_trans = df_jug[df_jug['stop_seq'] == transfer]
                if df_trans.shape[0] == 0:
                    df_trans = pd.DataFrame()
                    df_wait = pd.DataFrame()
                else:
                    df_trans = df_trans.loc[[df_trans['time_abs'].idxmin()]]
                    wait_time = (df_trans.time_abs - df_trans.time_pt).tolist()[0]
                    if wait_time < 0:
                        continue
                    df_wait = pd.DataFrame({
                        'stop1': df_trans.stop1, 
                        'stop_id': df_trans.stop_id, 
                        'time_pt': df_trans.time_abs,
                        't_time': [wait_time], 
                        'type': ['wait'],
                        'stop_seq': df_trans.stop_seq + 0.5
                    })
                    df_jug = df_jug[df_jug['stop_seq'] > transfer]
                    df_jug['time_pt'] = df_jug['time_pt'] + wait_time
                    df_jug = df_jug[df_jug['time_abs']  + 0.01 >= df_jug['time_pt']]
                    # 
                prev_node = transfer
                df_ent = pd.concat([df_prev_trans, df_trans, df_wait])
                df_complete_path = df_complete_path.append(df_ent)
        if len(np.unique(df_complete_path['stop_seq'])) < len(ls_seq):
            continue
        time = df_complete_path[df_complete_path['stop_seq'] == max(df_path['stop_seq'])].time_pt.tolist()[0]
        if time < prev_time:
            df_final_path = df_complete_path.copy()
            prev_time = time
    if prev_time > 9999:
        df_final_path = pd.DataFrame()
    return(df_final_path)


# For each transfer location, add back details of the transfer
# The details will be re-index by the 0.01 step of previous transfer sequence in the path
# The previous transfer link will be removed
# The transfer time is also recalculated into cumulative sum
# This is done iteratively for all the transfer nodes
def add_back_transfer_details(df_final_path, transfer_paths, transfer_node):
    for i in range(len(transfer_paths)):
        df_tran_path = transfer_paths[i]
        node = transfer_node[i]
        df_tran_path['time_pt'] = df_tran_path['t_time'].cumsum()
        if i != 0:
            prev_time = df_final_path[df_final_path['stop_seq'] == node-1].time_pt.tolist()[0]
            df_tran_path['time_pt'] = df_tran_path['time_pt'] + prev_time
        df_tran_path['stop_seq'] = [node + 0.01*i for i in range(df_tran_path.shape[0])]
        df_final_path = df_final_path[df_final_path['stop_seq'] != node]
        df_final_path = pd.concat([df_final_path, df_tran_path])
    return(df_final_path)


def final_path_wWait(path, df_stop_times, links):
    
    # Prepare the files: stops sequence, path DF, a list of transfer nodes
    df_stop_seq = generate_stop_sequence(df_stop_times)
    df_path, transfer_paths = generate_transit_paths(path, links)
    transfer_node = get_transfer_nodes(df_path)
    
    # Combining stops sequence and path
    df_path = df_path.merge(df_stop_seq.drop_duplicates(), how='left', on='stop_id')
    df_path['time_abs'] = [time_to_minute(x) for x in df_path['departure_time']]
    
    # Looking for the final path file
    df_final_path = produce_final_path(df_path, transfer_node)
    df_final_path = df_final_path.rename(columns={'stop_id':'stop2'})
    if df_final_path.shape[0] == 0:
        return(df_final_path)
    
    # Add back transfer details and formatting a bit
    df_final_path = add_back_transfer_details(df_final_path, transfer_paths, transfer_node)
    df_final_path = df_final_path.sort_values(by=['stop_seq'])
    df_final_path = df_final_path.drop(columns=['time_abs'])
    df_final_path.reset_index(drop=True, inplace=True)
    return(df_final_path)


def multiple_shortest_paths(DG, source, target, trip_id, k, kmax, df_stop_times, links):
    allPaths = pd.DataFrame()
    X = nx.shortest_simple_paths(DG, source, target)
    t_min = np.inf
    p_min = []
    for counter, path in enumerate(X):
        # Stop condition 1: have done kmax number of trials (some failed)
        if counter == kmax:
            break
        # Add the paths to final file
        df_final_path = final_path_wWait(path, df_stop_times, links)
        if df_final_path.shape[0] == 0:
            continue
        df_final_path['trip_id'] = trip_id
        df_final_path['trial'] = counter
        allPaths = allPaths.append(df_final_path)
        # Find shortest path                
        time = df_final_path.iloc[-1].time_pt
        if time < t_min:
            t_min = time
            p_min = counter
        # Stop condition 2: have obtained k number of successful trials
        if len(np.unique(allPaths['trial'])) == k:
            break
    return(allPaths, t_min, p_min)


# For now, this function is done for time network. Distance can be added later
def shortest_path_wWait(samp_in, df_stop_times, links, k, kmax):
    
    if samp_in.shape[0] == 0:
        return(None)
    samp_in = samp_in.drop(columns=['Unnamed: 0', 'geometry'])
    
    # Build time graph
    DG_time = nx.DiGraph()
    for ind, row in links.iterrows():
        DG_time.add_weighted_edges_from([(row['stop1'], row['stop2'], row['t_time'])], 
                                        ttype=row['type'])
        
    # Find shortest paths
    samp_in['t_time'] = 0
    samp_in['p_time'] = 'nan'
    #
    allPaths = pd.DataFrame()
    for ind, row in samp_in.iterrows():
        #
        source = row.o_stop
        target = row.d_stop
        #
        try:
            t_min = np.inf
            entPaths, t_min, p_min = multiple_shortest_paths(DG_time, source, target, 
                                                             row.trip_id, k, kmax, 
                                                             df_stop_times, links)
            allPaths = allPaths.append(entPaths)
            samp_in.loc[ind, 't_time'] = t_min
            samp_in.loc[ind, 'p_time'] = p_min
        except:
            samp_in.loc[ind, 't_time'] = 99999
            samp_in.loc[ind, 'p_time'] = 'Error, node not in network'
    return(samp_in, allPaths)


### ----------------------- SECTION 5 -----------------------
def export_link_shp(links, saveloc):
    links_shape = links[links['geotype'] != 'Missing']
    links_shape = gpd.GeoDataFrame(links_shape, geometry = links_shape.geometry)
    links_shape = links_shape.set_crs('EPSG:2163')
    links_shape.to_file(os.path.join(saveloc))
    
    
def prepare_ABM_dailyLinks(p_abm_links, p_abm_nodes, homeDir):
    
    # Set network output saving directory
    networkDir = os.path.join(homeDir, 'Data/Multimodal_network')
    if not os.path.exists(networkDir):
        os.mkdir(networkDir)
        
    # Import ABM links and nodes
    abm_links = gpd.read_file(p_abm_links)
    abm_nodes = gpd.read_file(p_abm_nodes)
    #
    abm_links = abm_links[['A', 'B', 'DISTANCE', 'WALKTIME', 
                           'TIME_EA', 'TIME_AM', 'TIME_MD', 'TIME_PM', 'TIME_EV', 
                           # 'SPD_EA', 'SPD_AM', 'SPD_MD', 'SPD_PM', 'SPD_EV', 
                           'geometry']]
    abm_nodes = abm_nodes[['N', 'geometry']]
    
    # Filter out zero speeds
    # It is found that SPD_EA, SPD_AM, SPD_MD, SPD_PM, SPD_EV all have 0 values in the same rows
    # These are again the same for <0 rows in TIME_EA, TIME_AM, TIME_MD, TIME_PM, and TIME_EV
    # For RTP 2020, Amd 3
    abm_links = abm_links[abm_links['TIME_EA'] >= 0]
    
    # Format ABM nodes
    abm_nodes = abm_nodes.to_crs('EPSG:4269')
    #
    abm_nodes['stop_id'] = 'ABM_' + abm_nodes['N'].astype("str")
    abm_nodes['stop_name'] = 'ABM Node: ' + abm_nodes['N'].astype("str")
    abm_nodes['stop_lat'] = [a.y for a in abm_nodes['geometry']]
    abm_nodes['stop_lon'] = [a.x for a in abm_nodes['geometry']]
    abm_nodes = abm_nodes[['stop_id','stop_name','stop_lat','stop_lon']]
    abm_nodes['stop_id_base'] = abm_nodes['stop_id']
    for ver in ['walk', 'auto_ea', 'auto_am', 'auto_md', 'auto_pm', 'auto_ev']:
        abm_nodes['stop_id'] = ver.split('_')[0] + '_' + abm_nodes['stop_id_base']
        abm_nodes.drop(columns=['stop_id_base']).to_csv(os.path.join(networkDir, 'nodes_abm_{}.csv'.format(ver)), 
                                                        index=False)
    
    # Format ABM links
    abm_links['stop1'] = 'ABM_' + abm_links['A'].astype("str")
    abm_links['stop2'] = 'ABM_' + abm_links['B'].astype("str")
    abm_links['t_dist'] = abm_links['DISTANCE']
    abm_links['type'] = 'automobile'
    abm_links['Computed'] = True
    abm_links['prov'] = 'ABM'
    
    # Save ABM walk links
    walk_links = abm_links.copy()
    walk_links = walk_links[['stop1','stop2','WALKTIME','t_dist','type','Computed','prov']]
    walk_links = walk_links.rename(columns={'WALKTIME': 't_time'})
    walk_links['type'] = 'walk'
    walk_links['stop1'] = 'walk_' + walk_links['stop1']
    walk_links['stop2'] = 'walk_' + walk_links['stop2']
    walk_links.to_csv(os.path.join(networkDir, 'links_abm_walk.csv'), index=False)
    
    # Save ABM auto links
    abm_links['stop1'] = 'auto_' + abm_links['stop1']
    abm_links['stop2'] = 'auto_' + abm_links['stop2']
    for prd in ['EA', 'AM', 'MD', 'PM', 'EV']:
        auto_links = abm_links.copy()
        auto_links = auto_links[['stop1','stop2','TIME_{}'.format(prd),'t_dist','type','Computed','prov']]
        auto_links = auto_links.rename(columns={'TIME_{}'.format(prd): 't_time'})
        auto_links.to_csv(os.path.join(networkDir, 'links_abm_auto_{}.csv'.format(prd.lower())), index=False)
    
    return True


def connect_node_stops(nodes, stops, accept_pair_dist, accept_connect_dist, connect_speed):
    
    # Format all the files 
    df_stops = gpd.GeoDataFrame(stops, geometry = gpd.points_from_xy(stops.stop_lon, stops.stop_lat))
    abm_nodes = gpd.GeoDataFrame(nodes, geometry = gpd.points_from_xy(nodes.stop_lon, nodes.stop_lat))
    #
    df_stops = df_stops.set_crs('EPSG:4269')
    abm_nodes = abm_nodes.set_crs('EPSG:4269')
    df_stops = df_stops.to_crs('EPSG:2163')
    abm_nodes = abm_nodes.to_crs('EPSG:2163')
    
    # Buffer the file by accept_pair_dist, spatial join and partition the data into two sets
    df_stops = gpd.GeoDataFrame(df_stops, geometry = df_stops.buffer(accept_pair_dist))
    #
    stop_node_pair = gpd.sjoin(df_stops, 
                               (abm_nodes[['stop_id', 'stop_lat', 'stop_lon', 'geometry']]
                                .rename(columns={
                                    'stop_id': 'node_id',
                                    'stop_lat': 'node_lat', 
                                    'stop_lon': 'node_lon'
                                })), 
                               how='left', predicate='contains')
    #
    direct_pair = stop_node_pair[~stop_node_pair['node_id'].isna()]
    connect_stops = stop_node_pair[stop_node_pair['node_id'].isna()]['stop_id']
    
    # Process the direct pairing stops (taken as 0 time connection)
    # Find distance between all pairs, and keep the one that is closest to each STOP
    # Then format
    direct_pair['dist'] = direct_pair.apply(lambda x: geopy.distance.geodesic((x.stop_lat, x.stop_lon),
                                                                              (x.node_lat, x.node_lon)).meters, 
                                            axis=1)
    direct_pair.reset_index(drop=True, inplace=True)
    direct_pair = direct_pair.loc[direct_pair.groupby('stop_id').dist.idxmin()]
    #
    direct_pair = pd.concat([(direct_pair[['stop_id','node_id','dist']]
                          .rename(columns={'stop_id':'stop1', 'node_id':'stop2', 'dist':'t_dist'})),
                         (direct_pair[['stop_id','node_id','dist']]
                          .rename(columns={'stop_id':'stop2', 'node_id':'stop1', 'dist':'t_dist'}))])
    direct_pair['t_dist'] = direct_pair['t_dist']/1609.344   # Change from m to mile
    direct_pair['t_time'] = 0   # Use 0 time
    direct_pair['type'] = 'node_stop_pair'
    direct_pair['Computed'] = False
    
    # Process the straight-line connection
    # Buffer again by accept_connect_dist, and then find possibly connected node-stop pairs in the range
    # Similar to above, keep only the pair with the smallest distance for each STOP
    # The distance is kept using the straight-line distance
    # Travel time is calculated using default travel speed provided (connect_speed)
    # Then format
    connect_stops = df_stops[df_stops['stop_id'].isin(connect_stops)]
    #
    connect_stops = gpd.GeoDataFrame(connect_stops, geometry = connect_stops.buffer(accept_connect_dist * 1609.344))
    # 
    connect_stops_pair = gpd.sjoin(connect_stops, 
                                   (abm_nodes[['stop_id', 'stop_lat', 'stop_lon', 'geometry']]
                                    .rename(columns={
                                        'stop_id': 'node_id',
                                        'stop_lat': 'node_lat', 
                                        'stop_lon': 'node_lon'
                                    })), 
                                   how='left', predicate='contains')
    #
    connect_pair = connect_stops_pair[~connect_stops_pair['node_id'].isna()]
    #
    connect_pair['dist'] = connect_pair.apply(lambda x: geopy.distance.geodesic((x.stop_lat, x.stop_lon),
                                                                                (x.node_lat, x.node_lon)).miles, 
                                              axis=1)
    #
    connect_pair.reset_index(drop=True, inplace=True)
    connect_pair = connect_pair.loc[connect_pair.groupby('stop_id').dist.idxmin()]
    #
    connect_pair = pd.concat([(connect_pair[['stop_id','node_id','dist']]
                               .rename(columns={'stop_id':'stop1', 'node_id':'stop2', 'dist':'t_dist'})),
                              (connect_pair[['stop_id','node_id','dist']]
                               .rename(columns={'stop_id':'stop2', 'node_id':'stop1', 'dist':'t_dist'}))])
    connect_pair['t_time'] = connect_pair['t_dist']/connect_speed*60
    connect_pair['type'] = 'node_stop_connector'
    connect_pair['Computed'] = False
    
    return(pd.concat([direct_pair, connect_pair]))


def auto_network_period(Time):
    time_period_dict = {
        'ea': ['Before 5am', '5am'],
        'am': ['6am', '7am', '8am', '9am'],
        'md': ['10am', '11am', '12noon', '1pm', '2pm'],
        'pm': ['3pm', '4pm', '5pm', '6pm'],
        'ev': ['7pm', '8pm', '9pm', '10pm', '11pm and later']
    }
    for prd in time_period_dict.keys():
        if Time in time_period_dict[prd]:
            return(prd)