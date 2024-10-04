# -*- coding: utf-8 -*-
"""
Created on Monday September, 2023

@author: Fizzy Huiying Fan
All rights reserved
"""

import os
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
# Spatial Features
from shapely.geometry import LineString
from shapely.geometry import Point

import geopandas as gpd
import geopy


def sample_pts_from_line(row, distance_delta):
    line = row.geometry
    distances = np.arange(0, line.length, distance_delta)
    if line.is_ring:
        end_point = Point(*line.coords[-1])
    else:
        end_point = line.boundary.geoms[1]
    points = [line.interpolate(distance) for distance in distances] + [end_point]
    pts_entry = pd.DataFrame({
        'stop1': row.stop1,
        'stop2': row.stop2,
        'geometry': points
    })
    return(pts_entry)

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        flat_list.append(element[0])
    return flat_list

def sample_points_from_network(NETWORK_PATH, NETWORK_PTS_PATH, distance_delta=30):
    t0 = time.time()
    # import files
    links = gpd.read_file(NETWORK_PATH)
    #
    if 'stop1' not in links.columns:
        links = links.rename(columns={'from':'stop1','to':'stop2'})
    # pts_dat = pd.DataFrame()
    if links.crs == 'EPSG:2163':
        links = links.set_crs('EPSG:2163')
    else:
        links = links.to_crs('EPSG:2163')
    pts_dat = pd.DataFrame()
    for ind, row in links.iterrows():
        if row.geometry.length == 0:
            continue
        pts_entry = sample_pts_from_line(row, distance_delta) # in meters
        pts_dat = pd.concat([pts_dat,pts_entry], ignore_index=True)
    print('- time used: {} minutes.'.format(str(round((time.time()-t0)/60, 2))))
    # Create GDF of points
    pts_dat = gpd.GeoDataFrame(pts_dat, geometry = pts_dat['geometry'])
    #
    pts_dat = pts_dat.set_crs('EPSG:2163')
    pts_dat = pts_dat.to_crs('EPSG:4269')
    #
    pts_dat['lon'] = pts_dat.apply(lambda a: a.geometry.x, axis=1)
    pts_dat['lat'] = pts_dat.apply(lambda a: a.geometry.y, axis=1)
    #
    pts_dat.to_file(os.path.join(NETWORK_PTS_PATH))
    return(None)

def sampling_points_from_network(TRANSIT_NETWORK_PATH, TRANSFER_NETWORK_PATH, NETWORK_PTS_PATH, distance_delta):
    print(' --------------- sampling points from links --------------- ')
    if not os.path.exists(NETWORK_PTS_PATH):
        os.mkdir(NETWORK_PTS_PATH)
    sample_points_from_network(TRANSIT_NETWORK_PATH, 
                                          os.path.join(NETWORK_PTS_PATH, 'transit_pts.shp'), 
                                          distance_delta)
    sample_points_from_network(TRANSFER_NETWORK_PATH, 
                                          os.path.join(NETWORK_PTS_PATH, 'transfer_pts.shp'), 
                                          distance_delta)
    return(None)

def get_wait_location(stop_id, stops_df):
    stop_row = stops_df[stops_df['stop_id'] == stop_id]
    if not stop_row.empty:
        return stop_row.iloc[0]['stop_lat'], stop_row.iloc[0]['stop_lon']
    return None, None

def interpolate_trajectory(row, pts_df, stops_df):
    # Helper function to interpolate timestamps
    def interpolate_timestamp(start_time, end_time, fraction):
        time_range = end_time - start_time
        return start_time + pd.Timedelta(seconds=time_range.total_seconds() * fraction)
    
    # If the mode is waiting, get the wait location and repeat it
    if row['mode'] == 'waiting':
        lat, lon = get_wait_location(int(float(row['stop1'])), stops_df)
        start_time = pd.to_datetime(row['time1'])
        return [(row['mode'], lat, lon, start_time + pd.Timedelta(seconds=i)) for i in range(int(row['tt_seconds']))]
    
    # Get the corresponding points for the segment
    if row['mode'] == 'riding':
        segment_points = pts_df[(pts_df['stop1'] == int(float(row['stop1']))) & (pts_df['stop2'] == int(float(row['stop2'])))]
    else: 
        row_stop1 = np.int64(np.float64(row['stop1'][4:]))
        row_stop2 = np.int64(np.float64(row['stop2'][4:]))
        segment_points = pts_df[(pts_df['stop1'] == row_stop1) & (pts_df['stop2'] == row_stop2)]
    
    # If no corresponding points, return empty list
    if segment_points.empty:
        return []
    
    # Sort the points (assuming they are in order in pts_df)
    segment_points = segment_points.sort_values(by=['lat', 'lon'])
    
    # Calculate the number of interpolated points
    num_points = int(row['tt_seconds'])
    
    # Interpolate the points
    interpolated_points = []
    start_time = pd.to_datetime(row['time1'])
    end_time = pd.to_datetime(row['time2'])
    for i in range(num_points):
        fraction = i / (num_points - 1) if num_points > 1 else 0
        idx = int(fraction * (len(segment_points) - 1))
        start_point = segment_points.iloc[idx]
        end_point = segment_points.iloc[min(idx + 1, len(segment_points) - 1)]
        lat = start_point['lat'] + fraction * (end_point['lat'] - start_point['lat'])
        lon = start_point['lon'] + fraction * (end_point['lon'] - start_point['lon'])
        timestamp = interpolate_timestamp(start_time, end_time, fraction)
        interpolated_points.append((row['mode'], lat, lon, timestamp))
    
    return interpolated_points


def generate_trajectory(traj, pts_transit, pts_walk, stops_df):
    output = []
    total=0
    absent=0
    for ind, row in traj.iterrows():
        if row['mode'] in ['riding']:
            points = interpolate_trajectory(row, pts_transit, stops_df)
        elif row['mode'] in ['walking', 'walk_ingress', 'walk_egress']:
            points = interpolate_trajectory(row, pts_walk, stops_df)
            total += 1
            if len(points) == 0: 
                absent += 1
        elif row['mode'] in ['waiting']: 
            points = interpolate_trajectory(row, None, stops_df)
        else: continue
        output.extend(points)
    if absent/total > 0.05:
        print(' - Attention on trip ', row.trip_id, '- ', round(100*(absent/total), 1), '% of trip segments are left out due to mismatch...')
    return output

def bi_directional_transfer(df, ls_cols):
    df_rev = df.rename(columns={ls_cols[0]:ls_cols[1], ls_cols[1]:ls_cols[0]})
    df = pd.concat([df, df_rev], ignore_index=True).groupby(ls_cols).min().reset_index()
    return(df)

def traj_second_by_second(PROC_GTFS_PATH, NETWORK_PTS_PATH, NEW_TRAJ_PATH, SEC_TRAJ_PATH):
    print('Start finalizing second by second trajectories...')
    t0 = time.time()
    # Import files
    pts_transit = gpd.read_file(os.path.join(NETWORK_PTS_PATH, 'transit_pts.shp'))
    pts_walk = gpd.read_file(os.path.join(NETWORK_PTS_PATH, 'transfer_pts.shp'))
    pts_walk = bi_directional_transfer(pts_walk, ['stop1','stop2'])
    stops = pd.read_csv(os.path.join(PROC_GTFS_PATH, 'stops.txt'))
    ls_traj = os.listdir(NEW_TRAJ_PATH)
    if not os.path.exists(SEC_TRAJ_PATH):
        os.mkdir(SEC_TRAJ_PATH)
    # iterate over ls_traj
    for p in tqdm(ls_traj, total=len(ls_traj)):
        traj = pd.read_csv(os.path.join(NEW_TRAJ_PATH, p))
        trajectory = generate_trajectory(traj, pts_transit, pts_walk, stops)
        trajectory_df = pd.DataFrame(trajectory, columns=['mode', 'lat', 'lon', 'timestamp'])
        trajectory_df.to_csv(os.path.join(SEC_TRAJ_PATH, p))
    print('Done! Total time used: ', round((time.time() - t0)/60, 1), 'minutes.')
    return(None)