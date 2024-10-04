# -*- coding: utf-8 -*-
"""
Created on July, 2023

@author: Fizzy Huiying Fan
All rights reserved
"""
import os
import glob
import psutil
import itertools
from itertools import chain

from tqdm import tqdm
import argparse
#
import numpy as np
import pandas as pd
import pickle

import math
from math import ceil
from random import shuffle
import time
from time import time as time_measure
from time import time
from datetime import datetime
#
import networkx as nx
#
import geopandas as gpd
from shapely.geometry import LineString
import geopy
import geopy.distance


# 2. Import RAPTOR functions
from program.std_raptor import raptor

def find_samp_stops(SAMP_PATH, GTFS_PATH, SAMP_INT_PATH, walk_thres):
    samp_in = pd.DataFrame()
    for p in glob.glob(os.path.join(SAMP_PATH, '*.csv')):
       entry = pd.read_csv(p)
       samp_in = pd.concat([samp_in, entry], axis=0)
    if samp_in.shape[0] == 0:
        return(samp_in)
    print(' - Looking for stops for each sample...')
    ## First find stops that are within walkable distance (0.5 mile)
    # Make stations geospatial file
    df_stops = pd.read_csv(os.path.join(GTFS_PATH, 'stops.csv'))
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
    samp_in.to_csv(os.path.join(SAMP_INT_PATH, 'sample_stops.csv'), index=False)
    return(None)



def run_raptor(GTFS_PATH, DICT_PATH, TRANSFER_PATH, SAVE_PATH, SAMP_INT_PATH,
               DATE_TOFILTER_ON, MAX_TRANSFER, WALKING_FROM_SOURCE, CHANGE_TIME_SEC, PRINT_ITINERARY):

  print('---------------------- 1. Files Import ----------------------')
  sample = pd.read_csv(os.path.join(SAMP_INT_PATH, 'sample_stops.csv'))
  stops_file = pd.read_csv(f'{GTFS_PATH}/stops.txt')
  trips_file = pd.read_csv(f'{GTFS_PATH}/trips.txt')
  stop_times_file = pd.read_csv(f'{GTFS_PATH}/stop_times.txt')
  transfers_file = pd.read_csv(f'{TRANSFER_PATH}')
  #
  with open(f'{DICT_PATH}/stops_dict_pkl.pkl', 'rb') as file:
    stops_dict = pickle.load(file) # Route_i: [stop_1, stop_2, ... stop_n]
  with open(f'{DICT_PATH}/stoptimes_dict_pkl.pkl', 'rb') as file:
    stoptimes_dict = pickle.load(file)
    # Route_i: [[(stop, time) for all stops in trip_1], 
    #           [(stop, time) for all stops in trip_2],
    #           ...,
    #           [(stop, time) for all stops in trip_m]]
    # Route_i: [[(stop_1, time_trip_1), (stop_2, time_trip_1),..., (stop_n, time_trip_1)], 
    #           [(stop_1, time_trip_2), (stop_2, time_trip_2),..., (stop_n, time_trip_2)], 
    #           ...,
    #           [(stop_1, time_trip_m), (stop_2, time_trip_m),..., (stop_n, time_trip_m)]]
    # Time = time stamp
  with open(f'{DICT_PATH}/transfers_dict_full.pkl', 'rb') as file:
    footpath_dict = pickle.load(file) # Stop_i: [(stop_1, timeInt),..., (stop_n, timeInt)]
  with open(f'{DICT_PATH}/routes_by_stop.pkl', 'rb') as file:
    routes_by_stop_dict = pickle.load(file) # Stop_i: [route_1, route_2, ... route_o]
  with open(f'{DICT_PATH}/idx_by_route_stop.pkl', 'rb') as file:
    idx_by_route_stop_dict = pickle.load(file)
    # [(route 1, stop 1): seq 1_1, 
    #  (route 1, stop 2): seq 1_2,
    #  ...  
    #  (route 1, stop n): seq 1_n, 
    #  ...
    #  (route m, stop n): seq m_n]


  print('---------------------- 2. System Set-up ----------------------')
  import sys
  class NoStdStreams(object):
      def __init__(self,stdout = None, stderr = None):
          self.devnull = open(os.devnull,'w')
          self._stdout = stdout or self.devnull or sys.stdout
          self._stderr = stderr or self.devnull or sys.stderr

      def __enter__(self):
          self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
          self.old_stdout.flush(); self.old_stderr.flush()
          sys.stdout, sys.stderr = self._stdout, self._stderr

      def __exit__(self, exc_type, exc_value, traceback):
          self._stdout.flush(); self._stderr.flush()
          sys.stdout = self.old_stdout
          sys.stderr = self.old_stderr
          self.devnull.close()


  print('---------------------- 3. Running RAPTOR ----------------------')
  df_sum = pd.DataFrame()
  df_paths = pd.DataFrame()
  log = pd.DataFrame()
  t = 0
  date_today = datetime.today().strftime('%Y%m%d')

  for ind, row in tqdm(sample.iterrows(), 
                      total=sample.shape[0]):
    # RAPTOR is doing something weird... it assumes all your network is scheduled for today...
    time = date_today + row.board_time[10:]
    with NoStdStreams():
      try:
        out, df_path =  raptor(row.o_stop, # SOURCE: int, 
                              row.d_stop, # DESTINATION: int, 
                              pd.to_datetime(time), # D_TIME, 
                              MAX_TRANSFER, 
                              WALKING_FROM_SOURCE,
                              CHANGE_TIME_SEC,
                              PRINT_ITINERARY,
                              routes_by_stop_dict, stops_dict, stoptimes_dict, 
                              footpath_dict, idx_by_route_stop_dict)
        df_sum_ent = pd.DataFrame({'trip_id': [row.trip_id], 
                                  'dep_time': [pd.to_datetime(time)],
                                  'arr_time': np.array([out]).flatten()[0]}) ## dimension messed up...
        df_sum = pd.concat([df_sum, df_sum_ent], ignore_index=True)
        df_path['trip_id'] = row.trip_id
        df_paths = pd.concat([df_paths, df_path], ignore_index=True)
        log_ent = pd.DataFrame({'trip_id': [row.trip_id],
                                'status': ['Success']})
        log = pd.concat([log, log_ent], ignore_index=True)
      except Exception as e:
        log_ent = pd.DataFrame({'trip_id': [row.trip_id],
                                'status': [e]})
        log = pd.concat([log, log_ent], ignore_index=True)


  print('---------------------- 4. Export outputs ----------------------')
  # Update date to the correct date
  DATE_TOFILTER_ON_dt = pd.to_datetime(DATE_TOFILTER_ON, format='%Y%m%d')
  date_today_dt = pd.to_datetime(date_today, format='%Y%m%d')

  # Calculate the difference between the two dates
  date_difference = date_today_dt - DATE_TOFILTER_ON_dt

  # Update the arrival_time column
  df_paths['arrival_time'] = df_paths['arrival_time'] - date_difference
  df_sum['dep_time'] = df_sum['dep_time'] - date_difference
  df_sum['arr_time'] = df_sum['arr_time'] - date_difference

  # Save output
  df_paths.to_csv(os.path.join(SAVE_PATH, 'paths.csv'), index=False)
  df_sum.to_csv(os.path.join(SAVE_PATH, 'summ.csv'), index=False)
  log.to_csv(os.path.join(SAVE_PATH, 'log.csv'), index=False)
  print(' - Done!')

  return(None)