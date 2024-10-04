# -*- coding: utf-8 -*-
"""
Created on Monday September, 2023

@author: Fizzy Huiying Fan
All rights reserved
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import os
import glob
from tqdm import tqdm
import zipfile
from math import ceil
import pickle

pd.options.mode.chained_assignment = None  # default='warn'

from program import gtfs_wrapper as gtfs_proc


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