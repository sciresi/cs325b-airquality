
# Matches EPA Stations to the nearest weather station data
# Code for reading GHCND .dly files attributed to https://gitlab.com/snippets/1838910

import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import csv
from os import listdir
from os.path import isfile, join
import collections

data_header_names = [
        "ID",
        "YEAR",
        "MONTH",
        "ELEMENT"]

data_header_col_specs = [
        (0,  11),
        (11, 15),
        (15, 17),
        (17, 21)]

data_header_dtypes = {
        "ID": str,
        "YEAR": int,
        "MONTH": int,
        "ELEMENT": str}

data_col_names = [[
        "VALUE" + str(i + 1),
        "MFLAG" + str(i + 1),
        "QFLAG" + str(i + 1),
        "SFLAG" + str(i + 1)]
        for i in range(31)]

data_col_names = sum(data_col_names, [])

data_replacement_col_names = [[
        ("VALUE", i + 1),
        ("MFLAG", i + 1),
        ("QFLAG", i + 1),
        ("SFLAG", i + 1)]
        for i in range(31)]

data_replacement_col_names = sum(data_replacement_col_names, [])
data_replacement_col_names = pd.MultiIndex.from_tuples(
        data_replacement_col_names,
        names=['VAR_TYPE', 'DAY'])

data_col_specs = [[
        (21 + i * 8, 26 + i * 8),
        (26 + i * 8, 27 + i * 8),
        (27 + i * 8, 28 + i * 8),
        (28 + i * 8, 29 + i * 8)]
                      for i in range(31)]
data_col_specs = sum(data_col_specs, [])
data_col_dtypes = [{
        "VALUE" + str(i + 1): int,
        "MFLAG" + str(i + 1): str,
        "QFLAG" + str(i + 1): str,
        "SFLAG" + str(i + 1): str}
                       for i in range(31)]
data_header_dtypes.update({k: v for d in data_col_dtypes for k, v in d.items()})


weather_st_to_years_data = {}    # dict storing the years of available data for each weather station
epa_to_weather_station = {}      # dictionary that maps (epa_station_id, year) -> (weather_station_id, distance) 
closest_weather_stations = set() # set of the relevant weather station ids

def epa_to_closest_weather_station(directory):
    '''
    Process the closest_weather_station_yyyy files. These uniquely cover each epa_station for each 
    year, so no need to iterate over epa daily data files themselves.
    '''
    
    prefix = "closest_weather_stations_"
    postfix = ".csv"
    years = ['2016', '2017', '2018', '2019']
    filenames = [ (directory + prefix + year + postfix) for year in years]

    for idx, f in enumerate(filenames): 
        with open(f) as csvfile:
            reader = csv.reader(csvfile,delimiter=',')
            line_count = 0
            incomplete_weather_data = 0
            for row in reader:
                if line_count == 0:
                    col_names = row
                else:
                    epa_station_id = row[2]
                    weather_station_id = row[4]
                    distance = row[3]
                    year = idx + 2016
                    epa_to_weather_station[(epa_station_id, year)] = (weather_station_id, distance)
                    closest_weather_stations.add(weather_station_id)

                    # double check we have weather data for the current closest station
                    assert(weather_st_to_years_data[weather_station_id][1] >= year)
                line_count += 1
        csvfile.close()
                                                                                                                                      
  
def weather_station_to_years_of_data(directory):
    ''' Processes all_weather_stations_yyyy.csv files to determine the years of data
    available for the weather stations.  Used to confirm that we have the necessary 
    year of weather data for each of the closest weather stations (in the year of interest). 
   
    As is - will not need to be run in the future. But, can/should edit to  process other 
    information in these all_weather_stations_yyyy.csv files
    '''
    prefix = "all_weather_stations_"
    postfix = ".csv"
    years = ['2016', '2017', '2018', '2019']
    filenames = [ (directory + prefix + year + postfix) for year in years]
    for idx, f in enumerate(filenames):  #idx = 0, 1, 2, 3
        with open(f) as csvfile:
            reader = csv.reader(csvfile,delimiter=',')
            line_count = 0
            for row in reader:
                if line_count == 0:
                    col_names = row
                else:
                    weather_station_id = row[1]
                    start_yr = int(row[2])
                    end_yr = int(row[3])
                    weather_st_to_years_data[weather_station_id] = (start_yr, end_yr)
                line_count += 1
        csvfile.close()

    # Save weather_st_to_years_data dict to file for late ruse
    np.save('weather_station_years_data.npy', weather_st_to_years_data)

    # Load
    read_dictionary = np.load('weather_station_years_data.npy', allow_pickle=True).item()





def read_ghcn_data_file(directory, filename, elements=None, include_flags=False, dropna='all'):
    ''' Reads in weather data from a GHCN .dly data file in the given directory with the 
    given filename into a pandas dataframe. Can specify a list of elements to include in the DF 
    via elements, otherwise all are included.
    Reformats the dataframe appropriately and saves just 2016-2019 data into a .csv file.
    '''
    
    df = pd.read_fwf(
        directory+filename,
        colspecs=data_header_col_specs + data_col_specs,
        names=data_header_names + data_col_names,
        index_col=data_header_names,
        dtype=data_header_dtypes )

    df.columns = data_replacement_col_names

    if not include_flags:
        df = df.loc[:, ('VALUE', slice(None))]
        df.columns = df.columns.droplevel('VAR_TYPE')
    
    df = df.stack(level='DAY').unstack(level='ELEMENT')

    if dropna:
        df.replace(-9999.0, pd.np.nan, inplace=True)
        df.dropna(how=dropna, inplace=True)

    df.index = pd.to_datetime(
            df.index.get_level_values('YEAR') * 10000 +
            df.index.get_level_values('MONTH') * 100 +
            df.index.get_level_values('DAY'),
            format='%Y%m%d')

    # Save only the relevant years, 2016 onwards
    df = df[df.index >= '2016-01-01']

    save_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/GHCND_weather/relevant_ghc/"
    csv_filename =  filename[:11] + ".csv"
    df.to_csv(save_dir + csv_filename,  encoding='utf-8')

    return df
        


def save_relevant_weather_data(data_dir):
    '''
    Loops through all the relevant weather stations, which are stored in closest_weather_stations,
    and saves the reformated .csv versions of the original data in a new directory that will be
    used going forward.

    Used once to create this new directory of relevant files, and should not have to be used again as is.
    '''
    for idx, weather_st_id in enumerate(closest_weather_stations):
        filename = weather_st_id + ".dly"
        w_id_df = read_ghcn_data_file(data_dir, filename, elements=None, include_flags=False, dropna='all')


if __name__ == "__main__":

    epa_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/epa/"
    ghcnd_base_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/GHCND_weather/"
    gh_data_dir = ghcnd_base_dir + "daily_data/ghcnd_hcn/"
    sent_path = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/sentinel/2016/"
    sent_file = "s2_2016_10_1002_171670012.tif"
    weath_file = "USC00457507.dly"
    
    weather_station_to_years_of_data(ghcnd_base_dir)
    epa_to_closest_weather_station(ghcnd_base_dir)
    #save_relevant_weather_data(gh_data_dir)
        
    

