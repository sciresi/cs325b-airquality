import datetime
import os
import pandas
import numpy as np

#assumes date has already been converted to_datetime and is in that default format
def epa_to_file_name(date_string, station_id):
    date_object =  datetime.datetime.strptime(date_string,"%Y-%m-%d")
    file_name = str(date_object.year)+"_"
    day_of_year = date_object - datetime.datetime(date_object.year,1,1)
    day_of_year = 1+int(day_of_year.days)
    if day_of_year < 10:
        file_name += "00"
    elif day_of_year < 100:
        file_name += "0"
    file_name += str(day_of_year) + "_"
    file_name += str(station_id) + ".tif"
    return file_name

#stacks all epa csv files into master csv file
def make_master_epa():
    files = os.listdir()
    first_file = True
    #reads in all relevant csvs
    for file in files:
        if file[-4:]==".csv" and file[:3] =="epa":
            new_df = pandas.read_csv(file)
            print(file)
            if not first_file:
                df = df.append(new_df,ignore_index=True)
            else:
                df = new_df
                first_file = False
    df['Date'] = pandas.to_datetime(df['Date'])
    df.to_csv("master_epa.csv",index=False)


#assumes other file is from modis and has modis filenames
#merges columns into master epa file
def add_columns_to_master_modis(other_file,columns):
    epa = pandas.read_csv("master_epa.csv")
    for column in columns:
        epa[column]=np.nan
    other = pandas.read_csv(other_file)
    for row in epa.index:
        date = epa['Date'][row]
        site = epa['Site ID'][row]
        modis_file = epa_to_file_name(date,site)
        other_row = other[other['Filename']==modis_file]
        for column in columns:
            epa.loc[row,column] = other_row[column][other_row.index[0]]
        
    epa.to_csv("master_epa.csv",index=False)
make_master_epa()
add_columns_to_master_modis("modis.csv", ["Blue [0,0]","Blue [0,1]","Blue [1,0]","Blue [1,1]","Green [0,0]","Green [0,1]","Green [1,0]","Green [1,1]"])   
