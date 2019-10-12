import os
import pandas
import numpy as np
#assumes it's in the same folder as all the epa csvs
#predicts that each datapoint's PM2.5 is the same as the reading before it at the same site

files = os.listdir()
first_file = True
file_num = 0
#reads in all relevant csvs
for file in files:
    if file[-4:]==".csv" and file != "broken_modis_channel_means.csv" and file !="modis_channel_means_revised.csv":
        new_df = pandas.read_csv(file)
        if not first_file:
            df = df.append(new_df,ignore_index=True)
        else:
            df = new_df
            first_file = False
        file_num+=1
df['Date'] = pandas.to_datetime(df.Date)
#makes sure dates are in order
df.sort_values(by = ['Date'],inplace=True,ascending=True)
#gets all sites
sites = set()
for site in df['Site ID']:
    sites.add(site)

mse = 0
num_predictions = 0
#loops through sites (each site's readings are ordered by date) and gets errors for each
for site in sites:
    site_df = df[df['Site ID']==site]
    num_readings = len(site_df)
    #lines up each date with the previous reading to get error
    days = np.zeros([num_readings+1])
    days_staggered = np.zeros([num_readings+1])
    days[:num_readings] = site_df['Daily Mean PM2.5 Concentration']
    days[0] = 0
    days_staggered[1:] = site_df['Daily Mean PM2.5 Concentration']
    days_staggered[num_readings] = 0
    diff = np.square(days-days_staggered)
    mse+=diff.sum()
    #-1 because we don't try to predict the first day's reading at any station
    num_predictions+=(num_readings-1)
#gets avverage
mse /= len(num_predictions)
print(mse)
    
