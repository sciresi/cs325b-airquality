#combines modis and csv data
#first part plots distrbution of modis channel averages on their own
#second part plots distribution of modis channel averages vs. corresponding PM2.5 for 2016
#assumes it's in a folder with the modis average csv file and the 2016 PM2.5 csvs
import datetime
import os
import pandas
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

def epa_to_file_name(date_string, station_id):
    date_object =  datetime.datetime.strptime(date_string,"%m/%d/%Y")
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
modis_df = pandas.read_csv("modis_channel_means_revised.csv")

#plot modis channel values by themselves

blue_df = modis_df[modis_df['Blue']>0]
plt.hist(blue_df['Blue'], bins = 150)
print(blue_df['Blue'].mean())
plt.title("Average Blue Channel Value in 2016 Modis Images")
plt.ylabel("# Images")
plt.xlabel("Average Blue Channel ")
plt.show()

green_df = modis_df[modis_df['Green']>0]
plt.hist(green_df['Green'], bins = 150)
print(green_df['Green'].mean())
plt.title("Average Green Value in 2016 Modis Images")
plt.ylabel("# Images")
plt.xlabel("Average Green Value")
plt.show()

color_df = blue_df[blue_df['Green']>0]
plt.scatter(color_df['Green'],color_df['Blue'])
plt.title("Average Green vs. Average Blue Value in 2016 Modis Images")
plt.xlabel("Average Green Value Brightness")
plt.ylabel("Average Blue Value Brightness")
plt.show()

#gathers all the csv files into one dataframe
files = os.listdir()
first_file = True
means_by_state = np.zeros(191)
file_num = 0
for file in files:
    if file[-6:]=="16.csv":
        print(file)
        new_df = pandas.read_csv(file)
        means_by_state[file_num]=new_df['Daily Mean PM2.5 Concentration'].mean()
        if not first_file:
            df = df.append(new_df,ignore_index=True)
        else:
            df = new_df
            first_file = False
        file_num+=1

#plots pm vs green, pm vs. blue
pm_list = []
green_list = []
blue_list = []
print(len(df))
for row in range(len(df)):
    if df['Date'][row][-2:]=="16" and df['Daily Mean PM2.5 Concentration'][row]>50:
        if row % 10000 == 0:
            print("At row:")
            print(str(row))
        file_name = epa_to_file_name(df['Date'][row],df['Site ID'][row])
        modis_row = modis_df[modis_df['Filename']==file_name]
        pm_list.append(df['Daily Mean PM2.5 Concentration'][row])
        green = modis_row['Green mean'][modis_row.index[0]]
        blue = modis_row['Blue mean'][modis_row.index[0]]
        green_list.append(green)
        blue_list.append(blue)
plot = plt.scatter(pm_list,green_list)
print(np.corrcoef(pm_list,green_list))
plt.title("MODIS Green Values vs. PM2.5")
plot.remove()
plt.show()
print(np.corrcoef(pm_list,blue_list))
 
plt.title("MODIS Blue Values vs. PM2.5")
plt.scatter(pm_list,blue_list)
plt.show()
    
