#combines modis and csv data FOR 2016 ONLY
#first part plots distrbution of modis channel averages on their own
#second part plots distribution of modis channel averages vs. corresponding PM2.5 for 2016
#assumes it's in a folder with the modis average csv file and the 2016 PM2.5 csvs
import datetime
import os
import pandas
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

def epa_to_modis_file_name(date_string, station_id):
    
    #handles dates where year is written with four digits
    if date_string[-4:].isnumeric():
        date_object =  datetime.datetime.strptime(date_string,"%m/%d/%Y")
    #handles dates where year is written with two digits
    else:
        date_object =  datetime.datetime.strptime(date_string,"%m/%d/%y")

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

#plot modis channel values by themselves and prints mean
#ignores invalid values
#plots blue by itself+reports mean
def plot_blue(modis_df):
    blue_df = modis_df[modis_df['Blue mean']>=0]
    plt.hist(blue_df['Blue mean'], bins = 150)
    print(blue_df['Blue mean'].mean())
    plt.title("Average Blue Channel Value in 2016 Modis Images")
    plt.ylabel("# Images")
    plt.xlabel("Average Blue Channel ")
    plt.show()
    

#plots green by itself+reports mean
def plot_green(modis_df):
    green_df = modis_df[modis_df['Green mean']>=0]
    plt.hist(green_df['Green mean'], bins = 150)
    print(green_df['Green mean'].mean())
    plt.title("Average Green Value in 2016 Modis Images")
    plt.ylabel("# Images")
    plt.xlabel("Average Green Value")
    plt.show()
    

#plots blue vs. green
#only includes pictures where both colors have valid values
def plot_blue_green(modis_df):
    blue_df = modis_df[modis_df['Blue mean']>=0]
    color_df = blue_df[blue_df['Green mean']>=0]
    plt.scatter(color_df['Green mean'],color_df['Blue mean'])
    plt.title("Average Green vs. Average Blue Value in 2016 Modis Images")
    plt.xlabel("Average Green Value Brightness")
    plt.ylabel("Average Blue Value Brightness")
    plt.show()

#gathers all the 2016 csv files into one dataframe
#returns dataframe
def get_epa(epa_directory, year = '2016'):
    files = os.listdir(epa_directory)
    first_file = True
    for file in files:
        if file[-8:]== year + ".csv":
            new_df = pandas.read_csv(epa_directory + file)
            if not first_file:
                df = df.append(new_df,ignore_index=True)
            else:
                df = new_df
                first_file = False
    return df

def get_modis_means(modis_means_filename, modis_means_directory):
    modis_df = pandas.read_csv(modis_means_directory + modis_means_filename)
    return modis_df

#plots pm vs green, pm vs. blue
#also reports pearson correlations between pm and green, pm and blue
#if under_fifty = True, only looks at PM under 50
#if false, only looks at PM over 50
def plot_pm_vs_modis(epa_df,modis_df,under_fifty = True):
    pm_list = []
    green_list = []
    blue_list = []
    for row in range(len(epa_df)):
        if (epa_df['Daily Mean PM2.5 Concentration'][row]>50 and not under_fifty) or (epa_df['Daily Mean PM2.5 Concentration'][row]<50 and  under_fifty):
            if row % 10000 == 0:
                print("At row:")
                print(str(row))
            
            file_name = epa_to_modis_file_name(epa_df['Date'][row],epa_df['Site ID'][row])
            modis_row = modis_df[modis_df['Filename']==file_name]
            pm_list.append(epa_df['Daily Mean PM2.5 Concentration'][row])
            green = modis_row['Green mean'][modis_row.index[0]]
            blue = modis_row['Blue mean'][modis_row.index[0]]
            green_list.append(green)
            blue_list.append(blue)
    
    plt.scatter(pm_list,green_list)
    print(np.corrcoef(pm_list,green_list))
    plt.title("MODIS Green Values vs. PM2.5")
    plt.show()

    print(np.corrcoef(pm_list,blue_list))
    plt.title("MODIS Blue Values vs. PM2.5")
    plt.scatter(pm_list,blue_list)
    plt.show()



if __name__ == "__main__":    

    means_file = "modis_means_2x2.csv"
    modis_dir = "/home/sarahciresi/gcloud/cs325b-airquality/DataVisualization/channel_means/"
    epa_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/epa/"
    
    #epa = get_epa(epa_dir)
    #modis = get_modis_means(means_file, modis_dir)
    #print(epa)
    #print(modis)
    #plot_blue(modis)
    #plot_green(modis)
    #plot_blue_green(modis)
    #plot_pm_vs_modis(epa,modis)
    
