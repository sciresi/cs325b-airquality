import os
import pandas
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import sys
sys.path.insert(0, '/home/sarahciresi/gcloud/cs325b-airquality/DataVisualization')
sys.path.insert(0, '/Users/sarahciresi/Documents/GitHub/Fall2019/cs325b-airquality/DataVisualization')
from  modis_vis_with_epa import get_modis_means, get_epa, epa_to_modis_file_name, plot_pm_vs_modis
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from read_tiff import flatten

def run_baseline_model(epa_data, modis_means):
    
    dates = set()
    for date in epa_data['Date']:
        dates.add(date)

    MSE = 0
    num_predictions = 0
    
    all_date_y_train = []
    all_date_y_test = []
    all_date_y_pred = []
    all_date_y_train_preds = []
    all_date_epa_site_train_order = []
    all_date_epa_site_test_order = []
    all_dates_train = []
    all_dates_test = []

    # Goes date by date to get MSE
    # Each date has a list of stations that have measuerments from that date

    for idx, date in enumerate(dates):
        if idx % 10 == 0:
            print("Processing date {}: {} ".format(idx, date))
        
        date_df = epa_data[epa_data['Date']==date]

        # X info is latitude, longitude; y is PM2.5; epa_set_ids tracks corresponnding site_ids
        X = []
        y = []
        epa_site_ids = []
        cur_date = []

        for i in range(len(date_df)):
            lat = np.radians(date_df['SITE_LATITUDE'][date_df.index[i]])
            long = np.radians(date_df['SITE_LONGITUDE'][date_df.index[i]])
            pm = date_df['Daily Mean PM2.5 Concentration'][date_df.index[i]]
            epa_site_id = date_df['Site ID'][date_df.index[i]]
            X.append([lat,long]) 
            y.append(pm)
            epa_site_ids.append(epa_site_id)
            cur_date.append(date)

        # Shuffle data and split into train/test sets
        X, y, epa_site_ids = shuffle(X, y, epa_site_ids)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False) # already shuffled
        X_train_, X_test_, epa_site_train, epa_site_test =  train_test_split(X, epa_site_ids, test_size=0.3, shuffle=False)
        _, _, cur_date_train, cur_date_test = train_test_split(X,  cur_date, test_size=0.3, shuffle=False)

        # nearest neighbors, as determined by haversine (distance between latitude,longitude coordinate pairs)
        knn = KNeighborsRegressor(n_neighbors=1,metric="haversine")
        knn.fit(X_train,y_train)
        y_pred = knn.predict(X_test)

        # Combine PM prediction from nearest neighbor with 2x2 aod data in simple linear regression model

        # Get the nearest neighbors of train data (not including point itself)
        y_train_nn_indices = knn.kneighbors(X_train)[1]
        y_train_nn_indices = [y for x in y_train_nn_indices for y in x] # flatten the list
        y_train_preds = np.asarray(y_train)[y_train_nn_indices]

        all_date_y_train_preds.append(y_train_preds.tolist())
        all_date_y_pred.append(y_pred)
        all_date_y_train.append(y_train)
        all_date_y_test.append(y_test)
        all_date_epa_site_train_order.append(epa_site_train)
        all_date_epa_site_test_order.append(epa_site_test)
        all_dates_train.append(cur_date_train)
        all_dates_test.append(cur_date_test)

    # Flatten list all lists
    all_date_y_train_preds = flatten(all_date_y_train_preds)
    all_date_y_pred = flatten(all_date_y_pred)
    all_date_y_train = flatten(all_date_y_train)
    all_date_y_test = flatten(all_date_y_test)
    all_date_epa_site_train_order = flatten(all_date_epa_site_train_order)
    all_date_epa_site_test_order = flatten(all_date_epa_site_test_order)
    all_dates_train = flatten(all_dates_train)
    all_dates_test = flatten(all_dates_test)

    X_aod_train = np.asarray(all_date_y_train_preds).reshape(-1,1)
    X_aod_test = np.asarray(all_date_y_pred).reshape(-1,1)

    num_sites_for_all_dates_train = len(all_date_epa_site_train_order)
    num_sites_for_all_dates_test = len(all_date_epa_site_test_order)
    
    green_means_train = np.zeros((num_sites_for_all_dates_train, 1))
    blue_means_train = np.zeros((num_sites_for_all_dates_train, 1))
    green_means_test = np.zeros((num_sites_for_all_dates_test, 1))
    blue_means_test  = np.zeros((num_sites_for_all_dates_test, 1))
    
    print("Beginning mean lookup")

    for idx, epa_site in enumerate(all_date_epa_site_train_order):
        modis_filename = epa_to_modis_file_name(all_dates_train[idx], epa_site)
        modis_row = modis_means[modis_means['Filename']==modis_filename]
        green_mean = modis_row['Green mean'][modis_row.index[0]]
        blue_mean = modis_row['Blue mean'][modis_row.index[0]]
        green_means_train[idx] = green_mean
        blue_means_train[idx] = blue_mean
            
    for idx, epa_site in enumerate(all_date_epa_site_test_order):
        modis_filename = epa_to_modis_file_name(all_dates_test[idx], epa_site)
        modis_row = modis_means[modis_means['Filename']==modis_filename]
        green_mean = modis_row['Green mean'][modis_row.index[0]]
        blue_mean = modis_row['Blue mean'][modis_row.index[0]]
        green_means_test[idx] = green_mean
        blue_means_test[idx] = blue_mean

    print("Finished mean lookup")

    X_aod_train = np.concatenate((X_aod_train, green_means_train, blue_means_train), axis=1)
    X_aod_test = np.concatenate((X_aod_test, green_means_test, blue_means_test), axis=1)

    print("Training LR")
    reg = LinearRegression().fit(X_aod_train, all_date_y_train)
    
    r2_score_train = reg.score(X_aod_train, all_date_y_train)
    r2_score_test = reg.score(X_aod_test, all_date_y_test)

    print("R2 train: {}".format(r2_score_train))
    print("R2 test: {}".format(r2_score_test))

    y_pred_lr = reg.predict(X_aod_test)
    
    diff = np.square(np.asarray(y_pred_lr) - np.asarray(all_date_y_test))
    MSE = diff.sum()
    num_predictions = len(all_date_y_test)
    #print("Adding squared error of {} for date {}.".format(diff.sum()/len(y_test), date))

    MSE = MSE/num_predictions   
    print("Mean squared error across all dates:  {}".format(MSE))
    
    
def plot_pm(epa_df, modis_df, mode="<50"):
    pm_list = []
    green_list = []
    blue_list = []
    print("Number of epa rows: {}".format(len(epa_df)))
            
    for row in range(len(epa_df)):
        if row % 1000 == 0:
            print("At row: x {}".format(str(row)))
        
        if (mode == "<50" and epa_df['Daily Mean PM2.5 Concentration'][row] < 50) or (mode == ">50" and epa_df['Daily Mean PM2.5 Concentration'][row] > 50) or (mode == "all"):
            file_name = epa_to_modis_file_name(epa_df['Date'][row],epa_df['Site ID'][row])
            modis_row = modis_df[modis_df['Filename']==file_name]
            pm_list.append(epa_df['Daily Mean PM2.5 Concentration'][row])
            green = modis_row['Green mean'][modis_row.index[0]]
            blue = modis_row['Blue mean'][modis_row.index[0]]
            green_list.append(green)
            blue_list.append(blue)
    
    plt.scatter(pm_list, green_list)
    print(np.corrcoef(pm_list, green_list))
    plt.title("MODIS Green Values vs. PM2.5")
    plt.ylabel("Mean Green Values")
    plt.xlabel("PM2.5 Concentration")
    plt.show()

    print(np.corrcoef(pm_list,blue_list))
    plt.title("MODIS Blue Values vs. PM2.5")
    plt.scatter(pm_list,blue_list)
    plt.ylabel("Mean Blue Values")
    plt.xlabel("PM2.5 Concentration")
    plt.show()


if __name__ == "__main__":

    means_file = "modis_means_2x2.csv"
    #modis_dir = "/home/sarahciresi/gcloud/cs325b-airquality/DataVisualization/channel_means/"
    #epa_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/epa/"
    modis_dir = "/Users/sarahciresi/Documents/GitHub/Fall2019/cs325b-airquality/DataVisualization/channel_means/"
    epa_dir = "/Users/sarahciresi/Documents/GitHub/Fall2019/cs325b-airquality/cs325b/data/epa/"

    epa_2016 = get_epa(epa_dir, year = "2016")
    modis_means_2016 = get_modis_means(means_file, modis_dir)
    run_baseline_model(epa_2016, modis_means_2016)
    #plot_pm(epa_2016, modis_means_2016)
