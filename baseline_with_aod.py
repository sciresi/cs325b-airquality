import os
import pandas
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import sys
sys.path.insert(0, '/home/sarahciresi/gcloud/cs325b-airquality/DataVisualization')
from  modis_vis_with_epa import get_modis_means, get_epa, epa_to_modis_file_name

''' Use modis_vis files instead
def get_aod_means(directory):
    aod_means = []
    file_name = epa_to_file_name(df['Date'][row],df['Site ID'][row])
    modis_row = modis_df[modis_df['Filename']==file_name]
    pm_list.append(df['Daily Mean PM2.5 Concentration'][row])
    modis_df = pandas.read_csv("modis_means_2x2.csv")
    return aod_means

dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/epa/"
files = os.listdir(dir)
first_file = True
file_num = 0

#reads in all relevant (epa?) csvs
for filen in files[:4]: ## change here to be all files
    print(filen)
    if filen[-4:]==".csv" and filen != "broken_modis_channel_means.csv" and filen !="modis_channel_means_revised.csv":
        new_df = pandas.read_csv(dir+filen, delimiter=',') 
        if not first_file:
            df = df.append(new_df,ignore_index=True)
        else:
            df = new_df
            first_file = False
        file_num+=1
'''

def run_baseline_model(epa_data, modis_means):
    
    dates = set()
    for date in epa_data['Date']:
        dates.add(date)

    MSE = 0
    num_predictions = 0

    # Goes date by date to get MSE
    # Each date has a list of stations that have measuerments from that date

    for idx, date in enumerate(dates):
        print("Processing date {} ".format(date))
        
        date_df = epa_data[epa_data['Date']==date]

        # X info is latitude, longitude; y is PM2.5; epa_set_ids tracks corresponnding site_ids
        X = []
        y = []
        epa_site_ids = []

        for i in range(len(date_df)):
            lat = np.radians(date_df['SITE_LATITUDE'][date_df.index[i]])
            long = np.radians(date_df['SITE_LONGITUDE'][date_df.index[i]])
            pm = date_df['Daily Mean PM2.5 Concentration'][date_df.index[i]]
            epa_site_id = date_df['Site ID'][date_df.index[i]]
            X.append([lat,long]) 
            y.append(pm)
            epa_site_ids.append(epa_site_id)

        # Shuffle data and split into train/test sets
        X, y, epa_site_ids = shuffle(X, y, epa_site_ids)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False) # already shuffled
        X_train_, X_test_, epa_site_train, epa_site_test =  train_test_split(X, epa_site_ids, test_size=0.3, shuffle=False)
              
        # nearest neighbors, as determined by haversine (distance between latitude,longitude coordinate pairs)
        knn = KNeighborsRegressor(n_neighbors=1,metric="haversine")
        knn.fit(X_train,y_train)
        y_pred = knn.predict(X_test)

        # Combine PM predictiton from nearest neighbor with 2x2 aod data in simple linear regression model

        # Get the nearest neighbors of train data (not including point itself)
        y_train_nn_indices = knn.kneighbors(X_train)[1]
        y_train_nn_indices = [y for x in y_train_nn_indices for y in x] # flatten the list
        y_train_preds = np.asarray(y_train)[y_train_nn_indices]

        # sanity check that it works for test data
        # y_test_nn_indices = knn.kneighbors(X_test)[1]
        # y_test_nn_indices = [y for x in y_test_nn_indices for y in x]
        # y_test_preds = np.asarray(y_train)[y_test_nn_indices]
        
        X_aod_train = np.asarray(y_train_preds).reshape(-1,1)
        X_aod_test = np.asarray(y_pred).reshape(-1,1)

        num_sites_for_date_train = len(epa_site_train)
        num_sites_for_date_test = len(epa_site_test)
        
        green_means_train = np.zeros((num_sites_for_date_train, 1))
        blue_means_train = np.zeros((num_sites_for_date_train, 1))
        green_means_test = np.zeros((num_sites_for_date_test, 1))
        blue_means_test  = np.zeros((num_sites_for_date_test, 1))
        
        for idx, epa_site in enumerate(epa_site_train):
            modis_filename = epa_to_modis_file_name(date, epa_site)
            modis_row = modis_means[modis_means['Filename']==modis_filename]
            green_mean = modis_row['Green mean'][modis_row.index[0]]
            blue_mean = modis_row['Blue mean'][modis_row.index[0]]
            green_means_train[idx] = green_mean
            blue_means_train[idx] = blue_mean

        for idx, epa_site in enumerate(epa_site_test):
            modis_filename = epa_to_modis_file_name(date, epa_site)
            modis_row = modis_means[modis_means['Filename']==modis_filename]
            green_mean = modis_row['Green mean'][modis_row.index[0]]
            blue_mean = modis_row['Blue mean'][modis_row.index[0]]
            green_means_test[idx] = green_mean
            blue_means_test[idx] = blue_mean
            
        X_aod_train = np.concatenate((X_aod_train, green_means_train, blue_means_train), axis=1)
        X_aod_test = np.concatenate((X_aod_test, green_means_test, blue_means_test), axis=1)
        
        reg = LinearRegression().fit(X_aod_train, y_train)

        #r2_score_train = reg.score(X_aod_train, y_train)
        #r2_score_test = reg.score(X_aod_test, y_test)

        #print("R2: {}".format(r2_score_train))
        #print("R2 test: {}".format(r2_score_test))

        y_pred = reg.predict(X_aod_test)
        
        diff = np.square(np.asarray(y_pred) - np.asarray(y_test))
        MSE += diff.sum()
        num_predictions += len(y_test)

        print("Adding squared error of {} for date {}.".format(diff.sum()/len(y_test), date))

    MSE = MSE/num_predictions   
    print("Mean squared error acrosss all dates:  {}".format(MSE))
    

if __name__ == "__main__":

    means_file = "modis_means_2x2.csv"
    modis_dir = "/home/sarahciresi/gcloud/cs325b-airquality/DataVisualization/channel_means/"
    epa_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/epa/"

    epa_2016 = get_epa(epa_dir, year = "2016")
    modis_means_2016 = get_modis_means(means_file, modis_dir)
    run_baseline_model(epa_2016, modis_means_2016)
