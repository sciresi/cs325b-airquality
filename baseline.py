import os
import pandas
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
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
dates = set()
for date in df['Date']:
    dates.add(date)

mse = 0
num_predictions = 0
#goes date by date to get mse
for date in dates:
    date_df = df[df['Date']==date]
    #x info is latitude, longitude
    #y info is pm2.5
    X = []
    y = []
    for i in range(len(date_df)):
        lat = np.radians(date_df['SITE_LATITUDE'][date_df.index[i]])
        long = np.radians(date_df['SITE_LONGITUDE'][date_df.index[i]])
        pm = date_df['Daily Mean PM2.5 Concentration'][date_df.index[i]]
        X.append([lat,long])
        y.append(pm)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #nearest neighbors, as determined by haversine (distance between latitude,longitude coordinate pairs)
    knn = KNeighborsRegressor(n_neighbors=1,metric="haversine")
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    diff = np.square(np.asarray(y_pred) - np.asarray(y_test))
    mse += diff.sum()
    num_predictions += len(y_test)
    
print(mse/num_predictions)
    

