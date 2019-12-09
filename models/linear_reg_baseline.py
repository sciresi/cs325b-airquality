import os
import pandas
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
files = os.listdir()
first_file = True
#reads in all relevant csvs
for file in files:
    if file[-4:]==".csv" and file != "broken_modis_channel_means.csv" and file !="modis_channel_means_revised.csv":
        new_df = pandas.read_csv(file)
        if not first_file:
            df = df.append(new_df,ignore_index=True)
        else:
            df = new_df
            first_file = False
print("Filtering for PM < 50")
df = df[df['Daily Mean PM2.5 Concentration']<50]
X = []
y = []
print(df.index)
for row in df.index:
    lat = df['SITE_LATITUDE'][row]
    long = df['SITE_LONGITUDE'][row]
    date_string = df['Date'][row]
    if date_string[-4:].isnumeric():
        date_object =  datetime.datetime.strptime(date_string,"%m/%d/%Y")
    #handles dates where year is written with two digits
    else:
        date_object =  datetime.datetime.strptime(date_string,"%m/%d/%y")
    X.append([lat,long,date_object.month])
    y.append(df['Daily Mean PM2.5 Concentration'][row])

print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
