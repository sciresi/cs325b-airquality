import os
import pandas
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class Small_Net(nn.Module):
    def __init__(self):
        super(Small_Net, self).__init__()
        #12,50,1 works for MSE ~ 1 on 10 datapoints
        #12,50,20,1 with lr .02 gets MSE ~21 on all of 2016
        
        self.fc1 = nn.Linear(17,50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def normalize_train(X_train):
    means = X_train.mean(axis=0)
    new_X = X_train-means
    var = X_train.var(axis=0)
    std = np.sqrt(var)
    #when there's no variance (all the same value, probably a lot of sentinel values for missing data)
    #skips the division by setting std = 1, thus avoiding division by 0
    for i in range(len(std)):
        if std[i]==0:
            std[i]=1
    new_X = new_X/std
    return new_X,means,std
def test_net(net, X_test,y_test,means,std):
    X_test = (X_test-means)/std
    input_data = torch.from_numpy(X_test).float()
    target_y = torch.from_numpy(y_test).unsqueeze(1).float()
    y_pred = net(input_data)
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, target_y)
    print("Test MSE:")
    print(loss.item())
    
def train_small_net(X_train, y_train):
    small_net = Small_Net()
    """small_net = nn.Sequential(nn.Linear(12,10),
                              nn.ReLU(),
                              nn.Linear(10,1),
                              )"""
    input_data = X_train
    input_data,means,std = normalize_train(input_data)
    input_data = torch.from_numpy(input_data).float()
    input_y = torch.from_numpy(y_train).unsqueeze(1).float()

    out = small_net(input_data)
    learning_rate = .02
    optimizer = optim.SGD(small_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    for epoch in range(100):
        y_pred = small_net(input_data)
        loss = loss_fn(y_pred, input_y)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return small_net, means, std
        
        
#should normalize!!!
def get_train_test_data(filename,under_fifty=True,filter_empty_temp=True):
    df = pandas.read_csv(filename)
    if under_fifty:
        df = df[df['Daily Mean PM2.5 Concentration']<50]
    else:
        df = df[df['Daily Mean PM2.5 Concentration']>50]
    df = df[df['TMAX'].notnull()]
    df = df[df['TMIN'].notnull()]
    df['PRCP'].fillna(-1,inplace=True)
    df['SNOW'].fillna(-1,inplace=True)
    df['SNWD'].fillna(-1,inplace=True)
    dates = pandas.to_datetime(df['Date'])
    months = dates.dt.month
    years = dates.dt.year
    X = np.stack((df['SITE_LATITUDE'],df['SITE_LONGITUDE'],months,years,
              df['Blue [0,0]'],df['Blue [0,1]'], df['Blue [1,0]'], df['Blue [1,1]'],
              df['Green [0,0]'],df['Green [0,1]'], df['Green [1,0]'], df['Green [1,1]'],
            df['PRCP'], df['SNOW'], df['SNWD'], df['TMAX'], df['TMIN']),axis=1)
    y = np.array(df['Daily Mean PM2.5 Concentration'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_train_test_data("master_epa_weather_modis_sentinel_2016.csv")
small_net, means, std = train_small_net(X_train, y_train)
test_net(small_net, X_test, y_test, means, std)
