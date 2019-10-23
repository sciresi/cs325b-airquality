import os
import pickle
import pandas
import csv
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/sarahciresi/gcloud/cs325b-airquality/DataVisualization')
sys.path.insert(0, '/Users/sarahciresi/Documents/GitHub/Fall2019/cs325b-airquality/DataVisualization')
from read_tiff import get_sentinel_img, get_sentinel_img_from_row
from pandarallel import pandarallel
from utils import remove_missing_sent, remove_full_missing_modis, remove_partial_missing_modis

class Small_CNN(nn.Module):
    def __init__(self):
        super(Small_CNN, self).__init__()
        #12,50,1 works for MSE ~ 1 on 10 datapoints
        #12,50,20,1 with lr .02 gets MSE ~21 on all of 2016

        in_channels = 3
        out_channels = 18
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=2, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(out_channels2, out_channels3, kernel_size=2, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # get in_dim into linear of out dim of conv
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 48)

        # regression 

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # regression
        
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
    small_cnn = Small_CNN()
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
    
    

def get_train_test_data(master_filename, sent_directory, threshold = None, sent_im_size = 200): #filter_empty_temp=True):
    
    pandarallel.initialize()
    
    df = pandas.read_csv(master_filename)
           
    if threshold != None:
        df = df[df['Daily Mean PM2.5 Concentration'] < threshold]
    
    total_num_samples = len(df)
    subset_num = 5000
    
    print("Looking at {}/{} samples".format(subset_num, total_num_samples))
    
    X = np.zeros((subset_num, sent_im_size, sent_im_size, 3))
    y = np.zeros((subset_num))
    
    df_subset = df.iloc[:subset_num]  
    count = 0

    imgs = df_subset.parallel_apply(get_sentinel_img_from_row, axis=1, dir_path = sent_directory, im_size = sent_im_size)
    print(imgs.shape)  
    np.save("sent_imgs_0-10000_X.npy", X)
    np.save("sent_imgs_0-10000_y.npy", y)
    '''
    for i, epa_station_date in df_subset.iterrows():
        if i % 1000 == 0:
            print("Processing sentinel image {} / {}".format(i, num_samples))
        sent_filename = str(epa_station_date['SENTINEL_FILENAME'])
        day_index = int(epa_station_date['SENTINEL_INDEX'])
        img, to_include =  get_sentinel_img(sent_directory, sent_filename, day_index, im_size)
        if to_include:
            X[count] = img
            y[count] = epa_station_date['Daily Mean PM2.5 Concentration']
            count += 1
    '''
    #np.save("sent_ims_apply.npy", X)
    #np.save("sent_ims_new_2.npy", X)
    #y = np.array(df['Daily Mean PM2.5 Concentration'])
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)    
    #return X_train, X_test, y_train, y_test
    
    return None, None, None, None

if __name__ == "__main__":

    #master_file = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/master_epa_weather_modis_sentinel_2016.csv"
    cleaned_mf = "data_csv_files/cleaned_data_temp_not_null.csv"
    sent_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/sentinel/2016/"

    X_train, X_test, y_train, y_test = get_train_test_data(cleaned_mf, sent_dir)
    
    #small_net, means, std = train_small_net(X_train, y_train)
    #test_net(small_net, X_test, y_test, means, std)
    #res = save_sentinel_data(master_file)
                       
                         
    
