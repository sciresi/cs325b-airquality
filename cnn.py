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
from read_tiff import get_sentinel_img, get_sentinel_img_from_row, save_sentinel_from_eparow
from pandarallel import pandarallel
from utils import remove_missing_sent, remove_full_missing_modis, remove_partial_missing_modis, save_df_to_csv, load_sentinel_npy_files, get_PM_from_row

class Small_CNN(nn.Module):
    def __init__(self):
        super(Small_CNN, self).__init__()
        #12,50,1 works for MSE ~ 1 on 10 datapoints
        #12,50,20,1 with lr .02 gets MSE ~21 on all of 2016

        in_channels = 13
        out_channels1 = 32
        out_channels2 = 64
        out_channels3 = 128
        
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=2, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=2, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(out_channels2, out_channels3, kernel_size=2, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 25 * 25, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # regression 

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.reshape(x.size(0), 128 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def normalize_train(X_train):
    
    means = np.mean(X_train, axis = 0)
    stdev = np.std(X_train, axis = 0)
    X_train -= means
    X_train /= stdev
    
    return X_train, means, stdev

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
    input_data_normalized, means, stdev = normalize_train(input_data)
    input_data_normalized = torch.from_numpy(input_data_normalized).float()
    input_y = torch.from_numpy(y_train).unsqueeze(1).float()
   
    #output = small_net(input_data)
    learning_rate = .002
    optimizer = optim.SGD(small_cnn.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    num_epochs = 100
    for epoch in range(num_epochs):
        y_pred = small_cnn(input_data_normalized)
        loss = loss_fn(y_pred, input_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {}/{}. Loss: {} ".format(epoch, num_epochs, loss.item()))
        
    return small_net,means, stdev
    
    

def get_train_test_data(master_filename, sent_directory, threshold = None, sent_im_size = 200): #filter_empty_temp=True):
    
    #pandarallel.initialize()
    
    df = pandas.read_csv(master_filename)
   
    if threshold != None:
        df = df[df['Daily Mean PM2.5 Concentration'] < threshold]
    
    total_num_samples = len(df)
    subset_num = total_num_samples
    
    print("Looking at 100000-{}/{} samples".format(subset_num, total_num_samples))
    
    #X = np.zeros((subset_num, sent_im_size, sent_im_size, 3))
    #y = np.zeros((subset_num))
    
    #df_subset = df.iloc[100000:]  
    #print("Starting to save all sentinel tensors")
    #df_subset.parallel_apply(save_sentinel_from_eparow, axis=1, dir_path = sent_directory, im_size = sent_im_size)
    #print("Finished saving all sentinel tensors")
    
    df_subset = df.iloc[:100]
    
    # Loads all .npy file into panda Series. Each element in pandas series is an image tensor (200, 200, 13)
    # For now, convert to np array of (num_samples, 200, 200, 13)
    print("Loading 1st 100 ims")
    npy_directory = '/home/sarahciresi/gcloud/cs325b-airquality/cs325b/images/s2/'
    #imgs_1000 = df_first_100.parallel_apply(load_sentinel_npy_files, axis=1, npy_files_dir_path = npy_directory)
    #X_train = np.array(imgs_1000.values.tolist())
    #y_train = df_first_100.parallel_apply(get_PM_from_row, axis=1)
    
    #print("Finished loading. X_train size {}".format(X_train.shape))
    #print("y_train size {}".format(y_train.shape))

        
    #imgs = df_subset.parallel_apply(get_sentinel_img_from_row, axis=1, dir_path = sent_directory, im_size = sent_im_size)
    #print(imgs.shape)  
    #np.save("sent_imgs_0-10000_X.npy", X)
    #np.save("sent_imgs_0-10000_y.npy", y)
    
    count = 0
    X = np.zeros((100, 200, 200, 13))
    y = np.array(df_subset['Daily Mean PM2.5 Concentration'])
    
    for i, row in df_subset.iterrows():
        if i % 20 == 0:
            print("Processing sentinel image {} / {}".format(i, 100))
        
        img = load_sentinel_npy_files(row, npy_directory)
        #sent_filename = str(epa_station_date['SENTINEL_FILENAME'])
        #day_index = int(epa_station_date['SENTINEL_INDEX'])
        #img, to_include =  get_sentinel_img(sent_directory, sent_filename, day_index, im_size)
        X[count] = img
        count += 1
   
    #np.save("sent_ims_1-100.npy", X)

    # Reshape from (m, h, w, channels) to (m, channels, h, w) 
    X = np.transpose(X, (0, 3, 1, 2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)    
    return X_train, X_test, y_train, y_test
    
    
if __name__ == "__main__":

    #master_file = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/master_epa_weather_modis_sentinel_2016.csv"
    cleaned_mf = "data_csv_files/new2_cleaned_data_temp_not_null.csv"
    sent_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/sentinel/2016/"

    X_train, X_test, y_train, y_test = get_train_test_data(cleaned_mf, sent_dir)
  
    small_net, means, std = train_small_net(X_train, y_train)
    #test_net(small_net, X_test, y_test, means, std)
    #res = save_sentinel_data(master_file)
                       
                         
    
