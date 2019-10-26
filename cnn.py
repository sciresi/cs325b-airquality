import os
import pickle
import pandas
import csv
import datetime
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/sarahciresi/gcloud/cs325b-airquality/DataVisualization')
sys.path.insert(0, '/Users/sarahciresi/Documents/GitHub/Fall2019/cs325b-airquality/DataVisualization')
import dataloader
from read_tiff import get_sentinel_img, get_sentinel_img_from_row, save_sentinel_from_eparow
from pandarallel import pandarallel
from utils import remove_missing_sent, remove_full_missing_modis, remove_partial_missing_modis, save_df_to_csv, load_sentinel_npy_files, get_PM_from_row

class Small_CNN(nn.Module):
    def __init__(self, device = "cpu"):
        super(Small_CNN, self).__init__()

        in_channels = 13
        out_channels1 = 32
        out_channels2 = 64
        out_channels3 = 128
        #out_channels3 = 256
        
        self.device = device
        
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=2, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=2, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(out_channels2, out_channels3, kernel_size=2, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(128 * 24 * 24, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1) 

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        #print(x.shape)
        x = x.reshape(x.size(0), 128 * 24 * 24)
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

    
def train(model, optimizer, loss_fn, dataloader):
    '''
    Trains the model for 1 epoch on ? batches.
    '''
    # Set model to train mode
    model.train()
    
    # Normalize data.. really do this somewhere else but here fore now
    
    # X_train, means, stdev =  normalize_train(X_train)
    # input_batch = torch.from_numpy(X_train).float()
    # labels_batch = torch.from_numpy(y_train).unsqueeze(1).float()
   
    loss_sum = 0
    loss_steps = 0
    num_batches = len(dataloader)
                      
    with tqdm(total=num_batches) as t:
        
        for i, sample_batch in enumerate(dataloader):
            input_batch, labels_batch = sample_batch['image'], sample_batch['pm']

            # Move to GPU if available       
            input_batch = input_batch.to(model.device, dtype=torch.float)
            labels_batch = labels_batch.to(model.device, dtype=torch.float)

            # Convert to torch Variables
            input_batch, labels_batch = Variable(input_batch), Variable(labels_batch)

            # Forward pass and calculate loss
            output_batch = model(input_batch) # y_pred
            loss = loss_fn(output_batch, labels_batch)

            print(labels[1])
            print(output_batch[1])

            # Compute gradients and perform parameter updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # update the average loss
            loss_sum += loss.item()
            loss_steps += 1

            t.set_postfix(loss='{:05.3f}'.format(loss_sum/loss_steps))
            t.update()
            
            print("Batch{}/{} MSE Loss: {} ".format(i, num_batches, loss.item()))
    
    return output_batch, loss.item()

def train_and_evaluate(model, optimizer, loss_fn, train_dataloader, val_dataloader, num_epochs):
    '''
    Trains the model and evaluates at every epoch
    '''
    
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        
        # Train on all batches
        y_pred, loss = train(model, optimizer, loss_fn, train_dataloader)

        # Evaluate on validation set
        #val_metrics = evaluate(model, loss_fn, val_dataloader)
        #val_MSE = val_metrics['MSE']
        #r2 = r2_score(y_train, y_pred.cpu().detach().numpy())
        #print("Epoch {}/{}: MSE Loss:  {:0.2f}   R2: {:0.2f}".format(epoch, num_epochs, loss, r2))

        print("Epoch {}/{}: MSE Loss:  {:0.2f}  ".format(epoch, num_epochs, loss))

    # Return last updated y_pred
    return y_pred

def get_train_val_test_data(master_filename, sent_directory, threshold = None, sent_im_size = 200): #filter_empty_temp=True):
    
    #pandarallel.initialize()
    
    df = pandas.read_csv(master_filename)
          
    if threshold != None:
        df = df[df['Daily Mean PM2.5 Concentration'] < threshold]
    
    total_num_samples = len(df)
    subset_num = 400
    
    print("Looking at {}/{} samples".format(subset_num, total_num_samples))
    
    #X = np.zeros((subset_num, sent_im_size, sent_im_size, 3))
    #y = np.zeros((subset_num))
    
    #df_subset = df.iloc[100000:]  
    #print("Starting to save all sentinel tensors")
    #df_subset.parallel_apply(save_sentinel_from_eparow, axis=1, dir_path = sent_directory, im_size = sent_im_size)
    #print("Finished saving all sentinel tensors")
    
    df_subset = df.iloc[:subset_num]
    
    # Loads all .npy file into panda Series. Each element in pandas series is an image tensor (200, 200, 13)
    # For now, convert to np array of (num_samples, 200, 200, 13)
    print("Loading {} images".format(subset_num))
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
    X = np.zeros((subset_num, 200, 200, 13))
    y = np.array(df_subset['Daily Mean PM2.5 Concentration'])
    
    for i, row in df_subset.iterrows():
        if i % 50 == 0:
            print("Processing sentinel image {} / {}".format(i, subset_num))
        
        img = load_sentinel_npy_files(row, npy_directory)
        #sent_filename = str(epa_station_date['SENTINEL_FILENAME'])
        #day_index = int(epa_station_date['SENTINEL_INDEX'])
        #img, to_include =  get_sentinel_img(sent_directory, sent_filename, day_index, im_size)
        X[count] = img
        count += 1
   
    #np.save("sent_ims_1-100.npy", X)

    # Reshape from (m, h, w, channels) to (m, channels, h, w) 
    X = np.transpose(X, (0, 3, 1, 2))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
if __name__ == "__main__":

    #X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_data(cleaned_mf, sent_dir)
    
    csv_ = "data_csv_files/cleaned_data_all_temp_new.csv"
    npy_dir = '/home/sarahciresi/gcloud/cs325b-airquality/cs325b/images/s2/'
    sent_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/sentinel/2016/"
    dataset = dataloader.SentinelDataset(master_csv_file=csv_, s2_npy_root_dir=npy_dir,
                                         s2_tif_root_dir = sent_dir)
    
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    val_dataloader = train_dataloader
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Small_CNN(device) 
    model.to(device)
    
    optimizer = optim.Adam(model.parameters())
    train_and_evaluate(model, optimizer, nn.MSELoss(), train_dataloader, val_dataloader, num_epochs=200)
    
