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
from dataloader import load_data
from read_tiff import get_sentinel_img, get_sentinel_img_from_row, save_sentinel_from_eparow
from pandarallel import pandarallel
from utils import remove_missing_sent, remove_full_missing_modis, remove_partial_missing_modis, remove_sent_and_save_df_to_csv, load_sentinel_npy_files, get_PM_from_row

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
        x = x.reshape(-1)
        
        return x

def normalize_train(X_train):
    
    means = np.mean(X_train, axis = 0)
    stdev = np.std(X_train, axis = 0)
    X_train -= means
    X_train /= stdev
    
    return X_train, means, stdev

    
def train(model, optimizer, loss_fn, dataloader):
    '''
    Trains the model for 1 epoch on all batches in the dataloader.
    '''
    # Set model to train mode
    model.train()
   
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

            # Compute gradients and perform parameter updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # update the average loss
            loss_sum += loss.item()
            loss_steps += 1

            # display average loss
            t.set_postfix(loss='{:05.3f}'.format(loss_sum/loss_steps))
            t.update()
            
            #if i % 200 == 0:
            print("Batch{}/{} MSE Loss: {} ".format(i, num_batches, loss.item()))
    
    return output_batch, loss.item()

def train_and_evaluate(model, optimizer, loss_fn, train_dataloader, val_dataloader, num_epochs):
    '''
    Trains the model and evaluates at every epoch
    '''
    
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        
        print("Epoch {}/{}".format(epoch, num_epochs))
              
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


if __name__ == "__main__":
    
    #save_to_csv= "data_csv_files/cleaned_data_all_temp_new_3.csv"
    #remove_sent_and_save_df_to_csv(cleaned_csv, save_to_csv)
    
    cleaned_csv = "data_csv_files/cleaned_data_all_temp_new_3.csv"
    npy_dir = '/home/sarahciresi/gcloud/cs325b-airquality/cs325b/images/s2/'
    sent_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/sentinel/2016/"

    dataloaders = load_data(cleaned_csv, npy_dir, sent_dir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Small_CNN(device) 
    model.to(device)
    
    optimizer = optim.Adam(model.parameters())
    train_and_evaluate(model, optimizer, nn.MSELoss(), dataloaders['train'], dataloaders['val'], num_epochs=5)
   
    