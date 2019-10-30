import os
import pandas
import csv
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
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
import utils

class Small_CNN(nn.Module):
    def __init__(self, device = "cpu"):
        super(Small_CNN, self).__init__()

        in_channels = 13
        out_channels1 = 32
        out_channels2 = 64
        out_channels3 = 128
        
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
    running_loss = 0
    summaries  = []
    batch_size = 32
    num_batches = len(dataloader)
    train_dataset_size = num_batches * batch_size
    
    print("Training for one epoch on {} batches.".format(num_batches))
          
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
       
            #if i % 100 == 0:
            # Move to cpu and convert to numpy
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()
            
            # Compute batch metrics
            r2 = r2_score(labels_batch, output_batch) 
            summary_batch = {'average r2': r2, 'average MSE loss': loss.item()}
            summaries.append(summary_batch)
            
            # Update the average loss
            loss_sum += loss.item()
            loss_steps += 1
            
            # loss = average loss over batch * num examples in batch 
            #-- unsure if this is not being computed correctly; much higher than it should be compared
            # to the version saved in mean_metrics
            running_loss += loss.item() * input_batch.size(0)  
            
            # Display batch loss and r2
            loss_str = '{:05.3f}'.format(loss.item())
            r2_str = '{:01.3f}'.format(r2)
            t.set_postfix(loss=loss_str, r2=r2_str) 
            t.update()
    
    # Epoch loss
    epoch_loss = running_loss / train_dataset_size
    
    # Save metrics
    mean_metrics = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]} 
    mean_metrics['epoch loss'] = epoch_loss
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in mean_metrics.items())
    print("Train metrics: {}".format(metrics_string))
    print("Average train loss over epoch: {} ".format(epoch_loss))

    return mean_metrics


def evaluate(model, loss_fn, dataloader):
    '''
    Evaluates the model for 1 epoch on all batches in the dataloader.
    '''
    # Set model to eval mode
    model.eval()
   
    summaries = []
    loss_sum = 0
    loss_steps = 0
    running_loss = 0
    batch_size = 32
    num_batches = len(dataloader)
    val_dataset_size = num_batches * batch_size
    
    print("Evaluating on {} batches".format(num_batches))
    
    with tqdm(total=num_batches) as t:

        with torch.no_grad():
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

                # Move to cpu and convert to numpy
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # Update the average loss
                loss_steps += 1
                loss_sum += loss.item()
                running_loss += loss.item() * input_batch.size(0)

                # Save metrics
                r2 = r2_score(labels_batch, output_batch) #.cpu().detach().numpy())
                summary_batch = {'average r2': r2, 'average MSE loss': loss.item()}
                summaries.append(summary_batch)

                # Display batch loss and r2
                loss_str = '{:05.3f}'.format(loss.item())
                r2_str = '{:01.3f}'.format(r2)
                t.set_postfix(loss=loss_str, r2=r2_str) 
                t.update()
    
    # Epoch loss
    epoch_loss = running_loss / val_dataset_size
   
    mean_metrics = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]}    
    mean_metrics['epoch loss'] = epoch_loss
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in mean_metrics.items())
    print("Evaluation metrics: {}".format(metrics_string))
    print("Average evaluation loss over epoch: {} ".format(epoch_loss))
    
    return mean_metrics


def train_and_evaluate(model, optimizer, loss_fn, train_dataloader, 
                       val_dataloader, num_epochs, saved_weights_file = None):
    '''
    Trains the model and evaluates at every epoch
    '''
    
    model_dir = "/home/sarahciresi/gcloud/cs325b-airquality/"

    # If a saved weights file for the model is specified, reload the weights
    if saved_weights_file is not None:
        saved_weights_path = os.path.join(model_dir, saved_weights_file + '.pth.tar')
        print("Restoring parameters from {}".format(saved_weights_path))
        utils.load_checkpoint(saved_weights_path, model, optimizer)

    best_val_r2 = 0.0
    
    all_train_losses = []
    all_val_losses = []
    all_train_r2 = []
    all_val_r2 = []
    
    for epoch in range(num_epochs):
        
        print("Running Epoch {}/{}".format(epoch, num_epochs))
              
        # Train on all batches
        train_mean_metrics = train(model, optimizer, loss_fn, train_dataloader)

        # Evaluate on validation set
        val_mean_metrics = evaluate(model, loss_fn, val_dataloader)
        
        # Save losses and r2 from this epoch
        all_train_losses.append( train_mean_metrics['epoch loss'] )
        all_val_losses.append( val_mean_metrics['epoch loss'] )
        all_train_r2.append( train_mean_metrics['average r2'] )
        all_val_r2.append( val_mean_metrics['average r2'] )
    
        val_r2 = val_mean_metrics['average r2']
        is_best = val_r2 > best_val_r2
        
        # Save current model weights from this epoch
        utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best, checkpoint=model_dir)

        # If best_eval, save to best_save_path
        if is_best:
            print("Found new best R2 value")
            best_val_r2 = val_r2

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights_regression.json")
            utils.save_dict_to_json(val_mean_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights_regression.json")
        utils.save_dict_to_json(val_mean_metrics, last_json_path)
    
    print("Train losses: {} ".format(all_train_losses))
    print("Val losses: {} ".format(all_val_losses))
    
    # Plot losses
    plt.figure(1)
    plt.plot(range(0, num_epochs), all_train_losses, label='train')
    plt.plot(range(0, num_epochs), all_val_losses, label='val')
    plt.legend(loc=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Average MSE Loss over all epochs")
    plt.show()
    plt.savefig("regression_losses_4.png")
     
    plt.figure(2)
    plt.plot(range(0, num_epochs), all_train_r2, label='train')
    plt.plot(range(0, num_epochs), all_val_r2, label='val')
    plt.legend(loc=2)
    plt.title("Average R2 over all epochs")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.show()
    plt.savefig("regression_r2_4.png")
        
    # Return train and eval metrics
    return train_mean_metrics, val_mean_metrics


if __name__ == "__main__":
    
    #save_to_csv= "data_csv_files/cleaned_data_all_temp_new_3.csv"
    #utils.remove_sent_and_save_df_to_csv(cleaned_csv, save_to_csv)
    
    cleaned_csv = "data_csv_files/cleaned_data_all_temp_new_3.csv"
    npy_dir = '/home/sarahciresi/gcloud/cs325b-airquality/cs325b/images/s2/'
    sent_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/sentinel/2016/"

    dataloaders = load_data(cleaned_csv, npy_dir, sent_dir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Small_CNN(device) 
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr= 0.0001)
    train_and_evaluate(model, optimizer, nn.MSELoss(), dataloaders['train'], dataloaders['val'], num_epochs=10)
   
    