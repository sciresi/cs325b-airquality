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

class CNN_combined(nn.Module):
    def __init__(self, device = "cpu"):
        super(CNN_combined, self).__init__()

        in_channels = 13
        out_channels1 = 32
        out_channels2 = 64
        out_channels3 = 128
        num_ff_features = 14
        
        self.device = device
        
        # Conv portion
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=5, stride=1, padding=1) # 11
        self.bn1 = nn.BatchNorm2d(out_channels1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=3, stride=1, padding=1) #7
        self.bn2 = nn.BatchNorm2d(out_channels2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(out_channels2, out_channels3, kernel_size=3, stride=1, padding=1) #5
        self.bn3 = nn.BatchNorm2d(out_channels3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=0.2)
        
        # Feed forward portion
        self.fffc1 = nn.Linear(num_ff_features, 100)
        self.fffc2 = nn.Linear(100,100)
        self.fffc3 = nn.Linear(100,100)
        self.dropfffc = nn.Dropout(p=0.2)
        
        # Recombined portion
        self.fc1 = nn.Linear(128 * 24 * 24 + 100, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1) 

    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)   
            
    def forward(self, x1, x2):
        
        # Conv
        x1 = self.conv1(x1)
        x1 = F.relu(self.bn1(x1))
        x1 = self.pool1(x1)
        x1 = self.conv2(x1)
        x1 = F.relu(self.bn2(x1))
        x1 = self.pool2(x1)
        x1 = self.conv3(x1)
        x1 = F.relu(self.bn3(x1))
        x1 = self.pool3(x1)
        x1 = x1.reshape(x1.size(0), 128 * 24 * 24)
        x1 = self.drop(x1)
        
        # FF 
        x2 = F.relu(self.fffc1(x2))
        x2 = F.relu(self.fffc2(x2))
        x2 = F.relu(self.fffc3(x2))
        x2 = self.dropfffc(x2)
        
        # Combined 
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape(-1)
        
        return x

def train(model, optimizer, loss_fn, dataloader, batch_size, epoch, scheduler=None):
    '''
    Trains the model for 1 epoch on all batches in the dataloader.
    '''
    
    summaries  = []
    batch_size = 32
    num_batches = len(dataloader)
    train_dataset_size = num_batches * batch_size
    
    # Set model to train mode
    model.train()
   
    print("Training for one epoch on {} batches.".format(num_batches))
          
    with tqdm(total=num_batches) as t:
        
        for i, sample in enumerate(dataloader):

            input_batch, labels_batch = sample['image'], sample['pm']
            lat, lon, mon = sample['lat'], sample['lon'], sample['month']
            prec, snow, snwd = sample['prec'], sample['snow'], sample['snwd']
            tmax, tmin = sample['tmax'], sample['tmin']
            mb1, mb2, mb3, mb4 = sample['mb1'], sample['mb2'], sample['mb3'], sample['mb4']
            mg1, mg2, mg3, mg4 = sample['mg1'], sample['mg2'], sample['mg3'], sample['mg4']
            
            #features = torch.stack((lat, lon, mon, tmax, tmin, prec, snow, snwd, ), dim=1)
            features = torch.stack((lat, lon, mon, tmax, tmin, prec, mb1, mb2, mb3, mb4, mg1, mg2, mg3, mg4), dim=1)
    
            # Move to GPU if available       
            input_batch = input_batch.to(model.device, dtype=torch.float)
            labels_batch = labels_batch.to(model.device, dtype=torch.float)
            features = features.to(model.device, dtype=torch.float)

            # Convert to torch Variables
            input_batch = Variable(input_batch)
            labels_batch = Variable(labels_batch)
            features = Variable(features)

            # Forward pass and calculate loss
            output_batch = model(input_batch, features) 
            loss = loss_fn(output_batch, labels_batch)

            # Compute gradients and perform parameter updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()
       
            # Move to cpu and convert to numpy
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()
            
            # Save predictions to compute r2 over full dataset
            curr_batch_size = output_batch.shape[0] # if last batch, may be less than full batch size
            indices = np.arange(curr_batch_size)
            #utils.save_predictions(indices, output_batch, labels_batch, 
            #                      curr_batch_size, "predictions/train_preds_epoch_" + str(epoch) + ".csv") 
           
            # Compute batch metrics
            r2 = r2_score(labels_batch, output_batch) 
            summary_batch = {'average r2': r2, 'average MSE loss': loss.item()}
            summaries.append(summary_batch)
           
            # Display batch loss and r2
            loss_str = '{:05.3f}'.format(loss.item())
            r2_str = '{:01.3f}'.format(r2)
            t.set_postfix(loss=loss_str, r2=r2_str) 
            t.update()
  
    # Save metrics
    mean_metrics = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in mean_metrics.items())
    print("Train metrics: {}".format(metrics_string))

    return mean_metrics


def evaluate(model, loss_fn, dataloader, batch_size, epoch):
    '''
    Evaluates the model for 1 epoch on all batches in the dataloader.
    '''
    
    summaries = []
    num_batches = len(dataloader)
    val_dataset_size = num_batches * batch_size
    
    # Set model to eval mode
    model.eval()
    
    print("Evaluating on {} batches".format(num_batches))
    
    with tqdm(total=num_batches) as t:
        with torch.no_grad():
            for i, sample in enumerate(dataloader):

                input_batch, labels_batch = sample['image'], sample['pm']
                lat, lon, mon = sample['lat'], sample['lon'], sample['month']
                prec, snow, snwd = sample['prec'], sample['snow'], sample['snwd']
                tmax, tmin = sample['tmax'], sample['tmin']
                mb1, mb2, mb3, mb4 = sample['mb1'], sample['mb2'], sample['mb3'], sample['mb4']
                mg1, mg2, mg3, mg4 = sample['mg1'], sample['mg2'], sample['mg3'], sample['mg4']
            
                features = torch.stack((lat, lon, mon, tmax, tmin, prec, mb1, mb2, mb3, mb4, mg1, mg2, mg3, mg4), dim=1)

                # Move to GPU if available       
                input_batch = input_batch.to(model.device, dtype=torch.float)
                labels_batch = labels_batch.to(model.device, dtype=torch.float)
                features = features.to(model.device, dtype=torch.float)

                # Convert to torch Variables
                input_batch = Variable(input_batch)
                labels_batch = Variable(labels_batch)
                features = Variable(features)

                # Forward pass and calculate loss
                output_batch = model(input_batch, features) 
                loss = loss_fn(output_batch, labels_batch)

                # Move to cpu and convert to numpy
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()
                
                # Save predictions to compute r2 over full dataset
                curr_batch_size = output_batch.shape[0]  
                indices = np.arange(curr_batch_size)
                utils.save_predictions(indices, output_batch, labels_batch, curr_batch_size, 
                                       "predictions/val_preds_epoch_" + str(epoch) + ".csv") 
          
                # Compute batch metrics
                r2 = r2_score(labels_batch, output_batch) #.cpu().detach().numpy())
                summary_batch = {'average r2': r2, 'average MSE loss': loss.item()}
                summaries.append(summary_batch)

                # Visualize this batches predictions
                # if i % 8 == 0:
                
                # Display batch loss and r2
                loss_str = '{:05.3f}'.format(loss.item())
                r2_str = '{:01.3f}'.format(r2)
                t.set_postfix(loss=loss_str, r2=r2_str) 
                t.update()
   
    mean_metrics = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]}    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in mean_metrics.items())
    print("Evaluation metrics: {}".format(metrics_string))
    
    return mean_metrics


def train_and_evaluate(model, optimizer, loss_fn, train_dataloader, val_dataloader, 
                       batch_size, num_epochs, model_dir=None, saved_weights_file=None):
    '''
    Trains the model and evaluates at every epoch
    '''

    best_val_r2 = 0.0
    all_train_losses, all_val_losses, all_train_r2, all_val_r2 = [], [], [], []
    
    # If a saved weights file for the model is specified, reload the weights
    if model_dir is not None and saved_weights_file is not None:
        saved_weights_path = os.path.join(model_dir, saved_weights_file + '.pth.tar')
        utils.load_checkpoint(saved_weights_path, model, optimizer)
        print("Restoring parameters from {}".format(saved_weights_path))

    for epoch in range(num_epochs):
        
        print("Running Epoch {}/{}".format(epoch, num_epochs))
              
        # Train model for one epoch
        train_mean_metrics = train(model, optimizer, loss_fn, train_dataloader, batch_size, epoch)

        # Evaluate on validation set
        val_mean_metrics = evaluate(model, loss_fn, val_dataloader, batch_size, epoch)
        
        # Save losses and r2 from this epoch
        all_train_losses.append( train_mean_metrics['average MSE loss'] )
        all_val_losses.append( val_mean_metrics['average MSE loss'] )
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
            best_val_r2 = val_r2
            print("Found new best R2 value of {}. Saving to checkpoint directory {}.".format(best_val_r2, model_dir))

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights_regression_unbal.json")
            utils.save_dict_to_json(val_mean_metrics, best_json_path)


    # Print average losses and R2 over train and validation sets
    print("Train losses: {} \n Validation losses: {}".format(all_train_losses, all_val_losses))
    print("Train mean R2s: {} \n Validation mean R2s: {}".format(all_train_r2, all_val_r2))
        
    # Plot losses and R2 over train and validation sets   
    num_ex = 20
    utils.plot_losses(all_train_losses, all_val_losses, num_epochs, num_ex, save_as="plots/loss_mmm_20000ex_unb.png")
    utils.plot_r2(all_train_r2, all_val_r2, num_epochs, num_ex, save_as="plots/r2_mmm_20000ex_unb.png")
                    
    # Return train and eval metrics
    return train_mean_metrics, val_mean_metrics


if __name__ == "__main__":
    
    #save_to_csv= "data_csv_files/cleaned_data_all_temp_new_3.csv"
    #utils.remove_sent_and_save_df_to_csv(cleaned_csv, save_to_csv)
    
    cleaned_csv = "data_csv_files/cleaned_data_all_temp_new_3.csv"
    npy_dir = '/home/sarahciresi/gcloud/cs325b-airquality/cs325b/images/s2/'
    sent_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/sentinel/2016/"
    chckpt_dir = "/home/sarahciresi/gcloud/cs325b-airquality/checkpt_overfit"
    
    lr = 0.0005
    reg = 1e-4
    batch_size = 64
    num_epochs = 30
    
    dataloaders = load_data(cleaned_csv, npy_dir, sent_dir, batch_size=batch_size, sample_balanced=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_combined(device) 
    model.apply(model.init_weights)
    model.to(device)
  
    
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=reg)
    train_and_evaluate(model, optimizer, nn.MSELoss(), dataloaders['train'], dataloaders['val'], 
                       batch_size=batch_size, num_epochs=num_epochs, model_dir = chckpt_dir)
    #                   , saved_weights_file="best_6_scratch")
   
