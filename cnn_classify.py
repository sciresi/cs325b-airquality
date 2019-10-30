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
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/sarahciresi/gcloud/cs325b-airquality/DataVisualization')
sys.path.insert(0, '/Users/sarahciresi/Documents/GitHub/Fall2019/cs325b-airquality/DataVisualization')
from dataloader import load_data
from pandarallel import pandarallel
import read_tiff
import utils

class Small_CNN_Classifier(nn.Module):
    def __init__(self, device = "cpu"):
        '''
        Initializes a CNN to perform classification rather than regression.
        Classifies PM2.5 above or below 12.
        '''
        super(Small_CNN_Classifier, self).__init__()

        in_channels = 13
        out_channels1 = 32
        out_channels2 = 64
        out_channels3 = 128
        
        self.device = device
        
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=5, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(out_channels2, out_channels3, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 24 * 24, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1) 
        self.sigmoid = nn.Sigmoid()
        
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
        x = self.sigmoid(x)
        x = x.reshape(-1)
        
        return x

    
def weighted_binary_cross_entropy(output, target, weights=None):
        
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))    


def train(model, optimizer, loss_fn, dataloader):
    '''
    Trains the model for 1 epoch on all batches in the dataloader.
    '''
    # Set model to train mode
    model.train()
   
    total_samples = 0
    total_correct = 0
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
            predictions = np.where(output_batch.data.cpu().numpy() < 0.5, 0, 1)
            #weights = torch.tensor([1, 50]).cuda()
            #loss = weighted_binary_cross_entropy(output_batch, labels_batch, weights=weights)
            loss = loss_fn(output_batch, labels_batch)
            
            # Print predictions to sanity check not predicting all 0
            '''if i % 20 == 0:
                print("Output: {}".format(output_batch))
                print("Predictions: {}".format(predictions))
                print("Labels: {}".format(labels_batch))
            '''        
            # Compute gradients and perform parameter updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
      
            # Move to cpu and convert to numpy
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()
            
            # Compute batch metrics
            total_samples += labels_batch.shape[0]
            num_correct = (predictions == labels_batch).sum().item()
            total_correct += num_correct
            
            batch_accuracy = num_correct / labels_batch.shape[0]
            summary_batch = {'accuracy': batch_accuracy, 'average CE loss': loss.item()}
            summaries.append(summary_batch)
            
            # Update the average loss
            loss_steps += 1
            loss_sum += loss.item()
            running_loss += loss.item() * input_batch.size(0)  
           
            # Display batch loss and r2
            loss_str = '{:05.3f}'.format(loss.item())
            acc_str = '{:05.3f}'.format(batch_accuracy)
            t.set_postfix(loss=loss_str, acc=acc_str) 
            t.update()
    
    # Epoch loss
    epoch_loss = running_loss / train_dataset_size
    loss_sum = loss_sum / loss_steps
    
    # Save metrics
    mean_metrics = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]}    
    mean_metrics['epoch loss'] = epoch_loss
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in mean_metrics.items())
    print("Train metrics: {}".format(metrics_string))
    print("Average train loss over epoch calculated via epoch_loss: {} ".format(epoch_loss))
    print("Average train loss calculated via loss_sum: {}".format(loss_sum))
    return mean_metrics


def evaluate(model, loss_fn, dataloader):
    '''
    Evaluates the model for 1 epoch on all batches in the dataloader.
    '''
    # Set model to eval mode
    model.eval()
   
    total_samples = 0
    total_correct = 0
    loss_sum = 0
    loss_steps = 0
    running_loss = 0
    summaries  = []
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
                predictions = np.where(output_batch.data.cpu().numpy() < 0.5, 0, 1)
                loss = loss_fn(output_batch, labels_batch)
                #weights = torch.tensor([1, 50]).cuda()
                #loss = weighted_binary_cross_entropy(output_batch, labels_batch, weights=weights)
                #loss = loss_fn(output_batch, labels_batch)
            
                # Print predictions to sanity check not predicting all 0
                '''if i % 20 == 0:
                    print("Output: {}".format(output_batch))
                    print("Predictions: {}".format(predictions))
                    print("Labels: {}".format(labels_batch))
                '''    
                # Move to cpu and convert to numpy
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # Compute batch metrics
                total_samples += labels_batch.shape[0]
                num_correct = (predictions == labels_batch).sum().item()
                total_correct += num_correct

                batch_accuracy = num_correct / labels_batch.shape[0]
                summary_batch = {'accuracy': batch_accuracy, 'average CE loss': loss.item()}
                summaries.append(summary_batch)

                # Update the average loss
                loss_steps += 1
                loss_sum += loss.item()
                running_loss += loss.item() * input_batch.size(0)  

                # Display batch loss and r2
                loss_str = '{:05.3f}'.format(loss.item())
                acc_str = '{:05.3f}'.format(batch_accuracy)
                t.set_postfix(loss=loss_str, acc=acc_str) 
                t.update()
    
    # Epoch loss
    epoch_loss = running_loss / val_dataset_size
    loss_sum = loss_sum / loss_steps
    
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
        saved_weights_path = os.path.join( model_dir, saved_weights_file + '.pth.tar')
        print("Restoring parameters from {}".format(saved_weights_path))
        utils.load_checkpoint(saved_weights_path, model, optimizer)

    best_acc = 0.0
    
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []
    
    for epoch in range(num_epochs):
        
        print("Running Epoch {}/{}".format(epoch, num_epochs))
              
        # Train on all batches
        train_mean_metrics = train(model, optimizer, loss_fn, train_dataloader)

        # Evaluate on validation set
        val_mean_metrics = evaluate(model, loss_fn, val_dataloader)
     
        # Save losses from this epoch
        all_train_losses.append(train_mean_metrics['epoch loss'])
        all_val_losses.append(val_mean_metrics['epoch loss'])
        
        all_train_accs.append(train_mean_metrics['accuracy'])
        all_val_accs.append(val_mean_metrics['accuracy'])
        
        val_acc = val_mean_metrics['accuracy']
        is_best = val_acc > best_acc
        
        # Save current model weights from this epoch
        utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()}, is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, save to best_save_path
        if is_best:
            print("Found new best accuracy")
            best_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights_classifier2.json")
            utils.save_dict_to_json(val_mean_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights_classifier2.json")
        utils.save_dict_to_json(val_mean_metrics, last_json_path)
    
    #print("Train losses: {} ".format(all_train_losses))
    #print("Val losses: {} ".format(all_val_losses))
    
    # Plot losses
    plt.figure(1)
    plt.plot(range(0, num_epochs), all_train_losses, label='train')
    plt.plot(range(0, num_epochs), all_val_losses, label='val')
    plt.legend(loc=2)
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("BCE Loss for 1000 Train and 1000 Val datapoints")
    plt.show()
    plt.savefig("classifier_losses.png")
    
    # Plot accuracies
    plt.figure(2)
    plt.plot(range(0, num_epochs), all_train_accs, label='train')
    plt.plot(range(0, num_epochs), all_val_accs, label='val')
    plt.legend(loc=2)
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Average accuracy over all batches for 1000 Train and 1000 Val datapoints")
    plt.show()
    plt.savefig("classifier_accuracies.png")
    
    # Return train and eval metrics
    return train_mean_metrics, val_mean_metrics


if __name__ == "__main__":
    
    #save_to_csv= "data_csv_files/cleaned_data_all_temp_new_3.csv"
    #utils.remove_sent_and_save_df_to_csv(cleaned_csv, save_to_csv)
    
    cleaned_csv = "data_csv_files/cleaned_data_all_temp_new_3.csv"
    npy_dir = '/home/sarahciresi/gcloud/cs325b-airquality/cs325b/images/s2/'
    sent_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/sentinel/2016/"

    dataloaders = load_data(cleaned_csv, npy_dir, sent_dir, classify=True, sample_balanced=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Small_CNN_Classifier(device) 
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    train_and_evaluate(model, optimizer, nn.BCELoss(),
                       dataloaders['train'], dataloaders['val'], 
                       num_epochs=30) 
   
    