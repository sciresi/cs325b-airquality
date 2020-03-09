import os
import sys
import time
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import load_data_new
import utils


class Sentinel_DNN(nn.Module):
    '''
    Sentinel DNN model.
    '''
    def __init__(self, device = "cpu"):
        super(Sentinel_DNN, self).__init__()

        num_input_features = 52   
        
        self.device = device
                
        self.fc1 = nn.Linear(num_input_features, 500)
        self.fc2 = nn.Linear(500, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, 2000)    
        self.fc4 = nn.Linear(2000, 1000)
        self.fc5 = nn.Linear(1000, 1) 
        self.drop = nn.Dropout(p=0.2) 


    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)   
            
    def forward(self, x1, x2):

        x = F.relu(self.fc1(x1))
        x = F.relu(self.bn2(self.fc2(x)))
        #x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # x = self.drop(x)
        x = x.reshape(-1)

        return x

    def _set_seeds(self, seed):
        """ Sets the seeds of various libraries """
        if self.device == "cuda:0":
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        
def train(model, optimizer, loss_fn, dataloader, batch_size, epoch, t_global_step):
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
            
            indices, inputs, features, labels = sample['index'], sample['image_stats'], sample['non_image'], sample['label']
            sites, dates, states = sample['site'], sample['month'], sample['state']
            
            # Move to GPU if available       
            inputs = inputs.to(model.device, dtype=torch.float)
            labels = labels.to(model.device, dtype=torch.float)
            features = features.to(model.device, dtype=torch.float)
                 
            # Forward pass and calculate loss
            outputs = model(inputs, features) 
            loss = loss_fn(outputs, labels)
            #loss = loss_fn(outputs, labels, 1, dataset='train') # Custom Weighted MSE loss

            # Compute gradients and perform parameter updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
       
            # Move to cpu and convert to numpy
            outputs = outputs.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            indices = indices.data.cpu().numpy()
            sites, dates = sites.data.cpu().numpy(), dates.data.cpu().numpy()
           
            # Compute batch metrics
            r2 = r2_score(labels, outputs) 
            summary_batch = {'average r2': r2, 'average MSE loss': loss.item()}
            summaries.append(summary_batch)
           
            # Display batch loss and r2
            loss_str = '{:05.3f}'.format(loss.item())
            r2_str = '{:01.3f}'.format(r2)
            t.set_postfix(loss=loss_str, r2=r2_str) 
            t.update()
            
            # Save predictions every 10 epochs to compute r2 over full dataset            
            if epoch % 10 == 0:
                curr_batch_size = outputs.shape[0]  
                utils.save_predictions(indices, outputs, labels, sites, dates, states, curr_batch_size, 
                                       "predictions/sent_dnn" + str(epoch) + ".csv") 
          
            del inputs, features, labels, outputs
            torch.cuda.empty_cache()
  
    # Save metrics
    mean_metrics = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in mean_metrics.items())
    print("Train metrics: {}".format(metrics_string))

    return mean_metrics, t_global_step


def evaluate(model, loss_fn, dataloader, dataset, batch_size, epoch, v_global_step):
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
                
                indices, inputs, features, labels = sample['index'], sample['image_stats'], sample['non_image'], sample['label']
                sites, dates, states = sample['site'], sample['month'], sample['state']

                # Move to GPU if available       
                inputs = inputs.to(model.device, dtype=torch.float)
                labels = labels.to(model.device, dtype=torch.float)
                features = features.to(model.device, dtype=torch.float)
                                
                # Forward pass and calculate loss
                outputs = model(inputs, features) 
                loss = loss_fn(outputs, labels)
                #loss = loss_fn(outputs, labels, 1, dataset='val') # Custom Weighted MSE loss
                
                # Move to cpu and convert to numpy
                outputs = outputs.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
                indices = indices.data.cpu().numpy()
                sites, dates = sites.data.cpu().numpy(), dates.data.cpu().numpy()
                
                # Save predictions to compute r2 over full dataset
                curr_batch_size = outputs.shape[0]  
                utils.save_predictions(indices, outputs, labels, sites, dates, states, curr_batch_size, 
                                       "predictions/sent_dnn" + dataset + "_16_mini_epoch_" + str(epoch) + ".csv") 
          
                # Compute batch metrics
                r2 = r2_score(labels, outputs)
                summary_batch = {'average r2': r2, 'average MSE loss': loss.item()}
                summaries.append(summary_batch)
               
                # Display batch loss and r2
                loss_str = '{:05.3f}'.format(loss.item())
                r2_str = '{:01.3f}'.format(r2)
                t.set_postfix(loss=loss_str, r2=r2_str) 
                t.update()

                del inputs, features, labels, outputs
                torch.cuda.empty_cache()
   
    mean_metrics = {metric: np.mean([x[metric] for x in summaries]) for metric in summaries[0]}    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in mean_metrics.items())
    print("Evaluation metrics: {}".format(metrics_string))
    
    return mean_metrics, v_global_step


def train_and_evaluate(model, optimizer, loss_fn, train_dataloader, 
                       val_dataloader, batch_size, num_epochs, num_train,
                       model_dir=None, saved_weights_file=None):
    '''
    Trains the model and evaluates at every epoch
    '''
    t_global = 0
    v_global = 0
    best_val_r2 = -1.0
    all_train_losses, all_val_losses, all_train_r2, all_val_r2 = [], [], [], []
    
    # If a saved weights file for the model is specified, reload the weights
    if model_dir is not None and saved_weights_file is not None:
        saved_weights_path = os.path.join(model_dir, saved_weights_file + '.pth.tar')
        utils.load_checkpoint(saved_weights_path, model, optimizer)
        print("Restoring parameters from {}".format(saved_weights_path))

    for epoch in range(num_epochs):
        
        print("Running Epoch {}/{}".format(epoch, num_epochs))
       
        epoch_start_time = time.time()
            
        # Train model for one epoch
        train_mean_metrics, t_global = train(model, optimizer, loss_fn, train_dataloader, batch_size, epoch, t_global)

        # Evaluate on validation set
        val_mean_metrics, v_global = evaluate(model, loss_fn, val_dataloader, 'val', batch_size, epoch, v_global)
        
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

        if epoch % 10 ==0:
            print("Train losses: {} \n Validation losses: {}".format(all_train_losses, all_val_losses))
            print("Train mean R2s: {} \n Validation mean R2s: {}".format(all_train_r2, all_val_r2))
       
        print("Epoch took --- %s seconds ---" % (time.time() - epoch_start_time))


    # Print average losses and R2 over train and validation sets
    print("Train losses: {} \n Validation losses: {}".format(all_train_losses, all_val_losses))
    print("Train mean R2s: {} \n Validation mean R2s: {}".format(all_train_r2, all_val_r2))
        
    # Plot losses and R2 over train and validation sets   
    utils.plot_losses(all_train_losses, all_val_losses, num_epochs, num_train, 
                      save_as="plots/loss_"+str(num_train)+"ex.png")
    utils.plot_r2(all_train_r2, all_val_r2, num_epochs, num_train, 
                  save_as="plots/r2_"+str(num_train)+"ex.png")
                    
    # Return train and eval metrics
    return train_mean_metrics, val_mean_metrics



def predict(model, loss_fn, dataloader, batch_size, num_epochs, 
            dataset='val', model_dir=None, saved_weights_file=None):
    
    # If a saved weights file for the model is specified, reload the weights
    if model_dir is not None and saved_weights_file is not None:
        saved_weights_path = os.path.join(model_dir, saved_weights_file + '.pth.tar')
        utils.load_checkpoint(saved_weights_path, model)
        print("Restoring parameters from {}".format(saved_weights_path))

    # Evaluate on validation or test set
    epoch = "final"
    mean_metrics, v_global_step = evaluate(model, loss_fn, dataloader, dataset, batch_size, epoch, 0)
    r2 = mean_metrics['average r2']
    print("Mean R2 for {} dataset: {}".format(dataset, r2))
    
    

def run_train():
    '''
    Runs the whole training process.
    '''
    npy_dir = utils.SENTINEL_FOLDER 
    checkpt_dir = "checkpoints/sent_dnn/"  

    train = os.path.join(utils.PROCESSED_DATA_FOLDER, "train_sites_DT_2000_stats_csv_2016.csv")
    val = os.path.join(utils.PROCESSED_DATA_FOLDER, "val_sites_DT_2000_stats_csv_2016.csv")
    
    lr = 0.00001
    reg = 5e-2    
    batch_size = 90 
    num_epochs = 50 
    num_train = 87962 
   
    print("Training model for {} epochs with batch size = {}, lr = {}, reg = {}.".format(num_epochs, batch_size, lr, reg))
   
    dataloaders = load_data_new(train, batch_size = batch_size, 
                                sample_balanced=False, num_workers=8,
                                val_nonimage_csv=val, stats_in_csv=True)   
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Sentinel_DNN(device)
    model.to(device)

    model._set_seeds(0)
    model.apply(model.init_weights)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=reg)
 
    train_and_evaluate(model, optimizer, nn.MSELoss() , dataloaders['train'], dataloaders['val'], 
                       batch_size=batch_size, num_epochs=num_epochs, num_train=num_train, 
                       model_dir = checkpt_dir) 
    
def run_test():
    '''
    Runs final evaluation on the test set.   
    '''
 
    npy_dir = utils.SENTINEL_FOLDER
    checkpt_dir = "checkpoints/sent_dnn/"    

    train_csv = os.path.join(utils.PROCESSED_DATA_FOLDER, "train_sites_master_csv_2016_2017.csv")   
    test_csv = os.path.join(utils.PROCESSED_DATA_FOLDER, "test_sites_master_csv_2016_2017.csv")
        
    dataloaders = load_data_new(train_csv, batch_size = 90, 
                                sample_balanced=False, num_workers=8,
                                train_images=npy_dir, test_images=npy_dir, 
                                test_nonimage_csv=test_csv)    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Sentinel_DNN(device)
    model.to(device)
    
    predict(model, nn.MSELoss(), dataloaders['test'], batch_size=90, num_epochs=1, 
            dataset='test', model_dir=checkpt_dir, saved_weights_file="best_weights")
    
    
if __name__ == "__main__":
    
    run_train()
    ## run_test()
