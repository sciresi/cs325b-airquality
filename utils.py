import pandas as pd
import os
import ast
import csv
import numpy as np
import json
import shutil
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

def load_csv_dfs(folder_path, blacklist = []):
    """
    Loads all .csv files from the specified folder and concatenates into one
    giant Pandas dataframe. Potential extension to different file types if we
    need it.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder from which to read .csv files.
    blacklist : list[str]
        List of filenames to ignore.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame for all of the .csv files in the specified folder.
    """
    
    df_list = []
    for filename in os.listdir(folder_path):
        if os.path.splitext(filename)[1] != ".csv":
            continue
        if filename in blacklist:
            continue
        file_path = os.path.join(folder_path, filename)
        df_list.append(pd.read_csv(file_path))
    return pd.concat(df_list)

def read_yaml(yaml_file):
    yaml_data = None
    print("Loading yaml data from {}".format(os.path.abspath(yaml_file)))
    with open(yaml_file, 'r') as input_file:
        yaml_data = yaml.load(input_file, Loader=yaml.Loader)
    return yaml_data

def get_directory_paths(folder_path):
    """
    Returns absolute paths to directories in the folder, excluding "." and "..".

    Parameters
    ----------
    folder : str
        Absolute path of the folder to search through.

    Returns
    -------
    directories : list
        List of absolute paths of directories contained within the folder.
    """

    directories = next(os.walk(folder_path))[1]
    
    # omit hidden folders (ones starting with .)
    directories = filter(lambda name : not name.startswith("."), directories)
    return [os.path.join(folder_path, directory) for directory in directories]

def clean_df(df):
    # TODO: add more filtering? change how we replace nans?
    df = df[df['TMAX'].notnull()]
    df = df[df['TMIN'].notnull()]
    df['PRCP'].fillna(-1,inplace=True)
    df['SNOW'].fillna(-1,inplace=True)
    df['SNWD'].fillna(-1,inplace=True)
    df = df[df['SENTINEL_INDEX'].notnull()]
    return df

def get_epa_features(row, filter_empty_temp=True):
    date = pd.to_datetime(row['Date'])
    month = date.month
    X = np.array([row['SITE_LATITUDE'],row['SITE_LONGITUDE'], month,
                  row['Blue [0,0]'],row['Blue [0,1]'], row['Blue [1,0]'], row['Blue [1,1]'],
                  row['Green [0,0]'],row['Green [0,1]'], row['Green [1,0]'], row['Green [1,1]'],
                  row['PRCP'], row['SNOW'], row['SNWD'], row['TMAX'], row['TMIN']])
    y = np.array(row['Daily Mean PM2.5 Concentration'])
    return X, y


def remove_partial_missing_modis(df):
    '''
    Given a df, removes the entries with any missing modis values
    '''
    df = df[df['Blue [0,0]'] != -1]
    df = df[df['Blue [0,1]'] != -1]
    df = df[df['Blue [1,0]'] != -1]
    df = df[df['Blue [1,1]'] != -1]
    df = df[df['Green [0,0]'] != -1]
    df = df[df['Green [0,1]'] != -1]
    df = df[df['Green [1,0]'] != -1]
    df = df[df['Green [1,1]'] != -1]

    return df

def remove_full_missing_modis(df):
    ''' 
    Given a df, removes the entries with fully missing modis values
    (i.e. all 8 pixel values are missing) 
    '''
    # Todo 
    pass
        
    
def remove_missing_sent(full_df):
    ''' 
    Given the full dataframe, removes the datapoints with corrupted sentinel files.
    The list of corrupted sentinel files to remove is given in file 
    "final_sent_mismatch.csv". 
    '''

    to_remove_csv = "/home/sarahciresi/gcloud/cs325b-airquality/data_csv_files/final_sent_mismatch.csv"
    to_remove_df = pd.read_csv(to_remove_csv)
    to_remove_df = to_remove_df.rename(columns={"0": "Filename"})
   
    print("Removing {} files from original df of length {}".format(len(to_remove_df), len(full_df)))

    # Should probably be able to hit this with an apply, but leaving for now
    for i, row in to_remove_df.iterrows():
        bad_file = row['Filename']
        full_df = full_df[full_df['SENTINEL_FILENAME'] != bad_file]
   
    print("After removing files, df of length {}".format(len(full_df)))

    return full_df


def remove_sent_and_save_df_to_csv(load_from_csv_filename, save_to_csv_filename):
    ''' 
    Given a .csv of the df of current datapoints, loads the df, then
    removes the missing sentinel from the updated bad-file list, and
    resaves to a .csv for later use.
    '''
    df = pd.read_csv(load_from_csv_filename)
    df = remove_missing_sent(df)
    df.to_csv(save_to_csv_filename)
    
    
    
def load_sentinel_npy_files(epa_row, npy_files_dir_path):
    ''' Reads in a sentinel.npy file which is a (h, w, 13) tensor of the sentinel image 
    for the day specified by the 'SENTINEL_INDEX' in epa_row.
    '''
    original_tif_filename = str(epa_row['SENTINEL_FILENAME'])
    index_in_original_tif =  int(epa_row['SENTINEL_INDEX'])
    npy_filename = original_tif_filename[:-4] + '_' + str(index_in_original_tif) + '.npy'
    full_npy_path = npy_files_dir_path + npy_filename
    img = np.load(full_npy_path)
    return img
 
    
def get_PM_from_row(epa_row):
    '''
    Given a row in a df, returns the PM2.5 concentration. To be used with pandarallel parallel_apply().
    '''
    pm_val =  float(epa_row['Daily Mean PM2.5 Concentration'])
    return pm_val


def save_dict_to_json(d, json_path):
    '''
    Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

        
def save_checkpoint(state, is_best, checkpoint):
    '''
    Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer 
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    '''
    
    filepath = os.path.join(checkpoint, 'last_6.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_6_scratch.pth.tar'))

        
def load_checkpoint(checkpoint, model, optimizer=None):
    '''
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    '''
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def plot_losses(train_losses, val_losses, num_epochs, num_ex, save_as):
    plt.clf()
    plt.plot(range(0, num_epochs), train_losses, label='train')
    plt.plot(range(0, num_epochs), val_losses, label='val')
    #plt.axis([0, num_epochs, 0, 1])
    plt.legend(loc=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Average MSE Loss of " + str(num_ex) + " over " + str(num_epochs) + " epochs.")
    plt.show()
    plt.savefig(save_as)
                                            
def plot_r2(train_r2, val_r2, num_epochs, num_ex, save_as):
    plt.clf()
    plt.plot(range(0, num_epochs), train_r2, label = 'train')
    plt.plot(range(0, num_epochs), val_r2, label = 'val')
    plt.axis([0, num_epochs, -0.2, 1])
    plt.legend(loc=2)
    plt.title("Average R2 of " + str(num_ex) + " over " + str(num_epochs) + " epochs.")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.show()
    plt.savefig(save_as)
    

def plot_filters_single_channel(t):
    
    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12
    
    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    t = t.data.cpu().numpy()
    npimg = np.array(t, np.float32)
    
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))
    
    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j], np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
   
    plt.tight_layout()
    plt.show()
    plt.savefig("image.png")
    
    
    
def plot_filters_multi_channel(t):
    
    #get the number of kernals
    num_kernels = t.shape[0]    
    
    #define number of columns for subplots
    num_cols = 12
    #rows = num of kernels
    num_rows = num_kernels
    
    #set the figure size
    fig = plt.figure(figsize=(num_cols,num_rows))
    
    #looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        
        #for each kernel, we convert the tensor to numpy 
        npimg = np.array(t[i].numpy(), np.float32)
        #standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        
    plt.savefig('myimage.png', dpi=100)    
    plt.tight_layout()
    plt.show()
    plt.savefig("image2.png")
    
    
def plot_weights(model, layer_num, single_channel = True, collated = False):
    #extracting the model features at the particular layer number
    layer = model.conv1
    
    #checking whether the layer is convolution layer or not 
    if isinstance(layer, nn.Conv2d):
        #getting the weight tensor data
        weight_tensor = layer.weight.data
        if single_channel:
            if collated:
                plot_filters_single_channel_big(weight_tensor)
            else:
                plot_filters_single_channel(weight_tensor)
        else:
            if weight_tensor.shape[1] == 3:
                plot_filters_multi_channel(weight_tensor)
            else:
                print("Can only plot weights with three channels with single channel = False")
    else:
        print("Can only visualize layers which are convolutional")

        
def save_predictions(indices, predictions, labels, batch_size, save_to):
    '''
    Method to save indices, labels, and predictions of a given batch. 
    All batches over entire epoch will be saved to the same file,
    so that each .csv file has the predictions over all samples.
    '''
    with open(save_to, 'a') as fd:
        writer = csv.writer(fd)
        for i in range(0, batch_size):
            index = indices[i]
            y_pred = predictions[i]
            y_true = labels[i]
            row = [index, y_pred, y_true]
            writer.writerow(row)

def strip_and_freeze(model):
    """
    Takes a PyTorch model, removes the last layer, and freezes the
    parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Model to freeze.

    Returns
    -------
    stripped_model : torch.nn.Sequential
        Same model but with last layer removed and parameters frozen.
    """
    layers = list(model.children())[:-1] # remove last layer
    stripped_model = torch.nn.Sequential(*layers)
    for param in stripped_model.parameters():
        param.requires_grad = False # freeze layer
    return stripped_model
   
def get_output_dim(model):
    """
    Takes a PyTorch model and returns the output dimension.

    Parameters
    ----------
    model : torch.nn.Module
        Model to report output dimension.

    Returns
    -------
    output_dim : int
        Dimension of the output from running model.forward()
    """
    last_layer = list(model.children())[-1]
    return last_layer.out_features

def compute_dataloader_mean_std(dataloader):
    """
    Iterates through a torch.utils.data.DataLoader and computes means and
    standard deviations. Useful for computing normalization constants over a
    training dataset.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader to iterate through to compute means and standard deviations.

    Returns
    -------
    normalizations : dict
        Dictionary mapping each input key (e.g., "non_image" or "image")
        to another dict of { "mean", "std" }.
    """
    pass
