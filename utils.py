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
from sklearn.metrics import r2_score

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

def rename_sentinel_files(folder_path):
    """
    Loads all Sentinel 2 files (.tif or .txt files starting with "s2") in
    the specified folder and renames them to omit the index. The original
    naming scheme for these files is

        (dataset)_(year)_(month)_(index)_(epa_station_id).[txt | tif]

    so this function simply renames every file to adhere to
        
        (dataset)_(year)_(month)_(epa_station_id).[txt | tif]

    so that given a (year, month, station_id) triple we can directly look
    up the file.

    Parameters
    ----------
    folder_path : str
        Path to the folder from which to read Sentinel 2 files
    """
    
    for filename in os.listdir(folder_path):
        if not filename.startswith("s2"):
            continue
        file_extension = os.path.splitext(filename)[1]
        if file_extension != ".tif" and file_extension != ".txt":
            continue
        file_parts = filename.split("_")
        if len(file_parts) < 5:
            continue
        new_filename = "_".join((*file_parts[:3], file_parts[4]))
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(old_file_path, new_file_path)
        
def rename_all_sentinel_files(sentinel_folder_path):
    """
        Loops through all the directories in the Sentinel data folder and
        renames them according to rename_sentinel_files.
        
        Parameters
        ----------
        sentinel_folder_path : str
            Absolute path to the Sentinel data folder
    """
    
    directories = next(os.walk(sentinel_folder_path))[1]
    
    # omit hidden folders (ones starting with .)
    directories = filter(lambda name : not name.startswith("."), directories)

    for directory in directories:
        rename_sentinel_files(os.path.join(sentinel_folder_path, directory))

def epa_row_to_sentinel_filename(row, extension = ""):
    """
    Takes in a row corresponding to an EPA measurement and returns the filename
    of the corresponding Sentinel image (based on the EPA station ID and date
    of the measurement).

    Parameters
    ----------
    row : pandas.Series
        Row storing an EPA measurement.
    extension : str, default ""
        Optional extension to add to the filename (e.g., ".tif" or ".txt").

    Returns
    -------
    filename : str
        The filename of the corresponding Sentinel image.
    """

    date = pd.to_datetime(row.Date)
    year = str(date.year)
    month = str(date.month)
    station_id = str(row["Site ID"])
    filename = "_".join(("s2", year, month, station_id))
    return filename + extension

def get_sentinel_dates(metadata_file_path):
    """
    Extracts the unique dates of Sentinel images listed by the metadata file.

    Parameters
    ----------
    metadata_file_path : str
        Path to the Sentinel metadata file storing image information.

    Returns
    -------
    dates : list[numpy.datetime64]
        List of dates stored in the metadata file.
    """

    metadata_file = open(metadata_file_path)
    dates = []
    seen_dates = set()
    for channel in ast.literal_eval(metadata_file.readline()):
        date = pd.to_datetime(channel.split("_")[1], yearfirst=True)
        if date in seen_dates:
            continue
        dates.append(date)
        seen_dates.add(date)
    metadata_file.close()
    return dates

def find_closest_sentinel_index(epa_date, sentinel_dates):
    """
    Finds the date of the closest Sentinel image (by time) to the date of the
    EPA measurement.

    Parameters
    ----------
    epa_date : numpy.datetime64
        Date of the EPA measurement.
    sentinel_dates : list[numpy.datetime64]
        List of dates of Sentinel images.

    Returns
    -------
    closest_index : int
        The index of the Sentinel image closest to the EPA measurement date.
    """

    epa_date = pd.to_datetime(epa_date, yearfirst=True)
    closest_index = None
    min_difference = None
    for index, sentinel_date in enumerate(sentinel_dates):
        difference = abs(epa_date - sentinel_date)
        if closest_index is None:
            closest_index = index
            min_difference = difference
        elif difference < min_difference:
            closest_index = index
            min_difference = difference
    return closest_index

def add_sentinel_info(row, metadata_folder_path, sentinel_dates):
    """
    Takes a row of a pandas.DataFrame storing an EPA measurement and adds the
    Sentinel image filename corresponding to the closest Sentinel image
    (by date) to that measurement, as well as the index of the image within the
    Sentinel file. If a Sentinel file does not exist for the (station ID, date)
    combination stored in the row, nothing is changed.

    Parameters
    ----------
    row : pandas.Series
        Row from a DataFrame storing EPA measurements.
    metadata_folder_path : str
        Full path to the Sentinel metadata folder/
    sentinel_dates : dict
        Dictionary mapping from Sentinel metadata filename to the dates of the
        images stored in that file.

    Returns
    -------
    row : pandas.Series
        The row with SENTINEL_FILENAME and SENTINEL_INDEX potentially modified.
    """

    sentinel_filename = epa_row_to_sentinel_filename(row)
    metadata_filename = sentinel_filename + ".txt"
    metadata_file_path = os.path.join(metadata_folder_path, metadata_filename)
    if not os.path.exists(metadata_file_path):
        return row
    epa_date = pd.to_datetime(row.Date)
    dates = sentinel_dates[sentinel_filename]
    closest_index = find_closest_sentinel_index(epa_date, dates)
    row.SENTINEL_FILENAME = sentinel_filename + ".tif"
    row.SENTINEL_INDEX = closest_index
    return row

def load_sentinel_dates(metadata_folder_path):
    """
    Loads all Sentinel metadata files from the specified folder and creates a
    dictionary that maps from the metadata file name to the list of dates that
    are present within that metadata file. These dates correspond to the dates
    that the Sentinel images were taken for that file.

    Parameters
    ----------
    metadata_folder_path : str
        Path to the Sentinel metadata folder from which to read metadata files.

    Returns
    -------
    file_to_dates_map : dict
        dict mapping Sentinel file name -> list of dates in that file.
    """

    file_to_dates_map = {}
    num_files = len(os.listdir(metadata_folder_path))
    for i, filename in enumerate(os.listdir(metadata_folder_path)):
        file_base, file_extension = os.path.splitext(filename)
        if not filename.startswith("s2") or file_extension != ".txt":
            continue
        metadata_file_path = os.path.join(metadata_folder_path, filename)
        dates = get_sentinel_dates(metadata_file_path)
        file_to_dates_map[file_base] = dates
    return file_to_dates_map

def clean_df(df):
    # TODO: add more filtering? change how we replace nans?
    df = df[df['TMAX'].notnull()]
    df = df[df['TMIN'].notnull()]
    df['PRCP'].fillna(-1,inplace=True)
    df['SNOW'].fillna(-1,inplace=True)
    df['SNWD'].fillna(-1,inplace=True)
    df = df[df['PRCP']>-1]
    #df = df[df['SNOW']>-1]
    #df = df[df['SNWD']>-1]
    df = df[df['SENTINEL_INDEX'].notnull()]
    
    # Fix indexing
    df = df.rename(columns={'Unnamed: 0': 'Index'}) # probably remove, wrong col
    df = df.drop(['Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1'], axis=1)
    
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

def get_epa_features_no_weather(row, filter_empty_temp=True):
    date = pd.to_datetime(row['Date'])
    month = date.month
    X = np.array([row['SITE_LATITUDE'],row['SITE_LONGITUDE'], month,
                  row['Blue [0,0]'],row['Blue [0,1]'], row['Blue [1,0]'], row['Blue [1,1]'],
                  row['Green [0,0]'],row['Green [0,1]'], row['Green [1,0]'], row['Green [1,1]']])
    y = np.array(row['Daily Mean PM2.5 Concentration'])
    return X, y

def get_epa_features_no_snow(row, filter_empty_temp=True):
    date = pd.to_datetime(row['Date'])
    month = date.month
    X = np.array([row['SITE_LATITUDE'],row['SITE_LONGITUDE'], month,
                  row['Blue [0,0]'],row['Blue [0,1]'], row['Blue [1,0]'], row['Blue [1,1]'],
                  row['Green [0,0]'],row['Green [0,1]'], row['Green [1,0]'], row['Green [1,1]'],
                  row['PRCP'], row['TMAX'], row['TMIN']])

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
        #writer.writerow(["Index", "Prediction", "Label"])
        for i in range(0, batch_size):
            index = indices[i]
            y_pred = predictions[i]
            y_true = labels[i]
            row = [index, y_pred, y_true]
            writer.writerow(row)

def mse_row(row):
    '''                                                                                                                                                                                                     
    Given a row of a df which is an example of the                                                                                                                                                          
    form (index, prediction, label),                                                                                                                                                                        
    computes the MSE of the example.                                                                                                                                                                        
    '''
    pred = row['Prediction']
    label = row['Label']
    mse = (label-pred)**2
    return mse


def compute_r2(predictions_csv):
    '''                                                 
    Takes in .csv created from save_predictions of (indices, predictions, labels)
    for each example, and calculates total R2 over the dataset.
    '''
    df = pd.read_csv(predictions_csv)
    
    indices = df['Index']
    predictions = df['Prediction']
    labels = df['Label']
    
    r2 = r2_score(labels, predictions)
    return r2
    
def plot_loss_histogram(predictions_csv):
    '''                                                 
    Takes in .csv created from save_predictions of (indices, predictions, labels)
    for each example, and calculates MSE of each example.     
    Then plots the histogram of the losses.  
    '''
    pandarallel.initialize()

    df = pd.read_csv(predictions_csv)
    mses = df.parallel_apply(mse_row, axis=1)
    df['MSE'] = mses

    # compute mean and stdev of MSEs                                                                                                                                                                        
    mean_mse = np.mean(mses)
    stddev_mse = np.std(mses)

    stdev1 = mean_mse + stddev_mse
    stdev2 = stdev1 + stddev_mse
    stdev3 = stdev2 + stddev_mse

    bins = 100
    plt.axis([0, 1600, 0, 30]) # really should be number of examples                                                                                                                                        
    plt.hist(mses, bins,alpha=.9,label = 'MSE loss')
    min_ylim, max_ylim = plt.ylim()
    plt.axvline(mean_mse, color='k', linestyle='dashed', linewidth=1)
    plt.text(mean_mse*1.1, max_ylim*0.9, 'Mean = {:.2f}'.format(mean_mse), fontsize=10)

    plt.axvline(stdev1, color='k', linestyle='dashed', linewidth=1)
    plt.text(stdev1*1.1, max_ylim*0.7,  'Mean+1stdv = {:.2f}'.format(stdev1), fontsize=10)

    plt.axvline(stdev2, color='k', linestyle='dashed', linewidth=1)
    plt.text(stdev2*1.1, max_ylim*0.6, 'Mean+2std = {:.2f}'.format(stdev2) , fontsize=10)

    plt.axvline(stdev3, color='k', linestyle='dashed', linewidth=1)
    plt.text(stdev3*1.1, max_ylim*0.5, 'Mean+3std = {:.2f}'.format(stdev3), fontsize=10)

    plt.savefig("loss-hist.png")
    plt.show() 
    
    above_three_stdv = df[df['MSE']>stdev3]
    sorted_above_three = above_three_stdv.sort_values(by=['Index'])
    print(sorted_above_three)
    
    
def load_prism_data():
    base_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/PRISM_weather/extracted_weather_data/"
    prism_2016_ppt = base_dir+ "PRISM_ppt_stable_4kmD2_2016.csv"
    prism_2016_tdmean = base_dir+ "PRISM_tdmean_stable_4kmD1_2016.csv"
    prism_2016_tmean = base_dir+ "PRISM_tmean_stable_4kmD1_2016.csv"

    ppt_df = pd.read_csv(prism_2016_ppt)
    print(ppt_df.columns)
    return ppt_df
    