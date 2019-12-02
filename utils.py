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
from pandarallel import pandarallel 
from scipy.stats.stats import pearsonr


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
    df = df.rename(columns={'Unnamed: 0': 'Index'}) 
    
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
        print("File doesn't exist {}".format(checkpoint))
        return 
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def plot_losses(train_losses, val_losses, num_epochs, num_ex, save_as):
    '''
    Method to plot train and validation losses over num_epochs epochs.
    '''
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
    '''
    Method to plot train and validation r2 values over num_epochs epochs.
    '''
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
    
        
def save_predictions(indices, predictions, labels, sites, dates, batch_size, save_to):
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
            site = sites[i]
            date = dates[i]
            row = [index, y_pred, y_true, site, date]
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
    pearson = pearsonr(labels, predictions)
    return r2, pearson 
    
def compute_r2_monthly(predictions_csv):
    '''                                                 
    Takes in .csv created from save_predictions of (indices, predictions, labels)
    for each example, and calculates total R2 over the dataset.
    '''
    df = fix_month_avg_preds_file(predictions_csv)
    
    predictions = df['Month Average']
    labels = df['Predicted Month Average']
    
    r2 = r2_score(labels, predictions)
    pearson = pearsonr(labels, predictions)
    return r2, pearson     
    
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

    plt.savefig("plots/loss_hist.png")
    plt.show() 
    
    above_three_stdv = df[df['MSE']>stdev3]
    sorted_above_three = above_three_stdv.sort_values(by=['Index'])
    print(sorted_above_three)
    
    
def get_month(row):
    date = pd.to_datetime(row['Date'])
    month = date.month
    return month


def resave_preds_with_month_and_site(predictions_csv, master_csv, averages_csv="averages.csv"):
    '''                                                                                                        
    Helper function to take  given predictions file/df and gather Site ID                       
    and month information for all datapoints in the file. Saves new .csv with these                            
    columns added and returns the df. To be used in computing monthly averages                              
    from daily PM predictions.                                                                                 
    '''
    new_pred_csv = "new_preds.csv"
    master_df = pd.read_csv(master_csv)
    pred_df = pd.read_csv(predictions_csv)

    new_pred_df = pd.DataFrame(columns=['Index', 'Prediction', 'Label', 'Month' ,'Site ID'])
    pred_df= pred_df.head(100) ## REMOVE ##                                                                    
    indices = pred_df['Index']
    for index in indices:
        pred_row = pred_df[pred_df['Index']==index]
        epa_row = master_df.loc[index]
        month = get_month(epa_row)
        site = epa_row['Site ID']
        pred_row['Month'] = month
        pred_row['Site ID'] = site
        new_pred_df = new_pred_df.append(pred_row)

    # Save new predictions file                                                                                
    new_pred_df.to_csv(new_pred_csv)

    return new_pred_df


def compute_pm_month_average_post(predictions_csv, master_csv, averages_csv="averages.csv"):
    '''                                                                                                        
    Computes PM monthly averages from daily predictions (after training). To be                                
    used when predicting daily average, then aggregating monthly after training.    
    
    Saves new predicted averages to separate .csv. Then reads from both averages.csv
    and predicted_averages.csv, joins the two df, and saves final df with both 
    true and predicted averages.
    '''
    predicted_avgs_csv = "predicted_avgs_csv.csv"
    predicted_and_true_avgs_csv = "predicted_and_true_avgs_csv.csv"

    master_df = pd.read_csv(master_csv)
    averages_df = pd.read_csv(averages_csv)
    new_pred_df = resave_preds_with_month_and_site(predictions_csv, master_csv, averages_csv)

    epa_stations = new_pred_df['Site ID'].unique()

    with open(predicted_avgs_csv, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(["Site ID", "Month", "Predicted Month Average"])

        for i, station_id in enumerate(epa_stations):
            station_datapoints = new_pred_df[new_pred_df['Site ID'] == station_id]

            for month in range(1,13):

                month_m_at_station_i = station_datapoints[station_datapoints['Month'] == month]
                if len(month_m_at_station_i) == 0:
                    continue
                pm_preds_for_month_m_at_station_i = month_m_at_station_i['Prediction']
                month_average_pred = np.mean(pm_preds_for_month_m_at_station_i)
                row = [station_id, month, month_average_pred]
                writer.writerow(row)

    # Now read from new averages file and merge with old                                                       
    average_df = pd.read_csv(averages_csv)
    average_df = average_df.set_index(["Site ID", "Month"])
    predicted_average_df = pd.read_csv(predicted_avgs_csv)
    predicted_average_df = predicted_average_df.set_index(["Site ID", "Month"])
    combined = pd.concat([average_df, predicted_average_df], axis=1)
    combined.to_csv(predicted_and_true_avgs_csv)

    
def compute_pm_month_average_pre(master_csv, averages_csv="averages.csv"):
    '''
    Computes PM monthly averages prior to training and adds to master sheet.
    To be used when trying to directly predict monthly average.
    '''
    
    pandarallel.initialize()

    df = pd.read_csv(master_csv)

    # Index on 'Month' and 'Site Id' to compute averages at each station for the month                                           
    months = df.parallel_apply(get_month, axis=1)
    df['Month'] = months

    epa_stations = df['Site ID'].unique()
    num_sites = len(epa_stations)

    with open(averages_csv, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(["Site ID", "Month", "Month Average"])
        for i, station_id in enumerate(epa_stations):
            if i%100 == 0:
                print("Getting monthly averages for site {}/{}".format(i, num_sites))

            station_datapoints = df[df['Site ID'] == station_id]

            for month in range(1,13):

                month_m_at_station_i = station_datapoints[station_datapoints['Month'] == month]
                pms_for_month_m_at_station_i = month_m_at_station_i['Daily Mean PM2.5 Concentration']
                month_average = np.mean(pms_for_month_m_at_station_i)
                row = [station_id, month, month_average]
                writer.writerow(row)

                
def fix_month_avg_preds_file(preds_true_file):
    '''
    Removes all null entries in the merged monthly averages file.
    '''
    df = pd.read_csv(preds_true_file)
    df = df[df['Month Average'].notnull()]
    df = df[df['Predicted Month Average'].notnull()]
    return df
    
    
def get_month_average(epa_row):
    averages_csv = "averages.csv"
    average_df = pd.read_csv(averages_csv)

    # from epa row get month and station id of datapoint                                                                         
    date = pd.to_datetime(epa_row['Date'])
    month = date.month
    station = epa_row['Site ID']

    # look up in averages file the corresponding average                                                                         
    average_row = average_df[(average_df['Site ID'] == station) & (average_df['Month'] == month)]
    average = average_row['Month Average']

    return average


def add_pm_month_average(master_csv):

    pandarallel.initialize()

    averages_csv = "averages.csv"

    master_df = pd.read_csv(master_csv)
    months = master_df.parallel_apply(get_month, axis=1)
    master_df['Month'] = months

    average_df = pd.read_csv(averages_csv)
    average_df = average_df.set_index(["Site ID", "Month"])

    combined = master_df.join(average_df, on=["Site ID", "Month"], how='left')
  
    save_to = "master_csv_with_averages.csv"
    combined.to_csv(save_to)



def average_analysis(averages_csv):

    average_df = pd.read_csv(averages_csv)
    for month in range(1, 13):
        month_df = average_df[average_df['Month']== month]
        pms = month_df['Month Average']
        mean_avg_pm_allsites = np.mean(pms)
        print("Month {} mean average over all sites: {}".format(month, mean_avg_pm_allsites))

        
def plot_predictions(predictions_csv, model_name):
    '''
    Method to plot true PM2.5 values vs. model predicted values based on predictions in
    the predictions_csv file.
    '''
    df = pd.read_csv(predictions_csv)
    
    indices = df['Index']
    predictions = df['Prediction']
    labels = df['Label']
   
    plt.scatter(labels, predictions, s=1)
    plt.xlabel("Real PM2.5 Values (μg/$m^3$')", fontsize=14)
    plt.ylabel("Model PM2.5 Predictions (μg/$m^3$')", fontsize=14)
    plt.axis([-10, 25, -10, 25])
    plt.title("True PM2.5 Values versus Model PM2.5 Predictions: " + model_name, fontsize=16)
    plt.savefig("plots/true_vs_preds_new_ " + model_name +".png")
    plt.show()

    
def plot_predictions_histogram(predictions_csv, model_name, dataset='val'):
    '''                                                 
    Takes in .csv created from save_predictions of (indices, predictions, labels)
    for each example. Then plots the histogram of the predictions vs. the labels.  
    '''
    plt.clf()
    df = pd.read_csv(predictions_csv)
    predictions = df['Prediction']
    labels = df['Label']
    bins = 75
   
    #plt.axis([0, 200, -10, 300]) # really should be number of examples                                                                                                                                        
    plt.hist(predictions, bins, alpha=.9,label = 'Predictions')
    plt.hist(labels, bins, alpha=.9,label = 'Labels')
    plt.xlabel("Prediction/Target Value")
    plt.ylabel("Frequency")
    plt.title("Histograms of Predicted Values versus Ground Truth Labels on Test Sites")
    plt.legend()
    plt.savefig("plots/"+dataset+"_predictions_hist.png")
    plt.show()

    
    
def highest_loss_analysis(predictions_csv):
    '''                                                 
    Takes in .csv created from save_predictions of (indices, predictions, labels, site id, month),
    for each example, and calculates MSE of each example.     
    Then determines examples with highest losses, and looks for trends. 
    '''
    pandarallel.initialize()

    df = pd.read_csv(predictions_csv)
    mses = df.parallel_apply(mse_row, axis=1)
    df['MSE'] = mses

    # Compute mean and stdev of MSEs
    mean_mse = np.mean(mses)
    stddev_mse = np.std(mses)

    stdev1 = mean_mse + stddev_mse
    stdev2 = stdev1 + stddev_mse
    stdev3 = stdev2 + stddev_mse
    
    # Determine outliers based on loss - defined as examples > 3 std. away from mean
    above_two_stdv = df[df['MSE']>stdev2]
    above_three_stdv = df[df['MSE']>stdev3]
    sorted_above_three = above_three_stdv.sort_values(by=['Index'])
    
    category = 'Site ID' # or Month
    category_outliers = above_three_stdv[category]
    unique_values = pd.unique(category_outliers)   #i.e. unique sites / unique months
    num_unique = len(unique_values)

    plt.hist(sites, num_unique, alpha=.9,label = 'Highest Loss examples by '+ category)
    plt.title('Highest Loss examples by '+ category ,fontname="Georgia", fontsize=16)
    plt.xlabel(category, fontname="Times New Roman", fontsize=12)
    plt.ylabel('Frequency',fontname="Times New Roman", fontsize=12)
    plt.savefig("plots/highest_losses_by_" + category + ".png")
    plt.show()
    
    n = 5 # Top 5 most fq sites most informative; 4 most fq months most informative
    most_fq_in_category = category_outliers.value_counts()[:n].index.tolist()  # Top 5 most fq sites most informative
    print(most_fq_in_category)
    print(category_outliers.mode()) # or for just one top-most frequent     

    
def get_outlier_info(master_csv):
    '''
    Look up information of an outlier found from highest_loss_analysis function
    '''
    
    site_id = 60371201 # CA
    df = pd.read_csv(master_csv)

    site_points = df[df['Site ID'] == site_id]
    twenty = site_points.head(20)                                                             
    most_fq_months = site_points['Month'].value_counts()[:12].index.tolist()

    preds_csv = "/Users/sarahciresi/Downloads/newest_combined_val_epoch_14.csv"
    preds_df = pd.read_csv(preds_csv)
    preds_at_site = preds_df[preds_df['Site ID'] == site_id]
    print(preds_at_site)

def get_mean_mse(predictions_csv):
    pandarallel.initialize()

    df = pd.read_csv(predictions_csv)
    mses = df.parallel_apply(mse_row, axis=1)
    df['MSE'] = mses

    # Compute mean and stdev of MSEs
    mean_mse = np.mean(mses)
    return mean_mse