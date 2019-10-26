import pandas as pd
import os
import ast
import numpy as np

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
    return df

def get_feature_vectors(df):
    # TODO: don't hardcode every single column!
    
    
    dates = pd.to_datetime(df['Date'])
    months = dates.dt.month
    years = dates.dt.year
    X = np.stack((df['SITE_LATITUDE'],df['SITE_LONGITUDE'],months,years,
                  df['Blue [0,0]'],df['Blue [0,1]'], df['Blue [1,0]'], df['Blue [1,1]'],
                  df['Green [0,0]'],df['Green [0,1]'], df['Green [1,0]'], df['Green [1,1]'],
                  df['PRCP'], df['SNOW'], df['SNWD'], df['TMAX'], df['TMIN']),axis=1)
    return X


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


def save_df_to_csv(df, csv_filename):
    ''' 
    Given a df of datapoints, saves to a .csv for later use.
    '''
    df.to_csv(csv_filename)
    
    
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
