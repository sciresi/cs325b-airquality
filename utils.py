import pandas as pd
import os

def load_csv_dfs(folder_path):
    """
    Loads all .csv files from the specified folder and concatenates into one
    giant Pandas dataframe. Potential extension to different file types if we
    need it.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder from which to read .csv files.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame for all of the .csv files in the specified folder.
    """
    
    df_list = []
    for file_name in os.listdir(folder_path):
        if os.path.splitext(file_name)[1] != ".csv":
            continue
        file_path = os.path.join(folder_path, file_name)
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
    
    for file_name in os.listdir(folder_path):
        if not file_name.startswith("s2"):
            continue
        file_extension = os.path.splitext(file_name)[1]
        if file_extension != ".tif" and file_extension != ".txt":
            continue
        file_parts = file_name.split("_")
        if len(file_parts) < 5:
            continue
        new_file_name = "_".join((*file_parts[:3], file_parts[4]))
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(old_file_path, new_file_path)
        
def rename_all_sentinel_files(sentinel_folder_path):
    """
        Loops through all the directories in the Sentinel data folder and
        renames them according to rename_sentinel_files.
    """
    
    directories = next(os.walk(sentinel_folder_path))[1]
    
    # omit hidden folders (ones starting with .)
    directories = filter(lambda name : not name.startswith("."), directories)
    
    for directory in directories:
        rename_sentinel_files(os.path.join(sentinel_folder_path, directory))