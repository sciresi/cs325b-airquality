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