import os
import numpy as np
import pandas as pd
import ast
import utils
from concurrent.futures import ThreadPoolExecutor
import sys
from pandarallel import pandarallel

try:
    import read_tiff
except ModuleNotFoundError:
    from DataVisualization import read_tiff

SENTINEL_CROP_SIZE = 200
NUM_FOLDER_THREADS = 2
NUM_SAVING_THREADS = 10
EXCLUDED_YEARS = [2018, 2019]

def save_sentinel_tif_to_npy(tif_file_path):
    """
    Reads in a .tif file storing Sentinel images from a given
    (epa station, month, year) and saves all channels from the same day to a
    .npy array.	

    Parameters
    ----------
    tif_file_path : str
        Absolute path to the .tif file to read in and save to .npy files.
    """
    dir_path = os.path.dirname(tif_file_path)
    filename = os.path.basename(tif_file_path)
    images = read_tiff.read_middle(dir_path, filename, SENTINEL_CROP_SIZE, SENTINEL_CROP_SIZE)
    assert images.shape[2] % read_tiff.NUM_BANDS_SENTINEL == 0, "{} measurement mismatch!".format(tif_file_path)
    num_measurements = images.shape[2] // read_tiff.NUM_BANDS_SENTINEL
    for measurement in range(num_measurements):
        band_start = read_tiff.NUM_BANDS_SENTINEL * measurement
        band_end = band_start + read_tiff.NUM_BANDS_SENTINEL
        image = images[:, :, band_start:band_end]
        save_name = os.path.splitext(filename)[0] +  "_{}.npy".format(measurement)
        save_path = os.path.join(dir_path, save_name)
        np.save(save_path, image)

def save_sentinel_npy(folder_path, num_threads=1):
    """
    Loads all .tif files in the specified Sentinel folder and saves each image
    to a .npy array.

    Parameters
    ----------
    folder_path : str
        Absolute path to the folder storing .tif files.
    num_threads : int, optional
        Number of threads to use for the job.
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for filename in os.listdir(folder_path):
            if not filename.startswith("s2"):
                continue
            file_extension = os.path.splitext(filename)[1]
            if file_extension != ".tif":
                continue
            file_path = os.path.join(folder_path, filename)
            executor.submit(save_sentinel_tif_to_npy, file_path)

def save_all_sentinel_npy(sentinel_folder_path, num_folder_threads=1,
                          num_saving_threads=1):
    """
    Loops through all the directories in the Sentinel data folder and
    saves images in the .tif files to .npy arrays.
    
    Parameters
    ----------
    sentinel_folder_path : str
        Absolute path to the Sentinel data folder
    num_folder_threads : int
        Number of threads to use to simultaneously process folders.
    num_saving_threads : int
        Number of threads to use in each folder for saving .tif files.
    """
    with ThreadPoolExecutor(max_workers=num_folder_threads) as executor:
        for directory in utils.get_directory_paths(sentinel_folder_path):
            if directory.endswith("2018") or directory.endswith("Metadata"):
                continue
            executor.submit(save_sentinel_npy, directory, num_saving_threads)

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
        Path to the folder from which to read Sentinel 2 files.
    """
    for filename in os.listdir(folder_path):
        if not filename.startswith("s2"):
            continue
        file_extension = os.path.splitext(filename)[1]
        if file_extension not in [".tif", ".txt", ".npy"]:
            continue
        file_parts = filename.split("_")
        if len(file_parts) < 5 or len(file_parts) == 5 and file_extension == ".npy":
            continue
        new_filename = "_".join((*file_parts[:3], *file_parts[4:]))
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

def add_sentinel_info(row, metadata_folder_path, sentinel_folder_path,
                      sentinel_dates):
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
        Full path to the Sentinel metadata folder
    sentinel_dates : dict
        Dictionary mapping from Sentinel metadata filename to the dates of the
        images stored in that file.
    sentinel_folder_path : str
        Full path to the folder storing Sentinel .npy files for each year.

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
    sentinel_filename += "_{}.npy".format(closest_index)
    year = epa_date.year
    sentinel_file_path = os.path.join(sentinel_folder_path, str(year), sentinel_filename)
    if not os.path.exists(sentinel_file_path):
        return row
    row.SENTINEL_FILENAME = sentinel_filename
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

def prism_column_to_date(column_name):
    """
    Takes a PRISM column name which represents a daily measurement and extracts
    the date in Month/Day/Year format.
    
    Parameters
    ----------
    column_name : str
        Name of the PRISM column.
        
    Returns
    -------
    date : str
        Date extracted from the column name in Month/Day/Year format.
    """
    name_parts = column_name.split("_")
    date = pd.to_datetime(name_parts[4])
    return "/".join(map(str, (date.month, date.day, date.year)))

def add_prism_df(prism_file_path, df_by_year, year, element):
    """
    Reads in the specified .csv file into a pandas.DataFrame, transposes the
    DataFrame so that there is a daily measurement for each row, and then
    joins the DataFrame to the aggregated yearly one.
    
    Parameters
    ----------
    prism_file_path : str
        Absolute path to the PRISM .csv file.
    df_by_year : dict from str -> pandas.DataFrame
        Stores a DataFrame of measurements for each year.
    year : str
        Year of the file.
    element : str
        Type of measurement this file stores (ppt, tdmean, tmean)
    """
    df = pd.read_csv(prism_file_path)
    day_columns = df.columns[4:-2] # each column is a daily measurement
    rename_map = { day : prism_column_to_date(day) for day in day_columns }
    df.rename(columns = rename_map, inplace=True) # rename columns to dates
    df.rename(columns = {"station_id" : "Site ID"}, inplace=True)
    melted = df.melt(id_vars = ["Site ID"], value_vars = rename_map.values(),
                     var_name = "Date", value_name = element) # transpose
    melted["Date"] = pd.to_datetime(melted["Date"])
    if year not in df_by_year:
        merged = melted # first DataFrame we've seen for this year
    else:
        yearly_df = df_by_year[year]
        merged = yearly_df.merge(melted, on = ["Site ID", "Date"], how = "outer")
    df_by_year[year] = merged

def gather_prism_data(prism_folder_path):
    """
    Aggregates the .csv files in the folder into a pandas.DataFrame that stores
    the daily measurements (dew point, measurement, and precipitation) for each
    station.
    
    Parameters
    ----------
    prism_folder_path : str
        Absolute path to the folder storing PRISM .csv files.
        
    Returns
    -------
    prism_df : pandas.DataFrame
        The aggregated DataFrame.
    """
    df_by_year = {}
    for filename in os.listdir(prism_folder_path):
        if not filename.endswith(".csv"):
            continue
        file_base = os.path.splitext(filename)[0]
        file_parts = file_base.split("_")
        element = file_parts[1] # tdmean, ppt, or tmean
        year = file_parts[-1]
        file_path = os.path.join(prism_folder_path, filename)
        add_prism_df(file_path, df_by_year, year, element)
    return df_by_year   
    
def main(argv):
    print("Will look for EPA data in ", utils.EPA_FOLDER)
    print("Will look for Sentinel data in ", utils.SENTINEL_FOLDER)
    print("Renaming Sentinel files...")
    rename_all_sentinel_files(utils.SENTINEL_FOLDER)
    if "--tif_to_npy" in argv:
        print("Saving all .tifs to .npy")
        save_all_sentinel_npy(utils.SENTINEL_FOLDER, NUM_FOLDER_THREADS, NUM_SAVING_THREADS)
    print("Loading dataframes...")
    epa_df = utils.load_csv_dfs(utils.EPA_FOLDER)
    print("Loading Sentinel dates...")
    dates = load_sentinel_dates(utils.SENTINEL_METADATA_FOLDER)
    new_df = epa_df.assign(SENTINEL_FILENAME = "", SENTINEL_INDEX = -1)
    del epa_df
    print("Adding Sentinel info...")
    pandarallel.initialize()
    new_df = new_df.parallel_apply(add_sentinel_info, axis=1, 
                                   metadata_folder_path=utils.SENTINEL_METADATA_FOLDER,
                                   sentinel_folder_path=utils.SENTINEL_FOLDER,
                                   sentinel_dates = dates)
    del dates
    new_df.to_csv(os.path.join(utils.EPA_FOLDER, "combined_clean") + ".csv")

if __name__ == "__main__":
    main(sys.argv)