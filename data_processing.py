import os
import numpy as np
import pandas as pd
import ast
import utils
from DataVisualization import read_tiff
from concurrent.futures import ThreadPoolExecutor

SENTINEL_CROP_SIZE = 200

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
	assert images.shape[2] % read_tiff.NUM_BANDS_SENTINEL == 0
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
	executor = ThreadPoolExecutor(max_workers=num_threads)
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
	executor = ThreadPoolExecutor(max_workers=num_folder_threads)
	for directory in utils.get_directory_paths(sentinel_folder_path):
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


def add_sentinel_info(row, metadata_folder_path, sentinel_folder_path
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
		The row with SENTINEL_FILENAME potentially modified.
	"""
	sentinel_filename = epa_row_to_sentinel_filename(row)
	metadata_filename = sentinel_filename + ".txt"
	metadata_file_path = os.path.join(metadata_folder_path, metadata_filename)
	if not os.path.exists(metadata_file_path):
		return row
	epa_date = pd.to_datetime(row.Date)
	dates = sentinel_dates[sentinel_filename]
	closest_index = find_closest_sentinel_index(epa_date, dates)
	sentinel_file_name += "_{}.npy".format(closest_index)
	sentinel_file_path = os.path.join(sentinel_folder_path, sentinel_file_name)
	if not os.path.exists(sentinel_file_path):
		return row
	row.SENTINEL_FILENAME = sentinel_file_name
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