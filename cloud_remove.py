import os
import numpy as np
import pandas as pd
import ast
import utils
import sys
import matplotlib.pyplot as plt
from pandarallel import pandarallel

HOME_FOLDER = os.path.expanduser("~")
REPO_NAME = "cs325b-airquality"
DATA_FOLDER = "data"
REPO_FOLDER = os.path.join(HOME_FOLDER, REPO_NAME)
SENTINEL_FOLDER = os.path.join(REPO_FOLDER, DATA_FOLDER, "sentinel")
PROCESSED_DATA_FOLDER = os.path.join(REPO_FOLDER, DATA_FOLDER, "processed_data")
YEAR = '2016'

def resave_master_csv_single_year(master_csv):
    
    df = pd.read_csv(master_csv) # train_csv / val_csv / test_csv
    df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    dates = pd.to_datetime(df['Date'])
    df['year'] = pd.DatetimeIndex(df['Date']).year
    df_2016 = df[df['year']==2016]
    df_2017 = df[df['year']==2017]
    
    df_2016 = df_2016.drop(columns=['year'])
    df_2017 = df_2017.drop(columns=['year'])
    
    #df_2016.to_csv(test_csv_2016)  # replace with save_to
    #df_2017.to_csv(test_csv_2017)  # replace with save_to

def resave_mini(master_csv, save_to, num_examples):

    df = pd.read_csv(master_csv) 
    df = df.drop(columns=['Unnamed: 0'])#, 'Unnamed: 0.1'])
    df = df.sample(n=num_examples)
    df.to_csv(save_to)

def get_means_row(row):
    npy_filename = str(row["SENTINEL_FILENAME"])
    tif_index =  int(row["SENTINEL_INDEX"])
    #npy_filename = npy_filename[:-4] + "_" + str(tif_index) + ".npy"
    npy_fullpath = os.path.join(SENTINEL_FOLDER, YEAR, npy_filename)
    image = np.load(npy_fullpath).astype(np.int16)
    means = np.mean(image, axis = (0,1))

    return means

def get_mins_row(row):
    npy_filename = str(row["SENTINEL_FILENAME"])
    tif_index =  int(row["SENTINEL_INDEX"])
    #npy_filename = npy_filename[:-4] + "_" + str(tif_index) + ".npy"
    npy_fullpath = os.path.join(SENTINEL_FOLDER, YEAR, npy_filename)
    image = np.load(npy_fullpath).astype(np.int16)
    mins = np.min(image, axis = (0,1))
    return mins
   
def get_maxes_row(row):
    npy_filename = str(row["SENTINEL_FILENAME"])
    tif_index =  int(row["SENTINEL_INDEX"])
    #npy_filename = npy_filename[:-4] + "_" + str(tif_index) + ".npy"
    npy_fullpath = os.path.join(SENTINEL_FOLDER, YEAR, npy_filename)
    image = np.load(npy_fullpath).astype(np.int16)
    maxes = np.max(image, axis = (0,1))
    return maxes    
    
def get_stdvs_row(row):
    npy_filename = str(row["SENTINEL_FILENAME"])
    tif_index =  int(row["SENTINEL_INDEX"])
    #npy_filename = npy_filename[:-4] + "_" + str(tif_index) + ".npy"
    npy_fullpath = os.path.join(SENTINEL_FOLDER, YEAR, npy_filename)
    image = np.load(npy_fullpath).astype(np.int16)
    stdvs = np.std(image, axis = (0,1))
    return stdvs

def check_is_cloudy_row(row, threshold=4000):

    npy_filename = str(row["SENTINEL_FILENAME"])
    tif_index =  int(row["SENTINEL_INDEX"])
    #npy_filename = npy_filename[:-4] + "_" + str(tif_index) + ".npy"
    npy_fullpath = os.path.join(SENTINEL_FOLDER, YEAR, npy_filename)
    image = np.load(npy_fullpath).astype(np.int16)
    means = np.mean(image, axis = (0,1))
    mean_b, mean_g, mean_r = means[1], means[2], means[3]
    mean_rgb = (mean_b+mean_g+mean_r)/3
    #below_threshold = mean_rgb < threshold
    
    ## Use decision tree on other bands as well as threshold 
    is_cloudy = DecisionTree(image, npy_filename) 
    is_cloudy = is_cloudy or mean_rgb > threshold ## 2000 

    return is_cloudy


def remove_sent_over_threshold(master_csv_year, to_threshold_csv, threshold=3000):
    '''
    i.e. master_csv_year  = train_csv_2016, 
         to_threshold_csv = train_csv_thresholded_2016_4000,
         threshold=4000
    '''
    pandarallel.initialize()
    
    df = pd.read_csv(master_csv_year, index_col=0)
    initial_len = len(df)
    is_cloudy = df.parallel_apply(check_is_cloudy_row, threshold=threshold, axis=1)
    df['Is cloudy'] = is_cloudy
    cloudy_df = df[df['Is cloudy'] == True]
    clear_df = df[df['Is cloudy'] == False]
    clear_df.to_csv(to_threshold_csv)
    clear_df_len = len(clear_df)
    
    print("Thresholded {} file at {} + used Decision Tree. \nInitial df was {} rows. Thresholded df is {} rows.".format(
        master_csv_year, threshold, initial_len, clear_df_len))


def normalize(arr):
    if np.max(arr) == np.min(arr):
        return arr
    return (arr - np.min(arr))*(255.0/(np.max(arr)-np.min(arr)))


def display_sample_images(thresholded_csv):
    
    # Get random sample of 5 images from csv
    df = pd.read_csv(thresholded_csv)#, index_col=0)
    df = df.sample(n=100)

    # Get corresponding .npy files
    for index, row in df.iterrows():
        print("Processing row {}...".format(index))
        npy_filename = str(row['SENTINEL_FILENAME'])
        day_idx =  int(row['SENTINEL_INDEX'])

        npy_filename = npy_filename[:-4] + "_" + str(day_idx) + ".npy"
        npy_fullpath = os.path.join(SENTINEL_FOLDER, year, npy_filename)
        image = np.load(npy_fullpath).astype(np.int16)
    
        # Normalize the inputs
        blues = image[:,:, 1]   # Band 2
        greens = image[:,:, 2]  # Band 3
        reds = image[:,:, 3]    # Band 4
    
        reds = normalize(reds)
        greens = normalize(greens)
        blues = normalize(blues)
       
        img = np.dstack((reds, greens, blues)).astype(int)

        means = np.mean(image, axis = (0,1)) # should be image.. messed this up
        mean = np.mean(image)
        
        # Check if cloudy
        is_cloudy = DecisionTree(image, npy_filename) 
        is_cloudy = is_cloudy or mean > 2000 
        
        imdir = "cloud_vis/cloudy/" if is_cloudy else "cloud_vis/clear/" 
        
        print("Image {}: Mean RGB: {}. Is cloudy: {}".format(npy_filename, mean, is_cloudy))

        # Display and save the image
        plt.imshow(img)
        plt.show()
        plt.savefig(imdir + npy_filename + '.png')

        
def save_img(filename):
    '''
    Save Sentinel image as RGB.
    '''
    fullpath = os.path.join(SENTINEL_FOLDER, year, filename)
    image = np.load(fullpath).astype(np.int16)
    blues = image[:,:, 1]   # Band 2
    greens = image[:,:, 2]  # Band 3
    reds = image[:,:, 3]    # Band 4
    reds = normalize(reds)
    greens = normalize(greens)
    blues = normalize(blues)
    img = np.dstack((reds, greens, blues)).astype(int)
    plt.imshow(img)
    plt.show()   
    plt.savefig("cloud_vis/" +filename+".png")

    
def save_stats(original_csv, csv_with_stats):
    '''
    Resaves df with stats over bands
    '''
    pandarallel.initialize(progress_bar=True)
    df = pd.read_csv(original_csv, index_col=0)
    means = df.parallel_apply(get_means_row, axis=1) 
    mins = df.parallel_apply(get_mins_row, axis=1)  
    maxes = df.parallel_apply(get_maxes_row, axis=1) 
    stdvs = df.parallel_apply(get_stdvs_row, axis=1) 
    df['means'] = means
    df['mins'] = mins
    df['maxes'] = maxes
    df['stdv'] = stdvs
    
    df.to_csv(csv_with_stats)
    
    

def DecisionTree(image, filename):
    '''
    Naive first pass at: 
    Given an image, implement decision tree from ---
    to classify whether cloudy image or not.
    
    Returns True if cloudy, False otherwise
    '''
    # 0    1      7    8   9    10   11   12
    # B1, B2, ... B8, B8A, B9, B10, B11, B12
    is_cloudy = False
    image = image/1000.
    im_w = image.shape[0]
    num_cloudy_pixels = 0
    
    for i in range(0, im_w):
        for j in range(0, im_w):
            is_cloudy_pixel = classify_pixel(image[i,j])
            if is_cloudy_pixel:
                num_cloudy_pixels += 1

    ## print("Number of cloudy pixels in image {}: {}/40000".format(filename, num_cloudy_pixels))

    is_cloudy = True if num_cloudy_pixels > 30000 else False
    return is_cloudy
    
    
def classify_pixel(pixel):
    '''
    Implements simple decision tree based off of Figure 5 in
    Ready-to-Use Methods for the Detection of Clouds (Hollstein, et al)
    to classify a pixel (1 x 1 x 13) as cloudy or not.
    '''
    if pixel[2] < 0.325:
        if pixel[8] < 0.166:
            if pixel[8] < 0.039: 
                is_cloudy = False # Water 
            else:
                is_cloudy = False # Shadow
        else:
            if pixel[10] < 0.011:
                is_cloudy = False  # Clear
            else:
                is_cloudy = True   # Cirrus
    else:
        if pixel[11] < 0.267:
            if pixel[3] < 0.674:
                is_cloudy = True # Cirrus
            else:
                is_cloudy = False # Snow
        else:
            if pixel[6] < 1.544:
                is_cloudy = True # Cloud
            else:
                is_cloudy = False # Snow
        
   
    return is_cloudy


def split(data_csv):
    '''
    Splits a given dataframe from data_csv into two.
    '''
    df = pd.read_csv(t, index_col=0)
    df = df.sample(frac=1)
    dflen = len(df)
    df1 = df.head(dflen//2)
    df2 = df.tail(dflen//2)
    base_fp = data_csv[:-4] 
    df1.to_csv(base_fp + "_split1.csv")
    df2.to_csv(base_fp + "_split2.csv")
    

if __name__ == "__main__":
    
    new_train_repaired = os.path.join(REPO_FOLDER, "train_repaired_sufficient.csv")
    new_train_repaired_stats = os.path.join(REPO_FOLDER, "train_repaired_sufficient_stats_2016.csv")
    new_val_repaired = os.path.join(REPO_FOLDER, "val_repaired_sufficient.csv")
    new_val_repaired_stats = os.path.join(REPO_FOLDER, "val_repaired_sufficient_stats_2016.csv")
    new_test_repaired = os.path.join(REPO_FOLDER, "test_repaired_sufficient.csv")
    new_test_repaired_stats = os.path.join(REPO_FOLDER, "test_repaired_sufficient_stats_2016.csv")

    #remove_sent_over_threshold(train_csv_thresholded_2016_3000, new_train, threshold=2000)  
    #remove_sent_over_threshold(val_csv_2016, val_csv_thresholded_2016_4000, threshold=4000)
    #remove_sent_over_threshold(test_csv_2016, new_test, threshold=2000)  

    #resave_mini(val_csv_thresholded_2016_3000, val_csv_thresholded_2016_3000_mini, num_examples=5000)
    # save_stats(new_test_repaired, new_test_repaired_stats)
    #split(new_train_w_stats) # splitting train_sites_DT_2000_stats_csv_2016.csv -> train_sites_DT_split1, train_sites_DT_split2 