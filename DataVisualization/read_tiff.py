import csv
import argparse
import time
import gdal
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import collections

NUM_BANDS_SENTINEL = 13

def normalize(arr):
    if np.max(arr) == np.min(arr):
        return arr
    return (arr - np.min(arr))*(255/(np.max(arr)-np.min(arr)))

def normalize_bin(arr):
    return (arr - np.min(arr))*(1/(np.max(arr)-np.min(arr)))

def flatten(l):
    return [num for item in l for num in (item if isinstance(item, list) else (item,))]
    
def read(dir_path, tif_path):
    '''
    Reads the full image from the tif file given by tif_path.
    '''
    gdal_dataset = gdal.Open(dir_path + tif_path)
    WW, HH = gdal_dataset.RasterXSize, gdal_dataset.RasterYSize 
    return read_middle(dir_path, tif_path, WW, HH)
    
def read_middle(dir_path, tif_path, w, h): 
    '''
    Reads the middle (w x h) image from the tif file of original size (WW x HH) given by tif_path using gdal 
    '''
    gdal_dataset = gdal.Open(dir_path + tif_path)
    WW, HH = gdal_dataset.RasterXSize, gdal_dataset.RasterYSize

    # Mid point minus half the width and height we want to read will give the top left corner
    if w > WW:
        return np.ones((w,h,2))*-1
        #raise Exception("Requested width exceeds original tif width.")
    if h > HH:
        return np.ones((w,h,2))*-1
        #raise Exception("Requested height exceeds original tif height.")

    gdal_result = gdal_dataset.ReadAsArray((WW - w)//2, (HH - h)//2, w, h)

    # If a tif file has only 1 band, then the band dimension will be removed
    if len(gdal_result.shape) == 2:
        gdal_result = np.reshape(gdal_result, [1] + list(gdal_result.shape))

    # gdal_result is a rank 3 tensor as follows (bands, height, width). Transpose to (h, w, b)
    return np.transpose(gdal_result, (1, 2, 0))
    

def display_modis(directory, filename):
    ''' 
    Takes in tif file named filename in the directory directory and uses gdal 
        to convert to an RGB image.
        Saves the file with the same filename as the original .tif file.
    '''
    #img = read(directory, filename)  
    img = read_middle(directory, filename, 2, 2)
    reds = np.zeros((img.shape[0],img.shape[1]))  
    img[img < -5000] = 0  # note missing values
    #img = normalize_bin(img) should we normalize?
    img = np.dstack((reds, img))
    
    plt.imshow(img)
    plt.show()
    plt.savefig('/home/sarahciresi/gcloud/cs325b-airquality/cs325b/images/modis/'+ filename + '.png')

    return img


def display_sentinel_gdal(dir_path, filename):
    ''' Takes in a Sentinel tif file named filename in the directory dir_path and uses gdal to convert to
        an RGB image.
    '''
    data = read(dir_path, filename)
    print("Processing Sentinel-2 image with shape {}".format(data.shape))
    num_measurements = data.shape[2]//NUM_BANDS_SENTINEL                     
 
    # for each daily measurement, get the corresponding blue, green, and red bands
    for day in range(0, num_measurements):
            
        blues = data[:,:, 1 + day * NUM_BANDS_SENTINEL]   # Band 2
        greens = data[:,:, 2 + day * NUM_BANDS_SENTINEL]  # Band 3
        reds = data[:,:, 3 + day * NUM_BANDS_SENTINEL]    # Band 4
    
        # Normalize the inputs
        reds = normalize(reds)
        greens = normalize(greens) 
        blues = normalize(blues)  

        img = np.dstack((reds, greens, blues)).astype(int)

        plt.imshow(img)
        plt.show()
        plt.savefig('/home/sarahciresi/gcloud/cs325b-airquality/cs325b/images/s2/gdal_'+ filename + '_m' + str(day) + '.png')
                
    return img


def display_sentinel_rast(dir_path, filename):
    ''' Takes in a Sentinel tif file named filename in the directory dir_path and uses rasterio 
        to convert to an RGB image.
    '''
    dataset = rasterio.open(dir_path + filename)
    num_measurements = dataset.count // NUM_BANDS_SENTINEL
    
    # for each daily measurement, get the corresponding blue, green, and red bands
    for day in range(0, num_measurements):
        blues = dataset.read(2 + day * NUM_BANDS_SENTINEL)
        greens = dataset.read(3 + day * NUM_BANDS_SENTINEL)
        reds = dataset.read(4 + day * NUM_BANDS_SENTINEL)

        # normalize the pixel values
        reds = normalize(reds)
        greens = normalize(greens)
        blues = normalize(blues)   
        
        img = np.dstack((reds,greens,blues)).astype(int)
        
        # save image of each measurement
        plt.imshow(img)
        plt.show()
        plt.savefig('/home/sarahciresi/gcloud/cs325b-airquality/cs325b/images/s2/rast_' + filename + '_m' + str(day) +'.png')    


def save_all_s2_imgs(directory):
    '''
    Converts and saves all Sentinel .tif files to images from the given directory.
    '''
    for idx, fname in enumerate(listdir(directory)):
        img = read(directory, fname)
        display_sentinel_gdal(directory, fname)


def save_all_modis_to_csv(csv_filename):
    '''
    Saves the blue and green channel values of the 4 center pixel values
    of every modis image (in all year directories) to the given csv named csv_filename.
    '''
    
    with open(csv_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Blue [0,0]", "Blue [0,1]", "Blue [1,0]", "Blue [1,1]",
                         "Green [0,0]", "Green [0,1]", "Green [1,0]", "Green [1,1]"])
        
        base_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/modis/"
        sub_dirs = ["2016_processed_100x100/", "2017_processed_100x100/", "2018_processed_100x100/", "2019_processed_100x100/"]
        for sub_dir in sub_dirs:

            full_dir = base_dir + sub_dir
            print("Saving 2x2 cropped modis images from directory: {} of size {} \n".format(full_dir, len(listdir(full_dir))))
            
            for idx, fname in enumerate(listdir(full_dir)):
                if idx % 100 == 0:
                    print("File {} name: {}".format(idx, fname))

                img = read_middle(full_dir, fname, 2, 2)

                blues = img[:,:,1]
                greens = img[:,:,0]

                # Mask missing values with small negative sentinel 
                blues[blues<-50000] = -1
                greens[greens<-50000] = -1
                num_pixels = greens.shape[0] * greens.shape[1]
                writer.writerow([blues[0,0], blues[0,1], blues[1,0], blues[1,1], greens[0,0], greens[0,1], 
                                 greens[1,0], greens[1,1]])

    csvfile.close()

        

def compute_means_all_files_modis(directory):
    '''
    Computes the mean value on the MODIS green and blue bands for all files in the directory given.
    '''

    #base_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/modis/"
    #sub_dirs = ["2016_processed_100x100/", "2017_processed_100x100/", "2018_processed_100x100/", "2019_processed_100x100/"]
    #for sub_dir in sub_dirs:
        
    #    full_dir = base_dir+sub_dir   
    #    print("Computing modis means for directory: {} of size {} \n".format(full_dir, len(listdir(full_dir))))

    means = []  # [filename, blue_mean, green_mean, # missing blue values, # missing green values]   
    for idx, fname in enumerate(listdir(directory)):
        if idx % 1000 == 0:
            print("File {} name: {}".format(idx, fname))

        img = read_middle(directory, fname, 2, 2)

        greens = img[:,:,0]
        blues = img[:,:,1]

        num_pixels = greens.shape[0] * greens.shape[1]

        # get the number of missing values in green and blue channels for whole image
        num_missing_g = np.where(greens.flatten() < - 50000)[0].shape[0]
        num_missing_b = np.where(blues.flatten() < - 50000)[0].shape[0]

        num_valid_g = num_pixels - num_missing_g
        num_valid_b = num_pixels - num_missing_b
        
        # compute the mean, not including missing pixels.. zero them out for the sum computation
        blues[blues < -50000] = 0
        greens[greens < -50000] = 0
        
        # avoid divide by 0 if full image is missing
        bm = np.sum(blues)
        gm = np.sum(greens)

        if (num_valid_b != 0):
            bm /= num_valid_b
        if (num_valid_g != 0):
            gm /= num_valid_g

        means.append([fname, bm, gm, num_missing_b, num_missing_g])

    # Save to .csv file for later use
    with open("modis_channel_means_revised.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Blue mean", "Green mean", "Num Missing Blue", "Num Missing Green"])
        writer.writerows(means)
    csvfile.close()

    
def compute_means_all_files_sentinel(directory):
    ''' 
    Computes the mean band value for each of the 13 Sentinel bands for each individual measurement in
    each .tif file in the directory. 
    '''
    print("Computing sentinel means for directory: {} of size {} \n".format(directory, len(listdir(directory))))

    means = [] # [filename, measurement, b1_mean_with_zeros, b2_mean, ... b13_mean,
               #  b1_m_without_zeros ... missing b1 values, ... # missing b13 values]
    means2= []
    
    for idx, fname in enumerate(listdir(directory)):
        if idx % 100 == 0:
            print("File {} name: {}".format(idx, fname))

        #img = read(directory, fname)
        im_size = 200
        im_size2 = 32
        img = read_middle(directory, fname, im_size, im_size)
        img32 = read_middle(directory, fname, im_size2, im_size2)

        # if the image was smaller than the requested size, a dummy img with sentinel values of -1
        # will be returned, and we won't include this image
        if np.array_equal(img, (np.ones((im_size, im_size, 2))*-1)):
            continue
        if np.array_equal(img32, (np.ones((im_size2, im_size2, 2))*-1)):
            continue
                    
        num_measurements = img.shape[2]//NUM_BANDS_SENTINEL
        num_pixels = img.shape[0] * img.shape[1]
        num_pixels2 = img32.shape[0] * img32.shape[1]
        
        # for each measurement, look at each of 1-13 bands, and compute the mean for that day
        for m in range(0, num_measurements):

            band_means_with_zeros = []
            band_means_without_zeros = []
            band_zero_val_counts = []

            ###
            band_means_with_zeros2 = []
            band_means_without_zeros2 = []
            band_zero_val_counts2 = []
            ###
            
            for band in range(0, NUM_BANDS_SENTINEL):
                band_n = img[:,:, band + m * NUM_BANDS_SENTINEL]
                num_zeros = np.where(band_n.flatten() == 0)[0].shape[0]  # changed from 500
                num_nonzero = num_pixels - num_zeros
                
                # compute the mean with zeros and without 
                band_mean_with_zeros = np.mean(band_n)
                band_mean_without_zeros = np.sum(band_n)  #np.mean(band_n)
                if (num_nonzero  != 0):
                    band_mean_without_zeros /= num_nonzero
                    
                band_means_with_zeros.append(band_mean_with_zeros)
                band_means_without_zeros.append(band_mean_without_zeros)
                band_zero_val_counts.append(num_zeros)

                ### do the same for smaller image size
                band_n2 = img32[:,:, band + m * NUM_BANDS_SENTINEL]
                num_zeros2 = np.where(band_n2.flatten() == 0)[0].shape[0]  # changed from 500
                num_nonzero2 = num_pixels2 - num_zeros2
                band_mean_with_zeros2 = np.mean(band_n2)
                band_mean_without_zeros2 = np.sum(band_n2)
                if (num_nonzero2  != 0):
                    band_mean_without_zeros2 /= num_nonzero2
                band_means_with_zeros2.append(band_mean_with_zeros2)
                band_means_without_zeros2.append(band_mean_without_zeros2)
                band_zero_val_counts2.append(num_zeros2)
                
            row = [fname, m, band_means_with_zeros, band_means_without_zeros, band_zero_val_counts]
            flattened_row = flatten(row)
            means.append(flattened_row)

            row2 = [fname, m, band_means_with_zeros2, band_means_without_zeros2, band_zero_val_counts2]
            flattened_row2 = flatten(row2)
            means2.append(flattened_row2)
            
    save_to_file = "sentinel_channel_means_" + str(im_size) + ".csv"
    with open(save_to_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Index", "B1 (with zeros)", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", 
                         "B11", "B12", "B13", "B1 (without zeros)", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10",
                         "B11", "B12", "B13", "Missing B1", "Missing B2", "Missing B3", "Missing B4",
                         "Missing B5", "Missing B6", "Missing B7", "Missing B8", "Missing B9", "Missing B10",
                         "Missing B11", "Missing B12", "Missing B13"])
        writer.writerows(means)
    csvfile.close()

    save_to_file = "sentinel_channel_means_" + str(im_size2) + ".csv"
    with open(save_to_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Index", "B1 (with zeros)", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10",
                         "B11", "B12", "B13", "B1 (without zeros)", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10",
                         "B11", "B12", "B13", "Missing B1", "Missing B2", "Missing B3", "Missing B4",
                         "Missing B5", "Missing B6", "Missing B7", "Missing B8", "Missing B9", "Missing B10",
                         "Missing B11", "Missing B12", "Missing B13"])
        writer.writerows(means2)
    csvfile.close()

if __name__ == "__main__":
     
    parser = argparse.ArgumentParser(description="Read the middle tile from a tif.")
    parser.add_argument('-p', '--tif_path', help='The path to the tif')
    parser.add_argument('-w','--width', default=1000, type=int, help='Tile width')
    parser.add_argument('-t','--height', default=5000, type=int, help='Tile height')
    parser.add_argument('-s','--type', default="modis", type=str, help="Image type")
    args = parser.parse_args()

    '''
    if(args.type == "modis"):
        img = display_modis(img, args.tif_path)

    elif(args.type =="s2"):
        display_sentinel_gdal(img, args.tif_path)
        display_sentinel_rast(dir_path, args.tif_path)
    '''
    
    #modis_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/modis/2016_processed_100x100/"
    modis_fp = "2016_076_171670012.tif"
    # display_modis(dp, fp)
    # compute_means_all_files_modis(modis_dir)

    sent_dir = "/home/sarahciresi/gcloud/cs325b-airquality/cs325b/data/sentinel/2016/"
    sent_fp = "s2_2016_9_176_482011039.tif" 
    #compute_means_all_files_sentinel(sent_dir)
    #display_sentinel_rast(sent_dir, sent_fp)
    #save_many_s2(sent_fp) 
    save_all_modis_to_csv("/home/sarahciresi/gcloud/cs325b-airquality/modis_2x2_all_years.csv")

