import argparse
import time
import gdal
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

def normalize(arr):
    return (arr - np.min(arr))*(255/(np.max(arr)-np.min(arr)))
        
def read(dir_path, tif_path, H, W):
    '''
    Reads the middle HxW image from the tif given by tif_path via gdal
    '''
    full_path = dir_path + tif_path
    gdal_dataset = gdal.Open(full_path)

    # x_size and y_size and the width and height of the entire tif in pixels
    x_size, y_size = gdal_dataset.RasterXSize, gdal_dataset.RasterYSize
    print("TIF Size (W, H): ", x_size, y_size)

    #gdal_result = gdal_dataset.ReadAsArray() 
    gdal_result = gdal_dataset.ReadAsArray(0, 0, x_size, y_size)
    
    # If a tif file has only 1 band, then the band dimension will be removed.
    if len(gdal_result.shape) == 2:
        gdal_result = np.reshape(gdal_result, [1] + list(gdal_result.shape))

    # gdal_result is a rank 3 tensor as follows (bands, height, width). Transpose to (h, w, b)
    return np.transpose(gdal_result, (1, 2, 0))


def resize_and_save_modis(img, filename):
    ''' Takes in a (w, h, 2) tensor and converts to a (w, h, 3) image)
    '''
    reds = np.zeros((img.shape[0],img.shape[1]))
    img = np.dstack((img, reds))

    plt.imshow(img)
    plt.show()
    plt.savefig('/home/sarahciresi/gcloud/cs325b/images/modis/'+ filename + '.png')
    
    return img

    
def display_sentinel_gdal(data, filename):
    ''' Takes in a (w, h, c) tensor and converts to a (w, h, 3) image
    using band1, band2, band3 as the channels
    '''
    print("Processing Sentinel-2 image with shape {}".format(data.shape))
    num_bands = 13
    num_measurements = data.shape[2]//num_bands                                 
 
    # for each daily measurement, get the corresponding blue, green, and red bands
    for day in range(0, num_measurements):
            
        reds = data[:,:, 2 + day * num_bands] # band1]
        greens = data[:,:, 3 + day * num_bands] #band2]
        blues = data[:,:, 4 + day * num_bands] #band3]
    
        # Normalize the inputs
        reds = normalize(reds)        #(reds - np.min(reds))*(255/(np.max(reds)-np.min(reds)))
        greens =  normalize(greens)   #(greens - np.min(greens))*(255/(np.max(greens)-np.min(greens)))
        blues =   normalize(blues)    #(blues - np.min(blues))*(255/(np.max(blues)-np.min(blues)))

        img = np.dstack((reds, greens, blues)).astype(int)

        plt.imshow(img)
        plt.show()
        plt.savefig('/home/sarahciresi/gcloud/cs325b/images/s2/gdal_'+ filename + '_m' + str(day) + '.png')
                
    return img

def display_sentinel_rast(dir_path, filename): #, b1, b2, b3):
    ''' '''

    filepath = dir_path + filename
    dataset = rasterio.open(filepath)
    num_bands = 13
    num_measurements = dataset.count // num_bands
    show((dataset,1))
    
    # for each daily measurement, get the corresponding blue, green, and red bands
    for day in range(0, num_measurements):
        blues = dataset.read(2 + day * num_bands)   # 2, 15, 28, etc.
        greens = dataset.read(3 + day * num_bands)
        reds = dataset.read(4 + day * num_bands)

        # normalize the pixel values
        reds = normalize(reds)       #(reds - np.min(reds))*(255/(np.max(reds)-np.min(reds)))
        greens = normalize(greens)   #(greens - np.min(greens))*(255/(np.max(greens)-np.min(greens)))
        blues = normalize(blues)     #(blues - np.min(blues))*(255/(np.max(blues)-np.min(blues)))
        
        img = np.dstack((reds,greens,blues)).astype(int)
        
        # save image of each measurement
        plt.imshow(img)
        plt.show()
        plt.savefig('/home/sarahciresi/gcloud/cs325b/images/s2/rast_' + filename + '_m' + str(day) +'.png')    


         
if __name__ == "__main__":
 
    #default_path = "/home/sarahciresi/gcloud/cs325b/data/modis/2016_processed_100x100/2016_001_181630023.tif"
    dir_path = "/home/sarahciresi/gcloud/cs325b/data/sentinel/2016/"
    file = "s2_2016_10_1002_171670012.tif"
    
    parser = argparse.ArgumentParser(description="Read the middle tile from a tif.")
    parser.add_argument('-p', '--tif_path',
                        default=file,
                        help='The path to the tif')
    parser.add_argument('-w','--width', default=1000, type=int, help='Tile width')
    parser.add_argument('-t','--height', default=5000, type=int, help='Tile height')
    parser.add_argument('-s','--type', default="modis", type=str, help="Image type")
    args = parser.parse_args()
    
    start_time = time.time()
    img = read(dir_path, args.tif_path, args.height, args.width)
    img_time = time.time() - start_time
    start_time = time.time()
    
    if(args.type == "modis"):
        img = resize_modis(img, args.tif_path)

    elif(args.type =="s2"):
        b1 = 2
        b2 = 3
        b3 = 4
        display_sentinel_gdal(img, args.tif_path)
        display_sentinel_rast(dir_path, args.tif_path)
        
    #img = imresize(np.squeeze(img2), 0.125)
    

