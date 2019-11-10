import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
from DataVisualization.read_tiff import save_sentinel_from_eparow
import pdb
import utils

BATCH_SIZE = 32


MIN_PM_VALUE = -9.7
MAX_PM_VALUE = 20.5

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        tensors = {"label" : torch.from_numpy(np.asarray(sample["label"])).to(dtype=torch.float)}
        
        if "image" in sample:
            # Swap channel axis
            image = sample["image"].transpose((2, 0, 1))
            tensors["image"] = torch.from_numpy(np.asarray(image)).to(dtype=torch.float)

        if "non_image" in sample:
            tensors["non_image"] = torch.from_numpy(np.asarray(sample["non_image"])).to(dtype=torch.float)
        
        return tensors

class CombinedDataset(Dataset):
    """
    Class encapsulating the dataset of EPA/Weather data and Sentimel images.
    """
    def __init__(self, master_csv_file, image_dir = None, threshold=None, classify=False):
        """
        Parameters
        ----------
        master_csv_file : str
            Path to csv file of EPA and weather data (with corrupted Sentinel
            files filtered out).
        image_dir : str
            Path to directory with image .npy files
        threshold : float
            Filter out data points with PM2.5 readings >= this value.
        classify : bool
            Whether to treat the dataset as a classification problem.
                
        Returns
        -------
        An instance of a CombinedDataset.
        """
        self.epa_df = pd.read_csv(master_csv_file)
        self.image_dir = image_dir
        self.transform = transforms.Compose([ToTensor()])
        self.classify = classify

        if threshold != None:
            self.epa_df = self.epa_df[self.epa_df['Daily Mean PM2.5 Concentration'] < threshold]
        
        if self.classify == True:
            
            self.epa_df.loc[self.epa_df['Daily Mean PM2.5 Concentration'] > 12.0, 'above_12'] = 1 
            self.epa_df.loc[self.epa_df['Daily Mean PM2.5 Concentration'] <= 12.0, 'above_12'] = 0 
            
        self.epa_df = utils.clean_df(self.epa_df)

    def __len__(self):
        return len(self.epa_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        epa_row = self.epa_df.iloc[idx]
        sample["non_image"], _ = utils.get_epa_features(epa_row)
        if self.image_dir:
            tif_filename = str(epa_row["SENTINEL_FILENAME"])
            tif_index =  int(epa_row["SENTINEL_INDEX"])
            npy_filename = tif_filename[:-4] + "_" + str(tif_index) + ".npy"
            npy_fullpath = os.path.join(self.image_dir, npy_filename)
        
            try:
                image = np.load(npy_fullpath)

            except (FileNotFoundError, ValueError) as exc:
                image = np.zeros((200,200, 13))
                
            sample["image"] = image
        
        if self.classify == False:
            pm = epa_row["Daily Mean PM2.5 Concentration"]
        else:
            pm = epa_row["above_12"]
            
        sample["label"] = pm
        sample = self.transform(sample)
     
        return sample
        

    def normalize(self, array):
        if np.max(array) == np.min(array):
            return array
        return (array - np.min(array))*(255/(np.max(array)-np.min(array))).astype(int)

class Normalize(object):
    """Normalize image Tensors."""

    def __call__(self, sample):
        image, lat, lon = sample['image'], sample['lat'], sample['lon']
        prec, snow, snwd = sample['prec'], sample['snow'], sample['snwd']
        tmin, tmax, month = sample['tmin'], sample['tmax'], sample['month']
        mb1, mb2, mb3, mb4 = sample['mb1'], sample['mb2'], sample['mb3'], sample['mb4']
        mg1, mg2, mg3, mg4 = sample['mg1'], sample['mg2'], sample['mg3'], sample['mg4']
        pm = sample['pm']
        
        img_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5, 
                                                       0.5,0.5,0.5,0.5,0.5,0.5,0.5],
                                              std=[0.229, 0.224, 0.225, 0.226,0.225,0.225, 
                                                   0.225,0.225,0.225,0.225,0.225,0.225,0.225])
                                                   
        image = img_normalization(image) 
        # normalize features 
        #pm = normalize_pm(pm, min_pm = MIN_PM_VALUE, max_pm = MAX_PM_VALUE)
        return {'image': image, 'lat': lat, 'lon':lon, 'prec': prec, 'snow':snow,
                'snwd': snwd, 'tmin':tmin, 'tmax':tmax, 'month': month, 
                'mb1': mb1, 'mb2': mb2, 'mb3': mb3, 'mb4': mb4, 
                'mg1': mg1, 'mg2': mg2, 'mg3': mg3, 'mg4': mg4, 
                'pm': pm} 


    
class SentinelDataset(Dataset):
    '''
    Class for loading the dataset, where sentinel images are the input, 
    and PM2.5 values from the master dataframe are the output.
    '''
    def __init__(self, master_csv_file, s2_npy_dir, s2_tif_dir, threshold=None, 
                 transform=None, classify=False, sample_balanced=False):
        ''' 
        master_csv_file:   master csv file (with corrupted sent. files filtered out)
        s2_npy_dir:        directory with sentinel .npy files
        s2_tif_dir:        directory with sentinel .tif files   
        threshold:         PM2.5 threshold. Default is none, should use 20.5.
        transform:         any data transforms to apply.
        '''
        self.epa_df = pd.read_csv(master_csv_file)
        self.epa_df = self.epa_df.sample(frac=1) # shuffle the df
        self.s2_npy_dir = s2_npy_dir
        self.s2_tif_dir = s2_tif_dir
        self.transform = transforms.Compose([ToTensor()]) #, Normalize()])
        self.classify = classify

        if threshold != None:
            self.epa_df = self.epa_df[self.epa_df['Daily Mean PM2.5 Concentration'] < threshold]
        
        if self.classify == True or sample_balanced == True:
            
            self.epa_df.loc[self.epa_df['Daily Mean PM2.5 Concentration'] > 12.0, 'above_12'] = 1 
            self.epa_df.loc[self.epa_df['Daily Mean PM2.5 Concentration'] <= 12.0, 'above_12'] = 0 
       
        if sample_balanced == True:
            above_12_df = self.epa_df[self.epa_df['above_12'] == 1].iloc[0:1000]
            below_12_df = self.epa_df[self.epa_df['above_12'] == 0].iloc[0:1000]
            self.epa_df = pd.concat([above_12_df, below_12_df], ignore_index=True)
            self.epa_df = self.epa_df.sample(frac=1)
            
        # for now, filter out null temps and prec here
        self.epa_df['PRCP'].fillna(-1,inplace=True)
        self.epa_df = self.epa_df[self.epa_df['PRCP']>-1]
        self.epa_df = self.epa_df[self.epa_df['TMAX'].notnull()]
        self.epa_df = self.epa_df[self.epa_df['TMIN'].notnull()]
    
    
    def __len__(self):
        return len(self.epa_df)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        epa_row = self.epa_df.iloc[idx]
        
        lat = float(epa_row['SITE_LATITUDE'])
        lon = float(epa_row['SITE_LONGITUDE'])
        prec = float(epa_row['PRCP'])
        snow = float(epa_row['SNOW'])
        snwd = float(epa_row['SNWD'])
        min_temp = float(epa_row['TMIN'])
        max_temp = float(epa_row['TMAX'])
        date = pd.to_datetime(epa_row['Date'])
        month = float(date.month)
        mod_b1, mod_b2 = float(epa_row['Blue [0,0]']), float(epa_row['Blue [0,1]'])
        mod_b3, mod_b4 = float(epa_row['Blue [1,0]']), float(epa_row['Blue [1,1]'])
        mod_g1, mod_g2 = float(epa_row['Green [0,0]']), float(epa_row['Green [0,1]'])
        mod_g3, mod_g4 = float(epa_row['Green [1,0]']), float(epa_row['Green [1,1]'])        
        
        tif_filename = str(epa_row['SENTINEL_FILENAME'])
        tif_index =  int(epa_row['SENTINEL_INDEX'])
        npy_filename = tif_filename[:-4] + '_' + str(tif_index) + '.npy'
        npy_fullpath = os.path.join(self.s2_npy_dir, npy_filename)
        
        image = np.load(npy_fullpath)
        image = self.normalize(image)
   
        #except (FileNotFoundError, ValueError) as exc:
        #    print(npy_fullpath)
        #   save_sentinel_from_eparow(epa_row, self.s2_tif_dir, im_size=200)
        #    image = np.zeros((200,200, 13))
        
        pm = epa_row['above_12'] if self.classify == True else epa_row['Daily Mean PM2.5 Concentration']
            
        sample = {'image': image, 'lat': lat, 'lon': lon, 'prec': prec, 'snow':snow, 
                  'snwd': snwd, 'tmin': min_temp, 'tmax': max_temp, 'month': month, 
                  'mb1': mod_b1, 'mb2': mod_b2, 'mb3': mod_b3, 'mb4': mod_b4, 
                  'mg1': mod_g1, 'mg2': mod_g2, 'mg3': mod_g3, 'mg4': mod_g4, 
                  'pm': pm}
        
        sample = self.transform(sample)
     
        return sample
    
    def normalize(self, array):
        
        # Get maxes and mins across all 13 channels
        maxs = np.max(array, axis = (0,1))
        mins = np.min(array, axis = (0,1))
        diffs = maxs-mins
        zeros = np.where(diffs==0)
        diffs[zeros] = maxs[zeros]
        
        # If any are still zero, max[c] = min[c] = 0, 
        # and we need to replace with ones to avoid divide by 0 errors
        zeros = np.where(diffs==0)
        diffs[zeros] = 1
        
        # Scale each channel individually to be between 0-255
        return (array - maxs)*(255/diffs)
    

def normalize_pm(pm, min_pm, max_pm):
    if min_pm == max_pm:
        return pm
    return (pm - min_pm)/(max_pm-min_pm)     
    
def get_sampler(dataset_size, train_end, proportion):
    split_length = int(np.floor(proportion * dataset_size))
    sampler_end = train_end
    sampler_start = sampler_end - split_length
    return SubsetRandomSampler(np.arange(sampler_start, sampler_end))

def load_data_new(train_nonimage_csv, batch_size = BATCH_SIZE, num_workers = 0, 
              **kwargs):
    """
    Reads in training, val, and test data as specified by the provided dict. 
    Returns a dictionary of torch.util.data.DataLoaders for train and
    (possibly) val and test.
    
    Parameters
    ----------
    train_nonimage_csv : str
        The path to the .csv file that stores ground data (e.g., weather and 
        EPA readings) as well as MODIS readings.
    
    **kwargs
        The following keyword arguments are supported and will influence the
        resulting data that is returned.
        
        * batch_size : int
            The minibatch size for training (and validation/testing).
            
        * num_workers : int
            The number of workers to use to load data (Note: introduces randomness
            that may not be seedable).
        
        * train_images: str
            The path to the .npy files that store Sentinel images.
          
        * val_nonimage_csv : str
            Same as "train_nonimage_csv" but for a specific set of validation
            data.
          
        * val_images : str
            Same as "train_images" but for a specific set of validation data.
            Only considered if val_nonimage_csv is provided.
          
        * test_nonimage_csv : str
            Same as "train_nonimage_csv" but for a specific set of test data.
          
        * test_images : str
            Same as "train_images" but for a specific set of test data.
            
        * split_train_val : float
            Only considered if val_nonimage_csv is not provided. Splits the
            training data into train/val sets, where the proportion of the
            training data is determined by the specified float.
            
        * split_train_test : float
            Only considered if test_nonimage_csv is not provided. Splits the
            training data into train/test sets, where the proportion of the
            training data is determined by the specified float.
            
    Returns
    -------
    dataloaders : dict
        Dictionary with the following possible keys:
        
        * "train" (required) : DataLoader for training data
        * "val" (optional) : DataLoader for validation data
        * "test" (optional) : DataLoader for testing data
    """
    print("Using {} workers to load data...".format(num_workers))
    train_dataset = CombinedDataset(train_nonimage_csv, kwargs.get("train_images"))
    train_end = len(train_dataset)
    if kwargs.get("test_nonimage_csv"):
        test_dataset = CombinedDataset(kwargs["test_nonimage_csv"], kwargs.get("test_images"))
        print("{} entries in test set".format(len(test_dataset)))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                    num_workers = num_workers)
    elif kwargs.get("split_train_test"):       
        test_sampler = get_sampler(len(train_dataset), train_end, kwargs["split_train_test"])
        print("{} entries in test set".format(len(test_sampler)))
        test_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler,
                                    num_workers = num_workers)
        train_end -= len(test_sampler)
    else:
        test_dataloader = None
        
    if kwargs.get("val_nonimage_csv"):
        val_dataset = CombinedDataset(kwargs["val_nonimage_csv"], kwargs.get(val_images))
        print("{} entries in validation set".format(len(val_dataset)))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, 
                                    num_workers = num_workers, shuffle=True)
    elif kwargs.get("split_train_val"):
        val_sampler = get_sampler(len(train_dataset), train_end, kwargs["split_train_val"])
        print("{} entries in validation set".format(len(val_sampler)))
        val_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                    num_workers = num_workers, sampler=val_sampler)
        train_end -= len(val_sampler)
    else:
        val_dataloader = None
    
    if train_end < len(train_dataset):
        train_sampler = SubsetRandomSampler(np.arange(train_end))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, 
                                      num_workers = num_workers)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                      num_workers = num_workers, shuffle=True)

    print("{} samples in training set".format(train_end))
        
    dataloaders = {"train" : train_dataloader}
    if test_dataloader:
        dataloaders["test"] = test_dataloader
    if val_dataloader:
        dataloaders["val"] = val_dataloader
    return dataloaders

# TODO: do we need this anymore?
def load_data(master_csv, npy_dir, sent_dir, batch_size, classify=False, sample_balanced=False):
    ''' 
    Function that loads the data.
    Loads df from the given master data file:    master_csv
    and sentinel images from the directory:      npy_dir
    Also takes in the directory storing sentinel 
    .tif files, to save .npy files if missing for some reason.
    
    If classify = True: loads in epa df so that there is an extra column
    with binary labels, indicating whether PM is > 12.0.
    
    Creates and returns separate dataloaders for train, val,
    and test datasets.
    '''
    dataset = SentinelDataset(master_csv_file=master_csv, s2_npy_dir=npy_dir, s2_tif_dir = sent_dir, 
                              threshold=20.5, classify=classify, sample_balanced=sample_balanced)
        
    # Split 60/20/20 for now to train on less
    split1, split2 = .1, .8 #.4, .4    #.6 .2
    full_ds_size = len(dataset) 
    indices = list(range(full_ds_size))
    split1 = int(np.floor(split1*full_ds_size))
    split2 = full_ds_size - int(np.floor(split2*full_ds_size))

    split1, split2 = 1000, 2000 #20000, 25000 #1000, 2000 #10000, 15000 #2000, 4000 #1000, 2000 #30000, 31000
    train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

    print("Looking at images {} - {} for train, {} - {} for val, and {} - {} for test."
          .format(0, split1-1, split1, split2-1, split2, full_ds_size))
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=4)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=2)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=0)

    dataloaders = { 'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    
    return dataloaders

def show_samples(dataset, num_samples):
    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]
        img = sample['image']
        img = img[1:4,:,:].numpy()
        img = img.transpose((1, 2, 0))
        img = normalize(img).astype(int)
        plt.imshow(img)
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        
        if i == 3:
            plt.show()
            break

def compute_dataloader_mean_std(dataloader):
    """
    Iterates through a torch.utils.data.DataLoader and computes means and
    standard deviations. Useful for computing normalization constants over a
    training dataset. This method first computes the sample first and second
    moments and uses that to calculate the sample standard deviation.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader to iterate through to compute means and standard deviations.

    Returns
    -------
    normalizations : dict
        Dictionary mapping each input key (e.g., "non_image" or "image")
        to another dict of { "mean", "std" }.
    """
    moments = {}
    counts = {}
    for batch in dataloader:
        for key in batch.keys():
            if key == "label" or key == "output": # no need to normalize output
                continue
            data = batch[key]
            num_dims = data.size(1) # channels for images, features for non_images
            first = moments[key].get("first", torch.empty(num_dims))
            second = normalizations[key].get("second", torch.empty(num_dims))
            data_count = np.prod((data.size()[0], *data.size()[2:])) # total elements to sum
            total_count = counts.key(key, 0) # total elements we've seen so far
            
            axes = [0, *range(2, len(data.size()))] # sum over all but axis = 1
            data_sum = data.sum(axes)
            data_ss = data.pow(2).sum(axes)
            first = (total_count * first + data_sum) / (total_count + data_count)
            second = (total_count * second + data_ss) / (total_count + data_count)

            moments[key]["first"] = first
            moments[key]["second"] = second
            counts[key] = total_count + data_count

    normalizations = {}
    for key in moments.keys():
        normalizations[key] = {}
        first_moment = moments[key]["first"]
        second_moment = moments[key]["second"]
        normalizations[key]["mean"] = first_moment
        normalizations[key]["std"] = torch.sqrt(second_moment - first_moment ** 2)
        
    return normalizations

