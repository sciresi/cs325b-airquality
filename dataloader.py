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



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        tensors = {"non_image" : torch.from_numpy(np.asarray(sample["non_image"])).to(dtype=torch.float),
                   "label" : torch.from_numpy(np.asarray(sample["label"])).to(dtype=torch.float)}
        
        if "image" in sample:
            # Swap channel axis
            image = sample["image"].transpose((2, 0, 1))
            tensors["image"] = torch.from_numpy(np.asarray(image)).to(dtype=torch.float)
        
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
    
def get_sampler(dataset_size, train_end, proportion):
    split_length = int(np.floor(proportion * dataset_size))
    sampler_end = train_end
    sampler_start = sampler_end - split_length
    return SubsetRandomSampler(np.arange(sampler_start, sampler_end))

def load_data(train_nonimage_csv, batch_size = BATCH_SIZE, num_workers = 0, 
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
    
#TODO: do we need this anymore?
def load_data_old(master_csv, npy_dir, sent_dir, classify=False):
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
    dataset = SentinelDataset(master_csv_file=master_csv, s2_npy_dir=npy_dir, s2_tif_dir = sent_dir, classify=classify)
    
    # Need to also normalize the data, currently is not normalized...
    
    # Split 60/20/20 for now to train on less
    split1, split2 = .1, .8 #.4, .4    #.6 .2
    full_ds_size = len(dataset) 
    indices = list(range(full_ds_size))
    split1 = int(np.floor(split1*full_ds_size))
    split2 = full_ds_size - int(np.floor(split2*full_ds_size))
    
    split1, split2 = 3000, 6000 #TODO: get rid of this
    train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

    print("Looking at images {} - {} for train, {} - {} for val, and {} - {} for test."
          .format(0, split1-1, split1, split2-1, split2, full_ds_size))
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=train_sampler, num_workers=4)
    val_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=val_sampler, num_workers=2)
    test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=test_sampler, num_workers=0)

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

            

