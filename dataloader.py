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
        tensors = {"index": sample["index"], "month": sample["month"], "site": sample["site"], "state": sample["state"],
                   "non_image" : torch.from_numpy(np.asarray(sample["non_image"])).to(dtype=torch.float),
                   "label" : torch.from_numpy(np.asarray(sample["label"])).to(dtype=torch.float)}
        
        if "image" in sample:
            # Swap channel axis from H x W x C  to  C x H x W
            image = sample["image"].transpose((2, 0, 1))
            
            # Remove bands?  keep 0-3, 9-11
            #first_4_bands = image[:4] 
            #last_3_bands = image[9:12] #[9:12]
            #image = np.concatenate((first_4_bands, last_3_bands), axis=0)
            
            tensors["image"] = torch.from_numpy(np.asarray(image)).to(dtype=torch.float)

        if "non_image" in sample:
            tensors["non_image"] = torch.from_numpy(np.asarray(sample["non_image"])).to(dtype=torch.float)
        
        return tensors

    
class CombinedDataset(Dataset):
    """
    Class encapsulating the dataset of EPA/Weather data and Sentimel images.
    """
    def __init__(self, master_csv_file, image_dir = None, threshold=None, 
                 classify=False, sample_balanced=False, predict_monthly=False):
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
        #self.epa_df = self.epa_df.sample(frac=1) # shuffle the df
        self.image_dir = image_dir
        self.transform = transforms.Compose([ToTensor(), Normalize()])
        self.classify = classify
        self.predict_monthly = predict_monthly

        self.epa_df['PRCP'].fillna(-1,inplace=True)
        self.epa_df = self.epa_df[self.epa_df['PRCP']>-1]
        self.epa_df['SNOW'].fillna(-1,inplace=True)
        self.epa_df['SNWD'].fillna(-1,inplace=True)
        self.epa_df = self.epa_df[self.epa_df['TMAX'].notnull()]
        self.epa_df = self.epa_df[self.epa_df['TMIN'].notnull()]
        
        if threshold != None:
            self.epa_df = self.epa_df[self.epa_df['Daily Mean PM2.5 Concentration'] < threshold]
            print("Thresholding at {}".format(threshold))
        
        if self.classify == True or sample_balanced == True:
            
            self.epa_df.loc[self.epa_df['Daily Mean PM2.5 Concentration'] > 12.0, 'above_12'] = 1 
            self.epa_df.loc[self.epa_df['Daily Mean PM2.5 Concentration'] <= 12.0, 'above_12'] = 0 
        
        if sample_balanced == True:
            above_12_df = self.epa_df[self.epa_df['above_12'] == 1]
            below_12_df = self.epa_df[self.epa_df['above_12'] == 0]
            print("{} initially has {} examples below 12 and {} examples above 12.".format(master_csv_file, 
                                                                                 len(below_12_df), 
                                                                                 len(above_12_df)))
            
            self.epa_df = self.epa_df.sample(frac=1)
            above_12_df = self.epa_df[self.epa_df['above_12'] == 1].iloc[0:7500]
            below_12_df = self.epa_df[self.epa_df['above_12'] == 0].iloc[0:7500]
            self.epa_df = pd.concat([above_12_df, below_12_df], ignore_index=True)
            self.epa_df = self.epa_df.sample(frac=1)
            
            print("After sampling, {} has {} examples below 12 and {} examples above 12.".format(master_csv_file, 
                                                                                 len(below_12_df), 
                                                                                 len(above_12_df)))
         
        #self.epa_df = utils.clean_df(self.epa_df) no more cleaning needed, just change index
        self.epa_df = self.epa_df.drop(['Unnamed: 0'], axis=1)
        self.epa_df = self.epa_df[self.epa_df['SENTINEL_INDEX'] != -1]
        self.epa_df = utils.clean_df(self.epa_df)
        print("Final dataset size for {} file after filtering:  {}".format(master_csv_file, len(self.epa_df)))
        
    def __len__(self):
        return len(self.epa_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        epa_row = self.epa_df.iloc[idx]
        sample["index"] = self.epa_df.index[idx]
           
        # Added     
        date = pd.to_datetime(epa_row['Date'])
        sample["month"] = date.month
        sample["site"] = epa_row['Site ID']
        sample["state"] = epa_row['STATE']
        #

        sample["non_image"], _ = utils.get_epa_features(epa_row) #no_snow
        if self.image_dir:
            year = str(date.year)
            npy_filename = str(epa_row["SENTINEL_FILENAME"])
            tif_index =  int(epa_row["SENTINEL_INDEX"])
            if year == "2016":
                npy_filename = npy_filename[:-4] + "_" + str(tif_index) + ".npy"
            npy_fullpath = os.path.join(self.image_dir, year, npy_filename)
        
            try:
                image = np.load(npy_fullpath).astype(np.int16)
                #image = self.normalize(image)

            except (FileNotFoundError, ValueError) as exc:
                image = np.zeros((200,200, 13))
                print("File {} not found.".format(npy_fullpath))
                
            sample["image"] = image
        
        # Label based on task type
        if self.classify == True:
            label = epa_row["above_12"]
        elif self.predict_monthly == True:
            label = epa_row["Month Average"]
        else:
            label = epa_row["Daily Mean PM2.5 Concentration"]

        sample["label"] = label
        
        sample = self.transform(sample)
        
        ## Select the 8 chosen bands
        ## Should be moved into ToTensor, here for now because post-normalization
        image = sample['image']
        first_4_bands = image[:4] 
        last_4_bands = image[9:]
        image = np.concatenate((first_4_bands, last_4_bands), axis=0)
        sample['image'] = image
        
        
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
    

class Normalize(object):
    """Normalize image Tensors."""

    def __call__(self, sample):
        index, image, non_image, label = sample['index'], sample['image'], sample['non_image'], sample['label'] 
        month, site, state = sample['month'], sample['site'], sample['state']
        
        img_norm = transforms.Normalize(mean=[3144.0764, 2940.7810, 2733.0339, 2820.7695, 2963.3057, 3402.0249,
                                        3641.9360, 3506.4553, 3780.6147, 1732.4203,  313.6926, 2383.2466, 1815.5107], 
                                        std=[2496.1377, 2527.8665, 2389.6580, 2587.3665, 2536.4597, 2420.5823,
                                        2437.2566, 2353.8274, 2421.9773, 1651.2279,  693.4088, 1381.2958, 1147.4915])
        
        ft_means = np.array([38.7123, -96.0561, 6.8902, -0.4955, -0.4929, -0.4907, -0.4956,
                                   -0.4731,-0.4710,-0.4700,-0.4747,26.2292,0.7070,6.7583,207.3810,86.5185])
        
        ft_stdvs = np.array([5.4067,17.2314,3.3965,0.5544,0.5546,0.5550,0.5543,0.5781,0.5783,
                                                  0.5787,0.5778,83.8406,11.0480,70.6240,104.9007,95.1414])
        
        non_image = (non_image-torch.from_numpy(ft_means).to(dtype=torch.float))/torch.from_numpy(ft_stdvs).to(dtype=torch.float)
                                                         
        image = img_norm(image) 
        
        return {'index':index, 'image': image, 'non_image':non_image, 'month':month, 'site':site, 'state':state,
                'label':label}


    
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
        #self.epa_df = self.epa_df.sample(frac=1) # shuffle the df
        self.s2_npy_dir = s2_npy_dir
        self.s2_tif_dir = s2_tif_dir
        self.transform = transforms.Compose([ToTensorOld()]) #, Normalize()])
        self.classify = classify

        if threshold != None:
            self.epa_df = self.epa_df[self.epa_df['Daily Mean PM2.5 Concentration'] < threshold]
        
        if self.classify == True or sample_balanced == True:
            
            self.epa_df.loc[self.epa_df['Daily Mean PM2.5 Concentration'] > 12.0, 'above_12'] = 1 
            self.epa_df.loc[self.epa_df['Daily Mean PM2.5 Concentration'] <= 12.0, 'above_12'] = 0 
       
        if sample_balanced == True:
            above_12_df = self.epa_df[self.epa_df['above_12'] == 1].iloc[0:200]
            below_12_df = self.epa_df[self.epa_df['above_12'] == 0].iloc[0:200]
            self.epa_df = pd.concat([above_12_df, below_12_df], ignore_index=True)
            self.epa_df = self.epa_df.sample(frac=1)
            
        # for now, filter out null temps and prec here
        self.epa_df['PRCP'].fillna(-1,inplace=True)
        self.epa_df = self.epa_df[self.epa_df['PRCP']>-1]
        self.epa_df = self.epa_df[self.epa_df['TMAX'].notnull()]
        self.epa_df = self.epa_df[self.epa_df['TMIN'].notnull()]
        self.epa_df = self.epa_df[self.epa_df['SENTINEL_INDEX'] != -1]
    
    
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
              sample_balanced=False, predict_monthly=False, **kwargs):
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
    train_dataset = CombinedDataset(train_nonimage_csv, kwargs.get("train_images"), 
                                    threshold=20.5, sample_balanced=sample_balanced,
                                   predict_monthly=predict_monthly)
    train_end = len(train_dataset)
    if kwargs.get("test_nonimage_csv"):
        test_dataset = CombinedDataset(kwargs["test_nonimage_csv"], kwargs.get("test_images"), threshold=20.5)
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
        val_dataset = CombinedDataset(kwargs["val_nonimage_csv"], kwargs.get("val_images"), threshold=20.5, 
                                       sample_balanced=sample_balanced, predict_monthly=predict_monthly)
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


def load_data(master_csv, npy_dir, sent_dir, batch_size, classify=False,
              sample_balanced=False, predict_monthly=False):
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
    #dataset = SentinelDataset(master_csv_file=master_csv, s2_npy_dir=npy_dir, s2_tif_dir = sent_dir, 
    #                         threshold=20.5, classify=classify, sample_balanced=sample_balanced)
    
    dataset = CombinedDataset(master_csv, image_dir=npy_dir, threshold=20.5, classify=classify, 
                              sample_balanced=sample_balanced, predict_monthly=predict_monthly)  
    
    # Split 60/20/20 for now to train on less
    split1, split2 = .1, .2 
    full_ds_size = len(dataset) 
    indices = list(range(full_ds_size))
    split1 = int(np.floor(split1*full_ds_size))
    split2 = full_ds_size - int(np.floor(split2*full_ds_size))

    split1, split2 =  5000, 8000 #60000, 90000 #5000, 10000 for monthly preds  #10000, 15000 for preds before
    train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

    print("Looking at images {} - {} for train, {} - {} for val, and {} - {} for test."
          .format(0, split1-1, split1, split2-1, split2, full_ds_size))
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=8)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=8)
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


def split_data_by_site(master_csv):
    '''
    Original function to split the 2016 data by site.
    Given the master csv file of all datapoints (for a given year), does the initial
    split of sites into train, val, and test.
    Saves that years data into train_site_data, val_site_data, and test_site_data
    master csv files.
    
    For future years data, will use same site split- so need to edit as appropriate.

    '''
    all_data = pd.read_csv(master_csv)
    all_data = utils.clean_df(all_data)
    epa_stations = all_data['Site ID'].unique()
    epa_stations = epa_stations.tolist()
    num_sites = len(epa_stations)  # used to compute indices for 60/20/20 split 
    '''
    random.shuffle(epa_stations)
    train_sites = epa_stations[:690]
    val_sites = epa_stations[690:920]
    test_sites = epa_stations[920:]
    '''
    ## in future, instead of above, change to split 
    ##based on .unique 'Site ID' rows from train_site_2016, etc.
    ## e.g.:
    #       train_data_2016 = pd.read_csv("train_sites_master_csv_2016.csv" 
    #       train_sites = train_data_2016['Site ID'].unique().tolist()
    #       train_data_2017 = all_data_2017[all_data_2017['Site ID'].isin(train_sites)]
    #       train_data_2017.to_csv("train_sites_master_csv_2017.csv")
    
    train_data = all_data[all_data['Site ID'].isin(train_sites)]
    val_data = all_data[all_data['Site ID'].isin(val_sites)]
    test_data = all_data[all_data['Site ID'].isin(test_sites)]
    
    train_data.to_csv("train_sites_master_csv_2016.csv")
    val_data.to_csv("val_sites_master_csv_2016.csv")
    test_data.to_csv("test_sites_master_csv_2016.csv")
    
    
    
#detach, move to cpu, call numpy, numpy.save(path, array)
#simultaneously save outputs
#embedding dataset which reads the numpy arrays at initialization
#implement len
#implement get_item

class EmbeddingDataset(Dataset):
    def __init__(self, labels_path, ns_labels_path, s_labels_path, embeddings_path):
        self.labels = np.load(labels_path)
        self.embeddings = np.load(embeddings_path)
        self.ns_labels = np.load(ns_labels_path)
        self.s_labels = np.load(s_labels_path)
    def __len__(self):
        return self.labels.size
    def __getitem__(self, idx):
        label = self.labels[idx]
        embed = self.embeddings[idx]
        #embed = 0
        return {'pm':label,'embed':embed,'ns_pred':self.ns_labels[idx],'s_pred':self.s_labels[idx]}

def load_embeddings(labels_path, ns_labels_path, s_labels_path, embeddings_path,batch_size):
    dataset = EmbeddingDataset(labels_path, ns_labels_path, s_labels_path, embeddings_path)
    full_ds_size = len(dataset) 
    indices = list(range(full_ds_size))
    split1, split2 = 64*240,64*300
    train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=4)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=2)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=0)
    
    dataloaders = { 'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    
    return dataloaders
            

