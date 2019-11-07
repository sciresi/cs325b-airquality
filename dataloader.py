import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/sarahciresi/gcloud/cs325b-airquality/DataVisualization')
from read_tiff import save_sentinel_from_eparow

MIN_PM_VALUE = -9.7
MAX_PM_VALUE = 20.5

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, lat, lon = sample['image'], sample['lat'], sample['lon']
        prec, snow, snwd = sample['prec'], sample['snow'], sample['snwd']
        tmin, tmax, month = sample['tmin'], sample['tmax'], sample['month']
        mb1, mb2, mb3, mb4 = sample['mb1'], sample['mb2'], sample['mb3'], sample['mb4']
        mg1, mg2, mg3, mg4 = sample['mg1'], sample['mg2'], sample['mg3'], sample['mg4']
        pm = sample['pm']
    
        # Swap channel axis
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(np.asarray(image)), 
                'lat': torch.from_numpy(np.asarray(lat)),
                'lon': torch.from_numpy(np.asarray(lon)),
                'prec': torch.from_numpy(np.asarray(prec)),
                'snow': torch.from_numpy(np.asarray(snow)),
                'snwd': torch.from_numpy(np.asarray(snwd)),
                'tmin': torch.from_numpy(np.asarray(tmin)),
                'tmax': torch.from_numpy(np.asarray(tmax)),
                'month': torch.from_numpy(np.asarray(month)),
                'mb1': torch.from_numpy(np.asarray(mb1)),
                'mb2': torch.from_numpy(np.asarray(mb2)),
                'mb3': torch.from_numpy(np.asarray(mb3)),
                'mb4': torch.from_numpy(np.asarray(mb4)),
                'mg1': torch.from_numpy(np.asarray(mg1)),
                'mg2': torch.from_numpy(np.asarray(mg2)),
                'mg3': torch.from_numpy(np.asarray(mg3)),
                'mg4': torch.from_numpy(np.asarray(mg4)),
                'pm': torch.from_numpy(np.asarray(pm))} 

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

            

