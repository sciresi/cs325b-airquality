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


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image,  pm = sample['image'], sample['pm']
    
        # Swap channel axis
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(np.asarray(image)), 'pm': torch.from_numpy(np.asarray(pm))} 
                #'epa_row': torch.from_numpy(epa_row), 'pm': torch.from_numpy(pm)}

class SentinelDataset(Dataset):
    '''
    Class for loading the dataset, where sentinel images are the input, 
    and PM2.5 values from the master dataframe are the output.
    '''
    def __init__(self, master_csv_file, s2_npy_dir, s2_tif_dir, threshold=None, transform=None, classify=False):
        ''' 
        master_csv_file:   master csv file (with corrupted sent. files filtered out)
        s2_npy_dir:        directory with sentinel .npy files
        s2_tif_dir:        directory with sentinel .tif files   
        threshold:         PM2.5 threshold. Default is none, should use 20.5.
        transform:         any data transforms to apply.
        '''
        self.epa_df = pd.read_csv(master_csv_file)
        self.s2_npy_dir = s2_npy_dir
        self.s2_tif_dir = s2_tif_dir
        self.transform = transforms.Compose([ToTensor()])
        self.classify = classify

        if threshold != None:
            self.epa_df = self.epa_df[self.epa_df['Daily Mean PM2.5 Concentration'] < threshold]
        
        if self.classify == True:
            
            self.epa_df.loc[self.epa_df['Daily Mean PM2.5 Concentration'] > 12.0, 'above_12'] = 1 
            self.epa_df.loc[self.epa_df['Daily Mean PM2.5 Concentration'] <= 12.0, 'above_12'] = 0 

    def __len__(self):
        return len(self.epa_df)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        epa_row = self.epa_df.iloc[idx]
        tif_filename = str(epa_row['SENTINEL_FILENAME'])
        tif_index =  int(epa_row['SENTINEL_INDEX'])
        npy_filename = tif_filename[:-4] + '_' + str(tif_index) + '.npy'
        npy_fullpath = os.path.join(self.s2_npy_dir, npy_filename)
        
        try:
            image = np.load(npy_fullpath)
      
        except (FileNotFoundError, ValueError) as exc:
            print(npy_fullpath)
            save_sentinel_from_eparow(epa_row, self.s2_tif_dir, im_size=200)
            image = np.zeros((200,200, 13))
        
        if self.classify == False:
            pm = epa_row['Daily Mean PM2.5 Concentration']
        else:
            pm = epa_row['above_12']
            
        sample = {'image': image, 'epa_row': epa_row, 'pm': pm}
        sample = self.transform(sample)
     
        return sample
        

    def normalize(self, array):
        if np.max(array) == np.min(array):
            return array
        return (array - np.min(array))*(255/(np.max(array)-np.min(array))).astype(int)

    
def load_data(master_csv, npy_dir, sent_dir, classify=False):
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

            

