import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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

    '''
    def __init__(self, master_csv_file, s2_npy_root_dir, s2_tif_root_dir, transform=None):
        ''' csv_file:
        root_dir: directory with sentinel .npy files
        '''
        self.epa_df = pd.read_csv(master_csv_file)
        self.s2_npy_root_dir = s2_npy_root_dir
        self.s2_tif_root_dir = s2_tif_root_dir
        self.transform = transforms.Compose([ToTensor()])


    def __len__(self):
        return len(self.epa_df)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        epa_row = self.epa_df.iloc[idx]
        tif_filename = str(epa_row['SENTINEL_FILENAME'])
        tif_index =  int(epa_row['SENTINEL_INDEX'])
        npy_filename = tif_filename[:-4] + '_' + str(tif_index) + '.npy'
        npy_fullpath = os.path.join(self.s2_npy_root_dir, npy_filename)
        
        try:
            image = np.load(npy_fullpath)
      
        except (FileNotFoundError, ValueError) as exc:
            print(npy_fullpath)
            save_sentinel_from_eparow(epa_row, self.s2_tif_root_dir, im_size=200)
        
        pm = epa_row['Daily Mean PM2.5 Concentration']
        #sample = {'image': image, 'epa_row': epa_row, 'pm': pm}
        sample = {'image': np.zeros((200,200, 13)), 'epa_row': epa_row, 'pm': pm}

        sample = self.transform(sample)
            
        return sample
        


    def normalize(self, array):
        if np.max(array) == np.min(array):
            return array
        return (array - np.min(array))*(255/(np.max(array)-np.min(array))).astype(int)

    
def show_samples(dataset, num_samples):
    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]
        #print(i, sample['image'].shape, sample['epa_row'].shape)
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


