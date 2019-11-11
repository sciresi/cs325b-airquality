import os
import pandas
import csv
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cnn
from tqdm import tqdm
from sklearn.metrics import r2_score
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import overfitting_nonsentinel_net as onn
print('hello')
import torch
from dataloader import load_data
import utils
"""def _set_seeds(self,seed):
        # Sets the seeds of various libraries 
        if self.use_cuda:
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)"""

#pretrained, does not train cnn or nonsentinel net!
class Blended_Component(nn.Module):
    
        
    def __init__(self):
        super(Blended_Component, self).__init__()
        self.fc1 = nn.Linear(184,1)
        #self._set_seeds(3)
    def forward(self,blended_feats):
        x = self.fc1(blended_feats)
        return x

#uses 2017 and 19 to train nonsentinel portion
#uses 16 to train sentinel and validate entire net
torch.manual_seed(0)

nonsentinel_means = np.array([ 39.08149721, -95.27314506,   5.98702434,  -0.53427844,  -0.53162205,  -0.53082625,
          -0.53546155,  -0.51394683,  -0.51152969,  -0.51126159,
  -0.51480852,  27.60078306,   1.34545631,  12.94090379, 195.88652635,
  76.13560177])

nonsentinel_std = np.array([  5.18903308,  16.93274656,   3.32542836,   0.55168541,   0.55215673,
   0.55237694,   0.55140683,   0.57626567,   0.57676487,  0.57696134,
   0.57598642,  83.77535907,  14.57672959,  99.58513944, 110.52559014,
  99.99687184])

X_train, X_val, y_train, y_val = onn.get_train_test_val_data([#"master_epa_weather_modis_sentinel_2016.csv",
                                                                         "master_epa_modis_weather_sentinel_2017.csv"
                                                                         #,"master_epa_modis_weather_2019.csv"
                                                                         ],nonsentinel_means,
                                                                         nonsentinel_std)

def train_blend(ns_net, cnn, loss_fn, dataloader, batch_size, epochs):
    num_batches = len(dataloader)
    val_dataset_size = num_batches * batch_size
    # Set model to eval mode
    ns_net.eval()
    cnn.eval()
    blend = Blended_Component()
    blend.to(device)
    blend.train()
    optimizer = optim.SGD(blend.parameters(), lr=.0001,weight_decay = .001)
    print("Evaluating on {} batches".format(num_batches))
    for epoch in range(epochs):
        print("new epoch")
        print(str(epoch))
        means = np.zeros(num_batches)
        with tqdm(total=num_batches) as t:
            for i, sample in enumerate(dataloader):
                    
                input_batch, labels_batch = sample['image'], sample['pm']
                lat, lon, mon = sample['lat'], sample['lon'], sample['month']
                prec, snow, snwd = sample['prec'], sample['snow'], sample['snwd']
                tmax, tmin = sample['tmax'], sample['tmin']
                mb1, mb2, mb3, mb4 = sample['mb1'], sample['mb2'], sample['mb3'], sample['mb4']
                mg1, mg2, mg3, mg4 = sample['mg1'], sample['mg2'], sample['mg3'], sample['mg4']
            
                features = torch.stack((lat, lon, mon,
                                        mb1, mb2, mb3, mb4,
                                        mg1, mg2, mg3, mg4,
                                        prec, snow, snwd, tmax, tmin), dim=1)
                    
                features = (features-torch.from_numpy(nonsentinel_means).double())/torch.from_numpy(nonsentinel_std).double()
                # Move to GPU if available
                
                input_batch = input_batch.to(device, dtype=torch.float)
                labels_batch = labels_batch.to(device, dtype=torch.float)
                features = features.to(device, dtype=torch.float)
                input_batch = Variable(input_batch)
                labels_batch = Variable(labels_batch)
                features = Variable(features)

                #Forward pass to get embeddings
                labels1, ns_embeddings = ns_net(features)
                labels2, s_embeddings = cnn(input_batch)

                blended_embeddings = torch.cat((ns_embeddings,s_embeddings),dim=1).detach()
                blended_embeddings = Variable(blended_embeddings)
                blended_embeddings = blended_embeddings.to(device, dtype=torch.float)
                print(blended_embeddings)
                y_pred = blend(blended_embeddings)
                print(y_pred)
                print(labels_batch)
                    
                loss = loss_fn(y_pred, labels_batch.reshape(y_pred.shape))
                #loss = Variable(loss, requires_grad = True)
                means[i]=loss.item()
                print(str(loss.item()))
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print(means.mean())            
                    

        t.update()
   
    return blend


#gets dataloader
batch_size = 64

cleaned_csv = "cleaned_data_all_temp_new_3.csv"
npy_dir = './np_dir/s2_npy/s2'

dataloaders_2016 = load_data(cleaned_csv, npy_dir, "missing", batch_size=batch_size, sample_balanced=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ns_net = onn.Small_Net()
ns_net.to(device)
ns_net.load_state_dict(torch.load("nonsentinel_net_1"))
ns_net.eval()

cnn_net = cnn.Small_CNN()
saved_weights_path = os.path.join("./checkpoints/", "best_6_scratch" + '.pth.tar')
print("Restoring parameters from {}".format(saved_weights_path))
utils.load_checkpoint(saved_weights_path, cnn_net)
cnn_net.to(device)
cnn_net.eval()

train_blend(ns_net, cnn_net, nn.MSELoss(), dataloaders_2016['train'], batch_size, 70)
