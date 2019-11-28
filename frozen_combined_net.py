#non-sentinel stats
#15.442212
#0.3843455249894836
#sentinel stats
#19.484118
#0.22320172957882922

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
from sklearn.metrics import r2_score, mean_squared_error
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import overfitting_nonsentinel_net as onn
print('hello')
import torch
from dataloader import load_data,load_embeddings
import utils
import random

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
    #.00001 is good for one-layer
        
    def __init__(self):
        super(Blended_Component, self).__init__()
        self.fc1 = nn.Linear(184,100)
        self.fc2 = nn.Linear(100,1)
        #self._set_seeds(3)
    def forward(self,blended_feats):
        x = self.fc1(blended_feats)
        x = F.leaky_relu(x)
        x = self.fc2(x)
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

def get_embeddings(ns_net,cnn,dataloader,batch_size):
    num_batches = len(dataloader)
    ns_net.eval()
    cnn.eval()
    ns_labels = np.zeros((1))
    s_labels = np.zeros((1))
    labels = np.zeros((1))
    embeddings = np.zeros((1))
    
    with tqdm(total=num_batches) as t:
        
        for i, sample in enumerate(dataloader):
            print("new batch")
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
            input_batch = input_batch.to(device, dtype=torch.float)
            labels_batch = labels_batch.to(device, dtype=torch.float)
            features = features.to(device, dtype=torch.float)
            input_batch = Variable(input_batch)
            labels_batch = Variable(labels_batch)
            features = Variable(features)
            ns_labels_batch, ns_embeddings = ns_net(features)
            s_labels_batch, s_embeddings = cnn(input_batch)

                
            blended_embeddings = torch.cat((ns_embeddings,s_embeddings),dim=1).detach().cpu().numpy()
            ns_labels_batch = ns_labels_batch.detach().cpu().numpy()
            s_labels_batch = s_labels_batch.detach().cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()

            if embeddings.size == 1:
                embeddings = blended_embeddings
                ns_labels = ns_labels_batch
                s_labels = s_labels_batch
                labels = labels_batch
                
            else:
                ns_labels = np.concatenate((ns_labels,ns_labels_batch),axis=0)
                s_labels = np.concatenate((s_labels,s_labels_batch),axis=0)
                embeddings = np.concatenate((embeddings,blended_embeddings),axis=0)
                labels = np.concatenate((labels,labels_batch),axis=0)
    ns_labels = ns_labels.reshape((ns_labels.size))
    print(ns_labels.shape)
    print(s_labels.shape)
    print(embeddings.shape)
    print(labels.shape)

    print("non-sentinel stats")
    print(str(mean_squared_error(labels,ns_labels)))
    print(str(r2_score(labels,ns_labels)))

    print("sentinel stats")
    print(str(mean_squared_error(labels,s_labels)))
    print(str(r2_score(labels,s_labels)))
    np.save("labels_big",labels)
    np.save("ns_labels_big",ns_labels)
    np.save("embeddings_big",embeddings)
    np.save("s_labels_big",s_labels)     

            

def train_blend(loss_fn, dataloader, batch_size, epochs, PATH = None):
    num_batches = len(dataloader)
    val_dataset_size = num_batches * batch_size
    
    blend = Blended_Component()
    #if PATH:
    #    blend.load_state_dict(torch.load(PATH))
    blend.to(device)
    blend.train()
    optimizer = optim.SGD(blend.parameters(), lr=.00001,weight_decay = .001)
    print("Evaluating on {} batches".format(num_batches))
    for epoch in range(epochs):
        print("epoch")
        print(str(epoch))
        mse = []
        with tqdm(total=num_batches) as t:
            for i, sample in enumerate(dataloader):
                
                labels_batch = sample['pm']
                embeddings = sample['embed']                
                # Move to GPU if available
                embeddings = embeddings.to(device, dtype=torch.float)
                labels_batch = labels_batch.to(device, dtype=torch.float)
                embeddings = Variable(embeddings)
                labels_batch = Variable(labels_batch)
                y_pred = blend(embeddings)
                loss = loss_fn(y_pred, labels_batch.reshape(y_pred.shape))
                mse.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print(str(np.array(mse).mean()))
        t.update()
   
    return blend

def _set_seeds(seed=3):
        """ Sets the seeds of various libraries """
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

_set_seeds()
#gets dataloader
batch_size = 64

cleaned_csv = "cleaned_data_all_temp_new_3.csv"
npy_dir = './np_dir/s2_npy/s2'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#if you already have embeddings (i.e. the outputs from the two frozen nets), use this to train a blended model

#dataloaders_embedding = load_embeddings("labels_test.npy", "ns_labels_test.npy",
#                                      "s_labels_test.npy", "embeddings_test.npy",batch_size)
#blend = train_blend(nn.MSELoss(), dataloaders_embedding['train'], batch_size,10000,PATH="blend_1000_1500")
#torch.save(blend.state_dict(),"blend_two_layer_net")


#if you don't already have embeddings, use this to get them!
dataloaders_2016 = load_data(cleaned_csv, npy_dir, "missing", batch_size=batch_size, sample_balanced=True)

ns_net = onn.Small_Net()
ns_net.to(device)
ns_net.load_state_dict(torch.load("nonsentinel_net_1"))
ns_net.eval()

cnn_net = cnn.Small_CNN()
saved_weights_path = os.path.join("./checkpoints/", "best_6_scratch.pth.tar")
print("Restoring parameters from {}".format(saved_weights_path))
utils.load_checkpoint(saved_weights_path, cnn_net)
cnn_net.to(device)
cnn_net.eval()

get_embeddings(ns_net,cnn_net,dataloaders_2016['train'], batch_size)

