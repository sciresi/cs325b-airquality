#four epochs total
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
import torch
from dataloader import load_data_new,load_embeddings
import utils
import random
import scipy.stats
import frozen_combined_net as fcn
import utils


class Full_Net(nn.Module):
    #when loading from a file
    
    def __init__(self, ns_net, s_net, blend):
        super(Full_Net, self).__init__()
        self.ns_net = ns_net
        self.s_net = s_net
        self.blend = blend
        
    def forward(self,ns_features, s_features):
        
        _, ns_embeddings = self.ns_net(ns_features)
        _, s_embeddings = self.s_net(s_features)
        blended_embeddings = torch.cat((ns_embeddings,s_embeddings),dim=1)
        x = self.blend(blended_embeddings)
        return x
        
def train_full_net(model, optimizer, loss_fn, train_dataloader, val_dataloader,
                   batch_size = 90, epochs = 20):
    model.to(device)
    best_val_r2=0.0

    num_batches = len(train_dataloader)
    val_num_batches = len(val_dataloader)
    
    for epoch in range(epochs):
        model.train()
        with tqdm(total=num_batches) as t:
            for i, sample in enumerate(train_dataloader):
                #can be used to speed up training by randomly skipping some batches
                #to test whether overall code works
                if True:
                    labels_batch = sample['label']
                    s_features = sample['image']
                    ns_features = sample['non_image'].double()
                    
                    # Move to GPU if available
                    labels_batch = labels_batch.to(device, dtype=torch.float)
                    s_features = s_features.to(device, dtype=torch.float)
                    ns_features = ns_features.to(device, dtype=torch.float)
                    
                    labels_batch = Variable(labels_batch)
                    s_features = Variable(s_features)
                    ns_features = Variable(ns_features)
                    
                    y_preds_batch = model(ns_features,s_features)
                    
                    loss = loss_fn(y_preds_batch,labels_batch.reshape(y_preds_batch.shape))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                t.update()
        
        val_r2 = model_eval(model,val_dataloader)
        if val_r2>best_val_r2:
            print("saving")
            torch.save(model.state_dict(),"checkpoints/unfrozen_best")
            best_val_r2 = val_r2
        torch.save(model.state_dict(),"checkpoints/unfrozen_last")

        t.update()
    return model

    

def model_eval(model,dataloader):
    num_batches = len(dataloader)
    pred_labels = np.zeros((1))
    labels = np.zeros((1))
    
    model.eval()
    with tqdm(total=num_batches) as t:
        for i, sample in enumerate(dataloader):
            labels_batch = sample['label']
            s_features = sample['image']
            ns_features = sample['non_image'].double()
            # Move to GPU if available
            labels_batch = labels_batch.to(device, dtype=torch.float)
            s_features = s_features.to(device, dtype=torch.float)
            ns_features = ns_features.to(device, dtype=torch.float)

            labels_batch = Variable(labels_batch)
            s_features = Variable(s_features)
            ns_features = Variable(ns_features)

            y_pred = model(ns_features,s_features)
            y_pred_np = y_pred.detach().cpu().numpy()
            labels_batch_np = labels_batch.detach().cpu().numpy()
            if labels.size == 1:
                pred_labels = y_pred_np
                labels = labels_batch_np
            else:
                pred_labels = np.concatenate((pred_labels,y_pred_np),axis=0)
                labels = np.concatenate((labels,labels_batch_np),axis=0)
            t.update()
        print("Fine-tuned MSE")
        print(str(mean_squared_error(labels,pred_labels)))
        print("Fine-tuned R2")
        print(str(r2_score(labels,pred_labels)))
        num_examples = labels.size
        pearson = scipy.stats.pearsonr(labels.reshape((num_examples)),pred_labels.reshape((num_examples)))
        print("Fine-tuned Pearson")
        print(str(pearson))
    
    
    return r2_score(labels,pred_labels)

def save_predictions(indices,predictions,labels,sites,dates,batch_size,save_to):
    with open(save_to, 'a') as fd:
        writer = csv.writer(fd)
        for i in range (0,batch_size):
            index = indices[i]
            y_pred = predictions[i]
            y_true = labels[i]
            site = sites[i]
            date = dates[i]
            row = [index, y_pred, y_true, site, date]
            writer.writerow(row)

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ns_net = onn.Small_Net()
    ns_net.to(device)
    ns_net.load_state_dict(torch.load("checkpoints/best_val_nonsentinel_savepoint"))
    ns_net.eval()

    cnn_net = cnn.Small_CNN()
    saved_weights_path = "checkpoints/sentinel_cnn/best_weights.pth.tar"
    print("Restoring parameters from {}".format(saved_weights_path))
    utils.load_checkpoint(saved_weights_path, cnn_net)
    cnn_net.to(device)
    cnn_net.eval()
    
    blend = fcn.Blended_Component()
    blend.load_state_dict(torch.load("checkpoints/blend_two_layer_net_spatial_best_2017"))
    blend.to(device)
    blend.eval()

    model = Full_Net(ns_net,cnn_net,blend)
    model.to(device)

    train_file = os.path.join(utils.PROCESSED_DATA_FOLDER, "train_sites_master_csv_2016_2017.csv")
    val_file = os.path.join(utils.PROCESSED_DATA_FOLDER, "val_sites_master_csv_2016_2017.csv")
    test_file = os.path.join(utils.PROCESSED_DATA_FOLDER, "test_sites_master_csv_2016_2017.csv")
    npy_dir = utils.SENTINEL_FOLDER 


    dataloaders = load_data_new(train_file, batch_size = 90, num_workers=8,
                                threshold=20.5,
                                train_images=npy_dir, val_images=npy_dir,
                                val_nonimage_csv=val_file) 



    #code for reloading from saved weights
    #model = Full_Net(onn.Small_Net(),cnn.Small_CNN(),fcn.Blended_Component())
    #model.load_state_dict(torch.load("unfrozen_best"))
    #model.to(device)

    batch_size = 90
    lr = 0.00001
    optimizer = optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.MSELoss()

    model = train_full_net(model, optimizer, loss_fn, dataloaders["train"], dataloaders["val"], batch_size = batch_size)

    print("Train statistics on fine-tuned model")
    model_eval(model,dataloaders['train'])

    print("Val statistics on fine-tuned model")
    model_eval(model,dataloaders['val'])


    dataloaders = load_data_new(train_file, batch_size = 90, num_workers=8,
                                threshold=20.5,
                                train_images=npy_dir, val_images=npy_dir,
                                val_nonimage_csv=test_file) 
    print("Test statistics on fine-tuned model")
    model_eval(model,dataloaders['val'])


