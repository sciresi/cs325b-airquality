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



#pretrained, does not train cnn or nonsentinel net!
class Blended_Component(nn.Module):
    #.00001 is good for one-layer
        
    def __init__(self):
        super(Blended_Component, self).__init__()
        self.fc1 = nn.Linear(200,100)
        self.fc2 = nn.Linear(100,1)
        #self._set_seeds(3)
    def forward(self,blended_feats):
        x = self.fc1(blended_feats)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x

#uses 2017 and 19 to train nonsentinel portion




def get_embeddings(ns_net,cnn,dataloader,batch_size,dataset):
    num_batches = len(dataloader)
    ns_net.eval()
    cnn.eval()
    ns_labels = np.zeros((1))
    s_labels = np.zeros((1))
    labels = np.zeros((1))
    embeddings = np.zeros((1))
    counter=0
    with tqdm(total=num_batches) as t:
        for i, sample in enumerate(dataloader):
            counter=counter+1
            input_batch, labels_batch = sample['image'], sample['label']
            features = sample["non_image"].double()
            np.set_printoptions(threshold=sys.maxsize)
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
            t.update()    
    ns_labels = ns_labels.reshape((ns_labels.size))
    

    #print("non-sentinel stats")
    #print(str(mean_squared_error(labels,ns_labels)))
    #print(str(r2_score(labels,ns_labels)))

    #print("sentinel stats")
    #print(str(mean_squared_error(labels,s_labels)))
    print(str(r2_score(labels,s_labels)))
    np.save("labels_big_spatial_"+dataset+"_17",labels)
    np.save("ns_labels_big_spatial_"+dataset+"_17",ns_labels)
    np.save("embeddings_big_spatial_"+dataset+"_17",embeddings)
    np.save("s_labels_big_spatial_"+dataset+"_17",s_labels)     


def blend_eval(blend,dataloader):
    num_batches = len(dataloader)
    pred_labels = np.zeros((1))
    labels = np.zeros((1))
    
    blend.eval()
    with tqdm(total=num_batches) as t:
        for i, sample in enumerate(dataloader):
            labels_batch = sample['pm']
            embeddings = sample['embed']
            embeddings = embeddings.to(device, dtype=torch.float)
            labels_batch = labels_batch.to(device, dtype=torch.float)
            embeddings = Variable(embeddings)        
            labels_batch = Variable(labels_batch)
            y_pred = blend(embeddings)
            y_pred_np = y_pred.detach().cpu().numpy()
            labels_batch_np = labels_batch.detach().cpu().numpy()
            if labels.size == 1:
                pred_labels = y_pred_np
                labels = labels_batch_np
            else:
                pred_labels = np.concatenate((pred_labels,y_pred_np),axis=0)
                labels = np.concatenate((labels,labels_batch_np),axis=0)
    pred_labels.reshape(labels.shape)
    
    print("blend mse")
    print(str(mean_squared_error(labels,pred_labels)))
    print("blend r2")
    print(str(r2_score(labels,pred_labels)))
    num_examples = labels.size
    pearson = scipy.stats.pearsonr(labels.reshape((num_examples)),pred_labels.reshape((num_examples)))
    print("blend Pearson")
    print(str(pearson))
    
    
def manual_filter(labels,pred_labels,threshold=20.5):
    new_labels = []
    new_preds = []
    for index in range(len(labels)):
        if labels[index]<threshold:
            new_labels.append(labels[index])
            new_preds.append(pred_labels[index])
    return np.asarray(new_labels),np.asarray(new_preds)
    
def non_blend_eval(labels_path, ns_labels_path, s_labels_path):

    labels = np.load(labels_path)
    s_pred_labels = np.load(s_labels_path)
    ns_pred_labels = np.load(ns_labels_path)


    print("sentinel mse")
    print(mean_squared_error(labels,s_pred_labels))
    print("sentinel  r2")
    print(str(r2_score(labels,s_pred_labels)))
    print("sentinel pearson")
    num_examples = labels.size
    print (scipy.stats.pearsonr(labels.reshape((num_examples)),s_pred_labels.reshape((num_examples))))
    

    print("nonsentinel mse")
    print(mean_squared_error(labels,ns_pred_labels))
    print("nonsentinel r2")
    print(str(r2_score(labels,ns_pred_labels)))
    print("nonsentinel pearson")
    num_examples = labels.size
    print (scipy.stats.pearsonr(labels.reshape((num_examples)),ns_pred_labels.reshape((num_examples))))
    



    
def train_blend(loss_fn, dataloader, val_dataloader, batch_size, epochs, PATH = None):
    print("IN TRAINING")
    print(len(dataloader))
    num_batches = len(dataloader)
    val_dataset_size = num_batches * batch_size
    
    blend = Blended_Component()
    if PATH:
        blend.load_state_dict(torch.load(PATH))
    blend.to(device)
    blend.train()
    optimizer = optim.SGD(blend.parameters(), lr=.00001,weight_decay = .001)
    val_mse_total = []
    mse_total = []
    r2_total = []
    val_r2_total = []
    best_val_r2 = 0
    for epoch in range(epochs):
        mse = []
        r2 = []
        val_mse = []
        val_r2 = []
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
                y_pred_np = y_pred.detach().cpu().numpy()
                labels_batch_np = labels_batch.detach().cpu().numpy()
                r2_batch = r2_score(labels_batch_np.reshape(y_pred_np.shape),y_pred_np)
                loss = loss_fn(y_pred, labels_batch.reshape(y_pred.shape))
                mse.append(loss.item())
                r2.append(r2_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        #get validation mse at end of each epoch
        blend.eval()
        val_num_batches = len(val_dataloader)
        with tqdm(total=val_num_batches) as t:
            for i, sample in enumerate(val_dataloader):
                labels_batch = sample['pm']
                embeddings = sample['embed']                
                # Move to GPU if available
                embeddings = embeddings.to(device, dtype=torch.float)
                labels_batch = labels_batch.to(device, dtype=torch.float)
                embeddings = Variable(embeddings)
                labels_batch = Variable(labels_batch)
                y_pred = blend(embeddings)
                y_pred_np = y_pred.detach().cpu().numpy()
                labels_batch_np = labels_batch.detach().cpu().numpy()
                loss = loss_fn(y_pred, labels_batch.reshape(y_pred.shape))
                r2_batch = r2_score(labels_batch_np.reshape(y_pred_np.shape),y_pred_np)
                val_mse.append(loss.item())
                val_r2.append(r2_batch)
        blend.train()  
        if epoch%5==0:
            print("epoch")
            print(str(epoch))
            print("train mse")
            print(str(np.array(mse).mean()))
            print("val mse")
            print(str(np.array(val_mse).mean()))
            print("train r2")
            print(str(np.array(r2).mean()))
            print("val r2")
            if np.array(val_r2).mean()>best_val_r2:
                print("saving")
                torch.save(blend.state_dict(),"checkpoints/blend_two_layer_net_spatial_best_2017")
                best_val_r2 = np.array(val_r2).mean()
            print(str(np.array(val_r2).mean()))
        val_mse_total.append(np.array(val_mse).mean())
        mse_total.append(np.array(mse).mean())
        r2_total.append(np.array(r2).mean())
        val_r2_total.append(np.array(val_r2).mean())
        t.update()
   
    return blend,mse_total,val_mse_total,r2_total,val_r2_total

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
if __name__ == "__main__":
    batch_size = 64

    cleaned_csv = "cleaned_data_all_temp_new_3.csv"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ns_net = onn.Small_Net()
    ns_net.to(device)
    ns_net.load_state_dict(torch.load("checkpoints/best_val_nonsentinel_savepoint"))
    ns_net.eval()


    #TODO: FIX THIS ONE
    cnn_net = cnn.Small_CNN()
    saved_weights_path = "checkpoints/sentinel_cnn/best_weights.pth.tar"
    print("Restoring parameters from {}".format(saved_weights_path))
    utils.load_checkpoint(saved_weights_path, cnn_net)
    cnn_net.to(device)
    cnn_net.eval()

    npy_dir = utils.SENTINEL_FOLDER 

    train_file = os.path.join(utils.PROCESSED_DATA_FOLDER, "train_sites_master_csv_2016_2017.csv")
    val_file = os.path.join(utils.PROCESSED_DATA_FOLDER, "val_sites_master_csv_2016_2017.csv")
    test_file = os.path.join(utils.PROCESSED_DATA_FOLDER, "test_sites_master_csv_2016_2017.csv")


    #if you don't already have embeddings, use this to get them!
    dataloaders_2016_2017 = load_data_new(train_file, batch_size = batch_size, num_workers=8,
                                      train_images=npy_dir, val_images=npy_dir,
                                threshold=20.5,
                                  val_nonimage_csv=val_file)
    print("Getting validation set embeddings")
    get_embeddings(ns_net,cnn_net,dataloaders_2016_2017['val'], batch_size, "val_2017")
    print("Getting training set embeddings")
    get_embeddings(ns_net,cnn_net,dataloaders_2016_2017['train'], batch_size, "train_2017")
    print("Getting test set embeddings")

    dataloaders_2016_2017 = load_data_new(train_file, batch_size = batch_size, num_workers=8,
                                threshold=20.5,
                                      train_images=npy_dir, val_images=npy_dir,
                                  val_nonimage_csv=test_file)
    get_embeddings(ns_net,cnn_net,dataloaders_2016_2017['val'], batch_size, "test_2017")

    print("Getting training set metrics on CNN and Non-Sentinel Net")
    non_blend_eval("labels_big_spatial_train_2017_17.npy", "ns_labels_big_spatial_train_2017_17.npy",
                                      "s_labels_big_spatial_train_2017_17.npy")
    print("Getting validation set metrics on CNN and Non-Sentinel Net")
    non_blend_eval("labels_big_spatial_val_2017_17.npy", "ns_labels_big_spatial_val_2017_17.npy",
                                      "s_labels_big_spatial_val_2017_17.npy")
    print("Getting test set metrics on CNN and Non-Sentinel Net")
    non_blend_eval("labels_big_spatial_test_2017_17.npy", "ns_labels_big_spatial_test_2017_17.npy",
                                      "s_labels_big_spatial_test_2017_17.npy")


    #if you already have embeddings (i.e. the outputs from the two frozen nets), use this to train a blended model
    blend = Blended_Component()
    #blend.load_state_dict(torch.load("blend_two_layer_net_big_spatial"))
    blend.to(device)
    dataloaders_embedding = load_embeddings("labels_big_spatial_train_2017_17.npy", "ns_labels_big_spatial_train_2017_17.npy",
                                      "s_labels_big_spatial_train_2017_17.npy", "embeddings_big_spatial_train_2017_17.npy",
                                        "labels_big_spatial_val_2017_17.npy", "ns_labels_big_spatial_val_2017_17.npy",
                                      "s_labels_big_spatial_val_2017_17.npy", "embeddings_big_spatial_val_2017_17.npy",64)


    blend,mse_total,val_mse_total,r2_total,val_r2_total = train_blend(nn.MSELoss(), dataloaders_embedding['train'],
                                                                 dataloaders_embedding['val'], batch_size,1000)
    print("Getting training set metrics on CNN and Non-Sentinel Net")
    blend_eval(blend,dataloaders_embedding['train'])
    print("Getting validation set metrics on CNN and Non-Sentinel Net")
    blend_eval(blend,dataloaders_embedding['val'])
    dataloaders_embedding = load_embeddings("labels_big_spatial_train_2017_17.npy", "ns_labels_big_spatial_train_2017_17.npy",
                                      "s_labels_big_spatial_train_2017_17.npy", "embeddings_big_spatial_train_2017_17.npy",
                                        "labels_big_spatial_test_2017_17.npy", "ns_labels_big_spatial_test_2017_17.npy",
                                      "s_labels_big_spatial_test_2017_17.npy", "embeddings_big_spatial_test_2017_17.npy",64)
    print("Getting test set metrics on CNN and Non-Sentinel Net")
    blend_eval(blend,dataloaders_embedding['val'])

