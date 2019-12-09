import os
import pandas
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import scipy.stats
from sklearn import datasets, linear_model
#from torch.utils.data import Dataset, DataLoader

#learning rate of .005 gets decent results on full dataset (training MSE around 8; val MSE close to 10)


torch.manual_seed(3)

class Small_Net(nn.Module):
    def __init__(self):
        super(Small_Net, self).__init__()
        
        self.fc1 = nn.Linear(16,100)
        self.fc2 = nn.Linear(116,100)
        self.fc3 = nn.Linear(200,100)
        self.fc4 = nn.Linear(200,100)
        self.fc5 = nn.Linear(200,100)
        self.fc6 = nn.Linear(200,100)
        self.fc7 = nn.Linear(200,100)
        self.final = nn.Linear(100,1)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(116)
        self.bn3 = nn.BatchNorm1d(200)
        self.bn4 = nn.BatchNorm1d(200)
        self.bn5 = nn.BatchNorm1d(200)
        self.bn6 = nn.BatchNorm1d(200)
        self.bn7 = nn.BatchNorm1d(200)
        
        
    def forward(self,x):
        #check if x and model are on cuda
        
        #x_1_input = self.bn1(x)
        x_1 = F.leaky_relu(self.fc1(x))
        x_2_input = self.bn2(torch.cat((x_1,x),1))
        x_2 = F.leaky_relu(self.fc2(x_2_input))
        x_3_input = self.bn3(torch.cat((x_2,x_1),1))
        x_3 = F.leaky_relu(self.fc3(x_3_input))
        x_4_input = self.bn4(torch.cat((x_3,x_2),1))
        x_4 = F.leaky_relu(self.fc4(x_4_input))
        x_5_input = self.bn5(torch.cat((x_4,x_3),1))
        x_5 = F.leaky_relu(self.fc5(x_5_input))
        x_6_input = self.bn6(torch.cat((x_5,x_4),1))
        x_6 = F.leaky_relu(self.fc6(x_6_input))
        x_7_input = self.bn7(torch.cat((x_6,x_5),1))
        x_7_output = self.fc7(x_7_input)
        x_7 = F.leaky_relu(x_7_output)
        x = self.final(x_7)
        return x, x_7_output

#assumes normalized data
def train_small_net(X_train, X_val, y_train, y_val, under_fifty=True, learning_rate = .005):
    small_net = Small_Net()
    input_data = X_train
    input_data = torch.from_numpy(input_data).float()
    input_y = torch.from_numpy(y_train).unsqueeze(1).float()

    input_data_batches = []
    input_y_batches = []
    num_batches = 10
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).unsqueeze(1).float()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    for batch in range(num_batches):
        data_batch = input_data[int(len(X_train)/num_batches*batch):int(len(X_train)/num_batches*(batch+1))]
        data_batch = data_batch.to(device)
        input_data_batches.append(data_batch)
        y_batch = input_y[int(len(X_train)/num_batches*batch):int(len(X_train)/num_batches*(batch+1))]
        y_batch = y_batch.to(device)
        print("batch size")
        print (len(y_batch))
        input_y_batches.append(y_batch)
        
    small_net.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    
    
    

    optimizer = optim.SGD(small_net.parameters(), lr=learning_rate, momentum = .9,weight_decay = .001)
    loss_fn = nn.MSELoss()
                               
    
    train_acc = []
    val_acc = []
    #epochs = 610
    epochs = 300
    best_val = 0
    for epoch in range(epochs):
        
        epoch_train_accs = np.zeros(num_batches)
        epoch_val_accs = np.zeros(num_batches)
        for batch in range(num_batches):
            input_data_batch = input_data_batches[batch]
            input_y_batch = input_y_batches[batch]
            y_pred,_ = small_net(input_data_batch)
            small_net.eval()
            y_val_pred,_ = small_net(X_val)
            small_net.train()
            val_loss = loss_fn(y_val_pred, y_val)
            loss = loss_fn(y_pred, input_y_batch)
            epoch_train_accs[batch] = loss.item()
            epoch_val_accs[batch] = val_loss.item()                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc.append(epoch_train_accs.mean())
        val_acc.append(epoch_val_accs.mean())
        if epoch % 5 == 2:
            print("Non-Sentinel Net epoch #:")
            print(epoch)
            print("training loss")
            print(epoch_train_accs.mean())
            print("val loss")
            print(epoch_val_accs.mean())
            small_net.eval()
            y_val_pred_np,_ = small_net(X_val)
            small_net.train()
            y_val_pred_np = y_val_pred_np.cpu().detach().numpy()
            y_val_np = y_val.cpu().detach().numpy()
            print(str(mean_squared_error(y_val_np,y_val_pred_np)))
            print("mean batch val r2")
            r2 = r2_score(y_val_np,y_val_pred_np)
            print(str(r2))
            if r2>best_val:
                print("SAVING")
                torch.save(small_net.state_dict(), "best_val_nonsentinel_savepoint")
                best_val=r2
            

            
                
    return small_net, train_acc, val_acc


means = np.array([ 38.96955041, -95.55518103,   6.18044079 , -0.53281437,  -0.53002186,
  -0.52982451,  -0.53317033,  -0.51143512 , -0.5089666,   -0.50960552,
  -0.51276996,  28.88975651,   1.24007246,  11.62563945, 196.37530146,
  78.43565678])

std = np.array([  5.27143519,  17.4077515,    3.35727818,   0.55251727,   0.55308288,
   0.55312935,   0.55245427 ,  0.57751112 ,  0.57814025,   0.57807082,
   0.57729462,  86.11508944,  13.85940652, 100.42227602, 109.87577457,
  99.77715125])      

def get_data(filename,under_fifty=True,filter_empty_temp=True, single_month = False, single_year = False,filter_modis=False):    
    df = pandas.read_csv(filename)
    df = df[df['TMAX'].notnull()]
    df = df[df['TMIN'].notnull()]
    print("before filter")
    print (df)
    if filter_modis:
        df = df[(df['Blue [0,0]'] !=-1) & (df['Blue [0,1]'] !=-1)
                & (df['Blue [1,0]']!=-1) & (df['Blue [1,1]'] != -1)
                & (df['Green [0,0]'] != -1) & (df['Green [0,1]']!=-1)
                & (df['Green [1,0]'] != -1) & (df['Green [1,1]'] != -1)]
    print (df)
    df['Date'] = pandas.to_datetime(df['Date'])
    if under_fifty:
        df = df[df['Daily Mean PM2.5 Concentration']<20.5]
        if single_month:
            df = df[df['Date'].dt.month==single_month]
            df = df[df['Date'].dt.year==single_year]
    else:
        df = df[df['Daily Mean PM2.5 Concentration']>20.5]

    df['PRCP'].fillna(-1,inplace=True)
    df = df[df['PRCP']>-1]
    df['SNOW'].fillna(-1,inplace=True)
    df['SNWD'].fillna(-1,inplace=True)
    months = df['Date'].dt.month
    X = np.stack((df['SITE_LATITUDE'],df['SITE_LONGITUDE'],months,
              df['Blue [0,0]'],df['Blue [0,1]'], df['Blue [1,0]'], df['Blue [1,1]'],
              df['Green [0,0]'],df['Green [0,1]'], df['Green [1,0]'], df['Green [1,1]'],
            df['PRCP'], df['SNOW'], df['SNWD'], df['TMAX'], df['TMIN']),axis=1)
    y = np.array(df['Daily Mean PM2.5 Concentration'])
    X = (X-means)/std
    return X,y

def eval(small_net,X,y):
    small_net.eval()
    num_examples = y.size
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).unsqueeze(1).float()
    
    y_pred_np = np.ndarray([1])
    batches = 3
    for index in range(batches):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        small_net.to(device)
        print(str(num_examples))
        print(str(int(index*(num_examples/batches))))
        print(str(int((index+1)*num_examples/batches)))
            
        X_batch = X[int(index*(num_examples/batches)):int((index+1)*num_examples/batches)]
        y_batch = y[int(index*(num_examples/batches)):int((index+1)*num_examples/batches)]
        X_batch = X_batch.to(device)
        y_pred_np_batch,_ = small_net(X_batch)
        y_pred_np_batch = y_pred_np_batch.cpu().detach().numpy()
        print(y_pred_np_batch.shape)
        print(y_pred_np.shape)
        print(y_pred_np.size)
        if y_pred_np.size==1:
            y_pred_np=y_pred_np_batch
        else:
            y_pred_np = np.concatenate((y_pred_np,y_pred_np_batch),axis=0)

    y_np = y.cpu().detach().numpy()

    print(y_pred_np.shape)
    print(y_np.shape)
    
    print("mse")
    print(str(mean_squared_error(y_np,y_pred_np)))
    print("r2")
    r2 = r2_score(y_np,y_pred_np)
    print(str(r2))
    pearson = scipy.stats.pearsonr(y_np.reshape((num_examples)),y_pred_np.reshape((num_examples)))
    print("Pearson")
    print(str(pearson))
    #plt.scatter(y_np,y_pred_np)
    #plt.xlabel("Real PM2.5 Values (μg/$m^3$')")
    #plt.ylabel("Nonsentinel Net PM2.5 Predictions (μg/$m^3$')")
    #plt.savefig("nonsentinel_real_vs_pred.png")

X_val, y_val = get_data("./processed_data/val_sites_master_csv_2017.csv")
X_train, y_train = get_data("./processed_data/train_sites_master_csv_2017.csv")

net = Small_Net()
#net.load_state_dict(torch.load("best_val_nonsentinel_savepoint"))
train_small_net(X_train,X_val,y_train,y_val,learning_rate=.005)

"""
for test_size in [.1,.75, .7, .6, .5]:
    X_train, y_train = get_data("total_train.csv")
    X_val, y_val = get_data("total_val.csv")

    print("NOW TRAINING THIS SIZE: "+str(test_size))
    

    for learning_rate in [.005,.0075,.01,.015,.02,.025]:
        if learning_rate == .005 or test_size<.92:
            print("NOW TRAINING THIS SIZE: "+str(test_size)+" AT THIS RATE: "+str(learning_rate))
            small_net, means, std, train_acc, val_acc = train_small_net(X_train, X_val, y_train, y_val, learning_rate=learning_rate)
            print(min(train_acc))
            plt.plot(train_acc,'r--', label = "Training MSE")
            plt.plot(val_acc,'b--', label = "Validation MSE")
            plt.legend()
            plt.savefig("size "+str(test_size)+"rate "+str(learning_rate)+".png")
            torch.save(small_net.state_dict(), "big_nonsentinel_lr_"+str(learning_rate))
plt.show()
plt.clf()"""



