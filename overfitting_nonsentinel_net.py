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
#summary
#.0005 is a good lr
#main code for the net has "MAIN" commented

torch.manual_seed(0)

#main
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
        
        x_1_input = self.bn1(x)
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
        x_7 = F.leaky_relu(self.fc7(x_7_input))
        x = self.final(x_7)
        
        return x
#main
def normalize_train(X_train):
    means = X_train.mean(axis=0)
    new_X = X_train-means
    var = X_train.var(axis=0)
    std = np.sqrt(var)
    #when there's no variance (all the same value, probably a lot of sentinel values for missing data)
    #skips the division by setting std = 1, thus avoiding division by 0
    for i in range(len(std)):
        if std[i]==0:
            std[i]=1
    new_X = new_X/std
    return new_X,means,std


def train_small_net(X_train, X_val, y_train, y_val, under_fifty=True, learning_rate = .025):
    small_net = Small_Net()
    input_data = X_train
    input_data,means,std = normalize_train(input_data)
    input_data = torch.from_numpy(input_data).float()
    input_y = torch.from_numpy(y_train).unsqueeze(1).float()

    #MAIN; ten batches
    input_data_batches = []
    input_y_batches = []
    num_batches = 10
    X_val = torch.from_numpy((X_val-means)/std).float()
    y_val = torch.from_numpy(y_val).unsqueeze(1).float()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    for batch in range(num_batches):
        data_batch = input_data[int(len(X_train)/num_batches*batch):int(len(X_train)/num_batches*(batch+1))]
        data_batch = data_batch.to(device)
        input_data_batches.append(data_batch)
        y_batch = input_y[int(len(X_train)/num_batches*batch):int(len(X_train)/num_batches*(batch+1))]
        y_batch = y_batch.to(device)
        input_y_batches.append(y_batch)
        
    small_net.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    #MAIN; mse, gradient descent, 610 epochs
    optimizer = optim.SGD(small_net.parameters(), lr=learning_rate, momentum = .9,weight_decay = .001)
    loss_fn = nn.MSELoss()
                               
    
    train_acc = []
    val_acc = []
    epochs = 610
    for epoch in range(epochs):
        
        epoch_train_accs = np.zeros(num_batches)
        epoch_val_accs = np.zeros(num_batches)
        for batch in range(num_batches):
            input_data_batch = input_data_batches[batch]
            input_y_batch = input_y_batches[batch]
            y_pred = small_net(input_data_batch)
            y_val_pred = small_net(X_val)
            val_loss = loss_fn(y_val_pred, y_val)
            loss = loss_fn(y_pred, input_y_batch)
            epoch_train_accs[batch] = loss.item()
            epoch_val_accs[batch] = val_loss.item()                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc.append(epoch_train_accs.mean())
        val_acc.append(epoch_val_accs.mean())
        

    y_val_pred = small_net(X_val)
    val_loss = loss_fn(y_val_pred, y_val)
    print("Val Mean squared error: %.2f"
              % mean_squared_error(y_val.detach().numpy(), y_val_pred.detach().numpy()))
    print("Check: Val Mean squared error: %.2f"
              % val_loss.item())
    print('Val Variance score: %.2f' % r2_score(y_val.detach().numpy(), y_val_pred.detach().numpy()))
    return small_net, means, std, train_acc, val_acc


      
#under_fifty has now been effectively reinterpreted as 20.5
def get_train_test_val_data(filenames,under_fifty=True,filter_empty_temp=True, single_month = False, single_year = False,filter_modis=False, test_portion=.1):
    
    df = pandas.DataFrame()
    for file in filenames:
        df = df.append(pandas.read_csv(file),ignore_index=True)
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
   

    if under_fifty:
        
        print(test_portion)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_portion, random_state=0)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.111111, random_state=0)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.222222, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test


for test_size in [.1,.75, .7, .6, .5]:
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_val_data(["master_epa_weather_modis_sentinel_2016.csv",
                                                                         "master_epa_modis_weather_sentinel_2017.csv",
                                                                         "master_epa_modis_weather_2019.csv"],
                                                                         filter_modis=False,test_portion = test_size)
    print(X_train.shape)
    print(y_train.shape)
    X_test=X_val
    y_test=y_val

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
plt.show()
plt.clf()
