import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from pandarallel import pandarallel 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

class NearestNeighborBaseline():
    '''
    Nearest Neighbor Baseline Model
    
    For each EPA site on a given date, predicts that that site's PM2.5 reading
    is the same as the PM2.5 reading of the next closest EPA site, 
    based on Haversine distance.
    
    '''
    def __init__(self, train_csv, threshold=20.5):
        self.train_df = pd.read_csv(train_csv)
        self.threshold = threshold 
        self.models = {}  # dict mapping each date to NN model
        self.dates = set()
        
    def train(self):
        '''
        Trains NN for each unique date and saves to list of models.
        '''
        
        self.train_df = self.train_df[self.train_df['Daily Mean PM2.5 Concentration'] < self.threshold]
    
        for date in self.train_df['Date']:
            self.dates.add(date)

        MSE = 0
        num_predictions = 0
        num_dates = len(self.dates)
        all_y_train, all_y_train_pred = [], []
        
        # Goes date by date to get nearest neighbor on that date
        # Each date has a list of stations that have measuerments from that date

        for idx, date in enumerate(self.dates):
            if idx % 10 == 0:
                print("Processing date {}/{}: {} ".format(idx, num_dates, date))
        
            tr_date_df = self.train_df[self.train_df['Date']==date]
       
            # X info is latitude, longitude; y is PM2.5; epa_set_ids tracks corresponnding site_ids
            X_train = []
            y_train = []

            for i in range(len(tr_date_df)):
                lat = np.radians(tr_date_df['SITE_LATITUDE'][tr_date_df.index[i]])
                long = np.radians(tr_date_df['SITE_LONGITUDE'][tr_date_df.index[i]])
                pm = tr_date_df['Daily Mean PM2.5 Concentration'][tr_date_df.index[i]]
                epa_site_id = tr_date_df['Site ID'][tr_date_df.index[i]]
                X_train.append([lat,long]) 
                y_train.append(pm)
                
            # Nearest neighbors for each date, as determined by haversine (distance between lat,lon coordinate pairs)
            knn = KNeighborsRegressor(n_neighbors=1, metric="haversine")
            knn.fit(X_train,y_train)
            
            # Get the nearest neighbors of train data (not including point itself)
            y_train_nn_indices = knn.kneighbors(X_train)[1]
            y_train_nn_indices = [y for x in y_train_nn_indices for y in x] # flatten the list
            y_train_pred = np.asarray(y_train)[y_train_nn_indices]

            all_y_train.append(y_train)
            all_y_train_pred.append(y_train_pred.tolist())
            
            # Save the model to dict of all models
            self.models[date] = knn

        all_y_train = utils.flatten(all_y_train)
        all_y_train_pred = utils.flatten(all_y_train_pred)

        r2_train = r2_score(all_y_train, all_y_train_pred)
        pearson_train = pearsonr(all_y_train, all_y_train_pred)
        MSE_train = ((np.asarray(all_y_train)-np.asarray(all_y_train_pred))**2).sum()/len(all_y_train)

        print("Train MSE all dates:  {}".format(MSE_train))
        print("Train r2 all dates:  {}".format(r2_train))
        print("Train pearson all dates:  {}".format(pearson_train))

                            
    def predict(self, test_csv):
        '''
        Uses models trained for each date to predict on the test sites given in test_
        '''
        
        self.test_df = pd.read_csv(test_csv)
        self.test_df = self.test_df[self.test_df['Daily Mean PM2.5 Concentration'] < self.threshold]

        all_y_test, all_y_pred = [], []
        all_states, all_months = [], []
        all_dists = []
        
        for idx, date in enumerate(self.dates):
           
            test_date_df = self.test_df[self.test_df['Date']==date]

            # X info is latitude, longitude; y is PM2.5; epa_set_ids tracks corresponnding site_ids
            X_test, y_test = [], []
            states, months = [], []
            
            for i in range(len(test_date_df)):
                lat = np.radians(test_date_df['SITE_LATITUDE'][test_date_df.index[i]])
                long = np.radians(test_date_df['SITE_LONGITUDE'][test_date_df.index[i]])
                pm = test_date_df['Daily Mean PM2.5 Concentration'][test_date_df.index[i]]
                state = test_date_df['STATE'][test_date_df.index[i]]
                month = test_date_df['Month'][test_date_df.index[i]]
                X_test.append([lat,long]) 
                y_test.append(pm)
                states.append(state)
                months.append(month)

            # Get the trained nn model for the current date
            knn = self.models[date]
            y_pred = knn.predict(X_test).tolist()
        
            # Get the distance of the nearest neighbor
            neighbor_distances_indices = knn.kneighbors(X_test)
            y_test_nn_distances = (neighbor_distances_indices[0].flatten()).tolist()
           
            # Append this date's predictions/labels to list of all predictions/labels
            all_y_test.append(y_test)
            all_y_pred.append(y_pred)
            
            # Save states, months, and distances for analysis
            all_states.append(states)
            all_months.append(months)
            all_dists.append(y_test_nn_distances)
            
        all_y_test = utils.flatten(all_y_test)
        all_y_pred = utils.flatten(all_y_pred)

        all_states = utils.flatten(all_states)
        all_months = utils.flatten(all_months)
        all_dists = utils.flatten(all_dists)
        
        r2 = r2_score(all_y_test, all_y_pred)
        pearson = pearsonr(all_y_test, all_y_pred)
        MSE = ((np.asarray(all_y_test)-np.asarray(all_y_pred))**2).sum()/len(all_y_test)
       
        print("Test MSE all dates:  {}".format(MSE))
        print("Test r2 all dates:  {}".format(r2))
        print("Test pearson all dates:  {}".format(pearson))

        return all_y_test, all_y_pred, all_dists, all_states, all_months, r2, pearson, MSE

       
                     
def save_predictions(labels, predictions, dists, months, states, save_to):
    '''
    For each datapoint, saves: labels, predictions, distance to nearest neighbor, month, state
    to the .csv file specified in save_to.
    '''
    
    df = pd.DataFrame()
    df['Index'] = np.arange(0,len(labels))
    df['Prediction'] = predictions
    df['Label'] = labels
    df['Distance from Nearest Site'] = dists
    df['Month'] = months
    df['State'] = states

    df.to_csv(save_to)
    
    
def plot_mse_vs_dist(predictions_csv):
    '''
    Plots MSE vs distance from nearest neighbor for predictions in predictions_csv,
    and returns the correlation coefficient between MSE and distance.
    '''
    
    pandarallel.initialize()

    df = pd.read_csv(predictions_csv)
    mses = df.parallel_apply(utils.mse_row, axis=1)
    dists = df['Distance from Nearest Site']
    pearson = pearsonr(dists, mses)
    plt.scatter(dists, mses, s=1)
    plt.ylabel("Mean Squared Error", fontsize=14)
    plt.xlabel("Distance to Next Closest EPA Site", fontsize=14)
    plt.title("Mean Squared Error versus Distance to Nearest EPA Site", fontsize=16)
    plt.savefig("plots/mse_vs_distance.png")
    plt.show()
    
    return pearson

def run_baseline():
    '''
    Runs the Nearest Neighbor baseline model, saves test set predictions, 
    and plots the predictions.
    '''
    
    train_csv = os.path.join(utils.PROCESSED_DATA_FOLDER, "train_sites_master_csv_2016_2017.csv")   
    test_csv = os.path.join(utils.PROCESSED_DATA_FOLDER, "test_sites_master_csv_2016_2017.csv")
    predictions_csv = "predictions/knn_predictions_new.csv"
    
    model = NearestNeighborBaseline(train_csv=train_csv)
    model.train()
    
    all_y_test, all_y_pred, all_dists, all_states, all_months, r2, pearson, MSE = model.predict(test_csv=test_csv)
    save_predictions(all_y_test, all_y_pred, all_dists, all_months, all_states, predictions_csv)
    
    ## utils.plot_predictions(predictions_csv, "Nearest Neighbor Baseline")
    ## pearson = plot_mse_vs_dist(predictions_csv)

    
if __name__ == "__main__":

    run_baseline()
    
    
    
