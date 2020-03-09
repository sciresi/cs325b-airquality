import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from pandarallel import pandarallel 
from scipy.stats.stats import pearsonr
import utils

PREDICTIONS_FOLDER = os.path.join(utils.HOME_FOLDER, utils.REPO_NAME, "predictions")

def get_month(row):
    date = pd.to_datetime(row['Date'])
    month = date.month
    return month

def compute_true_month_averages(master_csv, true_averages_csv):
    '''
    Computes the ground truth PM monthly averages from daily labels
    from a given master csv file and saves to the provided
    true_averages_csv file.
    '''
    pandarallel.initialize()
    
    df = pd.read_csv(master_csv)
    
    # Index on 'Month' and 'Site Id' to compute averages at each station for the month                
    months = df.parallel_apply(get_month, axis=1)
    df['Month'] = months

    epa_stations = df['Site ID'].unique()
    num_sites = len(epa_stations)

    with open(true_averages_csv, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(["Site ID", "Month", "Month Average"])
        for i, station_id in enumerate(epa_stations):
            
            station_datapoints = df[df['Site ID'] == station_id]

            for month in range(1,13):

                month_m_at_station_i = station_datapoints[station_datapoints['Month'] == month]
                pms_for_month_m_at_station_i = month_m_at_station_i['Daily Mean PM2.5 Concentration']
                month_average = np.mean(pms_for_month_m_at_station_i)
                row = [station_id, month, month_average]
                writer.writerow(row)
    
def compute_predicted_month_averages(preds_csv, master_csv, true_averages_csv, predicted_and_true_avgs_csv):
    '''
    - Computes the predicted PM monthly averages from daily predictions
        from a given predictions csv file.
        
    - Saves the new predictions to a separate predicted_averages_csv file.
    - Then reads from both true_averages_csv file and this new
       predicted_averages_csv file, joins the two dfs, and saves final
       df with both true and predicted averages to predicted_and_true_avgs_csv. 
    '''
    predicted_avgs_csv = "final_predictions_monthly_avgs.csv"  # intermediate file
    
    master_df = pd.read_csv(master_csv)
    preds_df = pd.read_csv(preds_csv)
    true_averages_df = pd.read_csv(true_averages_csv)
    
    epa_stations = preds_df['Site ID'].unique()  # Get unique EPA sites
    
    with open(predicted_avgs_csv, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(["Site ID", "Month", "Predicted Month Average"])

        for i, station_id in enumerate(epa_stations):
            station_datapoints = preds_df[preds_df['Site ID'] == station_id]

            for month in range(1,13):

                month_m_at_station_i = station_datapoints[station_datapoints['Month'] == month]
                if len(month_m_at_station_i) == 0:
                    continue
                pm_preds_for_month_m_at_station_i = month_m_at_station_i['Prediction']
                month_average_pred = np.mean(pm_preds_for_month_m_at_station_i)
                row = [station_id, month, month_average_pred]
                writer.writerow(row)

    # Now read from new averages file and merge with old
    #true_averages_df = pd.read_csv(averages_csv)
    true_averages_df = true_averages_df.set_index(["Site ID", "Month"])
    predicted_average_df = pd.read_csv(predicted_avgs_csv)
    predicted_average_df = predicted_average_df.set_index(["Site ID", "Month"])
    combined = pd.concat([true_averages_df, predicted_average_df], axis=1)
    combined.to_csv(predicted_and_true_avgs_csv)


def add_monthly_avgs_to_predictions_files(monthly_avgs_csv, preds_csv, combined_csv):
    '''
    Reads in the df with predicted and true monthly averages given 
    by monthly_avgs_csv, and finally adds both predicted and ground 
    truth monthly average back to the original predictions df given
    by preds_csv.
    
    Saves this final df to the csv given by combined_csv.
    '''
    preds_df = pd.read_csv(preds_csv)
    avgs_df = pd.read_csv(monthly_avgs_csv)
    avgs_df = avgs_df.set_index(["Site ID", "Month"])
    combined = preds_df.join(avgs_df, on=["Site ID", "Month"], how='left')
    combined = combined[combined['Month Average'].notnull()]
    combined = combined[combined['Predicted Month Average'].notnull()]
    combined.to_csv(combined_csv)
    
    
def compute_monthly_r2(preds_csv_with_monthly_avgs):
    '''
    Reads in a predictions df with monthly averages from 
    preds_csv_with_monthly_avgs and computes the final
    r2 and Pearson for monthly aggregated predictions.
    '''
    df = pd.read_csv(preds_csv_with_monthly_avgs)
    labels = df['Month Average']
    predictions = df['Predicted Month Average']
    r2 = r2_score(labels, predictions)
    pearson = pearsonr(labels, predictions)
    return r2, pearson
    

def run_month_loop(val_master, val_preds_csv, true_averages_csv, both_monthly_avgs_csv, preds_and_avgs_csv):
    '''
    Runs full monthly aggregation loop to convert from daily predictions to monthly
    predictions, and compute final monthly r2 and pearson scores.
    '''
    compute_true_month_averages(val_master, true_averages_csv)
    compute_predicted_month_averages(val_preds_csv, val_master, true_averages_csv, both_monthly_avgs_csv)
    add_monthly_avgs_to_predictions_files(both_monthly_avgs_csv, val_preds_csv, preds_and_avgs_csv)
    r2, p = compute_monthly_r2(preds_and_avgs_csv)
    return r2, p


if __name__ == "__main__":

    val_master = os.path.join(utils.PROCESSED_DATA_FOLDER, "val_sites_DT_and_thresh_2000_csv_2016.csv")
    val_preds = os.path.join(PREDICTIONS_FOLDER, "combined_val_16_mini_epoch_13.csv")
    r2, p = run_month_loop(val_master, val_preds, "final_true_avgs.csv", "final_monthly_avgs.csv", "final_preds_and_avgs.csv")
    print("Monthly averages R2: {}".format(r2))
    