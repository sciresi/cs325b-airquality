import pandas
import matplotlib.pyplot as plt
import numpy as np

sent_df = pandas.read_csv("channel_means/sentinel_channel_means_200.csv")

b1 = sent_df['B1']
b2 = sent_df['B2']
b3 = sent_df['B3']
b4 = sent_df['B4']
b5 = sent_df['B5']
b6 = sent_df['B6']
b7 = sent_df['B7']
b8 = sent_df['B8']
b8a = sent_df['B9']
b9 = sent_df['B10']
b10 = sent_df['B11']
b11 = sent_df['B12']
b12 = sent_df['B13']

array = [b1,b2,b3,b4,b5,b6,b7,b8,b8a,b9,b10,b11,b12]
bins=150

def plot_cirrus():
    
    plt.hist(b10,bins,alpha=.5,label = 'B10 (Cirrus)')
    plt.xlabel("Band 10 Mean Brightness")
    plt.ylabel("# Images")
    plt.title("Distribution of Band 10 Mean Brightness")
    plt.legend(loc='upper right')
    plt.savefig("sent_dist_cirrus.png")
    plt.show()

def plot_rgb():
    '''
    Plots distrubtion for just RGB bands (B2,B3,B4)
    '''
    plt.hist(b2,bins,alpha=.5,label = 'B2 (Blue)')
    plt.hist(b3,bins,alpha=.5,label = 'B3 (Green)')
    plt.hist(b4,bins,alpha=.5,label = 'B4 (Red)')
    plt.xlabel("Bands B2 (Blue), B3 (Green), and B4 (Red) Mean Brightness")
    plt.ylabel("# Images")
    plt.title("Distribution of Sentinel-2 Band Mean Brightness (200x200)")
    plt.legend(loc='upper right')
    plt.savefig("sent_dist_rgb.png")
    plt.show()
                           
    
def plot_rest():
    '''
    Plots all bands excluding Cirrus (Band 10).
    '''
    plt.hist(b1,bins,alpha=.5,label = 'B1 (Aerosols)')
    plt.hist(b2,bins,alpha=.5,label = 'B2 (Blue)')
    plt.hist(b3,bins,alpha=.5,label = 'B3 (Green)')
    plt.hist(b4,bins,alpha=.5,label = 'B4 (Red)')
    plt.hist(b5,bins,alpha=.5,label = 'B5 (Red Edge 1)')
    plt.hist(b6,bins,alpha=.5,label = 'B6 (Red Edge 2)')
    plt.hist(b7,bins,alpha=.5,label = 'B7 (Red Edge 3)')
    plt.hist(b8,bins,alpha=.5,label = 'B8 (NIR)')
    plt.hist(b8a,bins,alpha=.5,label = 'B8a (Red Edge 4)')
    plt.hist(b9,bins,alpha=.5,label = 'B9 (Water Vapor)')
    plt.hist(b11,bins,alpha=.5,label = 'B11 (SWIR 1)')
    plt.hist(b12,bins,alpha=.5,label = 'B12 (SWIR 2)')

    plt.xlabel("Bands 1-9 and 11-12 Mean Brightness")
    plt.ylabel("# Images")
    plt.title("Distribution of Sentinel-2 Band Mean Brightness (200x200)")
    plt.legend(loc='upper right')
    plt.savefig("sent_dist.png")
    plt.show()

plot_cirrus()
#plot_rest()
#plot_rgb()
