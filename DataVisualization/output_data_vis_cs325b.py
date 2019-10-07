#code to get csv files:
#gsutil cp gs://es262-airquality/epa/* .

import os
import pandas
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt



files = os.listdir()
first_file = True
means_by_state = np.zeros(191)
file_num = 0
states = set()
for file in files:
    if file[-4:]==".csv":
        print(file)
        new_df = pandas.read_csv(file)
        print(new_df['STATE'][0])
        states.add(new_df['STATE'][0])
        means_by_state[file_num]=new_df['Daily Mean PM2.5 Concentration'].mean()
        if not first_file:
            df = df.append(new_df,ignore_index=True)
        else:
            df = new_df
            first_file = False
        file_num+=1
print(states)
df_min = df[df['Daily Mean PM2.5 Concentration']==-9.7]
print(df_min)

print(df['Daily Mean PM2.5 Concentration'].mean())
print(df['Daily Mean PM2.5 Concentration'].var())
print(df)

pm=df['Daily Mean PM2.5 Concentration']

plt.hist(pm, bins=150)
plt.title("Distribution of PM2.5 Readings")
plt.xlabel("PM2.5 (micrograms/cubic meter)")
plt.ylabel("# Readings")
plt.show()

plt.hist(pm[pm<50], bins=150)
plt.title("Distribution of PM2.5 Readings  (Excluding >50)")
plt.xlabel("PM2.5 (micrograms/cubic meter)")
plt.ylabel("# Readings")
plt.show()

df_sixteen = df[df['Date'].str[-2:]=="16"]['Daily Mean PM2.5 Concentration']
print(df_sixteen)
print(df_sixteen.mean())
print(df_sixteen.var())
plt.hist(df_sixteen[df_sixteen<50], bins=50)
plt.title("Distribution of PM2.5 Readings  (Excluding >50)- 2016")
plt.xlabel("PM2.5 (micrograms/cubic meter)")
plt.ylabel("# Readings")
plt.show()

df_seventeen = df[df['Date'].str[-2:]=="17"]['Daily Mean PM2.5 Concentration']
print(df_seventeen)
print(df_seventeen.mean())
print(df_seventeen.var())
plt.hist(df_seventeen[df_seventeen<50], bins=50)
plt.title("Distribution of PM2.5 Readings  (Excluding >50)- 2017")
plt.xlabel("PM2.5 (micrograms/cubic meter)")
plt.ylabel("# Readings")

plt.show()
df_eighteen = df[df['Date'].str[-2:]=="18"]['Daily Mean PM2.5 Concentration']
print(df_eighteen)
print(df_eighteen.mean())
print(df_eighteen.var())
plt.hist(df_eighteen[df_eighteen<50], bins=50)
plt.title("Distribution of PM2.5 Readings  (Excluding >50)- 2018")
plt.xlabel("PM2.5 (micrograms/cubic meter)")
plt.ylabel("# Readings")
plt.show()

df_nineteen = df[df['Date'].str[-2:]=="19"]['Daily Mean PM2.5 Concentration']
print(df_nineteen)
print(df_nineteen.mean())
print(df_nineteen.var())
plt.hist(df_nineteen[df_nineteen<50], bins=50)
plt.title("Distribution of PM2.5 Readings  (Excluding >50)- 2019")
plt.xlabel("PM2.5 (micrograms/cubic meter)")
plt.ylabel("# Readings")
plt.show()


state_means = []
for state in states:
    df_state = df[df['STATE'] == state]
    state_mean = df_state['Daily Mean PM2.5 Concentration'].mean()
    state_means.append(state_mean)
    print(state)
    print(state_mean)

plt.hist(state_means, bins=25)
plt.title("Distribution of Mean PM2.5 Readings by State")
plt.xlabel("PM2.5 (micrograms/cubic meter)")
plt.ylabel("# Readings")
plt.show()

dates = []
date_means = []

for year in range(16,20):
    df_year = df[df['Date'].str[-2:] == str(year)]
    for month in range(1,13):
        if year<19 or month<10:
            date = str(month)+"/"+str(year)
            dates.append(date)
            if month<10:
                df_date = df_year[df_year['Date'].str[0:3]== "0"+str(month)+"/"]
            else:
                df_date = df_year[df_year['Date'].str[0:3]== str(month)+"/"]
            print(month)
            print(df_date)
            date_mean = df_date['Daily Mean PM2.5 Concentration'].mean()
            date_means.append(date_mean)
print(date_means)
plt.xticks(rotation=70)
plt.plot(dates,date_means)
plt.title("Mean PM2.5 by Month")
plt.xlabel("Month")
plt.ylabel("PM2.5 (micrograms/cubic meter)")
plt.show()


