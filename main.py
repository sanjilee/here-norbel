import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

path = '/Users/sanjilee/development/here-norbel/data_chicago_hackathon_2024/probe_data/2/*.csv'
roundabout_csv = '/Users/sanjilee/development/here-norbel/data_chicago_hackathon_2024/hamburg_extra_layers/hamburg_rounsabouts.csv'
roundabouts = pd.read_csv(roundabout_csv)
roundabout_11 = roundabouts[roundabouts['bbox'] == 2]
for fname in glob.glob(path):
    df = pd.read_csv(fname)
    filtered_data = pd.DataFrame()
    grouped = df.groupby(['traceid'])
    for name, group in grouped:
        group['sampledate'] = pd.to_datetime(group['sampledate'], format="%Y-%m-%d %H:%M:%S")
        group = group.sort_values('sampledate')
        time_diff = group['sampledate'].diff().dt.total_seconds()
        heading_diff = group['heading'].diff()
        derivative = heading_diff / time_diff
        group = group[derivative > 20]
        filtered_data = pd.concat([filtered_data, group])
    traceid_max = grouped['speed'].max()
    useful = traceid_max[traceid_max >= 60].index
    filtered_data = filtered_data[filtered_data['traceid'].isin(useful)]
print(filtered_data)
plt.figure(figsize = (10, 6))
plt.scatter(filtered_data['longitude'], filtered_data['latitude'], c = 'blue', marker = 'o', s = 10)
plt.scatter(roundabout_11['longitude'], roundabout_11['latitude'], c = 'red', marker = 'x', s = 20)
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.grid(True)
plt.show()

