import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.cluster import KMeans
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import os

def process_file(fname):
    df = pd.read_csv(fname)
    coords_list = []
    df['sampledate'] = pd.to_datetime(df['sampledate'])
    for _, group in df.groupby(['traceid']):
        group = group[group['speed'] != 0].sort_values('sampledate')
        if group['speed'].max() < 60:
            continue
        time_diff = group['sampledate'].diff().dt.total_seconds()
        heading_diff = group['heading'].diff()
        derivative = heading_diff / time_diff
        filtered_group = group[derivative < -30]
        coords_list.extend(filtered_group[['latitude', 'longitude']].values.tolist())
    return coords_list

def main():
    start = time.time()
    path = '/Users/neilteje/development/here-hackathon/here-norbel/data_chicago_hackathon_2024/probe_data/0/*.csv'
    roundabout_csv = '/Users/neilteje/development/here-hackathon/here-norbel/data_chicago_hackathon_2024/hamburg_extra_layers/hamburg_rounsabouts.csv'
    roundabouts = pd.read_csv(roundabout_csv)
    roundabout = roundabouts[roundabouts['bbox'] == 0]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_file, glob.glob(path)))
    coords_list = [coord for sublist in results for coord in sublist]
    coords = np.array(coords_list)
    print(coords)
    print("---- %s seconds ----" % (time.time() - start))

    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=420, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)

    filtered_data = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    filtered_data['cluster'] = cluster_labels

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(filtered_data['longitude'], filtered_data['latitude'], 
                          c=filtered_data['cluster'], cmap='viridis', 
                          marker='o', s=10, alpha=0.6)
    plt.scatter(roundabout['longitude'], roundabout['latitude'], 
                c='red', marker='x', s=50, label='Roundabouts')
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Clustered Probe Data with Roundabouts')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()