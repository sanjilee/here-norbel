import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import os
from kneed import KneeLocator
import requests
import random
from scipy.spatial import KDTree
import matplotlib.patches as patches

API_KEY = "61VlR0WusRIndHjGVKEPStGeHggSOmtJW1wKG6aTMqA"
# Create a directory to save the images
output_dir = "centroid_images"
os.makedirs(output_dir, exist_ok=True)


def process_file(fname):
    df = pd.read_csv(fname)
    coords_list = []
    for _, group in df.groupby(['traceid']):
        if group['speed'].max() < 60:
            continue
        heading_diff = group['heading'].diff()
        filtered_group = group[(heading_diff < -60) & (heading_diff > -90)]
        coords_list.extend(filtered_group[['latitude', 'longitude']].values.tolist())
    return coords_list

def filter_points_by_distance(coords, threshold=0.0005):
    tree = KDTree(coords)
    to_keep = []
    for i, point in enumerate(coords):
        dist, idx = tree.query(point, k=2)  
        if dist[1] <= threshold: 
            to_keep.append(True)
        else:
            to_keep.append(False)
    return coords[to_keep]

def plot_cluster_squares_and_centroids(ax, filtered_data, cluster_labels, square_size=0.009):
    half_size = square_size / 2 

    unique_clusters = np.unique(cluster_labels)
    
    for cluster in unique_clusters:
        cluster_data = filtered_data[filtered_data['cluster'] == cluster]
        centroid_lat = cluster_data['latitude'].mean()
        centroid_lon = cluster_data['longitude'].mean()
        lower_left_lon = centroid_lon - half_size
        lower_left_lat = centroid_lat - half_size
        square = patches.Rectangle((lower_left_lon, lower_left_lat), square_size, square_size, 
                                   linewidth=2, edgecolor='black', facecolor='none', label=f'Cluster {cluster} Square')
        ax.add_patch(square)
        ax.scatter(centroid_lon, centroid_lat, c='red', marker='x', s=100, label=f'Cluster {cluster} Centroid')

def main():
    for i in range (15):
        output_dir = f"centroid_images/{i}"
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        path = f'data_chicago_hackathon_2024/probe_data/{i}/*.csv'
        roundabout_csv = 'data_chicago_hackathon_2024/hamburg_extra_layers/hamburg_rounsabouts.csv'
        roundabouts = pd.read_csv(roundabout_csv)
        roundabout = roundabouts[roundabouts['bbox'] == i]
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(process_file, glob.glob(path)))
        
        coords_list = [coord for sublist in results for coord in sublist]
        coords = np.array(coords_list)

        # Filter points by distance threshold
        filtered_coords = filter_points_by_distance(coords, threshold=0.0005)
        
        print(filtered_coords)
        print(len(filtered_coords))

        distortions = []
        k_range = range(2, 9)  

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init = 10)
            kmeans.fit(filtered_coords)
            distortions.append(kmeans.inertia_)

        optimal_clusters = (KneeLocator(k_range, distortions, curve='convex', direction='decreasing')).elbow + 2

        kmeans = KMeans(n_clusters=optimal_clusters, random_state=420, n_init=10)
        cluster_labels = kmeans.fit_predict(filtered_coords)

        filtered_data = pd.DataFrame(filtered_coords, columns=['latitude', 'longitude'])
        filtered_data['cluster'] = cluster_labels
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(filtered_data['longitude'], filtered_data['latitude'], 
                            c=filtered_data['cluster'], cmap='viridis', 
                            marker='o', s=10, alpha=0.6)
        
        ax.scatter(roundabout['longitude'], roundabout['latitude'], 
                    c='red', marker='x', s=50, label='Roundabouts')
        centroids = kmeans.cluster_centers_
        for centroid in centroids:
            lat = centroid[1]
            lon = centroid[0]
            image_data = fetch_map_image(lon, lat, API_KEY)

            if image_data:
                # Save the image
                image_filename = os.path.join(output_dir, f"map_image_{lat},{lon}.png")
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_data)
                print(f"Saved image {i + 1} at coordinates ({lat}, {lon})")

        plot_cluster_squares_and_centroids(ax, filtered_data, cluster_labels, square_size=0.009)
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Clustered Probe Data {i} with Roundabouts using {optimal_clusters} clusters')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{i}_cluster_graphs.png")

def fetch_map_image(lat, lon, api_key):
    """Fetch map image from HERE API given latitude and longitude."""
    url = f"https://image.maps.hereapi.com/mia/v3/base/mc/center:{lat},{lon};zoom=15/256x256/png?apiKey={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to fetch image for coordinates ({lat}, {lon}). Status code: {response.status_code}")
        return None

if __name__ == "__main__":
    main()
