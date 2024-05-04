import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.spatial.transform import Rotation as R
from my_plot_funcs import plot_save_segmented_data_euler, plot_save_segmented_data_quaternions


FS_LABEL = 15
FS_TITLE = 15
FS_TICKS = 14
LW =  1.25
DPI_PNG = 600
PLOT_SHOW = False

def convert_quaternion_to_euler(data):
    # q = w,x,y,z
    norms = np.linalg.norm(data, axis=1)
    valid_indices = np.where(norms > 0)[0]
    normalized_quaternions = data[valid_indices] / norms[valid_indices][:, np.newaxis]

    rotation = R.from_quat(normalized_quaternions)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    euler_angles_unwrapped = np.unwrap(euler_angles * np.pi / 180) * 180 / np.pi

    euler_angles_matrix = np.full((data.shape[0], 3), np.nan)
    euler_angles_matrix[valid_indices] = euler_angles_unwrapped
    return np.array(euler_angles_matrix)


def pd_2_numpy_and_segment(pd_frame, dict):
    t = pd_frame['Time (Seconds)'].to_numpy()
    # Calculate time differences (dt) between consecutive samples
    # dt = np.diff(t)
    # # Calculate the mean value of dt, which is the average sampling period
    # mean_sampling_period = np.mean(dt)
    # print(f"mean: {1.0 / mean_sampling_period}")
    x = pd_frame['X.1'].to_numpy()
    y = pd_frame['Y.1'].to_numpy()
    z = pd_frame['Z.1'].to_numpy()
    q_w = pd_frame['W'].to_numpy()
    q_x = pd_frame['X'].to_numpy()
    q_y = pd_frame['Y'].to_numpy()
    q_z = pd_frame['Z'].to_numpy()
    list_segmented_stitches = [None] * 8
    i = 0
    for key, value in dict.items():
        t_start, t_end = value 
        mask = (t >= t_start) & (t <= t_end)
        tmp = np.zeros((t[mask].shape[0], 11))
        tmp[:,0] = t[mask]
        tmp[:,1] = x[mask]
        tmp[:,2] = y[mask]
        tmp[:,3] = z[mask]  
        tmp[:,4] = q_w[mask]
        tmp[:,5] = q_x[mask]
        tmp[:,6] = q_y[mask]
        tmp[:,7] = q_z[mask]
        tmp[:,8] = convert_quaternion_to_euler(tmp[:, 4:8])[:, 0]
        tmp[:,9] = convert_quaternion_to_euler(tmp[:, 4:8])[:, 1]
        tmp[:,10] = convert_quaternion_to_euler(tmp[:, 4:8])[:, 2]
        list_segmented_stitches[i] = tmp
        i = i + 1
    return list_segmented_stitches

def cluster_data_points(data, number_of_clusters):
    # Selecting a subset of data for clustering
    selected_data = data[:, 1:4]
    # Initialize KMeans with specified number of clusters
    kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=15, random_state=42)
    # Fitting KMeans on the selected data
    kmeans.fit(selected_data)
    # Getting the labels and centroids from KMeans
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    # Initialize list to hold information about each cluster
    cluster_info = []
    # Iterating over each cluster to gather information
    for i in range(kmeans.n_clusters):
        # Identifying indices of points belonging to the current cluster
        cluster_points_indices = np.where(labels == i)[0]
        # Selecting the actual points belonging to the current cluster
        cluster_points = selected_data[cluster_points_indices]
        # Getting the centroid of the current cluster
        cluster_centroid = centroids[i]
        # Adding information about the current cluster to the list
        cluster_info.append({
            'cluster_index': i,
            'cluster_points': cluster_points,
            'cluster_centroid': cluster_centroid,
            # Here 'cluster_indices_in_data' explicitly represents the indices of the points in the original dataset
            'cluster_indices_in_data': cluster_points_indices.tolist()
        })
    # Sorting the clusters based on the number of points they contain
    sorted_cluster_info = sorted(cluster_info, key=lambda x: len(x['cluster_points']), reverse=True)
    return sorted_cluster_info

def get_nan_percentatge(data):
    nan_count = np.count_nonzero(np.isnan(data))
    total_count = data.size
    return round((nan_count / total_count) * 100.0, 2)

def get_data_loss(subject_id, segment, data, tool_name):
    np_mat_raw = data[segment]
    tmp = np.zeros(2)
    tmp[0] = get_nan_percentatge(np_mat_raw)
    tmp[1] = np_mat_raw.shape[0]
    #np.save(f"Data_Loss/S_{subject_id}_{tool_name}_{segment+1}", tmp)
    return tmp[0]


def plot_clustered_data_3D_g2(subject_id, tool_name, segment_id, cluster_info, data, ax, cluster_color):
    for i in range(len(cluster_info)):
        #color = colors[i % len(colors)]
        color = cluster_color[i]
        indices = cluster_info[i]['cluster_indices_in_data']
        ax.scatter(data[indices, 1], data[indices, 2], data[indices, 3], color=color, label=f'Cluster {i}', s=1)
    ax.set_xlabel('[m]')
    ax.set_ylabel('[m]')
    ax.set_zlabel('[m]')
    ax.set_title(f'Stitch {segment_id+1}')

def cluster_data_points_DBSCAN(data, eps = 0.02, min_samples = 10 ):
    # Selecting a subset of data for clustering
    selected_data = data[:, 1:4]
    
    dbscan  = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Fitting DBSCAN on the selected data
    dbscan.fit(selected_data)

    #get the labels
    labels = dbscan.labels_
    print(labels)
    
    # Initialize list to hold information about each cluster
    cluster_info = []
    # Iterating over each cluster to gather information
    print('Number of cluster unique:', len(np.unique(labels)))
    for i in range(len(np.unique(labels))):
        # Identifying indices of points belonging to the current cluster
        cluster_points_indices = np.where(labels == i)[0]
        # Selecting the actual points belonging to the current cluster
        cluster_points = selected_data[cluster_points_indices]
        
        
        # Adding information about the current cluster to the list
        cluster_info.append({
            'cluster_index': i,
            'cluster_points': cluster_points.tolist(), #à voir si pose problème après!!! Avant on avait un ndarray
            #'cluster_means': cluster_means,
            # Here 'cluster_indices_in_data' explicitly represents the indices of the points in the original dataset
            'cluster_indices_in_data': cluster_points_indices.tolist()
        })
    # Sorting the clusters based on the number of points they contain
    sorted_cluster_info = sorted(cluster_info, key=lambda x: len(x['cluster_points']), reverse=True)
    return sorted_cluster_info

def get_list_segments_with_clusters_DBSCAN(subject_id, tool_name, data, clusters_info, clustering_algo, plot_enabled=True):
    list_segments_clusters = [None] * 8
    list_segments_clusters_info = [None] * 8
    
    if plot_enabled: fig = plt.figure(figsize=(15, 9))
    for s in range(8):
        if plot_enabled:
            ax = fig.add_subplot(2, 4, s+1, projection='3d')
            palette = sns.color_palette("deep", len(clusters_info['Subject_'+str(subject_id)][s]))
            plot_clustered_data_3D_g2(subject_id, tool_name, s, clusters_info['Subject_'+str(subject_id)][s], data[s], ax, palette)


        clusters = [None] * len(clusters_info['Subject_'+str(subject_id)][s])
        for c in range(len(clusters_info['Subject_'+str(subject_id)][s])):
            indices = clusters_info['Subject_'+str(subject_id)][s][c]['cluster_indices_in_data']
            clusters[c] = data[s][indices]
            # print(f"cluster:{c+1} - {len(clusters_info_tw[c]['cluster_points'])} - {list_clustered_tw[c].shape}")
        list_segments_clusters[s] = clusters #for each stitch gives segmented data corresponding to each cluster separated
        list_segments_clusters_info[s] = clusters_info['Subject_'+str(subject_id)][s] #for each stitch gives general cluster info
    
    if plot_enabled:
        fig.suptitle(f'3-D Scatter Plot of Clustered Data using DBSCAN ({tool_name})')
        plt.tight_layout()

        directory = f"{clustering_algo}/Plots/Clustering_step1/S_{subject_id}/"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}{tool_name}.png", dpi=DPI_PNG)  
        
        if (PLOT_SHOW):
            plt.show()
        plt.close(fig)

    return list_segments_clusters, list_segments_clusters_info

#Manual segmentation functions
def select_points_manually2(df_manual, df_data, delta_video_tracking=[0,0,0]):
    df_manual['Nbr_selection'] = 1 #video 1
    df_manual.loc[df_manual['Start_event_s'] > 962, ['Nbr_selection']] = 2 #video 2
    df_manual.loc[df_manual['Start_event_s'] > 1924, ['Nbr_selection']] = 3 #video 3

    nbr_video = len(df_manual['Nbr_selection'].unique())
    df_manual['Start_event_s2'] = df_manual['Start_event_s']
    df_manual['End_event_s2'] = df_manual['End_event_s']
    for i in range(nbr_video):
        df_manual.loc[df_manual['Nbr_selection'] == (i+1), ['Start_event_s2', 'End_event_s2']] += delta_video_tracking[i]
        
    intervals = list(zip(df_manual['Start_event_s2'], df_manual['End_event_s2']))
    
    condition = "|".join([f"(`Time (Seconds)` > {debut} & `Time (Seconds)` < {fin})" for (debut, fin) in intervals])
    
    df_filtered = df_data.query(f"not ({condition})")
    return df_filtered

def adjust_time_on_nbr_videos(df):
    if (len(df['Video_nbr'].unique())>1):
        df['Start_event'] = df['Start_event'].astype(float)
        df['End_event'] = df['End_event'].astype(float)

        df.loc[df['Video_nbr'] == 2, ['Start_event', 'End_event']] += 16.02
        df.loc[df['Video_nbr'] == 3, ['Start_event', 'End_event']] += 32.04
        return df.round(3)
    else: return df

def convert_to_seconds2(time):
    minutes = int(time)
    decimal = round(time-minutes,3)
    decimal_100 = decimal*100
    seconds = int(decimal_100)
    ms = decimal_100 - seconds
    total_seconds = minutes * 60 + seconds + ms
    return total_seconds