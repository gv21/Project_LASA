#step 1
import pandas as pd 
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
from scipy.signal import savgol_filter
from time_stitches import *
from scipy.spatial.transform import Rotation as R
from features import *
from scipy.spatial.transform import Rotation as R
from my_plot_funcs import *
TOOLS = ["The Tweezers", "Needle Holder"]
CLUSTER_COLORS = ["green", "blue", "red"]

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

"""def convert_quaternion_to_euler(data):
    # q = w,x,y,z
    norms = np.linalg.norm(data, axis=1)
    valid_indices = norms > 0  
    normalized_quaternions = data[valid_indices] / norms[valid_indices, np.newaxis]
    euler_angles_matrix = np.full((data.shape[0], 3), np.nan)
    for i, quat in zip(np.where(valid_indices)[0], normalized_quaternions):
        rotation = R.from_quat(quat)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        euler_angles_unwrapped = np.unwrap(euler_angles * np.pi / 180) * 180 / np.pi  
        euler_angles_matrix[i] = euler_angles_unwrapped
    return np.array(euler_angles_matrix)"""

#Gaëlle's version
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
#end of Gaëlle's version

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

def get_list_segments_with_clusters(subject_id, tool_name, number_of_clusters, data, plot_enabled):
    list_segments_clusters = [None] * 8
    list_segments_clusters_info = [None] * 8
    for s in range(8): 
        clusters_info = cluster_data_points(data[s], number_of_clusters)
        if(plot_enabled):
            plot_clustered_data_3D(subject_id, tool_name, s, clusters_info, data[s]) 
        clusters = [None] * number_of_clusters
        for c in range(number_of_clusters):
            indices = clusters_info[c]['cluster_indices_in_data']
            clusters[c] = data[s][indices]
            # print(f"cluster:{c+1} - {len(clusters_info_tw[c]['cluster_points'])} - {list_clustered_tw[c].shape}")
        list_segments_clusters[s] = clusters
        list_segments_clusters_info[s] = clusters_info

    return list_segments_clusters, list_segments_clusters_info

def plot_segmented_signals(subject_id, number_of_clusters, data_tw, data_nh, data_tw_raw, data_nh_raw):
    for s in range(8): 
        clusters_info_tw = cluster_data_points(data_tw[s], number_of_clusters)
        list_clusters_tw = [None] * number_of_clusters
        for c in range(number_of_clusters):
            indices = clusters_info_tw[c]['cluster_indices_in_data']
            list_clusters_tw[c] = data_tw[s][indices]

        clusters_info_nh = cluster_data_points(data_nh[s], number_of_clusters)
        list_clusters_nh = [None] * number_of_clusters
        for c in range(number_of_clusters):
            indices = clusters_info_nh[c]['cluster_indices_in_data']
            list_clusters_nh[c] = data_nh[s][indices]

        data_loss_tw = get_data_loss(subject_id, s, data_tw_raw, "tweezers")
        data_loss_nh = get_data_loss(subject_id, s, data_nh_raw, "needle holder")
        plot_save_segmented_data_quaternions(subject_id, s, list_clusters_tw, list_clusters_nh, data_loss_tw, data_loss_nh)  
        plot_save_segmented_data_euler(subject_id, s, list_clusters_tw, list_clusters_nh, data_loss_tw, data_loss_nh)  

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

def derivative_with_time(t, data):
    # Calculate dt, the difference in time between each point
    dt = np.diff(t, axis=0)
    # Calculate the derivative of the position data
    derivative = np.diff(data[:, 1:], axis=0) / dt[:, None]
    # Prepend a zero row to make the derivative array match the original in size
    derivative_padded = np.vstack([derivative[0], derivative])
    # Reinsert the time column
    derivative_with_time = np.hstack([data[:, 0].reshape(-1, 1), derivative_padded])
    return derivative_with_time

def get_derivatives(t, data):
    velocity = derivative_with_time(t, data)
    acceleration = derivative_with_time(t, velocity)
    jerk = derivative_with_time(t, acceleration)
    return (velocity, acceleration, jerk)
    
def get_list_derivatives_for_segments(data):
    list_np_segments_d1_X = [None] * 8 
    list_np_segments_d2_X = [None] * 8 
    list_np_segments_d3_X = [None] * 8 

    for s in range(8):
        np_d1_X = np.empty((data[s].shape[0], data[s].shape[1]))
        np_d2_X = np.empty((data[s].shape[0], data[s].shape[1]))
        np_d3_X = np.empty((data[s].shape[0], data[s].shape[1]))

        np_d1_X[:, 0] = data[s][:, 0]
        np_d1_X[:, 1:4] = get_derivatives(data[s][:, 0], data[s][:, 1:4])[0]
        np_d1_X[:, 4:8] = data[s][:, 4:8]
        np_d1_X[:, 8:] = get_derivatives(data[s][:, 0], data[s][:, 8:])[0]

        np_d2_X[:, 0] = data[s][:, 0]
        np_d2_X[:, 1:4] = get_derivatives(data[s][:, 0], data[s][:, 1:4])[1]
        np_d2_X[:, 4:8] = data[s][:, 4:8]
        np_d2_X[:, 8:] = get_derivatives(data[s][:, 0], data[s][:, 8:])[1]

        np_d3_X[:, 0] = data[s][:, 0]
        np_d3_X[:, 1:4] = get_derivatives(data[s][:, 0], data[s][:, 1:4])[2]
        np_d3_X[:, 4:8] = data[s][:, 4:8]
        np_d3_X[:, 8:] = get_derivatives(data[s][:, 0], data[s][:, 8:])[2]

        list_np_segments_d1_X[s] = np_d1_X
        list_np_segments_d2_X[s] = np_d2_X
        list_np_segments_d3_X[s] = np_d3_X

    return list_np_segments_d1_X, list_np_segments_d2_X, list_np_segments_d3_X

def get_features(subject_id, derivatives_segments_tw, derivatives_segments_nh, 
                 list_clusters_info_tw, list_clusters_info_nh, save =True):
    # just for the task cluster

    directory = f"Kmeans/OT_Features/S_{subject_id}/"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
 
    NUM_OF_METRICS = 8
    NUMBER_OF_SEGMENTS = 8 
    NUMBER_OF_TOOLS = 2
    NUMBER_OF_CLUSTERS = 3

    data_metrics = np.zeros((NUMBER_OF_SEGMENTS, NUMBER_OF_CLUSTERS, NUM_OF_METRICS, NUMBER_OF_TOOLS))

    for s in range(NUMBER_OF_SEGMENTS):
        unclustered_t_tw = derivatives_segments_tw[0][s][:, 0]
        unclustered_t_nh = derivatives_segments_nh[0][s][:, 0]

        unclustered_position_tw = derivatives_segments_tw[0][s][:, 1:4]
        unclustered_position_nh = derivatives_segments_nh[0][s][:, 1:4]

        unclustered_d1_position_tw = derivatives_segments_tw[1][s][:, 1:4]
        unclustered_d1_position_nh = derivatives_segments_nh[1][s][:, 1:4]
    
        unclustered_d2_position_tw = derivatives_segments_tw[2][s][:, 1:4]
        unclustered_d2_position_nh = derivatives_segments_nh[2][s][:, 1:4]
    
        unclustered_d3_position_tw = derivatives_segments_tw[3][s][:, 1:4]
        unclustered_d3_position_nh = derivatives_segments_nh[3][s][:, 1:4]

        unclustered_quaternion_tw = derivatives_segments_tw[0][s][:, 4:8]
        unclustered_quaternion_nh = derivatives_segments_nh[0][s][:, 4:8] 
        
        unclustered_euler_tw = derivatives_segments_tw[0][s][:, 8:]
        unclustered_euler_nh = derivatives_segments_nh[0][s][:, 8:] 

        for c in range(NUMBER_OF_CLUSTERS):
            #for c in range(number_of_clusters): #Two loops, one not used? 
            indices_tw = list_clusters_info_tw[s][c]['cluster_indices_in_data']
            indices_nh = list_clusters_info_nh[s][c]['cluster_indices_in_data']

            t_tw = unclustered_t_tw[indices_tw]
            t_nh = unclustered_t_nh[indices_nh]
            position_tw = unclustered_position_tw[indices_tw]
            position_nh = unclustered_position_nh[indices_nh]
            quaternion_tw = unclustered_quaternion_tw[indices_tw]
            quaternion_nh = unclustered_quaternion_nh[indices_nh]
            euler_tw = unclustered_euler_tw[indices_tw]
            euler_nh = unclustered_euler_nh[indices_nh]

            d1_position_tw = unclustered_d1_position_tw[indices_tw]
            d1_position_nh = unclustered_d1_position_nh[indices_nh]
            d2_position_tw = unclustered_d2_position_tw[indices_tw]
            d2_position_nh = unclustered_d2_position_nh[indices_nh]
            d3_position_tw = unclustered_d3_position_tw[indices_tw]
            d3_position_nh = unclustered_d3_position_nh[indices_nh]

            # path length
            path_length_tw = get_path_length(position_tw)
            path_length_nh = get_path_length(position_nh)

            # total rotation
            total_rotation_tw_x = get_total_rotation(euler_tw)[0]
            total_rotation_nh_x = get_total_rotation(euler_nh)[0]

            total_rotation_tw_y = get_total_rotation(euler_tw)[1]
            total_rotation_nh_y = get_total_rotation(euler_nh)[1]
       
            total_rotation_tw_z = get_total_rotation(euler_tw)[2]
            total_rotation_nh_z = get_total_rotation(euler_nh)[2]    

            # economy of volume
            EoV_tw = get_economy_of_volume(position_tw, path_length_tw)
            EoV_nh = get_economy_of_volume(position_nh, path_length_nh)

            # jerk
            jerk_tw = get_jerk(t_tw, d1_position_tw, d2_position_tw, d3_position_tw)
            jerk_nh = get_jerk(t_tw, d1_position_nh, d2_position_nh, d3_position_nh)

            # mean and std of d1 and d2 position
            norm_d1_position_tw_mean = get_mean_std_velocity_norm(d1_position_tw)[0]
            norm_d1_position_nh_mean = get_mean_std_velocity_norm(d1_position_nh)[0]

            data_metrics[s, c, 0, 0] = t_tw.shape[0] * (1.0 / 120.0)
            data_metrics[s, c, 0, 1] = t_nh.shape[0] * (1.0 / 120.0)

            data_metrics[s, c, 1, 0] = path_length_tw
            data_metrics[s, c, 1, 1] = path_length_nh

            data_metrics[s, c, 2, 0] = total_rotation_tw_x
            data_metrics[s, c, 2, 1] = total_rotation_nh_x

            data_metrics[s, c, 3, 0] = total_rotation_tw_y
            data_metrics[s, c, 3, 1] = total_rotation_nh_y

            data_metrics[s, c, 4, 0] = total_rotation_tw_z
            data_metrics[s, c, 4, 1] = total_rotation_nh_z

            data_metrics[s, c, 5, 0] = jerk_tw
            data_metrics[s, c, 5, 1] = jerk_nh

            data_metrics[s, c, 6, 0] = EoV_tw
            data_metrics[s, c, 6, 1] = EoV_nh

            data_metrics[s, c, 7, 0] = norm_d1_position_tw_mean
            data_metrics[s, c, 7, 1] = norm_d1_position_nh_mean

            if save: np.save(f"{directory}/ot_metrics.npy", data_metrics)
            else:
                print('Data metrics are: ') 
                print(data_metrics)

def get_np_mat_specific_cluste(list_segments_with_clusters, cluster_id):
    list_clusters = [None] * 8
    for segment in range(8):
        tmp = np.zeros((list_segments_with_clusters[segment][cluster_id].shape[0], list_segments_with_clusters[segment][cluster_id].shape[1])) 
        tmp = list_segments_with_clusters[segment][cluster_id]
        list_clusters[segment] = tmp
    return list_clusters

number_of_clusters = 3
target_subjects = [1]

i = 0
for subject in target_subjects:
    
    dict_segment_time = segments_time[i]
    
    tweezers_raw = pd.read_csv(f'Data/Sync_data/S_{subject}_TW_raw.csv')
    tweezers_rec = pd.read_csv(f'Data/Sync_data/S_{subject}_TW_reconstructed.csv')
    needle_holder_raw = pd.read_csv(f'Data/Sync_data/S_{subject}_NH_raw.csv')
    needle_holder_rec = pd.read_csv(f'Data/Sync_data/S_{subject}_NH_reconstructed.csv')

    print (1)
    list_np_segmented_tw_raw = pd_2_numpy_and_segment(tweezers_raw, dict_segment_time)
    list_np_segmented_tw_rec = pd_2_numpy_and_segment(tweezers_rec, dict_segment_time)
    list_np_segmented_nh_raw = pd_2_numpy_and_segment(needle_holder_raw, dict_segment_time)
    list_np_segmented_nh_rec = pd_2_numpy_and_segment(needle_holder_rec, dict_segment_time)

    print(2)
    plot_segmented_signals(subject, number_of_clusters, list_np_segmented_tw_rec, list_np_segmented_nh_rec, list_np_segmented_tw_raw, list_np_segmented_nh_raw)

    list_segments_with_clusters_tw, list_clusters_info_tw = get_list_segments_with_clusters(subject, "tweezers", number_of_clusters, list_np_segmented_tw_rec, True) 
    list_segments_with_clusters_nh, list_clusters_info_nh = get_list_segments_with_clusters(subject, "needle holder", number_of_clusters, list_np_segmented_nh_rec, True) 

    #duplica
    #plot_segmented_signals(subject, number_of_clusters, list_np_segmented_tw_rec, list_np_segmented_nh_rec, list_np_segmented_tw_raw, list_np_segmented_nh_raw)
    
    print(3)
    studied_cluster = 0 #we need to study all clusters
    tmp_tw = get_np_mat_specific_cluste(list_segments_with_clusters_tw, studied_cluster)
    tmp_nh = get_np_mat_specific_cluste(list_segments_with_clusters_nh, studied_cluster)
    plot_a_cluster_in_signal_euler(subject, tmp_tw, tmp_nh)

    print(4)
    list_np_segments_d1_X_tw, list_np_segments_d2_X_tw, list_np_segments_d3_X_tw = get_list_derivatives_for_segments(list_np_segmented_tw_rec)
    list_np_segments_d1_X_nh, list_np_segments_d2_X_nh, list_np_segments_d3_X_nh = get_list_derivatives_for_segments(list_np_segmented_nh_rec)

    derivatives_segments_tw = [list_np_segmented_tw_rec, list_np_segments_d1_X_tw, list_np_segments_d2_X_tw, list_np_segments_d3_X_tw]
    derivatives_segments_nh = [list_np_segmented_nh_rec, list_np_segments_d1_X_nh, list_np_segments_d2_X_nh, list_np_segments_d3_X_nh]

    get_features(subject, derivatives_segments_tw, derivatives_segments_nh, list_clusters_info_tw, list_clusters_info_nh)
    i = i + 1


