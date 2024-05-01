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
from features import get_path_length
from scipy.spatial.transform import Rotation as R
from my_plot_funcs import * #only use plot_a_cluster_in_signal_euler and const defined at the beginning of the file
TOOLS = ["The Tweezers", "Needle Holder"]
PLOT_SHOW = False
#CLUSTER_COLORS = ["green", "blue", "red"]
from functions import pd_2_numpy_and_segment
import seaborn as sns

from sklearn.cluster import DBSCAN
import json
import time


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

def plot_save_segmented_data_quaternions_g(subject_id, segment_id, data_tw, data_nh, data_loss_tw, data_loss_nh, clusterinf_algo, save = True):
    directory = f"{clusterinf_algo}/Plots/Signals/Quaternion Representation/"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    fig, axs = plt.subplots(7, 2, sharex=True, figsize=(12, 10)) 
    # Plot data (tweezers) considering clustering
    axs[0, 0].set_title(f'Quat {TOOLS[0]} (Sub{subject_id}, Seg{segment_id+1}) - Loss {data_loss_tw}[%]', fontsize=FS_TITLE) 
    for i in range(7):
        palette_tw = sns.color_palette("deep", len(data_tw))
        #palette_tw = ['orange', 'blue', 'red', 'green', 'yellow', 'pink', 'purple']
        for c in range (len(data_tw)):
            axs[i,0].scatter(data_tw[c][:, 0], data_tw[c][:, i+1], facecolor=palette_tw[c], s=1)
        """axs[i, 0].scatter(data_tw[0][:, 0], data_tw[0][:, i+1], facecolor=CLUSTER_COLORS[0], s=1) 
        axs[i, 0].scatter(data_tw[1][:, 0], data_tw[1][:, i+1], facecolor=CLUSTER_COLORS[1], s=1) 
        axs[i, 0].scatter(data_tw[2][:, 0], data_tw[2][:, i+1], facecolor=CLUSTER_COLORS[2], s=1)""" 
        axs[i, 0].tick_params(axis='x', labelsize=FS_TICKS)  
        axs[i, 0].tick_params(axis='y', labelsize=FS_TICKS) 
    # Plot data (needle holder) considering clustering
    axs[0, 1].set_title(f'Quat {TOOLS[1]} (Sub{subject_id}, Seg{segment_id+1}) - Loss {data_loss_nh}[%]', fontsize=FS_TITLE) 
    for i in range(7):
        palette_nh = sns.color_palette("deep", len(data_nh))
        for c in range (len(data_nh)):
            axs[i,1].scatter(data_nh[c][:, 0], data_nh[c][:, i+1], facecolor=palette_nh[c], s=1)
        """axs[i, 1].scatter(data_nh[0][:, 0], data_nh[0][:, i+1], facecolor=CLUSTER_COLORS[0], s=1) 
        axs[i, 1].scatter(data_nh[1][:, 0], data_nh[1][:, i+1], facecolor=CLUSTER_COLORS[1], s=1) 
        axs[i, 1].scatter(data_nh[2][:, 0], data_nh[2][:, i+1], facecolor=CLUSTER_COLORS[2], s=1)"""        
        axs[i, 1].tick_params(axis='x', labelsize=FS_TICKS)  
        axs[i, 1].tick_params(axis='y', labelsize=FS_TICKS)   
    # Set labels
    axs[0, 0].set_ylabel(f'[m]', fontsize=FS_LABEL)
    axs[1, 0].set_ylabel(f'[m]', fontsize=FS_LABEL)
    axs[2, 0].set_ylabel(f'[m]', fontsize=FS_LABEL)
    axs[3, 0].set_ylabel(f'q_w', fontsize=FS_LABEL)
    axs[4, 0].set_ylabel(f'q_x', fontsize=FS_LABEL)
    axs[5, 0].set_ylabel(f'q_y', fontsize=FS_LABEL)
    axs[6, 0].set_ylabel(f'q_z', fontsize=FS_LABEL)
    axs[6, 0].set_xlabel(f'Time [s]', fontsize=FS_LABEL)
    axs[6, 1].set_xlabel(f'Time [s]', fontsize=FS_LABEL)
    plt.tight_layout()
    if save: plt.savefig(f"{directory}/S_{subject_id}_signals_{segment_id+1}.png", dpi=DPI_PNG)  
    if (PLOT_SHOW):
        plt.show()
    if save: plt.close(fig)

def plot_save_segmented_data_euler_g(subject_id, segment_id, data_tw, data_nh, data_loss_tw, data_loss_nh, clustering_algo, save =True):
    directory = f"{clustering_algo}/Plots/Signals/Euler Representation/"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    fig, axs = plt.subplots(6, 2, sharex=True, figsize=(12, 10)) 
    # Plot data (tweezers) considering clustering
    axs[0, 0].set_title(f'Euler {TOOLS[0]} (Sub{subject_id}, Seg{segment_id+1}) - Loss {data_loss_tw}[%]', fontsize=FS_TITLE) 
    for i in range(6):
        palette_tw = sns.color_palette("deep", len(data_tw))
        for c in range (len(data_tw)):
            axs[i,0].scatter(data_tw[c][:, 0], data_tw[c][:, i+4], facecolor=palette_tw[c], s=1)
        """axs[i, 0].scatter(data_tw[0][:, 0], data_tw[0][:, i+1+4], facecolor=CLUSTER_COLORS[0], s=1) 
        axs[i, 0].scatter(data_tw[1][:, 0], data_tw[1][:, i+1+4], facecolor=CLUSTER_COLORS[1], s=1) 
        axs[i, 0].scatter(data_tw[2][:, 0], data_tw[2][:, i+1+4], facecolor=CLUSTER_COLORS[2], s=1)"""
        axs[i, 0].tick_params(axis='x', labelsize=FS_TICKS)  
        axs[i, 0].tick_params(axis='y', labelsize=FS_TICKS) 
    # Plot data (needle holder) considering clustering
    axs[0, 1].set_title(f'Euler {TOOLS[1]} (Sub{subject_id}, Seg{segment_id+1}) - Loss {data_loss_nh} [%]', fontsize=FS_TITLE) 
    for i in range(6):
        palette_nh = sns.color_palette("deep", len(data_nh))
        for c in range (len(data_nh)):
            axs[i,1].scatter(data_nh[c][:, 0], data_nh[c][:, i+1+4], facecolor=palette_nh[c], s=1)
        """axs[i, 1].scatter(data_nh[0][:, 0], data_nh[0][:, i+1+4], facecolor=CLUSTER_COLORS[0], s=1) 
        axs[i, 1].scatter(data_nh[1][:, 0], data_nh[1][:, i+1+4], facecolor=CLUSTER_COLORS[1], s=1) 
        axs[i, 1].scatter(data_nh[2][:, 0], data_nh[2][:, i+1+4], facecolor=CLUSTER_COLORS[2], s=1)"""       
        axs[i, 1].tick_params(axis='x', labelsize=FS_TICKS)  
        axs[i, 1].tick_params(axis='y', labelsize=FS_TICKS)     
    # Set labels
    axs[0, 0].set_ylabel(f'$x\,[m]$', fontsize=FS_LABEL)
    axs[1, 0].set_ylabel(f'$y\,[m]$', fontsize=FS_LABEL) 
    axs[2, 0].set_ylabel(f'$z\,[m]$', fontsize=FS_LABEL)
    axs[3, 0].set_ylabel(f'$r_x [deg]$', fontsize=FS_LABEL)
    axs[4, 0].set_ylabel(f'$r_y [deg]$', fontsize=FS_LABEL)
    axs[5, 0].set_ylabel(f'$r_z [deg]$', fontsize=FS_LABEL)
    axs[5, 0].set_xlabel(f'Time [s]', fontsize=FS_LABEL)
    axs[5, 1].set_xlabel(f'Time [s]', fontsize=FS_LABEL)
    plt.tight_layout()
    if save: plt.savefig(f"{directory}/S_{subject_id}_signals_{segment_id+1}.png", dpi=DPI_PNG)  
    if (PLOT_SHOW):
        plt.show()
    if save: plt.close(fig)


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

def plot_segmented_signals_DBSCAN(subject_id, data_tw, data_nh, data_tw_raw, data_nh_raw, nh_cluster, tw_cluster, save =True):
    clustering_algo = 'DBSCAN'
    for s in range(8): #can go until 8, I put 2 so that I don't have all the plots
        #clusters_info_tw = cluster_data_points_DBSCAN(data_tw[s], eps = 20, min_samples=200)
        print(s)
        print('cluster tw')
        list_clusters_tw = [None] * len(tw_cluster['Subject_'+str(subject_id)][s])#to change when run on several subjects
        for c in range(len(tw_cluster['Subject_'+str(subject_id)][s])):
            indices = tw_cluster['Subject_'+str(subject_id)][s][c]['cluster_indices_in_data']
            list_clusters_tw[c] = data_tw[s][indices]

        print('cluster nh')
        #clusters_info_nh = cluster_data_points_DBSCAN(data_nh[s], eps = 20, min_samples=200)
        list_clusters_nh = [None] * len(nh_cluster['Subject_'+str(subject_id)][s])
        for c in range(len(nh_cluster['Subject_'+str(subject_id)][s])):
            indices = nh_cluster['Subject_'+str(subject_id)][s][c]['cluster_indices_in_data']
            list_clusters_nh[c] = data_nh[s][indices]

        print('data loss')
        #Plot function only consider 3 clusters, gives errors if less
        data_loss_tw = get_data_loss(subject_id, s, data_tw_raw, "tweezers", save = save)
        data_loss_nh = get_data_loss(subject_id, s, data_nh_raw, "needle holder", save = save)
        #Plot function only consider 3 clusters, gives errors if less
        print('plot save quaternions')
        plot_save_segmented_data_quaternions_g(subject_id, s, list_clusters_tw, list_clusters_nh, data_loss_tw, data_loss_nh, clustering_algo, save=save) 
        print('plot_save euler') 
        plot_save_segmented_data_euler_g(subject_id, s, list_clusters_tw, list_clusters_nh, data_loss_tw, data_loss_nh,clustering_algo, save =save)

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
                 list_clusters_info_tw, list_clusters_info_nh, clustering_algo, save =True):
    # just for the task cluster

    if save:
        directory = f"{clustering_algo}/OT_Features/S_{subject_id}/"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
 
    NUM_OF_METRICS = 8
    NUMBER_OF_SEGMENTS = 8 
    NUMBER_OF_TOOLS = 2

    #find the maximum number of cluster
    NUMBER_OF_CLUSTERS = 1 #there is at least one cluster
    for s in range (NUMBER_OF_SEGMENTS):
        nbr_clust_tw = len(list_clusters_info_tw[s])
        nbr_clust_nh = len(list_clusters_info_nh[s])
        max_ = max(nbr_clust_tw, nbr_clust_nh)
        if max_>NUMBER_OF_CLUSTERS: NUMBER_OF_CLUSTERS=max_

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
            indices_tw = []
            if (c< len(list_clusters_info_tw[s])) : 
                indices_tw = list_clusters_info_tw[s][c]['cluster_indices_in_data']
            
            indices_nh = []
            if (c< len(list_clusters_info_nh[s])):
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
            total_rotation_tw_x = get_total_rotation_g(euler_tw)[0]
            total_rotation_nh_x = get_total_rotation_g(euler_nh)[0]

            total_rotation_tw_y = get_total_rotation_g(euler_tw)[1]
            total_rotation_nh_y = get_total_rotation_g(euler_nh)[1]
       
            total_rotation_tw_z = get_total_rotation_g(euler_tw)[2]
            total_rotation_nh_z = get_total_rotation_g(euler_nh)[2]    

            # economy of volume
            EoV_tw = get_economy_of_volume_g(position_tw, path_length_tw)
            EoV_nh = get_economy_of_volume_g(position_nh, path_length_nh)

            # jerk
            jerk_tw = get_jerk_g(t_tw, d1_position_tw, d2_position_tw, d3_position_tw)
            jerk_nh = get_jerk_g(t_tw, d1_position_nh, d2_position_nh, d3_position_nh)

            # mean and std of d1 and d2 position
            norm_d1_position_tw_mean = get_mean_std_velocity_norm_g(d1_position_tw)[0]
            norm_d1_position_nh_mean = get_mean_std_velocity_norm_g(d1_position_nh)[0]

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

def get_nan_percentatge(data):
    nan_count = np.count_nonzero(np.isnan(data))
    total_count = data.size
    return round((nan_count / total_count) * 100.0, 2)

def get_data_loss(subject_id, segment, data, tool_name, save =True):
    np_mat_raw = data[segment]
    tmp = np.zeros(2)
    tmp[0] = get_nan_percentatge(np_mat_raw)
    tmp[1] = np_mat_raw.shape[0]
    if save: np.save(f"DBSCAN/Data_Loss/S_{subject_id}_{tool_name}_{segment+1}", tmp)
    return tmp[0]



target_subjects = [1,19,23,7,24,26]

#run only if the clustering has not bees stored yet
perform_clustering = False #if clustering has already be done, ie clustering stored in jason files then put False
start_time = time.time()
if perform_clustering:
    i =0
    nh_clusters_all_sub = {}
    tw_clusters_all_sub = {}
    for subject in target_subjects:
        print('Subject:', subject)
        dict_segment_time = segments_time[i]
        
        tweezers_raw = pd.read_csv(f'Data/Sync_data/S_{subject}_TW_raw.csv')
        tweezers_rec = pd.read_csv(f'Data/Sync_data/S_{subject}_TW_reconstructed.csv')
        needle_holder_raw = pd.read_csv(f'Data/Sync_data/S_{subject}_NH_raw.csv')
        needle_holder_rec = pd.read_csv(f'Data/Sync_data/S_{subject}_NH_reconstructed.csv')

        list_np_segmented_tw_raw = pd_2_numpy_and_segment(tweezers_raw, dict_segment_time)
        list_np_segmented_tw_rec = pd_2_numpy_and_segment(tweezers_rec, dict_segment_time)
        list_np_segmented_nh_raw = pd_2_numpy_and_segment(needle_holder_raw, dict_segment_time)
        list_np_segmented_nh_rec = pd_2_numpy_and_segment(needle_holder_rec, dict_segment_time)

        print('Run for TW')
        tw_sub_cluster = []
        for n in range (8):
            print('Clustering stitch ', n)
            tw_ordered_info_cluster = cluster_data_points_DBSCAN(list_np_segmented_tw_rec[n], eps = 0.02, min_samples=200)
            tw_sub_cluster.append(tw_ordered_info_cluster)
        #sub_cluster[s][c]{dict} s= stitches, c=cluster
        tw_clusters_all_sub['Subject_'+str(subject)] = tw_sub_cluster

        print('Run for NH')
        sub_cluster = []
        for m in range (8):
            print(m)
            ordered_info_cluster = cluster_data_points_DBSCAN(list_np_segmented_nh_rec[m], eps = 0.02, min_samples=200)
            sub_cluster.append(ordered_info_cluster)
        #sub_cluster[s][c]{dict} s= stitches, c=cluster
        nh_clusters_all_sub['Subject_'+str(subject)] = sub_cluster
        i= i+1
    
    with open('DBSCAN_nh_clustering.json', 'w') as f:
        json.dump(nh_clusters_all_sub, f)

    with open('DBSCAN_tw_clustering.json', 'w') as f:
        json.dump(tw_clusters_all_sub, f)
end_time = time.time()
duration_clustering = end_time - start_time

print("Duration time for clustering:", duration_clustering, "secondes") 
print("Duration time for clustering:", duration_clustering/3600, "hours") 


#Load stored clusters
with open('DBSCAN_nh_clustering.json', 'r') as f:
    nh_clusters_all_sub = json.load(f)

with open('DBSCAN_tw_clustering.json', 'r') as f:
    tw_clusters_all_sub = json.load(f)


start_time_2nd = time.time()
save = True #if we want to save plots and metrics
i =0
clustering_algo = 'DBSCAN'
for subject in target_subjects:
    print('Subject:', subject)
    dict_segment_time = segments_time[i]
        
    tweezers_raw = pd.read_csv(f'Data/Sync_data/S_{subject}_TW_raw.csv')
    tweezers_rec = pd.read_csv(f'Data/Sync_data/S_{subject}_TW_reconstructed.csv')
    needle_holder_raw = pd.read_csv(f'Data/Sync_data/S_{subject}_NH_raw.csv')
    needle_holder_rec = pd.read_csv(f'Data/Sync_data/S_{subject}_NH_reconstructed.csv')

    list_np_segmented_tw_raw = pd_2_numpy_and_segment(tweezers_raw, dict_segment_time)
    list_np_segmented_tw_rec = pd_2_numpy_and_segment(tweezers_rec, dict_segment_time)
    list_np_segmented_nh_raw = pd_2_numpy_and_segment(needle_holder_raw, dict_segment_time)
    list_np_segmented_nh_rec = pd_2_numpy_and_segment(needle_holder_rec, dict_segment_time)

    print('Plot segmented signals DBSCAN')

    plot_segmented_signals_DBSCAN(subject, list_np_segmented_tw_rec, list_np_segmented_nh_rec, list_np_segmented_tw_raw, list_np_segmented_nh_raw, nh_clusters_all_sub, tw_clusters_all_sub,save =save)

    print('end plot segmented signals DBSCAN')
    list_segments_with_clusters_tw, list_clusters_info_tw = get_list_segments_with_clusters_DBSCAN(subject, "tweezers", list_np_segmented_tw_rec, tw_clusters_all_sub, clustering_algo)
    list_segments_with_clusters_nh, list_clusters_info_nh = get_list_segments_with_clusters_DBSCAN(subject, "needle_holder", list_np_segmented_nh_rec, nh_clusters_all_sub, clustering_algo) 
    
    print('Study a cluster')
    studied_cluster = 0 #we only want the main cluster
    tmp_tw = get_np_mat_specific_cluste(list_segments_with_clusters_tw, studied_cluster)
    tmp_nh = get_np_mat_specific_cluste(list_segments_with_clusters_nh, studied_cluster)
    plot_a_cluster_in_signal_euler(subject, tmp_tw, tmp_nh, save = save)
 
    print('Get derivatives')
    list_np_segments_d1_X_tw, list_np_segments_d2_X_tw, list_np_segments_d3_X_tw = get_list_derivatives_for_segments(list_np_segmented_tw_rec)
    list_np_segments_d1_X_nh, list_np_segments_d2_X_nh, list_np_segments_d3_X_nh = get_list_derivatives_for_segments(list_np_segmented_nh_rec)

    derivatives_segments_tw = [list_np_segmented_tw_rec, list_np_segments_d1_X_tw, list_np_segments_d2_X_tw, list_np_segments_d3_X_tw]
    derivatives_segments_nh = [list_np_segmented_nh_rec, list_np_segments_d1_X_nh, list_np_segments_d2_X_nh, list_np_segments_d3_X_nh]

    get_features(subject, derivatives_segments_tw, derivatives_segments_nh, list_clusters_info_tw, list_clusters_info_nh, clustering_algo='DBSCAN', save =save)
    i = i + 1

end_time_2nd = time.time()
duration_clustering_2nd = end_time_2nd - start_time_2nd

print("Duration time parse data:", duration_clustering_2nd, "secondes") 
print("Duration time parse data:", duration_clustering_2nd/3600, "hours") 
