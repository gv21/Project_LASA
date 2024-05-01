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

TOOLS = ["The Tweezers", "Needle Holder"]
CLUSTER_COLORS = ["green", "blue", "red"]
FS_LABEL = 15
FS_TITLE = 15
FS_TICKS = 14
LW =  1.25
DPI_PNG = 600
PLOT_SHOW = False

# Enable LaTeX for all text rendering in Matplotlib
"""plt.rcParams['text.usetex'] = True
# Optionally, specify the default font to be used in the plots
plt.rcParams['font.family'] = 'serif'"""

def plot_save_segmented_data_quaternions(subject_id, segment_id, data_tw, data_nh, data_loss_tw, data_loss_nh, clustering_algo ='Kmeans', save = True):
    directory = f"{clustering_algo}/Plots/Signals/Quaternion Representation/"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    fig, axs = plt.subplots(7, 2, sharex=True, figsize=(12, 10)) 
    # Plot data (tweezers) considering clustering
    axs[0, 0].set_title(f'{TOOLS[0]} (Sub. {subject_id}, Seg. {segment_id+1}) - Loss {data_loss_tw} [\%]', fontsize=FS_TITLE) 
    for i in range(7):
        axs[i, 0].scatter(data_tw[0][:, 0], data_tw[0][:, i+1], facecolor=CLUSTER_COLORS[0], s=1) 
        axs[i, 0].scatter(data_tw[1][:, 0], data_tw[1][:, i+1], facecolor=CLUSTER_COLORS[1], s=1) 
        axs[i, 0].scatter(data_tw[2][:, 0], data_tw[2][:, i+1], facecolor=CLUSTER_COLORS[2], s=1) 
        axs[i, 0].tick_params(axis='x', labelsize=FS_TICKS)  
        axs[i, 0].tick_params(axis='y', labelsize=FS_TICKS) 
    # Plot data (needle holder) considering clustering
    axs[0, 1].set_title(f'{TOOLS[1]} (Sub. {subject_id}, Seg. {segment_id+1}) - Loss {data_loss_nh} [\%]', fontsize=FS_TITLE) 
    for i in range(7):
        axs[i, 1].scatter(data_nh[0][:, 0], data_nh[0][:, i+1], facecolor=CLUSTER_COLORS[0], s=1) 
        axs[i, 1].scatter(data_nh[1][:, 0], data_nh[1][:, i+1], facecolor=CLUSTER_COLORS[1], s=1) 
        axs[i, 1].scatter(data_nh[2][:, 0], data_nh[2][:, i+1], facecolor=CLUSTER_COLORS[2], s=1)         
        axs[i, 1].tick_params(axis='x', labelsize=FS_TICKS)  
        axs[i, 1].tick_params(axis='y', labelsize=FS_TICKS)     
    # Set labels
    axs[0, 0].set_ylabel(f'$x\,[m]$', fontsize=FS_LABEL)
    axs[1, 0].set_ylabel(f'$y\,[m]$', fontsize=FS_LABEL)
    axs[2, 0].set_ylabel(f'$z\,[m]$', fontsize=FS_LABEL)
    axs[3, 0].set_ylabel(f'$q_w$', fontsize=FS_LABEL)
    axs[4, 0].set_ylabel(f'$q_x$', fontsize=FS_LABEL)
    axs[5, 0].set_ylabel(f'$q_y$', fontsize=FS_LABEL)
    axs[6, 0].set_ylabel(f'$q_z$', fontsize=FS_LABEL)
    axs[6, 0].set_xlabel(f'Time [s]', fontsize=FS_LABEL)
    axs[6, 1].set_xlabel(f'Time [s]', fontsize=FS_LABEL)
    plt.tight_layout()
    if save: plt.savefig(f"{directory}/S_{subject_id}_signals_{segment_id+1}.png", dpi=DPI_PNG)  
    if (PLOT_SHOW):
        plt.show()
    if save: plt.close(fig)

def plot_save_segmented_data_euler(subject_id, segment_id, data_tw, data_nh, data_loss_tw, data_loss_nh, clustering_algo = 'Kmeans',save =True):
    directory = f"{clustering_algo}/Plots/Signals/Euler Representation/"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    fig, axs = plt.subplots(6, 2, sharex=True, figsize=(12, 10)) 
    # Plot data (tweezers) considering clustering
    axs[0, 0].set_title(f'{TOOLS[0]} (Sub. {subject_id}, Seg. {segment_id+1}) - Loss {data_loss_tw} [\%]', fontsize=FS_TITLE) 
    for i in range(6):
        axs[i, 0].scatter(data_tw[0][:, 0], data_tw[0][:, i+1+4], facecolor=CLUSTER_COLORS[0], s=1) 
        axs[i, 0].scatter(data_tw[1][:, 0], data_tw[1][:, i+1+4], facecolor=CLUSTER_COLORS[1], s=1) 
        axs[i, 0].scatter(data_tw[2][:, 0], data_tw[2][:, i+1+4], facecolor=CLUSTER_COLORS[2], s=1) 
        axs[i, 0].tick_params(axis='x', labelsize=FS_TICKS)  
        axs[i, 0].tick_params(axis='y', labelsize=FS_TICKS) 
    # Plot data (needle holder) considering clustering
    axs[0, 1].set_title(f'{TOOLS[1]} (Sub. {subject_id}, Seg. {segment_id+1}) - Loss {data_loss_nh} [\%]', fontsize=FS_TITLE) 
    for i in range(6):
        axs[i, 1].scatter(data_nh[0][:, 0], data_nh[0][:, i+1+4], facecolor=CLUSTER_COLORS[0], s=1) 
        axs[i, 1].scatter(data_nh[1][:, 0], data_nh[1][:, i+1+4], facecolor=CLUSTER_COLORS[1], s=1) 
        axs[i, 1].scatter(data_nh[2][:, 0], data_nh[2][:, i+1+4], facecolor=CLUSTER_COLORS[2], s=1)         
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

def plot_a_cluster_in_signal_euler(subject_id, data_tw, data_nh, clustering_algo='Kmeans', save = True):
    directory =''
    if clustering_algo=='Kmeans': directory = f"Kmeans/Plots/Signals/Euler Representation/A Cluster/"
    else: directory = f"{clustering_algo}/Plots/Signals/Euler Representation/A Cluster/"
    
    if not os.path.exists(directory) and save:
        os.makedirs(directory, exist_ok=True)
    for s in range(8): 
        """if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)""" #already did it before
        fig, axs = plt.subplots(6, 2, sharex=True, figsize=(12, 10)) 
        # Plot data (tweezers) considering clustering
        axs[0, 0].set_title(f'{TOOLS[0]} (Sub. {subject_id}, Seg. {s+1})', fontsize=FS_TITLE) 
        for i in range(6):
            if (i < 3):
                axs[i, 0].scatter(data_tw[s][:, 0], data_tw[s][:, i+1], facecolor=CLUSTER_COLORS[0], s=1) 
            else:
                axs[i, 0].scatter(data_tw[s][:, 0], data_tw[s][:, i+4], facecolor=CLUSTER_COLORS[0], s=1) 
            axs[i, 0].tick_params(axis='x', labelsize=FS_TICKS)  
            axs[i, 0].tick_params(axis='y', labelsize=FS_TICKS) 
        # Plot data (needle holder) considering clustering
        axs[0, 1].set_title(f'{TOOLS[1]} (Sub. {subject_id}, Seg. {s+1})', fontsize=FS_TITLE) 
        for i in range(6):
            if (i < 3):
                axs[i, 1].scatter(data_nh[s][:, 0], data_nh[s][:, i+1], facecolor=CLUSTER_COLORS[0], s=1)    
            else:
                axs[i, 1].scatter(data_nh[s][:, 0], data_nh[s][:, i+1+4], facecolor=CLUSTER_COLORS[0], s=1)    
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
        if save: plt.savefig(f"{directory}/S_{subject_id}_signals_{s+1}.png", dpi=DPI_PNG)  
        if (PLOT_SHOW):
            plt.show()
        if save: plt.close(fig)


def plot_clustered_data_3D(subject_id, tool_name, segment_id, cluster_info, data, clustering_algo='Kmeans'):
    directory = f"{clustering_algo}/Plots/Segmented/Clusters/S_{subject_id}/{tool_name}/{segment_id+1}"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    colors = CLUSTER_COLORS 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, info in enumerate(cluster_info):
        color = colors[i % len(colors)]
        indices = info['cluster_indices_in_data']
        ax.scatter(data[indices, 1], data[indices, 2], data[indices, 3], c=color, label=f'Cluster {i}', s=1)
    ax.set_xlabel('$x\,[m]$')
    ax.set_ylabel('$y\,[m]$')
    ax.set_zlabel('$z\,[m]$')
    plt.title(f'3-D Scatter Plot of Clustered Data (Stitch {segment_id+1})')
    plt.tight_layout()
    plt.savefig(f"{directory}/clusters.png", dpi=DPI_PNG)  
    if (PLOT_SHOW):
        plt.show()
    plt.close(fig)

