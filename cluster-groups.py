#Step 3
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # Import StandardScaler
from numpy.polynomial.polynomial import Polynomial

# Enable LaTeX for all text rendering in Matplotlib
"""plt.rcParams['text.usetex'] = True
# Optionally, specify the default font to be used in the plots
plt.rcParams['font.family'] = 'serif'"""

GROUPS_NAMES = ["E", "I", "N"]

target_subjects = [1]

metrics_tw = np.empty((8*len(target_subjects), 8))
metrics_nh = np.empty((8*len(target_subjects), 8))

# load data
subject_number = 0
for subject in target_subjects:
    data = np.load(f"OT_Features/S_{subject}/ot_metrics.npy")

    metrics_tw[subject_number*8:(subject_number+1)*8, :] = data[:, 0, :, 0]
    metrics_nh[subject_number*8:(subject_number+1)*8, :] = data[:, 0, :, 1]

    subject_number = subject_number + 1

def determine_expertise(score):
    if (score < 3): return GROUPS_NAMES[0]
    elif( score >= 3 and score < 6): return GROUPS_NAMES[1]
    else: return GROUPS_NAMES[2]

def get_expertise(subject):
    subjects_info = np.load("subject_info_modified.npy")
    for row in subjects_info:
        if row[0] == subject:
            score = row[-1]
            return score

def cluster_data(data, number_of_cluster, target_subjects):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=number_of_cluster, random_state=0).fit(data_scaled)
    # print("Cluster Centers:", kmeans.cluster_centers_)
    # print("Labels:", kmeans.labels_)
    pca = PCA(n_components=4)

    reduced_data = pca.fit_transform(data_scaled)
    print(np.cumsum(pca.explained_variance_ratio_))

    fig = plt.figure()
    plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['green', 'blue', 'red']
    for i in range(number_of_cluster):  
        cluster_data = reduced_data[kmeans.labels_ == i]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], color=colors[i], label=f'Cluster {i+1}', alpha=0.5, s=250)
        
        # for j, (x, y) in enumerate(cluster_data):
        #     subject_index = np.where(kmeans.labels_ == i)[0][j] // 8  # Calculate subject index
        #     expertise = determine_expertise(get_expertise(target_subjects[subject_index]))
        #     plt.text(x, y, f'{expertise}', color='black', fontsize=12)

    centers = pca.transform(kmeans.cluster_centers_)
    print(centers.shape)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='X', s=200, color='black', label='Centers')
    
    plt.title('Cluster visualization with PCA-reduced data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def plot_metric_progress_subplots_with_fit_and_expertise(metrics, subject_ids, metric_index=0, degree=2):
    num_subjects = len(subject_ids)
    num_cols = 3
    num_rows = (num_subjects + num_cols - 1) // num_cols
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
    axs = axs.flatten()
    
    for i, subject in enumerate(subject_ids):
        expertise_level = determine_expertise(get_expertise(subject))
        start_index = i * 8
        end_index = (i + 1) * 8
        metric_values = metrics[start_index:end_index, metric_index]
        segment_numbers = np.arange(1, 9)
        
        coefs = Polynomial.fit(segment_numbers, metric_values, degree)
        x_fit = np.linspace(segment_numbers.min(), segment_numbers.max(), 100)
        y_fit = coefs(x_fit)
        
        axs[i].plot(segment_numbers, metric_values, label=f'Subject {subject}', marker='o', linestyle='-')
        axs[i].plot(x_fit, y_fit, linestyle='--', color='gray', label='Fit')
        
        axs[i].set_title(f'Subject {subject}: ({expertise_level})')
        axs[i].set_xlabel('Segment Number')
        axs[i].set_ylabel('Metric Value')
        axs[i].set_xticks(range(1, 9))
        axs[i].legend()
        axs[i].grid(True)
    
    for ax in axs[num_subjects:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()

cluster_data(metrics_tw, 3, target_subjects)
cluster_data(metrics_nh, 3, target_subjects)
# plot_metric_progress_subplots_with_fit_and_expertise(metrics_tw, target_subjects, metric_index=3, degree=4)