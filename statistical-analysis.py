#Step 2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon
from scipy.stats import ranksums
from scipy.stats import shapiro
import statistics 
import scipy.stats as stats

GROUPS_NAMES = ["E", "I", "N"]

def test_normality(data):
    if (len(data) > 3): # needs at least 3 samples
        SIGNIFICANCE_LEVEL_NORMALITY = 0.05
        stat, p = shapiro(data)
        flag = False
        if (p > SIGNIFICANCE_LEVEL_NORMALITY):
            flag = True
            msg_mean = statistics.mean(data)
            msg_std = statistics.stdev(data)
            q3, q1 = np.percentile(data, [75 ,25])
            iqr = q3 - q1
            msg_median = statistics.median(data)
            msg_iqr = iqr
        else:
            flag = False
            msg_mean = statistics.mean(data)
            msg_std = statistics.stdev(data)
            q3, q1 = np.percentile(data, [75 ,25])
            iqr = q3 - q1
            msg_median = statistics.median(data)
            msg_iqr = iqr
        return (flag, round(stat, 3), round(p, 3), round(msg_mean, 2), round(msg_std, 2), round(msg_median, 2), round(msg_iqr, 2))
    
    else: return (False, )

def statistical_test(data):
    res = ranksums(data[0], data[1]) # Mann-Whitney U Test
    stat = res.statistic
    p_value = res.pvalue
    df = len(data[0]) + len(data[1]) - 2
    return stat, p_value, df 

def sns_box_plot(grouped_data):
    colors_box = ['#FFFFFF', '#FFFFFF', '#FFFFFF']
    colors_dots = ['darkgreen', 'darkblue', 'darkred']
    axis = plt.subplot()
    sns.set(style='whitegrid')
    boxplot_n = sns.boxplot(data=grouped_data, width=0.35, linewidth=1.65, saturation=1, palette=colors_box,
        medianprops=dict(color="black", alpha=0.8, linewidth=0.8),
        whiskerprops=dict(linestyle="--", color="black", alpha=0.9, linewidth=0.6),
        capwidths=0.05,
        capprops=dict(color="black", alpha=0.9),
        boxprops=dict(edgecolor="black", alpha=0.9, linewidth=0.6),
        flierprops=dict(markerfacecolor="red", markeredgecolor="red", alpha=1, markersize=7, marker='+'),
        showfliers=True,
        showmeans=True,)
    sns.stripplot(data=grouped_data, marker="o", alpha=0.9, palette=colors_dots, dodge=True, jitter=True,  ax=boxplot_n)
    return axis

def statistics_and_plots(grouped_data, metric_name, metric_quantity, tool_name, clustering_algo ='DBSCAN'):  
    # return: statistics, (flag, round(stat, 3), round(p, 3), round(msg_mean, 2), round(msg_std, 2), round(msg_median, 2), round(msg_iqr, 2))
    modified_group_names = [None] * len(grouped_data)
    descriptive_statistics = [None] * len(grouped_data)
    for i in range(len(grouped_data)):
        descriptive_statistics[i] = test_normality(grouped_data[i])
        if (test_normality(grouped_data[i])[0]):
            modified_group_names[i] = GROUPS_NAMES[i] + "*"
        else:
            modified_group_names[i] = GROUPS_NAMES[i]

    for i in range(len(grouped_data)):
        print(f">> {GROUPS_NAMES[i]} - descriptive statistics (n = {len(grouped_data[i])}): {descriptive_statistics[i]}")
    print("-----")

    axis = sns_box_plot(grouped_data)
    
    # Calculate the mean values and plot it
    mean_values = [np.mean(group) for group in grouped_data]
    axis.plot([0, 1], [mean_values[0], mean_values[1]], linewidth=0.75, color='coral')
    axis.plot([1, 2], [mean_values[1], mean_values[2]], linewidth=0.75, color='coral')

    axis.set_title(f"{tool_name}")
    xtick_labels = [f"{name} (n={len(data)})" for name, data in zip(modified_group_names, grouped_data)]
    axis.set_xticklabels(xtick_labels)
    axis.set_xlabel(f"Groups", fontsize=12) 
    axis.set_ylabel(f"{metric_name} {metric_quantity}", fontsize=12) 
    axis.tick_params(axis='both', labelsize=12)  
    # axis.set_ylim(0, upper_limit)  # Assuming you want the lower limit to start at 0

    # significant difference
    # return: (statistics, p_value, df)
    test_result_01 = statistical_test([grouped_data[0], grouped_data[1]])
    test_result_12 = statistical_test([grouped_data[1], grouped_data[2]])
    test_result_02 = statistical_test([grouped_data[0], grouped_data[2]])
    print(f"+ Statistics test (E & I): {test_result_01}")
    print(f"+ Statistics test (I & N): {test_result_12}")
    print(f"+ Statistics test (E & N): {test_result_02}")
    annotate_significance(test_result_01[1], [grouped_data[0], grouped_data[1]], [0, 1])
    annotate_significance(test_result_12[1], [grouped_data[1], grouped_data[2]], [1, 2])
    annotate_significance(test_result_02[1], [grouped_data[0], grouped_data[2]], [0, 2])

    axis.grid(False)
    axis.set_frame_on(True)
    plt.tight_layout()
    directory = f"{clustering_algo}/Plots/Statistics_Metrics/"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    file_name = f"{directory}/{get_metrc_name_quantity(metric)[0]}_{tool_name}.png"
    plt.tight_layout()
    plt.savefig(file_name, dpi=900, bbox_inches='tight', transparent=True)  

def annotate_significance(p_val, data, cols):
    enabled_annotation = True
    x1, x2 = cols[0], cols[1] # columns
    max0 = max(data[0])
    max1 = max(data[1])
    if (max0 > max1): max_both = max0
    else: max_both = max1

    y, h = 1.05*max_both, 0.005 
    if (enabled_annotation and (p_val < 0.05)):
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c='k')
        plt.text((x1 + x2) * 0.5, 1.0*(y + h), f"$ \dagger $", ha='center', va='bottom', color='k', fontsize=14)
    elif (enabled_annotation and (p_val <= 0.01)):
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c='k')
        plt.text((x1 + x2) * 0.5, 1.0*(y + h), f"$ \ddagger $", ha='center', va='bottom', color='k', fontsize=14)    
        
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
        
def get_metrc_name_quantity(metric_id):
    if (metric_id == 0):
        return "Time", f"$ [s] $"
    elif (metric_id == 1):
        return "Path Length", f"$ [m] $" 
    elif (metric_id == 2):
        return "Total Rotation X", f"$ [deg] $" 
    elif (metric_id == 3):
        return "Total Rotation Y", f"$ [deg] $" 
    elif (metric_id == 4):
        return "Total Rotation Z", f"$ [deg] $" 
    elif (metric_id == 5):
        return "Jerk", f"$ [-] $"  
    elif (metric_id == 6):
        return "Economy of Volume", f"$ [\%] $"  
    elif (metric_id == 7):
        return "Mean of Velocity", f"$ [m/s] $" 

    
# tools, samples, metrics
METRICS  = 8
expert = np.empty((2, 16, METRICS))
intermediate = np.empty((2, 16, METRICS))
novice = np.empty((2, 16, METRICS))
expert_subjects = 0
intermediate_subjects = 0
novice_subjects = 0

target_subjects = [1]

for subject in target_subjects:
    metrics = np.load(f"OT_Features/S_{subject}/ot_metrics.npy")
    if (determine_expertise(get_expertise(subject)) == GROUPS_NAMES[0]):
        expert[0, (expert_subjects*8):(expert_subjects+1)*8, 0:METRICS] = metrics[0:8, 0, 0:METRICS, 0]
        expert[1, (expert_subjects*8):(expert_subjects+1)*8, 0:METRICS] = metrics[0:8, 0, 0:METRICS, 1]
        expert_subjects = expert_subjects + 1
    elif (determine_expertise(get_expertise(subject)) == GROUPS_NAMES[1]):
        intermediate[0, (intermediate_subjects*8):(intermediate_subjects+1)*8, 0:METRICS] = metrics[0:8, 0, 0:METRICS, 0]
        intermediate[1, (intermediate_subjects*8):(intermediate_subjects+1)*8, 0:METRICS] = metrics[0:8, 0, 0:METRICS, 1]
        intermediate_subjects = intermediate_subjects + 1
    elif (determine_expertise(get_expertise(subject)) == GROUPS_NAMES[2]):
        novice[0, (novice_subjects*8):(novice_subjects+1)*8, 0:METRICS] = metrics[0:8, 0, 0:METRICS, 0]
        novice[1, (novice_subjects*8):(novice_subjects+1)*8, 0:METRICS] = metrics[0:8, 0, 0:METRICS, 1]
        novice_subjects = novice_subjects + 1       

FIG_SIZE = (5, 4)
tools = ["The Tweezers", "Needle Holder"] 
clustering_algo ='DBSCAN'
for tool in tools:
    for metric in range(6):
        if (tool == tools[0]):
            grouped_data = [expert[0, :, metric], intermediate[0, :, metric], novice[0, :, metric]]
        elif (tool == tools[1]):
            grouped_data = [expert[1, :, metric], intermediate[1, :, metric], novice[1, :, metric]]

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIG_SIZE, sharex=False, sharey=False, squeeze=True)
        statistics_and_plots(grouped_data, 
                            get_metrc_name_quantity(metric)[0], 
                            get_metrc_name_quantity(metric)[1],
                            tool, clustering_algo=clustering_algo)

        # plt.show()

        