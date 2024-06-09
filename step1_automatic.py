from step1_functions import *
from time_stitches import *
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

save = True
DPI_PNG = 600
PLOT_SHOW= True
plot = True
subjects = [1,19, 23,7, 24, 26, 10, 13, 16, 17, 20, 27]
i = 0

#prepare folder for figures
if save:
    directory = f"Step1_Auto_position_plots"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

for subject in (subjects):
    selected_points_nh4,selected_points_tw4 = run_step1_per_subject(subject,i)

    if plot:
        start_t_task = full_segments_time[i]['1'][0]
        end_t_task = full_segments_time[i]['8'][1]

        needle_holder_rec = pd.read_csv(f'Data/Sync_data/S_{subject}_NH_reconstructed.csv')
        tweezers_rec = pd.read_csv(f'Data/Sync_data/S_{subject}_TW_reconstructed.csv')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        sns.lineplot(x=needle_holder_rec['Time (Seconds)'], y=needle_holder_rec['X.1'], ax=ax1)
        sns.lineplot(x=selected_points_nh4['Time (Seconds)'], y=selected_points_nh4['X.1'], ax=ax1)
        ax1.axvline(x=start_t_task, color='black', linestyle='--')
        ax1.axvline(x=end_t_task, color='black', linestyle='--')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Position')
        ax1.set_title(f'Needle Holder - Subject {subject}')
        ax1.legend(['x', 'x_selected'])

        # Tracer les données pour 'tw'
        sns.lineplot(x=tweezers_rec['Time (Seconds)'], y=tweezers_rec['X.1'], ax=ax2)
        sns.lineplot(x=selected_points_tw4['Time (Seconds)'], y=selected_points_tw4['X.1'], ax=ax2)
        ax2.axvline(x=start_t_task, color='black', linestyle='--')
        ax2.axvline(x=end_t_task, color='black', linestyle='--')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Position')
        ax2.legend(['x', 'x_selected'])
        ax2.set_title(f'The Tweezers - Subject {subject}')

        if save:
            plt.savefig(f"{directory}/S_{subject}_x.png", dpi=DPI_PNG)  
        if (PLOT_SHOW):
            plt.show()
        if save: plt.close(fig)

    i = i+1