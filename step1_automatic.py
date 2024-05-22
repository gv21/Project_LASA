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

for subject in subjects:
    needle_holder_rec = pd.read_csv(f'Data/Sync_data/S_{subject}_NH_reconstructed.csv')
    tweezers_rec = pd.read_csv(f'Data/Sync_data/S_{subject}_TW_reconstructed.csv')

    medians_nh = (needle_holder_rec[['X.1', 'Y.1', 'Z.1']]).median()
    medians_tw = (tweezers_rec[['X.1', 'Y.1', 'Z.1']]).median()

    """X Position"""
    selected_xpos_nh, selected_xpos_tw = remove_extreme_using_x_pos(needle_holder_rec, tweezers_rec, prop_nh=0.5, prop_tw=0.5)
    time_event_xpos_nh, time_event_xpos_tw = compute_event_time_pos(full_segments_time[i], needle_holder_rec, tweezers_rec, selected_xpos_nh,
                                                                selected_xpos_tw)

    pairs_t_xpos_nh = compute_pairs_time(time_event_xpos_nh[0], time_event_xpos_nh[1])
    pairs_t_xpos_tw = compute_pairs_time(time_event_xpos_tw[0], time_event_xpos_tw[1])

    pairs_adjusted_t_xpos_nh = adjust_time_pair(needle_holder_rec, pairs_t_xpos_nh, medians_nh['X.1'], time_allowed = 6)
    pairs_adjusted_t_xpos_tw = adjust_time_pair(tweezers_rec, pairs_t_xpos_tw, medians_tw['X.1'], time_allowed=6)

    mask_extreme_xpos_event_removed_nh = remove_extreme(needle_holder_rec, pairs_adjusted_t_xpos_nh)
    mask_extreme_xpos_event_removed_tw = remove_extreme(tweezers_rec, pairs_adjusted_t_xpos_tw)

    """X Velocity"""
    mask_min_max_nh, mask_min_max_tw = compute_mask_using_v(needle_holder_rec, tweezers_rec, smooth_window=60, prop_nh=0.4, prop_tw=0.4)
    time_event_nh, time_event_tw = compute_event_time(full_segments_time[i], needle_holder_rec, tweezers_rec, mask_min_max_nh[0],  
                                                            mask_min_max_nh[1], mask_min_max_tw[0], mask_min_max_tw[1])
                
    pairs_t_nh = compute_pairs_time(time_event_nh[0], time_event_nh[1])
    pairs_t_tw = compute_pairs_time(time_event_tw[0], time_event_tw[1])

    pairs_adjusted_t_nh = adjust_time_pair(needle_holder_rec, pairs_t_nh, medians_nh['X.1'], time_allowed = 6)
    pairs_adjusted_t_tw = adjust_time_pair(tweezers_rec, pairs_t_tw, medians_tw['X.1'], time_allowed = 6)

    mask_extreme_v_event_removed_nh = remove_extreme(needle_holder_rec, pairs_adjusted_t_nh)
    mask_extreme_v_event_removed_tw = remove_extreme(tweezers_rec, pairs_adjusted_t_tw)
    mask_extreme_v_glass_nh = remove_extreme(needle_holder_rec, pairs_adjusted_t_tw) #we don't want to keep event when the other tool is in the glass
    mask_extreme_v_glass_tw = remove_extreme(tweezers_rec, pairs_adjusted_t_nh) #we don't want to keep event when the other tool is in the glass

    """Select points"""
    mask_glass_nh = mask_extreme_xpos_event_removed_nh & mask_extreme_v_event_removed_nh & selected_xpos_nh
    mask_glass_tw = mask_extreme_xpos_event_removed_tw & mask_extreme_v_event_removed_tw & selected_xpos_tw
    selected_points_nh = needle_holder_rec[mask_glass_nh & mask_extreme_v_glass_nh] #v is the most discriminative in this case compared to xpos
    selected_points_tw = tweezers_rec[mask_glass_tw & mask_extreme_v_glass_tw] #v is the most discriminative in this case compared to xpos

    #so far we excluded events where the tool is reaching the glass, we then want a more precise selection of points
    """2nd selection based on z position"""
    std_nh =  (selected_points_nh[['X.1', 'Y.1', 'Z.1']]).std()
    std_tw =  (selected_points_tw[['X.1', 'Y.1', 'Z.1']]).std()
    if (subject == 19): #subject 19 is treated separately, two different ref level of z position during the task 
        selected_zpos_nh, selected_zpos_tw = remove_extreme_using_z_pos2_sub19(needle_holder_rec, tweezers_rec)
    else:
        selected_zpos_nh, selected_zpos_tw = remove_extreme_using_z_pos2(needle_holder_rec, tweezers_rec, std_nh, std_tw, medians_nh, medians_tw)
    time_event_zpos_nh, time_event_zpos_tw = compute_event_time_pos(full_segments_time[i], needle_holder_rec, tweezers_rec, selected_zpos_nh,
                                                                selected_zpos_tw)

    pairs_t_zpos_nh = compute_pairs_time(time_event_zpos_nh[0], time_event_zpos_nh[1])
    pairs_t_zpos_tw = compute_pairs_time(time_event_zpos_tw[0], time_event_zpos_tw[1])

    pairs_adjusted_t_zpos_nh = adjust_time_pair(selected_points_nh, pairs_t_zpos_nh, medians_nh['Z.1'], time_allowed = 6)
    pairs_adjusted_t_zpos_tw = adjust_time_pair(selected_points_tw, pairs_t_zpos_tw, medians_tw['Z.1'], time_allowed=6)

    mask_extreme_zpos_event_removed_nh = remove_extreme(selected_points_nh, pairs_adjusted_t_zpos_nh)
    mask_extreme_zpos_event_removed_tw = remove_extreme(selected_points_tw, pairs_adjusted_t_zpos_tw)

    
    selected_points_nh2 = selected_points_nh[mask_extreme_zpos_event_removed_nh]
    selected_points_tw2 = selected_points_tw[mask_extreme_zpos_event_removed_tw]

    """2nd selection based on y position"""
    selected_ypos_nh, selected_ypos_tw = remove_extreme_using_y_pos(needle_holder_rec, tweezers_rec, std_nh, std_tw, medians_nh, medians_tw)
    time_event_ypos_nh, time_event_ypos_tw = compute_event_time_pos(full_segments_time[i], needle_holder_rec, tweezers_rec, selected_ypos_nh,
                                                                selected_ypos_tw)

    pairs_t_ypos_nh = compute_pairs_time(time_event_ypos_nh[0], time_event_ypos_nh[1])
    pairs_t_ypos_tw = compute_pairs_time(time_event_ypos_tw[0], time_event_ypos_tw[1])

    pairs_adjusted_t_ypos_nh = adjust_time_pair(selected_points_nh2, pairs_t_ypos_nh, medians_nh['Y.1'], time_allowed = 6)
    pairs_adjusted_t_ypos_tw = adjust_time_pair(selected_points_tw2, pairs_t_ypos_tw, medians_tw['Y.1'], time_allowed=6)

    mask_extreme_ypos_event_removed_nh = remove_extreme(selected_points_nh2, pairs_adjusted_t_ypos_nh)
    mask_extreme_ypos_event_removed_tw = remove_extreme(selected_points_tw2, pairs_adjusted_t_ypos_tw)

    
    selected_points_nh3 = selected_points_nh2[mask_extreme_ypos_event_removed_nh]
    selected_points_tw3 = selected_points_tw2[mask_extreme_ypos_event_removed_tw]

    """2nd selection based on x position"""
    #x position ref line of subject 1 is moving during the task, we need to treat it separately
    if (subject!=1):
        selected_xpos2_nh, selected_xpos2_tw = remove_extreme_using_x_pos_on_selected_data(needle_holder_rec, tweezers_rec, std_nh, std_tw, medians_nh, medians_tw)
    else:
        selected_xpos2_nh, selected_xpos2_tw  = remove_extreme_using_x_pos_on_selected_data_sub1(needle_holder_rec, tweezers_rec, selected_points_nh, selected_points_tw)
    time_event_xpos2_nh, time_event_xpos2_tw = compute_event_time_pos(full_segments_time[i], needle_holder_rec, tweezers_rec, selected_xpos2_nh,
                                                                selected_xpos2_tw)

    pairs_t_xpos2_nh = compute_pairs_time(time_event_xpos2_nh[0], time_event_xpos2_nh[1])
    pairs_t_xpos2_tw = compute_pairs_time(time_event_xpos2_tw[0], time_event_xpos2_tw[1])

    pairs_adjusted_t_xpos2_nh = adjust_time_pair(selected_points_nh3, pairs_t_xpos2_nh, medians_nh['X.1'], time_allowed = 6)
    pairs_adjusted_t_xpos2_tw = adjust_time_pair(selected_points_tw3, pairs_t_xpos2_tw, medians_tw['X.1'], time_allowed=6)

    mask_extreme_xpos2_event_removed_nh = remove_extreme(selected_points_nh3, pairs_adjusted_t_xpos2_nh)
    mask_extreme_xpos2_event_removed_tw = remove_extreme(selected_points_tw3, pairs_adjusted_t_xpos2_tw)

    selected_points_nh4 = selected_points_nh3[mask_extreme_xpos2_event_removed_nh]
    selected_points_tw4 = selected_points_tw3[mask_extreme_xpos2_event_removed_tw]

    """Manual correction of incorrectly segmented points """
    if (subject==1):
        mask_tw = (selected_points_tw4['Time (Seconds)']>1275) & (selected_points_tw4['Time (Seconds)']<1287)
        selected_points_tw4 = selected_points_tw4[~mask_tw]
    if (subject==13):
        mask_tw = (selected_points_tw4['Time (Seconds)']>679) & (selected_points_tw4['Time (Seconds)']<711.5)
        selected_points_tw4 = selected_points_tw4[~mask_tw]
    """end"""

    if plot:
        start_t_task = full_segments_time[i]['1'][0]
        end_t_task = full_segments_time[i]['8'][1]

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

        """fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        sns.lineplot(x=needle_holder_rec['Time (Seconds)'], y=needle_holder_rec['Y.1'], ax=ax1)
        sns.lineplot(x=selected_points_nh4['Time (Seconds)'], y=selected_points_nh4['Y.1'], ax=ax1)
        ax1.axvline(x=start_t_task, color='black', linestyle='--')
        ax1.axvline(x=end_t_task, color='black', linestyle='--')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Position')
        ax1.set_title(f'Needle Holder - Subject {subject}')
        ax1.legend(['y', 'y_selected'])

        # Tracer les données pour 'tw'
        sns.lineplot(x=tweezers_rec['Time (Seconds)'], y=tweezers_rec['Y.1'], ax=ax2)
        sns.lineplot(x=selected_points_tw4['Time (Seconds)'], y=selected_points_tw4['Y.1'], ax=ax2)
        ax2.axvline(x=start_t_task, color='black', linestyle='--')
        ax2.axvline(x=end_t_task, color='black', linestyle='--')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Position')
        ax2.legend(['y', 'y_selected'])
        ax2.set_title(f'The Tweezers - Subject {subject}')

        if save:
            plt.savefig(f"{directory}/S_{subject}_y.png", dpi=DPI_PNG)  
        if (PLOT_SHOW):
            plt.show()
        if save: plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        sns.lineplot(x=needle_holder_rec['Time (Seconds)'], y=needle_holder_rec['Z.1'], ax=ax1)
        sns.lineplot(x=selected_points_nh4['Time (Seconds)'], y=selected_points_nh4['Z.1'], ax=ax1)
        ax1.axvline(x=start_t_task, color='black', linestyle='--')
        ax1.axvline(x=end_t_task, color='black', linestyle='--')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Position')
        ax1.set_title(f'Needle Holder - Subject {subject}')
        ax1.legend(['z', 'z_selected'])

        # Tracer les données pour 'tw'
        sns.lineplot(x=tweezers_rec['Time (Seconds)'], y=tweezers_rec['Z.1'], ax=ax2)
        sns.lineplot(x=selected_points_tw4['Time (Seconds)'], y=selected_points_tw4['Z.1'], ax=ax2)
        ax2.axvline(x=start_t_task, color='black', linestyle='--')
        ax2.axvline(x=end_t_task, color='black', linestyle='--')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Position')
        ax2.legend(['z', 'z_selected'])
        ax2.set_title(f'The Tweezers - Subject {subject}')

        if save:
            plt.savefig(f"{directory}/S_{subject}_z.png", dpi=DPI_PNG) 
        if (PLOT_SHOW):
            plt.show() 
        if save: plt.close(fig)"""

    i = i+1