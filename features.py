from scipy.spatial.transform import Rotation as R
import numpy as np

#Soheil's features
def get_path_length(data):
    # Calculate the differences in positions between consecutive samples
    differences = np.diff(data, axis=0)
    # Calculate the Euclidean distance (path length) between consecutive points
    distances = np.linalg.norm(differences, axis=1)
    # Sum up the distances to get the total path length
    total_path_length = np.sum(distances)
    return total_path_length

def get_total_rotation(data):
    angle_differences = np.diff(data, axis=0)
    total_rotations = np.sum(angle_differences, axis=0)
    return total_rotations
    
def get_economy_of_volume(position, path_length):
    MV = 1
    for j in range(3):
        MV = MV * (np.max(position[:,j]) - np.min(position[:,j]))
    return 100.0 * ((MV ** (1/3)) / path_length)

def get_jerk(t, d1_position, d2_position, d3_position):
    norm_d1_position = np.linalg.norm(d1_position, axis=1) #d1_position = v (x,y,z)
    norm_d3_position = np.linalg.norm(d3_position, axis=1) #d3_position = jerk (x,y,z)

    norm_d1_position_max = np.max(norm_d1_position)
    frequency = 120.0
    dt = t.shape[0] * (1.0 / frequency)
    tmp = np.trapz(y=norm_d3_position**2, x=None, dx=1.0/frequency) 
    return -np.log(((dt**3) / (norm_d1_position_max**2)) * tmp)

def get_mean_std_velocity_norm(d1_position):
    norm_d1_position = np.linalg.norm(d1_position, axis=1)
    return np.mean(norm_d1_position), np.std(norm_d1_position)

#Soheil's features adapted
def get_economy_of_volume_g(position, path_length):
    if ((len(position)==0) or (path_length==0)): 
        return np.nan
    else:   
        MV = 1
        for j in range(3):
            MV = MV * (np.max(position[:,j]) - np.min(position[:,j]))
        return 100.0 * ((MV ** (1/3)) / path_length)
    
def get_total_rotation_g(data):
    if len(data)==0:
        return [np.nan, np.nan, np.nan]
    else:
        angle_differences = np.diff(data, axis=0)
        total_rotations = np.sum(angle_differences, axis=0)
        return total_rotations

def get_mean_std_velocity_norm_g(d1_position):
    if len(d1_position)==0:
        return np.nan, np.nan
    else:
        norm_d1_position = np.linalg.norm(d1_position, axis=1)
        return np.mean(norm_d1_position), np.std(norm_d1_position)
    

def get_jerk_g(t, d1_position, d2_position, d3_position):
    if len(d1_position)==0 and len(d2_position)==0 and len(d3_position)==0:
        return np.nan
    elif (len(t)==0): return np.nan
    else:
        norm_d1_position = np.linalg.norm(d1_position, axis=1)
        norm_d3_position = np.linalg.norm(d3_position, axis=1)

        norm_d1_position_max = np.max(norm_d1_position)
        frequency = 120.0
        dt = t.shape[0] * (1.0 / frequency)
        tmp = np.trapz(y=norm_d3_position**2, x=None, dx=1.0/frequency) 
        return -np.log(((dt**3) / (norm_d1_position_max**2)) * tmp)