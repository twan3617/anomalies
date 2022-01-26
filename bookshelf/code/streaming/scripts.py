### file for scripts
### idea: have separate functions for cleaning data from Building_Model.mat, for performing welch then vmd, and for concatenating data together into features_seq.
### this gives us more flexibility if we want to modify things halfway through the process.
### data processing function 

import pandas as pd 
import numpy as np 
from scipy.signal import welch
import stumpy
from stumpy.floss import _cac
from vmdpy import VMD



def exp_list_process(data_dict, cases, num_healthy):
    '''
    data_dict should be a dictionary with keys Sensorn (n between 1 and num_sensors)
    num_sensors is number of sensors you want (between 1 and 24; just takes first num_sensors sensors)
    cases should be a list with elements of the form ['damage_area', 'damage_type', 'voltage']
    '''
    num_sensors = len(data_dict.keys())
    df_list = dict()

    for sensor_num in range(1,num_sensors+1):
        sensor_str = 'Sensor'+str(sensor_num)
        df = data_dict[sensor_str]
        df_list_temp = []
        for num in cases:
            query1 = (df['damage_location']==num[0]).values
            query2 = (df['damage_level']==num[1]).values
            query3 = (df['voltage_level']==num[2]).values

            df_query = df[query1 & query2 & query3]
            df_list_temp.append(df_query.reset_index(drop=True)) # Organize cases per sensor
            
        df_list[sensor_str] = pd.concat(df_list_temp) # Concatenate cases across each sensor
        
        # Removing 50-num_healthy D00 (heahlthy i.e. non damage) cases 
        # Because a lot of the damage case only have 5 events recorded in total
        # We shall have 50-num_healthy D00 --> 5 DB0 --> 5 DBB --> 5 DBB cases in a sequence
        df_list[sensor_str] =  df_list[sensor_str].iloc[50-num_healthy:,:].reset_index(drop=True)
    
    return df_list 

def welch_vmd(df_list, signal_length, nperseg, fs, vmd, alpha, tau, DC, init, tol):
    '''
    Input: 
    data: df_list, a dict with Sensorn as key and all of the data (including labels).
    vmd: a natural number (if vmd > 0, perform vmd with num_modes = vmd; if 0, do not perform vmd)
    
    In terms of the data pipeline, this should be placed after the correct experimental data has been pulled out of Building_Model.mat into df_list. Just iterate over the number of sensors. 
    This gives us flexibility in case we want to experiment with different combinations of normal/anomalous data etc. 
    '''  
    first_item = next(iter(df_list.values()))
    num_experiments = len(first_item)
    
    if vmd >= 1: 
        num_sensors = len(df_list.keys())
        building_features = dict()
        for sensor_num in range(1,num_sensors+1):
            # For Each Sensor
            sensor_str = f'Sensor{sensor_num}'
            sensor_values = df_list[sensor_str].iloc[:,:signal_length].values
            print(f'Processing Sensor {sensor_num}')
            building_features[sensor_str] = dict() # Initialize nested dictionary (sensor_num : experiment_num : values)

            # For Each Experiment
            for i in range(num_experiments):
                experiment_str = f'Experiment{i+1}'
                data_row = sensor_values[i,:]
                data_row_demeaned = data_row - np.mean(data_row) # Remove "DC" component (i.e. de-mean)
                u, _, _ = VMD(data_row_demeaned, alpha, tau, vmd, DC, init, tol)  # Return K x n-array from n-array input
                if i%5 == 0:
                    print(f'\tProcessed VMD for Experiment Num (i.e. Data Row) {i}')

                vmd_list = []
                for j in range(vmd):
                    # For Each VMD Signal More Feature Engineering
                    # Paper for welch: http://bobweigel.net/csi763/images/Welch_1967.pdf
                    pxx = welch(u[j,:], fs=fs, nperseg=nperseg) # Returns len 2 tuple
                    vmd_list.append(np.log(pxx[1]/max(pxx[1]))) # Feature Engineering
                
                building_features[sensor_str][experiment_str] = np.array(vmd_list)
        return building_features
    
    if vmd == 0: 
        building_features = dict()
        welch_list = []

        for i in range(num_experiments):
            experiment_str = f'Experiment{i+1}'
            data_row = df_list.iloc[i,:]
            data_row_demeaned = data_row - np.mean(data_row) # Remove "DC" component (i.e. de-mean)
            
            pxx = welch(data_row_demeaned, fs=1600, nperseg=nperseg) # Returns len 2 tuple
            welch_list.append(np.log(pxx[1]/max(pxx[1]))) # Feature Engineering
            
            building_features[experiment_str] = np.array(welch_list)

            welch_list.clear()


def data_sequencing(building_features, vmd):
    num_sensors= len(building_features.keys())
    num_experiments = len(building_features['Sensor1'].keys())

    if vmd >= 1:
        building_features_seq = dict()
        for i in range(1,num_sensors+1):
            sensor_str = f'Sensor{i}'
            
            stack_list = []
            for j in range(1,num_experiments+1):
                experiment_str = f'Experiment{j}'
                stack_list.append(building_features[sensor_str][experiment_str])
                
            building_features_seq[sensor_str] = np.hstack(stack_list)
        return building_features_seq 

    if vmd == 0: 
        return np.hstack([building_features[f'Experiment{i+1}'] for i in range(num_experiments)]).squeeze()
        
