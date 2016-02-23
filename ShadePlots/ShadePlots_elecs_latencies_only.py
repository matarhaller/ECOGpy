from __future__ import division
import pandas as pd
import os
import numpy as np
import sys
import cPickle as pickle
import loadmat
import pdb
from scipy import stats

def shadeplots_elecs_stats():
    """ 
    calculates mean, max, min, latency, median, and std on the mean trace for trial for all electrodes in an active cluster
    OLD - uses electrodes and windows from PCA/Stats/single_electrode_windows_withdesignation_EDITED.csv
    NOW - uses electrodes and windows from PCA/csvs_FINAL/final_windows.csv (after going through and editing them)
    calculates both stimulus and response locked parameters
    """

    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'

    #filename = os.path.join(SJdir,'PCA', 'Stats', 'single_electrode_windows_csvs', 'single_electrode_windows_withdesignation_EDITED.csv')
    filename = os.path.join(SJdir, 'PCA', 'csvs_FINAL', 'final_windows.csv')
    df = pd.read_csv(filename)
                

    for s_t in df.groupby(['subj','task']):

        subj, task = s_t[0]

        #load data
        filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent.mat') 
        data_dict = loadmat.loadmat(filename)

        active_elecs, Params, srate, RT, data_trials = [data_dict.get(k) for k in ['active_elecs','Params','srate','RTs','data_percent']]
        srate = float(srate)
        data_all = data_trials.mean(axis = 1) #mean across trials, (new shape is elecs x time)
        bl_st = -500/1000*srate #in data points

        maxes, lats, RTs, RTs_median, RTs_min, lats_static, lats_min_static, lats_semi_static = [dict() for i in range(8)]

        RT = RT + abs(bl_st) #RTs are calculated from stim/cue onset, need to account for bl in HG_elecMTX_percent 

        for row in s_t[1].itertuples():
            _, subj, task, elec, pattern, cluster, start_idx, end_idx, start_idx_resp, end_idx_resp = row #in datapoints
            eidx = np.in1d(active_elecs, elec)
            data = data_all[eidx,:].squeeze() #mean trace


            #define start and end indices based on electrode type
            if any([(pattern == 'S'), (pattern == 'sustained'), (pattern == 'S+sustained'), (pattern == 'SR')]):
                start_idx = start_idx + abs(bl_st)
                end_idx = end_idx + abs(bl_st)

            if pattern == 'R': 
                start_idx = start_idx + abs(bl_st)
                end_idx = end_idx + abs(bl_st)

            if pattern == 'D':
                start_idx = start_idx + abs(bl_st)
                end_idx = np.median(RT) + end_idx_resp

            if start_idx == end_idx:
                continue  #for SR elecs that only have response activity - don't calculate a mean value

            #calculate stats (mean trace)

            maxes[elec] = data[start_idx:end_idx].max()
            lats[elec] = (data[start_idx:end_idx].argmax()+1)/srate*1000 #within HG window

            RTs[elec] = (RT+bl_st).mean()/srate*1000 #from stimulus onset (adjusted for all subjects)
            RTs_median[elec] = np.median(RT+bl_st)/srate*1000 #from stimulus onset (adjusted for all subjects)
            RTs_min[elec] = np.min(RT+bl_st)/srate*1000 #from stimulus onset (adjusted for all subjects)

            lats_static[elec] = (data[abs(bl_st)::].argmax()+1)/srate*1000 #from stimulus onset to end (adjusted for all subjects)
            lats_semi_static[elec] = (data[start_idx::].argmax()+1)/srate*1000 #from HG onset


        data_dict = {'maxes':maxes, 'lats':lats, 'RTs':RTs, 'RTs_median': RTs_median, 'RTs_min' : RTs_min, 'lats_static' : lats_static, 'lats_semi_static' : lats_semi_static}

        #update csv file        
        for k in data_dict.keys():
            if k in ['bl_st', 'srate','active_elecs']:
                data_dict.pop(k, None)

        df_values = pd.DataFrame(data_dict)

        #save dataframe with values for all elecs for subject/task - later combined into mean_traces_all_elecs.csv in elec_values.ipynb
        filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'smoothed', 'mean_traces', 'csv_files', '_'.join([subj, task]) + '.csv')
        df_values.to_csv(filename)
       
if __name__ == '__main__':
    shadeplots_elecs_stats()
