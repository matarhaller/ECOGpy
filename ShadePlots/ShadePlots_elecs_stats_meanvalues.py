from __future__ import division
import pandas as pd
import os
import numpy as np
import sys
import cPickle as pickle
import loadmat
import pdb
from scipy import stats

def shadeplots_elecs_stats(resplocked = False):
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

    #df = df.query("subj not in ['GP27', 'GP44', 'ST28']") #drop unused subjects

    if resplocked:
        for s_t in df.groupby(['subj','task']):

            subj, task = s_t[0]
            #load data
            filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent.mat') 
            data_dict = loadmat.loadmat(filename)

            active_elecs, Params, srate, RT, data_trials = [data_dict.get(k) for k in ['active_elecs','Params','srate','RTs','data_percent']]
            srate = float(srate)
            data_all = data_trials.mean(axis = 1) #mean across trials, (new shape is elecs x time)
            
            bl_st = -500/1000*srate

            medians, means, stds, maxes, lats, lats_pro, RTs, mins, lats_min, RTs_median, RTs_min = [dict() for i in range(11)]

            RT = RT  + abs(bl_st) #RTs are calculated from stim onset, need to account for bl in HG_elecMTX_percent 

            for row in s_t[1].itertuples():
                _, subj, task, cluster, pattern, elec, start_idx, end_idx, start_idx_resp, end_idx_resp = row
                eidx = np.in1d(active_elecs, elec)
                data = data_trials[eidx,:].squeeze()

                #only do response electrodes
                if pattern == 'R': 
                    start_idx_resp = start_idx_resp
                    end_idx_resp = end_idx_resp

                    if start_idx_resp == end_idx_resp:
                        continue  #for inactive R elecs (not clear why on spreadsheet)

                    #create data matrix
                    data_resp = np.empty((data_trials.shape[1], end_idx_resp-start_idx_resp))
                    for j, r in enumerate(RT):
                        tmp = data[j, r + start_idx_resp : r + end_idx_resp]
                        data_resp[j,:] = tmp
                    data_resp = data_resp.mean(axis = 1) #mean acros trials, new shape is elecs x time

                    #calculate stats (mean trace)
                    means[elec] = data_resp.mean()
                    stds[elec] = data_resp.std()
                    maxes[elec] = data_resp.max()
                    lats[elec] = (data_resp.argmax()+1)/srate*1000
                    lats_min[elec] = (data_resp.argmin()+1)/srate*1000 #convert to ms
                    medians[elec] = stats.nanmedian(data_resp)
                    mins[elec] = data_resp.min()
                    RTs[elec] = (RT+Params['bl_st']/1000*srate).mean()/srate*1000 #from stimulus onset (adjusted for all subjects)
                    RTs_median[elec] = np.median(RT+Params['bl_st']/1000*srate)/srate*1000 #from stimulus onset (adjusted for all subjects)
                    RTs_min[elec] = np.min(RT+Params['bl_st']/1000*srate)/srate*1000 #from stimulus onset (adjusted for all subjects)


                #save stats (mean traces)
                filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'smoothed', 'mean_traces', 'data', ''.join([subj, '_', task, '_resplocked.p']))
                data_dict = {'means':means, 'stds':stds, 'maxes':maxes, 'lats':lats, 'srate': srate, 'bl_st':bl_st, 'RTs':RTs, 'medians' : medians, 'mins': mins, 'lats_min':lats_min, 'RTs_median': RTs_median, 'RTs_min': RTs_min}

                with open(filename, 'w') as f:
                    pickle.dump(data_dict, f)
                    f.close()

                #update csv file        
                for k in data_dict.keys():
                    if k in ['bl_st', 'srate','active_elecs']:
                        data_dict.pop(k, None)

                df_values = pd.DataFrame(data_dict)

                #save dataframe with values for all elecs for subject/task
                filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'smoothed', 'mean_traces', 'csv_files', '_'.join([subj, task, 'resplocked']) + '.csv')
                df_values.to_csv(filename)

                
    else: #not response locked 
        for s_t in df.groupby(['subj','task']):

            subj, task = s_t[0]

            #if ((subj == 'ST1') and (task == 'SelfAud') and (cluster == 2)): #drop bc garbage cluster
            #    continue

            #load data
            filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent.mat') 
            data_dict = loadmat.loadmat(filename)

            active_elecs, Params, srate, RT, data_trials = [data_dict.get(k) for k in ['active_elecs','Params','srate','RTs','data_percent']]
            srate = float(srate)
            data_all = data_trials.mean(axis = 1) #mean across trials, (new shape is elecs x time)
            bl_st = -500/1000*srate #in data points

            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'unsmoothed', 'data', ''.join([subj, '_', task, '.p'])) #for medians and means
            data_dict = pickle.load(open(filename, 'rb')) #keys are medians, means, for single trial values

            medians, means, stds, maxes, lats, RTs, mins, lats_min, RTs_median, RTs_min, lats_static, lats_min_static, lats_semi_static = [dict() for i in range(13)]

            RT = RT + abs(bl_st) #RTs are calculated from stim/cue onset, need to account for bl in HG_elecMTX_percent 

            for row in s_t[1].itertuples():
                _, subj, task, elec, pattern, cluster, start_idx, end_idx, start_idx_resp, end_idx_resp = row #in datapoints
                eidx = np.in1d(active_elecs, elec)
                data = data_all[eidx,:].squeeze() #mean trace

               
                #define start and end indices based on electrode type
                if any([(pattern == 'S'), (pattern == 'sustained'), (pattern == 'S+sustained'), (pattern == 'SR')]):
                    start_idx = start_idx + abs(bl_st)
                    end_idx = end_idx + abs(bl_st)

                if pattern == 'R': #fixed so can use stim locked onsets/offsets
                    start_idx = start_idx + abs(bl_st)
                    end_idx = end_idx + abs(bl_st)
                    
                if pattern == 'D':
                    start_idx = start_idx + abs(bl_st)
                    end_idx = np.median(RT) + end_idx_resp

                if start_idx == end_idx:
                    continue  #for inactive R elecs (not clear why on spreadsheet)

                #calculate stats (mean trace)
                means[elec] = np.nanmean(data_dict['means'][elec]) #from single trials
                medians[elec] = np.nanmean(data_dict['medians'][elec]) #from single trials
                maxes[elec] = data[start_idx:end_idx].max()
                lats[elec] = (data[start_idx:end_idx].argmax()+1)/srate*1000
                lats_min[elec] = (data[start_idx:end_idx].argmin()+1)/srate*1000
                stds[elec] = data[start_idx:end_idx].std()
                mins[elec] = data[start_idx:end_idx].min()
                RTs[elec] = (RT+bl_st).mean()/srate*1000 #from stimulus onset (adjusted for all subjects)
                RTs_median[elec] = np.median(RT+bl_st)/srate*1000 #from stimulus onset (adjusted for all subjects)
                RTs_min[elec] = np.min(RT+bl_st)/srate*1000 #from stimulus onset (adjusted for all subjects)
                lats_static[elec] = (data[abs(bl_st)::].argmax()+1)/srate*1000 #from stimulus onset to end (adjusted for all subjects)
                lats_min_static[elec] = (data[abs(bl_st)::].argmin()+1)/srate*1000 #from stimulus onset to end (adjusted for all subjects)
                lats_semi_static[elec] = (data[start_idx::].argmax()+1)/srate*1000

            #save stats (mean traces)
            #filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'smoothed', 'mean_traces', 'data', ''.join([subj, '_', task, '.p']))
            data_dict = {'means':means, 'stds':stds, 'maxes':maxes, 'lats':lats, 'srate': srate, 'bl_st':bl_st, 'RTs':RTs, 'medians' : medians, 'mins': mins, 'lats_min':lats_min, 'RTs_median': RTs_median, 'RTs_min' : RTs_min, 'lats_static' : lats_static, 'lats_min_static' : lats_min_static, 'lats_semi_static' : lats_semi_static}

            #with open(filename, 'w') as f:
            #    pickle.dump(data_dict, f)
            #    f.close()

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
