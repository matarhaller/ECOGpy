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
    calculates mean, peak, latency, and std per trial for all electrodes in an active cluster - added medians and coefficient of variation and mins
    uses windows for individual electrodes from PCA/Stats/single_electrode_windows_withdesignation.csv
    saves pickle file with numbers per trial in ShadePlots_hclust/elecs/significance_windows
    *** runs on unsmoothed data (12/11/14)***
    """

    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'

    #filename = os.path.join(SJdir,'PCA', 'Stats', 'single_electrode_windows_csvs', 'single_electrode_windows_withdesignation_EDITED.csv')
    filename = os.path.join(SJdir, 'PCA', 'csvs_FINAL', 'mean_traces_all_subjs_dropSR.csv')
    df = pd.read_csv(filename)

    for s_t in df.groupby(['subj','task']):

        subj, task = s_t[0]
        #load data
        #filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent_unsmoothed.mat')
        filename = os.path.join(SJdir, 'Subjs',subj, task, 'HG_elecMTX_zscore.mat')
        data_dict = loadmat.loadmat(filename)

        active_elecs, Params, srate, RT, data_all = [data_dict.get(k) for k in ['active_elecs','Params','srate','RTs','data_zscore']]
        bl_st = Params['bl_st']
        bl_st = bl_st/1000*srate
        
        if task in ['DecisionAud', 'DecisionVis']:
            bl_st = 500/1000*srate #remove cue from baseline - start/end_idx are relative to cue onset) - change 12/24 - okay with RT 12/25

        cofvar, maxes_rel, medians, means, stds, maxes, lats, sums, lats_pro, RTs, num_dropped, mins, lats_min = [dict() for i in range(13)]
        
        RT = RT + abs(bl_st) #RTs are calculated from stim/cue onset, need to account for bl in HG_elecMTX_percent (for 500, not 1000 baseline 12/25)

        for row in s_t[1].itertuples():
            _, subj, task, elec, pattern, cluster, start_idx, end_idx, start_idx_resp, end_idx_resp, RTs_values, RTs_median, RTs_min, lats_values, lats_semi_static, lats_static, max_vals, ROI = row
            eidx = np.in1d(active_elecs, elec)
            data = data_all[eidx,:,:].squeeze()

            st_resp = 0

            #define start and end indices based on electrode type
            if any([(pattern == 'S'), (pattern == 'sustained'), (pattern == 'S+sustained'), (pattern == 'SR')]):
                start_idx = start_idx + abs(bl_st)
                end_idx = end_idx + abs(bl_st)
                if start_idx == end_idx:
                	continue #for SR elecs that dont' have stimlocked (CP9, e91)

                num_to_drop = 0

                #calculate stats (single trials)
                means[elec] = data[:,start_idx:end_idx].mean(axis = 1)
                stds[elec] = data[:,start_idx:end_idx].std(axis = 1)
                maxes[elec] = data[:,start_idx:end_idx].max(axis = 1)
                lats[elec] = data[:,start_idx:end_idx].argmax(axis = 1)
                lats_min[elec] = data[:, start_idx:end_idx].argmin(axis = 1)

                sums[elec] = data[:, start_idx:end_idx].sum(axis = 1)
                lats_pro[elec] = lats[elec] / len(np.arange(start_idx, end_idx))
                RTs[elec] = RT
                num_dropped[elec] = num_to_drop

                medians[elec] = stats.nanmedian(data[:,start_idx:end_idx], axis = 1)
                maxes_rel[elec] = maxes[elec]-means[elec]
                cofvar[elec] = stds[elec]/means[elec]
                mins[elec] = data[:,start_idx:end_idx].min(axis = 1)

                #update dataframe
                #ix = np.where([(df.subj == subj) & (df.task == task) & (df.elec == elec)])[1][0]
                #df.ix[ix,'dropped'] = num_to_drop


            if pattern == 'R':
                start_idx_resp = start_idx_resp + abs(st_resp)
                end_idx_resp = end_idx_resp + abs(st_resp)

                if start_idx_resp == end_idx_resp:
                	continue  #for inactive R elecs (not clear why on spreadsheet)

                #create data matrix
                data_resp = np.empty(data.shape)
                for j, r in enumerate(RT):
                    tmp = data[j, r + start_idx_resp : r + end_idx_resp]
                    tmp = np.pad(tmp, (0, data.shape[1]-len(tmp)), 'constant', constant_values = -999)
                    data_resp[j,:] = tmp
                data_resp[data_resp == -999] = np.nan

                nanidx = np.isnan(np.nanmean(data_resp, axis = 1)) #if start > end
                if np.any(nanidx):

                    #drop equivalent number of long RTs
                    num_to_drop = np.sum(nanidx)
                    i = np.argpartition(RT, -num_to_drop)[-num_to_drop :] #find the indices of the longest RTs
                    nanidx[i] = True #mark the long trials as bad too
                    num_dropped[elec] = num_to_drop * 2 #dropping both ends of RT distribution

                    #calculate params for (single trials)
                    data_resp[nanidx,:] = np.nan
                    means[elec] = np.nanmean(data_resp, axis = 1)
                    stds[elec] = np.nanstd(data_resp, axis = 1)
                    maxes[elec] = np.nanmax(data_resp, axis = 1)
                    sums[elec] = np.nansum(data_resp, axis = 1)

                    medians[elec] = stats.nanmedian(data_resp, axis = 1)
                    maxes_rel[elec] = maxes[elec]-means[elec]
                    cofvar[elec] = stds[elec]/means[elec]
                    mins[elec] = np.nanmin(data_resp, axis = 1)

                    data_resp[nanidx,0] = -999
                    tmp_lat = np.nanargmax(data_resp, axis = 1)
                    tmp_lat = np.ndarray.astype(tmp_lat, dtype = float)
                    tmp_lat[nanidx] = np.nan
                    lats[elec] = tmp_lat
                    lats_pro[elec] = tmp_lat / np.sum(~np.isnan(data_resp), axis = 1)

                    data_resp[nanidx,0] = 9999
                    tmp_lat = np.nanargmin(data_resp, axis = 1)
                    tmp_lat = np.ndarray.astype(tmp_lat, dtype = float)
                    tmp_lat[nanidx] = np.nan
                    lats_min[elec] = tmp_lat

                    tmp_RT = np.ndarray.astype(RT, dtype = float)
                    tmp_RT[nanidx] = np.nan
                    RTs[elec] = tmp_RT

                else:
                    num_to_drop = 0
                    num_dropped[elec] = num_to_drop
                    lats[elec] = np.nanargmax(data_resp, axis = 1)
                    lats_min[elec] = np.nanargmin(data_resp, axis = 1)

                    lats_pro[elec] = np.nanargmax(data_resp, axis = 1) / np.sum(~np.isnan(data_resp), axis = 1)
                    RTs[elec] = RT
                    means[elec] = np.nanmean(data_resp, axis = 1)
                    stds[elec] = np.nanstd(data_resp, axis = 1)
                    maxes[elec] = np.nanmax(data_resp, axis = 1)
                    sums[elec] = np.nansum(data_resp, axis = 1)
                    mins[elec] = np.nanmin(data_resp, axis = 1)

                    medians[elec] = stats.nanmedian(data_resp, axis = 1)
                    maxes_rel[elec] = maxes[elec] - means[elec]
                    cofvar[elec] = stds[elec]/means[elec]

                #update dataframe
                #ix = np.where([(df.subj == subj) & (df.task == task) & (df.elec == elec)])[1][0]
                #df.ix[ix,'dropped'] = num_to_drop * 2 #dropping both ends of RT distribution

            if pattern == 'D':
                start_idx = start_idx + abs(bl_st)
                end_idx_resp = end_idx_resp + abs(st_resp)

                #create data matrices
                data_dur = np.empty(data.shape)
                for j, r in enumerate(RT):
                    tmp = data[j, start_idx : r + end_idx_resp]
                    tmp = np.pad(tmp, (0, data.shape[1]-len(tmp)), 'constant', constant_values = -999)
                    data_dur[j,:] = tmp
                data_dur[data_dur == -999] = np.nan

		                
                nanidx = np.isnan(np.nanmean(data_dur, axis = 1)) #if start > end
                if np.any(nanidx):

                    #drop equivalent number of long RTs
                    num_to_drop = np.sum(nanidx)
                    i = np.argpartition(RT, -num_to_drop)[-num_to_drop :] #find the indices of the longest RTs
                    nanidx[i] = True #mark the long trials as bad too
                    num_dropped[elec] = num_to_drop * 2 #dropping both ends of RT distribution

                    #calculate params for single trials
                    data_dur[nanidx, :] = np.nan
                    means[elec] = np.nanmean(data_dur, axis = 1)
                    stds[elec] = np.nanstd(data_dur, axis = 1)
                    maxes[elec] = np.nanmax(data_dur, axis = 1)
                    sums[elec] = np.nansum(data_dur, axis = 1)

                    medians[elec] = stats.nanmedian(data_dur, axis = 1)
                    maxes_rel[elec] = maxes[elec] - means[elec]
                    cofvar[elec] = stds[elec]/means[elec]
                    mins[elec] = np.nanmin(data_dur, axis = 1)

                    data_dur[nanidx,0] = -999
                    tmp_lat = np.nanargmax(data_dur, axis = 1)
                    tmp_lat = np.ndarray.astype(tmp_lat, dtype = float)
                    tmp_lat[nanidx] = np.nan
                    lats[elec] = tmp_lat
                    lats_pro[elec] = tmp_lat / np.sum(~np.isnan(data_dur), axis = 1)

                    data_dur[nanidx, 0] = 9999
                    tmp_lat = np.nanargmin(data_dur, axis = 1)
                    tmp_lat = np.ndarray.astype(tmp_lat, dtype = float)
                    tmp_lat[nanidx] = np.nan
                    lats_min[elec] = tmp_lat

                    tmp_RT = np.ndarray.astype(RT, dtype = float)
                    tmp_RT[nanidx] = np.nan
                    RTs[elec] = tmp_RT
                else:
                    num_to_drop = 0
                    num_dropped[elec] = num_to_drop
                    means[elec] = np.nanmean(data_dur, axis = 1)
                    stds[elec] = np.nanstd(data_dur, axis = 1)
                    maxes[elec] = np.nanmax(data_dur, axis = 1)
                    sums[elec] = np.nansum(data_dur, axis = 1)

                    medians[elec] = stats.nanmedian(data_dur, axis = 1)
                    maxes_rel[elec] = maxes[elec] - means[elec]
                    cofvar[elec] = stds[elec]/means[elec]
                    mins[elec] = np.nanmin(data_dur, axis = 1)

                    lats[elec] = np.nanargmax(data_dur, axis = 1)
                    lats_min[elec] = np.nanargmin(data_dur, axis = 1)
                    lats_pro[elec] = np.nanargmax(data_dur, axis = 1) / np.sum(~np.isnan(data_dur), axis = 1)
                    RTs[elec] = RT

                #update dataframe
                #ix = np.where([(df.subj == subj) & (df.task == task) & (df.elec == elec)])[1][0]
                #df.ix[ix,'dropped'] = num_to_drop * 2 #dropping both ends of RT distribution

        #save stats (single trials)
        filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'unsmoothed', 'data', ''.join([subj, '_', task, '.p']))
        data_dict = {'active_elecs': active_elecs, 'lats_pro': lats_pro, 'sums':sums, 'means':means, 'stds':stds, 'maxes':maxes, 'lats':lats, 'srate': srate, 'bl_st':bl_st,'RTs':RTs, 'dropped':num_dropped, 'maxes_rel' : maxes_rel, 'medians' : medians, 'variations': cofvar, 'mins': mins, 'lats_min':lats_min}
        
        with open(filename, 'w') as f:
            pickle.dump(data_dict, f)
            f.close()

        #save csv file (without dropping trials)
        for k in data_dict.keys():
            if k in ['bl_st', 'srate','active_elecs', 'dropped']:
                continue
            data = pd.DataFrame(data_dict[k])
        
            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'zscore', 'csv_files', '_'.join([subj, task, k]) + '.csv')
            data.to_csv(filename, index = False)

    #save dataframe with dropped trials
    #filename = os.path.join(SJdir,'PCA', 'Stats', 'single_electrode_windows_withdesignation_EDITED_dropped_unsmoothed.csv')
    #df.to_csv(filename)
    
    
if __name__ == '__main__':
    shadeplots_elecs_stats()
