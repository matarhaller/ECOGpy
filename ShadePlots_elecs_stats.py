from __future__ import division
import pandas as pd
import os
import numpy as np
import sys
import cPickle as pickle
import loadmat

def shadeplots_elecs_stats():
    """ 
    calculates mean, peak, latency, and std per trial for all electrodes in an active cluster
    uses windows for individual electrodes from PCA/Stats/single_electrode_windows_withdesignation.csv
    saves pickle file with numbers per trial in ShadePlots_hclust/elecs/significance_windows
    """

    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'

    filename = os.path.join(SJdir,'PCA', 'Stats', 'single_electrode_windows_withdesignation.csv')
    df = pd.read_csv(filename)

    filename = os.path.join(SJdir, 'PCA', 'duration_dict_500_FINAL', 'stim_resp_cluster_windows_withduration_EDITED.xlsx') #to get windows for mean cluster trace
    df_cluster = pd.read_excel(filename)

    for s_t in df.groupby(['subj','task']):
        subj, task = s_t[0]
        #load data
        filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX.mat')
        data_dict = loadmat.loadmat(filename)

        active_elecs, Params, srate, RTs, data_all = [data_dict.get(k) for k in ['active_elecs','Params','srate','RTs','data']]
        bl_st = Params['bl_st']
        means = dict(); stds = dict(); maxes = dict(); lats = dict(); sums = dict(); lats_pro = dict()

        for row in s_t[1].itertuples():
            _, _, subj, task, cluster, pattern, elec, start_idx, end_idx, start_idx_resp, end_idx_resp = row
            eidx = np.in1d(active_elecs, elec)
            data = data_all[eidx,:,:].squeeze()

            st_resp = 0

            #define start and end indices based on electrode type
            if any([(pattern == 'S'), (pattern == 'sustained'), (pattern == 'S+sustained'), (pattern == 'SR')]):
                start_idx = start_idx + abs(bl_st)
                end_idx = end_idx + abs(bl_st)
                if start_idx == end_idx:
                	continue #for SR elecs that dont' have stimlocked (CP9, e91)

                #calculate stats (single trials)
                means[elec] = data[:,start_idx:end_idx].mean(axis = 1)
                stds[elec] = data[:,start_idx:end_idx].std(axis = 1)
                maxes[elec] = data[:,start_idx:end_idx].max(axis = 1)
                lats[elec] = data[:,start_idx:end_idx].argmax(axis = 1)
                sums[elec] = data[:, start_idx:end_idx].sum(axis = 1)
                lats_pro[elec] = lats[elec] / len(np.arange(start_idx, end_idx))

            if pattern == 'R':
                start_idx_resp = start_idx_resp+abs(st_resp)
                end_idx_resp = end_idx_resp+abs(st_resp)

                if start_idx_resp == end_idx_resp:
                	continue  #for inactive R elecs (not clear why on spreadsheet)

                #create data matrix
                #data_resp = np.empty(data.shape)
                data_resp = np.empty((RTs.shape[0], data.shape[1]))
                for j, r in enumerate(RTs):
                    tmp = data[j, r + start_idx_resp : r + end_idx_resp]
                    tmp = np.pad(tmp, (0, data.shape[1]-len(tmp)), 'constant', constant_values = -999)
                    data_resp[j,:] = tmp
                data_resp[data_resp == -999] = np.nan

                #calculate stats (single trials)
                means[elec] = np.nanmean(data_resp, axis = 1)
                stds[elec] = np.nanstd(data_resp, axis = 1)
                maxes[elec] = np.nanmax(data_resp, axis = 1)
                sums[elec] = np.nansum(data_resp, axis = 1)

                nanidx = np.isnan(np.nansum(data_resp, axis = 1)) #if start > end
                if np.any(nanidx):
                    data_resp[nanidx,0] = -999
                    tmp_lat = np.nanargmax(data_resp, axis = 1)
                    tmp_lat = np.ndarray.astype(tmp_lat, dtype = float)
                    tmp_lat[nanidx] = np.nan
                    lats[elec] = tmp_lat
                    lats_pro[elec] = tmp_lat / np.sum(~np.isnan(data_resp), axis = 1)
                else:
                    lats[elec] = np.nanargmax(data_resp, axis = 1)
                    lats_pro[elec] = np.nanargmax(data_resp, axis = 1) / np.sum(~np.isnan(data_resp), axis = 1)

            if pattern == 'D':
                start_idx = start_idx + abs(bl_st)
                end_idx_resp = end_idx_resp+abs(st_resp)

                start_idx_cluster = df_cluster[(df_cluster.subj == subj) & (df_cluster.task == task) & (df_cluster.cluster == cluster)].start_idx 
                end_idx_resp_cluster = df_cluster[(df_cluster.subj == subj) & (df_cluster.task == task) & (df_cluster.cluster == cluster)].end_idx_resp
                
                #create data matrices
                #data_dur = np.empty(data.shape)
                data_dur = np.empty((RTs.shape[0], data.shape[1]))
                for j, r in enumerate(RTs):
                    tmp = data[j, start_idx : r + end_idx_resp]
                    tmp = np.pad(tmp, (0, data.shape[1]-len(tmp)), 'constant', constant_values = -999)
                    data_dur[j,:] = tmp
                data_dur[data_dur == -999] = np.nan

                #calculate stats (single trials)
                means[elec] = np.nanmean(data_dur, axis = 1)
                stds[elec] = np.nanstd(data_dur, axis = 1)
                maxes[elec] = np.nanmax(data_dur, axis = 1)
                sums[elec] = np.nansum(data_dur, axis = 1)
		                
                nanidx = np.isnan(np.nansum(data_dur, axis = 1)) #if start > end
                if np.any(nanidx):
                    data_dur[nanidx,0] = -999
                    tmp_lat = np.nanargmax(data_dur, axis = 1)
                    tmp_lat = np.ndarray.astype(tmp_lat, dtype = float)
                    tmp_lat[nanidx] = np.nan
                    lats[elec] = tmp_lat
                    lats_pro[elec] = tmp_lat / np.sum(~np.isnan(data_dur), axis = 1)

                else:
                    lats[elec] = np.nanargmax(data_dur, axis = 1)
                    lats_pro[elec] = np.nanargmax(data_dur, axis = 1) / np.sum(~np.isnan(data_dur), axis = 1)
                
        #save stats (single trials)
        filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'data', ''.join([subj, '_', task, '.p']))
        data_dict = {'active_elecs': active_elecs, 'lats_pro': lats_pro, 'sums':sums, 'means':means, 'stds':stds, 'maxes':maxes, 'lats':lats, 'srate': srate, 'bl_st':bl_st,'RTs':RTs}

        with open(filename, 'w') as f:
            pickle.dump(data_dict, f)
            f.close()

if __name__ == '__main__':
    shadeplots_elecs_stats()