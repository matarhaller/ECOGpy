from __future__ import division
import pandas as pd
import os
import numpy as np
import sys
import cPickle as pickle
import loadmat
import pdb
from scipy import stats

def shadeplots_elecs_stats_surr(id_num = 99):

    """ 
    calculates params per electrode on surrogate data. Surrogate data is HG windows concatenated and circshifted. Only active HG included.
    calculates mean, peak, latency, and std per trial for all electrodes in an active cluster - added medians and coefficient of variation and mins
    uses windows for individual electrodes from PCA/Stats/single_electrode_windows_csvs/single_electrode_windows_withdesignation_EDITED.csv
    saves pickle file with numbers per trial in ShadePlots_hclust/elecs/significance_windows/unsmoothed
    Uses unsmoothed data
    No latencies for duration elecs
    Added fake data with trial index 12/18/14
    """

    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'

    saveDir_csv = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'unsmoothed', 'csv_files', 'orig', 'surr_' + str(id_num))
    saveDir_data= os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'unsmoothed', 'data', 'surr_' + str(id_num))

    if not(os.path.exists(saveDir_csv)) and not(os.path.exists(saveDir_data)):
        os.mkdir(saveDir_csv)
        os.mkdir(saveDir_data)
        print('making:\n%s\n%s' %(saveDir_csv, saveDir_data))

    else:
        print('either %s\n or %s\n already exists!\nterminating...' %(saveDir_csv, saveDir_data))
        return

    filename = os.path.join(SJdir,'PCA', 'Stats', 'single_electrode_windows_csvs', 'single_electrode_windows_withdesignation_EDITED.csv')
    df = pd.read_csv(filename)

    for s_t in df.groupby(['subj','task']):

        subj, task = s_t[0]
        #load data
        filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent_unsmoothed.mat')
        data_dict = loadmat.loadmat(filename)

        active_elecs, Params, srate, RT, data_all = [data_dict.get(k) for k in ['active_elecs','Params','srate','RTs','data_percent']]
        bl_st = Params['bl_st']
        bl_st = bl_st/1000*srate

        if task in ['DecisionAud', 'DecisionVis']:
            bl_st = 500/1000*srate #remove cue from baseline - start/end_idx are relative to cue onset) - change 12/24 - okay with RT 12/25
        
        RT = RT + abs(bl_st) #RTs are calculated from stim/cue onset, need to account for bl in HG_elecMTX_percent (for 500, not 1000 baseline 12/25)

        maxes_idx, medians_idx, cofvar, maxes_rel, medians, means, stds, maxes, lats, sums, lats_pro, RTs, num_dropped, mins, lats_min = [dict() for i in range(15)]

        for row in s_t[1].itertuples():
            _, _, subj, task, cluster, pattern, elec, start_idx, end_idx, start_idx_resp, end_idx_resp, _, _ = row
            eidx = np.in1d(active_elecs, elec)
            data = data_all[eidx,:,:].squeeze()

            st_resp = 0

            #define start and end indices based on electrode type
            if any([(pattern == 'S'), (pattern == 'sustained'), (pattern == 'S+sustained'), (pattern == 'SR')]):
                start_idx = start_idx + abs(bl_st)
                end_idx = end_idx + abs(bl_st)
                if start_idx == end_idx:
                	continue #for SR elecs that dont' have stimlocked (CP9, e91)
                
                print('%s %s %i %s\n' %(subj, task, elec, pattern))
                
                data_idx = np.ones_like(data[:, start_idx:end_idx])
                data_idx = (data_idx.transpose() * range(data_idx.shape[0])).transpose() #each trial labeled by trial number

                #make surrogate dataset based on activity window
                data_surr = data[:, start_idx:end_idx].flatten() #take HG windows
                randidx = np.random.randint(len(data_surr))
                data_surr = np.roll(data_surr, randidx) #cirshift data 
                data_surr = data_surr.reshape((data.shape[0], -1)) #reshape data into matrix                

                data_idx = np.roll(data_idx.flatten(), randidx) #circshift trial labels
                data_idx = data_idx.reshape((data.shape[0], -1))

                #calculate stats (single trials)
                means[elec] = data_surr.mean(axis = 1)
                stds[elec] = data_surr.std(axis = 1)
                maxes[elec] = data_surr.max(axis = 1)
                lats[elec] = data_surr.argmax(axis = 1)
                lats_min[elec] = data_surr.argmin(axis = 1)

                sums[elec] = data_surr.sum(axis = 1)
                lats_pro[elec] = lats[elec] / len(np.arange(start_idx, end_idx))
                RTs[elec] = RT
                
                medians[elec] = stats.nanmedian(data_surr, axis = 1)
                maxes_rel[elec] = maxes[elec]-means[elec]
                cofvar[elec] = stds[elec]/means[elec]
                mins[elec] = data_surr.min(axis = 1)
        
                medians_idx[elec] = stats.nanmedian(data_idx, axis = 1)
                maxes_idx[elec] = data_idx.max(axis = 1)

            if pattern == 'R':
                start_idx_resp = start_idx_resp + abs(st_resp)
                end_idx_resp = end_idx_resp + abs(st_resp)

                if start_idx_resp == end_idx_resp:
                	continue  #for inactive R elecs (not clear why on spreadsheet)

                print('%s %s %i %s\n' %(subj, task, elec, pattern))

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
                    data_resp[nanidx,:] = np.nan

                    #drop nan from RTs
                    tmp_RT = np.ndarray.astype(RT, dtype = float)
                    tmp_RT[nanidx] = np.nan
                    RTs[elec] = tmp_RT
                    
                    #make index matrix
                    data_idx = np.ones_like(data_resp)
                    data_idx = (data_idx.transpose() * range(data_idx.shape[0])).transpose()

                    #make surrogate data
                    data_surr = data_resp.flatten() #take HG window
                    data_surr_drop = np.isnan(data_surr) #for dropping trials from data_idx
                    data_surr = data_surr[~np.isnan(data_surr)] #remove nan (also drops trials that are completely nan)
                    randidx = np.random.randint(len(data_surr))
                    data_surr = np.roll(data_surr, randidx) #circshift
                    data_surr = data_surr.reshape((data_resp.shape[0],-1)) #reshape
                    data_surr = np.insert(data_surr, nanidx, np.empty((1, data_surr.shape[1]))*np.nan, axis = 0) #insert nan rows (shape of _surr == _resp)

                    data_idx = data_idx.flatten() 
                    data_idx = data_idx[~data_surr_drop] #drop nan trials
                    data_idx = np.roll(data_idx, randidx) #circshift
                    data_idx = data_idx.reshape((data_resp.shape[0], -1)) #reshape
                    data_idx = np.insert(data_idx, nanidx, np.empty((1, data_resp.shape[1])) * np.nan, axis = 0) #insert nan rows (shape of _idx == _resp)

                else: 
                    #make surrogate data
                    data_surr = data_resp.flatten() #take HG window
                    data_surr_drop = np.isnan(data_surr) #for dropping trials from data_idx
                    data_surr = data_surr[~np.isnan(data_surr)] #remove nan 
                    randidx = np.random.randint(len(data_surr))
                    data_surr = np.roll(data_surr, randidx) #circshift
                    data_surr = data_surr.reshape((data_resp.shape[0],-1)) #reshape                    
                    RTs[elec] = RT

                    data_idx = np.ones_like(data_resp)
                    data_idx = (data_idx.transpose() * range(data_idx.shape[0])).transpose() #each trial labeled by trial number
                    data_idx = data_idx.flatten()
                    data_idx = data_idx[~data_surr_drop] #drop nan trials based on data_surr
                    data_idx = np.roll(data_idx, randidx) #circshift
                    data_idx = data_idx.reshape((data_resp.shape[0], -1)) #reshape
                    
                #reshape data_surr with nan
                data_resp_surr = np.empty_like(data_resp)
                for j in range(data_surr.shape[0]):
                    tmp = data_surr[j,:]
                    tmp = np.pad(tmp, (0, data_resp.shape[1]-len(tmp)), 'constant', constant_values = -999)
                    data_resp_surr[j,:] = tmp
                data_resp_surr[data_resp_surr == -999] = np.nan

                #reshape data_idx with nan
                data_idx_surr = np.empty_like(data_resp)
                for j in range(data_idx.shape[0]):
                    tmp = data_idx[j,:]
                    tmp = np.pad(tmp, (0, data_resp.shape[1]-len(tmp)), 'constant', constant_values = -999)
                    data_idx_surr[j,:] = tmp
                data_idx_surr[data_idx_surr == -999] = np.nan
                                
                #calculate params for (single trials)
                means[elec] = np.nanmean(data_resp_surr, axis = 1)
                stds[elec] = np.nanstd(data_resp_surr, axis = 1)
                maxes[elec] = np.nanmax(data_resp_surr, axis = 1)
                sums[elec] = np.nansum(data_resp_surr, axis = 1)

                medians[elec] = stats.nanmedian(data_resp_surr, axis = 1)
                maxes_rel[elec] = maxes[elec]-means[elec]
                cofvar[elec] = stds[elec]/means[elec]
                mins[elec] = np.nanmin(data_resp_surr, axis = 1)

                medians_idx[elec] = stats.nanmedian(data_idx_surr, axis = 1)
                maxes_idx[elec] = np.nanmax(data_idx_surr, axis = 1)

            if pattern == 'D':
                start_idx = start_idx + abs(bl_st)
                end_idx_resp = end_idx_resp + abs(st_resp)
                
                print('%s %s %i %s\n' %(subj, task, elec, pattern))

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
                    data_dur[nanidx, :] = np.nan

                    #drop nan from RTs
                    tmp_RT = np.ndarray.astype(RT, dtype = float)
                    tmp_RT[nanidx] = np.nan
                    RTs[elec] = tmp_RT
                else:
                    RTs[elec] = RT     

                #make surrogate data
                data_surr = data_dur.flatten() #take HG window
                data_surr_drop = np.isnan(data_surr) #for data_idx dropping points based on data_surr
                data_surr = data_surr[~np.isnan(data_surr)] #drop nan datapoints
                randidx = np.random.randint(len(data_surr))
                data_surr = np.roll(data_surr, randidx) #circshift

                #reshape data_surr with nan buffer at end
                data_dur_surr = np.empty_like(data_dur)
                start = 0
                for j in range(data_dur.shape[0]):
                    trial_length = sum(~np.isnan(data_dur[j,:]))
                    if j>0:
                        start = end
                    end = start + trial_length
                    if trial_length>0: #not a nan trial
                        tmp = data_surr[start:end]
                        tmp = np.pad(tmp, (0, data_dur.shape[1]-len(tmp)), 'constant', constant_values = -999)
                        data_dur_surr[j,:] = tmp
                    else: #nan trial
                        data_dur_surr[j,:] = -999
                data_dur_surr[data_dur_surr == -999] = np.nan

                #make surrogate data for idx
                data_idx = np.ones_like(data_dur)
                data_idx = (data_idx.transpose() * range(data_idx.shape[0])).transpose() #trials x time with index for trial data
                data_idx = data_idx.flatten() #take HG window
                data_idx = data_idx[~data_surr_drop]#remove datapoints that are missing in data_surr (to get same number of points)
                data_idx = np.roll(data_idx, randidx) #circshift

                #reshape data_idx with nan
                data_dur_idx = np.empty_like(data_dur)
                start = 0
                for j in range(data_dur.shape[0]):
                    trial_length = sum(~np.isnan(data_dur[j,:]))
                    if j>0:
                        start = end
                    end = start + trial_length
                    if trial_length>0: #not a nan trial
                        tmp = data_idx[start:end]
                        tmp = np.pad(tmp, (0, data_dur.shape[1]-len(tmp)), 'constant', constant_values = -999)
                        data_dur_idx[j,:] = tmp
                    else: #nan trial
                        data_dur_idx[j,:] = -999
                data_dur_idx[data_dur_idx == -999] = np.nan

                #calculate params for single trials
                means[elec] = np.nanmean(data_dur_surr, axis = 1)
                stds[elec] = np.nanstd(data_dur_surr, axis = 1)
                maxes[elec] = np.nanmax(data_dur_surr, axis = 1)
                sums[elec] = np.nansum(data_dur_surr, axis = 1)

                medians[elec] = stats.nanmedian(data_dur_surr, axis = 1)
                maxes_rel[elec] = maxes[elec] - means[elec]
                cofvar[elec] = stds[elec]/means[elec]
                mins[elec] = np.nanmin(data_dur_surr, axis = 1)

                medians_idx[elec] = stats.nanmedian(data_dur_idx, axis = 1)
                maxes_idx[elec] = np.nanmax(data_dur_idx, axis = 1)

        #save stats (single trials)
        filename = os.path.join(saveDir_data, ''.join([subj, '_', task, '_surr.p']))
        data_dict = {'active_elecs': active_elecs, 'lats_pro': lats_pro, 'sums':sums, 'means':means, 'stds':stds, 'maxes':maxes, 'lats':lats, 'srate': srate, 'bl_st':bl_st,'RTs':RTs, 'dropped':num_dropped, 'maxes_rel' : maxes_rel, 'medians' : medians, 'variations': cofvar, 'mins': mins, 'lats_min':lats_min, 'medians_idx':medians_idx, 'maxes_idx':maxes_idx}

        with open(filename, 'w') as f:
            pickle.dump(data_dict, f)
            f.close()

        #save csv file 
        for k in data_dict.keys():
            if k in ['bl_st', 'srate','active_elecs', 'dropped']:
                continue
            data = pd.DataFrame(data_dict[k])
        
            filename = os.path.join(saveDir_csv, '_'.join([subj, task, k]) + '_surr.csv') #drops trials
            data.to_csv(filename, index = False)

    #save dataframe with dropped trials
    filename = os.path.join(SJdir,'PCA', 'Stats', 'single_electrode_windows_withdesignation_EDITED_dropped_surr_' + str(id_num) + '.csv')
    df.to_csv(filename)    
    
if __name__ == '__main__':
    shadeplots_elecs_stats_surr()
