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

    means = dict(); stds = dict(); maxes = dict(); lats = dict(); sums = dict(); lats_pro = dict()

    for s_t in df.groupby(['subj','task']):
        subj, task = s_t[0]
        #load data
        filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX.mat')
        data_dict = loadmat.loadmat(filename)

        active_elecs, Params, srate, RTs, data = [data_dict.get(k) for k in ['active_elecs','Params','srate','RTs','data']]

        for row in s_t[1].itertuples():
            _, _, subj, task, cluster, pattern, elec, start_idx, end_idx, start_idx_resp, end_idx_resp = row

            #define start and end indices based on electrode type
            if any([(pattern == 'S'), (pattern == 'sustained'), (pattern == 'S+sustained'), (pattern == 'SR')]):
                start = start_idx
                end = end_idx

            if pattern == 'R':
                start = start_idx_resp
                end = end_idx_resp

            if pattern == 'D':
                start = start_idx
                end = end_idx_resp

            bl_st = Params['bl_st']
            start = start + abs(bl_st)
            end = end + abs(bl_st)

            data = data[active_elecs == elec, :, :].squeeze() #single trials for that electrode

            #calculate stats (single trials)
            means[elec] = data[:,start:end].mean(axis = 1)
            stds[elec] = data[:,start:end].std(axis = 1)
            maxes[elec] = data[:,start:end].max(axis = 1)
            lats[elec] =  data[:,start:end].argmax(axis = 1)
            sums[elec] = data[:, start:end].sum(axis = 1)
            lats_pro[elec] = lats / len(np.arange(start, end))

        #save stats (single trials)
        filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'data', ''.join([subj, '_', task, '.p']))
        data_dict = {'elec': elec, 'pattern':pattern, 'lats_pro': lats_pro, 'sums':sums, 'means':means, 'stds':stds, 'maxes':maxes, 'lats':lats, 'cdata': cdata, 'start_idx': start_idx, 'end_idx':end_idx, 'srate': srate, 'bl_st':bl_st,'RTs':RTs}

        with open(filename, 'w') as f:
            pickle.dump(data_dict, f)
            f.close()

if __name__ == '__main__':
    shadeplots_elecs_stats()
