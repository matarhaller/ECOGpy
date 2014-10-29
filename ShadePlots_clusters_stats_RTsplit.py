from __future__ import division
import pandas as pd
import os
import numpy as np
import sys
import cPickle as pickle
from scipy import stats

def shadeplots_clusters_stats_RTsplit():
    """ 
    calculates mean, peak, latency, and std per trial
    saves pickle file with numbers per trial in ShadePlots_hclust/significance_windows
    """

    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'

    filename = os.path.join(SJdir,'PCA','duration_dict_500_FINAL', 'stim_resp_cluster_windows_withduration_EDITED.xlsx')
    df = pd.ExcelFile(filename)
    df = df.parse('stim_resp_cluster_windows_withd')

    mean_p_stim = list()
    std_p_stim = list()
    max_p_stim = list()
    lat_p_stim = list()
    sum_p_stim = list()
    mean_p_resp = list()
    std_p_resp = list()
    max_p_resp = list()
    sum_p_resp = list()
    lat_p_resp = list()
    lat_pro_p_stim = list()
    lat_pro_p_resp = list()
    subjs = list(); tasks = list(); clusts = list(); patterns = list()

    mean_slow_mean_stim = list(); mean_fast_mean_stim = list()
    std_fast_mean_stim = list(); std_slow_mean_stim = list()
    max_slow_mean_stim = list(); max_fast_mean_stim = list()
    lat_slow_mean_stim = list(); lat_fast_mean_stim = list()
    sum_slow_mean_stim = list(); sum_fast_mean_stim = list()
    lat_pro_slow_mean_stim = list(); lat_pro_fast_mean_stim = list()
    lat_pro_slow_std_stim = list(); lat_pro_fast_std_stim = list()
    lat_slow_std_stim = list(); lat_fast_std_stim = list()


    mean_slow_mean_resp = list(); mean_fast_mean_resp = list()
    std_fast_mean_resp = list(); std_slow_mean_resp = list()
    max_slow_mean_resp = list(); max_fast_mean_resp = list()
    lat_slow_mean_resp = list(); lat_fast_mean_resp = list()
    sum_slow_mean_resp = list(); sum_fast_mean_resp = list()
    lat_pro_slow_mean_resp = list(); lat_pro_fast_mean_resp = list()
    lat_pro_slow_std_resp = list(); lat_pro_fast_std_resp = list()
    lat_slow_std_resp = list(); lat_fast_std_resp = list()


    for row in df.itertuples():
        i, subj, task, cluster, pattern, start_idx, end_idx, start_idx_resp, end_idx_resp, _ = row
 
        #load data
        filename = os.path.join(SJdir, 'PCA','ShadePlots_hclust', 'significance_windows', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
        with open(filename, 'r') as x:
            data_dict = pickle.load(x)
            x.close()

        if any([(pattern == 'S'), (pattern == 'sustained'), (pattern == 'S+sustained')]):

            means, maxes, lats, lats_pro, sums, stds, RTs, medians, variations = [data_dict.get(k) for k in ['means','maxes','lats','lats_pro','sums','stds', 'RTs', 'medians','variations']]

            #calculate median RT to split trials
            slowidx = np.where(RTs > np.median(RTs))[0]
            fastidx = np.where(RTs < np.median(RTs))[0]

            if slowidx.shape > fastidx.shape:
                slowidx = slowidx[0:len(fastidx)]
            if slowidx.shape < fastidx.shape:
                fastidx = fastidx[0:len(slowidx)]

            #calculate stats (single trials)
            means_slow = means[slowidx]
            maxes_slow = maxes[slowidx]
            lats_slow = lats[slowidx]
            lats_pro_slow = lats_pro[slowidx]
            sums_slow = sums[slowidx]        
            stds_slow = stds[slowidx]
            medians_slow = medians[slowidx]
            vars_slow = variations[slowidx]

            means_fast = means[fastidx]
            maxes_fast = maxes[fastidx]
            lats_fast = lats[fastidx]
            lats_pro_fast = lats_pro[fastidx]
            sums_fast = sums[fastidx]
            stds_fast = stds[fastidx]
            medians_fast = medians[fastidx]
            vars_fast = variations[fastidx]

            #calculate ttests
            mean_p_stim.append(stats.ttest_ind(means_slow, means_fast)[1])
            std_p_stim.append(stats.ttest_ind(stds_slow, stds_fast)[1])
            max_p_stim.append(stats.ttest_ind(maxes_slow, maxes_fast)[1])
            lat_p_stim.append(stats.ttest_ind(lats_slow, lats_fast)[1])
            sum_p_stim.append(stats.ttest_ind(sums_slow, sums_fast)[1])
            lat_pro_p_stim.append(stats.ttest_ind(lats_pro_slow, lats_pro_fast)[1])
            medians_p_stim.append(stats.ttest_ind(medians_slow, medians_fast)[1])
            vars_p_stim.append(stats.ttest_ind(vars_slow, vars_fast)[1])

            mean_p_resp.append(np.nan)
            std_p_resp.append(np.nan)
            max_p_resp.append(np.nan)
            sum_p_resp.append(np.nan)
            lat_p_resp.append(np.nan)
            lat_pro_p_resp.append(np.nan)
            vars_p_resp.append(np.nan)
            medians_p_resp.append(np.nan)

            #calculate mean values for fast vs slow
            mean_slow_mean_stim.append(means_slow.mean())
            mean_fast_mean_stim.append(means_fast.mean())
            std_fast_mean_stim.append(stds_fast.mean())
            std_slow_mean_stim.append(stds_slow.mean())
            max_slow_mean_stim.append(maxes_slow.mean())
            max_fast_mean_stim.append(maxes_fast.mean())
            lat_slow_mean_stim.append(lats_slow.mean())
            lat_fast_mean_stim.append(lats_fast.mean())
            sum_slow_mean_stim.append(sums_slow.mean())
            sum_fast_mean_stim.append(sums_fast.mean())
            lat_pro_slow_mean_stim.append(lats_pro_slow.mean())
            lat_pro_fast_mean_stim.append(lats_pro_fast.mean())
            lat_pro_slow_std_stim.append(lats_pro_slow.std())
            lat_pro_fast_std_stim.append(lats_pro_fast.std())
            lat_slow_std_stim.append(lats_slow.std())
            lat_fast_std_stim.append(lats_fast.std())
            median_slow_mean_stim.append(medians_slow.mean())
            median_fast_mean_stim.append(medians_fast.mean())
            var_slow_mean_stim.append(vars_slow.mean())
            var_fast_mean_stim.append(vars_fast.mean())


            mean_slow_mean_resp.append(np.nan)
            mean_fast_mean_resp.append(np.nan)
            std_fast_mean_resp.append(np.nan)
            std_slow_mean_resp.append(np.nan)
            max_slow_mean_resp.append(np.nan)
            max_fast_mean_resp.append(np.nan)
            lat_slow_mean_resp.append(np.nan)
            lat_fast_mean_resp.append(np.nan)
            sum_slow_mean_resp.append(np.nan)
            sum_fast_mean_resp.append(np.nan)
            lat_pro_slow_mean_resp.append(np.nan)
            lat_pro_fast_mean_resp.append(np.nan)
            lat_pro_slow_std_resp.append(np.nan)
            lat_pro_fast_std_resp.append(np.nan)
            lat_slow_std_resp.append(np.nan)
            lat_fast_std_resp.append(np.nan)
            median_slow_mean_stim.append(medians_slow.mean())
            median_fast_mean_stim.append(medians_fast.mean())
            var_slow_mean_stim.append(vars_slow.mean())
            var_fast_mean_stim.append(vars_fast.mean())

            subjs.append(subj)
            tasks.append(task)
            clusts.append(cluster)
            patterns.append(pattern)

            #save stats (single trials)
            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'significance_windows', 'RTsplit', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            keys = ['subj','task','pattern','RTs','lats_pro_slow','lats_pro_fast','sums_slow','sums_fast','means_slow','means_fast','maxes_slow','maxes_fast','lats_slow','lats_fast','stds_slow','stds_fast', 'medians_slow','medians_fast','vars_slow','vars_fast']
            values = [subj,task, pattern, RTs, lats_pro_slow, lats_pro_fast, sums_slow, sums_fast, means_slow, means_fast, maxes_slow, maxes_fast, lats_slow, lats_fast, stds_slow, stds_fast, medians_slow, medians_fast, vars_slow, vars_fast]
            data_dict = dict(zip(keys, values))
            with open(filename, 'w') as f:
                pickle.dump(data_dict, f)
                f.close()

        if pattern == 'R':
            means, maxes, lats, sums, stds, lats_pro, RTs, medians, variations = [data_dict.get(k) for k in ['means','maxes','lats','sums','stds', 'lats_pro', 'RTs', 'medians', 'variations']]

            #calculate median RT to split trials
            slowidx = np.where(RTs > np.median(RTs))[0]
            fastidx = np.where(RTs < np.median(RTs))[0]

            if slowidx.shape > fastidx.shape:
                slowidx = slowidx[0:len(fastidx)]
            if slowidx.shape < fastidx.shape:
                fastidx = fastidx[0:len(slowidx)]

            #calculate stats (single trials)
            means_slow = means[slowidx]
            maxes_slow = maxes[slowidx]
            lats_slow = lats[slowidx]
            lats_pro_slow = lats_pro[slowidx]
            sums_slow = sums[slowidx]        
            stds_slow = stds[slowidx]
            medians_slow = medians[slowidx]
            vars_slow = variations[slowidx]

            means_fast = means[fastidx]
            maxes_fast = maxes[fastidx]
            lats_fast = lats[fastidx]
            lats_pro_fast = lats_pro[fastidx]
            sums_fast = sums[fastidx]
            stds_fast = stds[fastidx]
            medians_fast = medians[fastidx]
            vars_fast = variations[fastidx]

            #calculate ttests
            mean_p_resp.append(stats.ttest_ind(means_slow, means_fast)[1])
            std_p_resp.append(stats.ttest_ind(stds_slow, stds_fast)[1])
            max_p_resp.append(stats.ttest_ind(maxes_slow, maxes_fast)[1])
            lat_p_resp.append(stats.ttest_ind(lats_slow, lats_fast)[1])
            sum_p_resp.append(stats.ttest_ind(sums_slow, sums_fast)[1])
            lat_pro_p_resp.append(stats.ttest_ind(lats_pro_slow, lats_pro_fast)[1])
            median_p_resp.append(stats.ttest_ind(medians_slow, medians_fast)[1])
            var_p_resp.append(stats.ttest_ind(vars_slow, vars_fast)[1])

            mean_p_stim.append(np.nan)
            std_p_stim.append(np.nan)
            max_p_stim.append(np.nan)
            sum_p_stim.append(np.nan)
            lat_p_stim.append(np.nan)
            lat_pro_p_stim.append(np.nan)
            median_p_stim.append(np.nan)
            var_p_stim.append(np.nan)

            #calculate mean values for fast vs slow
            mean_slow_mean_resp.append(means_slow.mean())
            mean_fast_mean_resp.append(means_fast.mean())
            std_fast_mean_resp.append(stds_fast.mean())
            std_slow_mean_resp.append(stds_slow.mean())
            max_slow_mean_resp.append(maxes_slow.mean())
            max_fast_mean_resp.append(maxes_fast.mean())
            lat_slow_mean_resp.append(lats_slow.mean())
            lat_fast_mean_resp.append(lats_fast.mean())
            sum_slow_mean_resp.append(sums_slow.mean())
            sum_fast_mean_resp.append(sums_fast.mean())
            lat_pro_slow_mean_resp.append(lats_pro_slow.mean())
            lat_pro_fast_mean_resp.append(lats_pro_fast.mean())
            lat_pro_slow_std_resp.append(lats_pro_slow.std())
            lat_pro_fast_std_resp.append(lats_pro_fast.std())
            lat_slow_std_resp.append(lats_slow.std())
            lat_fast_std_resp.append(lats_fast.std())
            median_slow_mean_resp.append(medians_slow.mean())
            median_fast_mean_resp.append(medians_fast.mean())
            var_slow_mean_resp.append(vars_slow.mean())
            var_fast_mean_resp.append(vars_fast.mean())

            mean_slow_mean_stim.append(np.nan)
            mean_fast_mean_stim.append(np.nan)
            std_fast_mean_stim.append(np.nan)
            std_slow_mean_stim.append(np.nan)
            max_slow_mean_stim.append(np.nan)
            max_fast_mean_stim.append(np.nan)
            lat_slow_mean_stim.append(np.nan)
            lat_fast_mean_stim.append(np.nan)
            sum_slow_mean_stim.append(np.nan)
            sum_fast_mean_stim.append(np.nan)
            lat_pro_slow_mean_stim.append(np.nan)
            lat_pro_fast_mean_stim.append(np.nan)
            lat_pro_slow_std_stim.append(np.nan)
            lat_pro_fast_std_stim.append(np.nan)
            lat_slow_std_stim.append(np.nan)
            lat_fast_std_stim.append(np.nan)
            median_slow_mean_stim.append(np.nan)
            median_fast_mean_stim.append(np.nan)
            var_slow_mean_stim.append(np.nan)
            var_fast_mean_stim.append(np.nan)

            subjs.append(subj)
            tasks.append(task)
            clusts.append(cluster)
            patterns.append(pattern)

            #save stats (single trials)
            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'significance_windows', 'RTsplit', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            keys = ['subj','task','pattern','RTs','lats_pro_slow','lats_pro_fast','sums_slow','sums_fast','means_slow','means_fast','maxes_slow','maxes_fast','lats_slow','lats_fast','stds_slow','stds_fast', 'medians_slow','medians_fast','vars_slow','vars_fast']
            values = [subj,task, pattern, RTs, lats_pro_slow, lats_pro_fast, sums_slow, sums_fast, means_slow, means_fast, maxes_slow, maxes_fast, lats_slow, lats_fast, stds_slow, stds_fast, medians_slow, medians_fast, vars_slow, vars_fast]
            data_dict = dict(zip(keys, values))
            
            with open(filename, 'w') as f:
                pickle.dump(data_dict, f)
                f.close()
        
        if pattern == 'SR':
            sums_stim, sums_resp, means_stim, means_resp, stds_stim, stds_resp, maxes_stim, maxes_resp, lats_stim, lats_resp, lats_pro_stim, lats_pro_resp, RTs = [data_dict.get(k) for k in ['sums_stim','sums_resp', 'means_stim', 'means_resp', 'stds_stim', 'stds_resp', 'maxes_stim', 'maxes_resp', 'lats_stim', 'lats_resp', 'lats_pro_stim', 'lats_pro_resp', 'RTs']]

            #calculate median RT to split trials
            slowidx = np.where(RTs > np.median(RTs))[0]
            fastidx = np.where(RTs < np.median(RTs))[0]

            if slowidx.shape > fastidx.shape:
                slowidx = slowidx[0:len(fastidx)]
            if slowidx.shape < fastidx.shape:
                fastidx = fastidx[0:len(slowidx)]

            #calculate stats (single trials) - stim
            means_stim_slow = means_stim[slowidx]
            maxes_stim_slow = maxes_stim[slowidx]
            lats_stim_slow = lats_stim[slowidx]
            lats_pro_stim_slow = lats_pro_stim[slowidx]
            sums_stim_slow = sums_stim[slowidx]        
            stds_stim_slow = stds_stim[slowidx]

            means_stim_fast = means_stim[fastidx]
            maxes_stim_fast = maxes_stim[fastidx]
            lats_stim_fast = lats_stim[fastidx]
            lats_pro_stim_fast = lats_pro_stim[fastidx]
            sums_stim_fast = sums_stim[fastidx]
            stds_stim_fast = stds_stim[fastidx]

            #calculate stats (single trials) = resp
            means_resp_slow = means_resp[slowidx]
            maxes_resp_slow = maxes_resp[slowidx]
            lats_resp_slow = lats_resp[slowidx]
            lats_pro_resp_slow = lats_pro_resp[slowidx]
            sums_resp_slow = sums_resp[slowidx]        
            stds_resp_slow = stds_resp[slowidx]

            means_resp_fast = means_resp[fastidx]
            maxes_resp_fast = maxes_resp[fastidx]
            lats_resp_fast = lats_resp[fastidx]
            lats_pro_resp_fast = lats_pro_resp[fastidx]
            sums_resp_fast = sums_resp[fastidx]
            stds_resp_fast = stds_resp[fastidx]

            #calculate ttests
            mean_p_stim.append(stats.ttest_ind(means_stim_slow, means_stim_fast)[1])
            std_p_stim.append(stats.ttest_ind(stds_stim_slow, stds_stim_fast)[1])
            max_p_stim.append(stats.ttest_ind(maxes_stim_slow, maxes_stim_fast)[1])
            lat_p_stim.append(stats.ttest_ind(lats_stim_slow, lats_stim_fast)[1])
            lat_pro_p_stim.append(stats.ttest_ind(lats_pro_stim_slow, lats_pro_stim_fast)[1])
            sum_p_stim.append(stats.ttest_ind(sums_stim_slow, sums_stim_fast)[1])

            mean_p_resp.append(stats.ttest_ind(means_resp_slow, means_resp_fast)[1])
            std_p_resp.append(stats.ttest_ind(stds_resp_slow, stds_resp_fast)[1])
            max_p_resp.append(stats.ttest_ind(maxes_resp_slow, maxes_resp_fast)[1])
            lat_p_resp.append(stats.ttest_ind(lats_resp_slow, lats_resp_fast)[1])
            lat_pro_p_resp.append(stats.ttest_ind(lats_pro_resp_slow, lats_pro_resp_fast)[1])
            sum_p_resp.append(stats.ttest_ind(sums_resp_slow, sums_resp_fast)[1])

            #calculate mean values for fast vs slow
            mean_slow_mean_stim.append(means_stim_slow.mean())
            mean_fast_mean_stim.append(means_stim_fast.mean())
            std_fast_mean_stim.append(stds_stim_fast.mean())
            std_slow_mean_stim.append(stds_stim_slow.mean())
            max_slow_mean_stim.append(maxes_stim_slow.mean())
            max_fast_mean_stim.append(maxes_stim_fast.mean())
            lat_slow_mean_stim.append(lats_stim_slow.mean())
            lat_fast_mean_stim.append(lats_stim_fast.mean())
            sum_slow_mean_stim.append(sums_stim_slow.mean())
            sum_fast_mean_stim.append(sums_stim_fast.mean())
            lat_pro_slow_mean_stim.append(lats_pro_stim_slow.mean())
            lat_pro_fast_mean_stim.append(lats_pro_stim_fast.mean())
            lat_pro_slow_std_stim.append(lats_pro_stim_slow.std())
            lat_pro_fast_std_stim.append(lats_pro_stim_fast.std())
            lat_slow_std_stim.append(lats_stim_slow.std())
            lat_fast_std_stim.append(lats_stim_fast.std())

            mean_slow_mean_resp.append(means_resp_slow.mean())
            mean_fast_mean_resp.append(means_resp_fast.mean())
            std_fast_mean_resp.append(stds_resp_fast.mean())
            std_slow_mean_resp.append(stds_resp_slow.mean())
            max_slow_mean_resp.append(maxes_resp_slow.mean())
            max_fast_mean_resp.append(maxes_resp_fast.mean())
            lat_slow_mean_resp.append(lats_resp_slow.mean())
            lat_fast_mean_resp.append(lats_resp_fast.mean())
            sum_slow_mean_resp.append(sums_resp_slow.mean())
            sum_fast_mean_resp.append(sums_resp_fast.mean())
            lat_pro_slow_mean_resp.append(lats_pro_resp_slow.mean())
            lat_pro_fast_mean_resp.append(lats_pro_resp_fast.mean())
            lat_pro_slow_std_resp.append(lats_pro_resp_slow.std())
            lat_pro_fast_std_resp.append(lats_pro_resp_fast.std())
            lat_slow_std_resp.append(lats_resp_slow.std())
            lat_fast_std_resp.append(lats_resp_fast.std())

            subjs.append(subj)
            tasks.append(task)
            clusts.append(cluster)
            patterns.append(pattern)

            #save stats (single trials)
            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'significance_windows', 'RTsplit', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            keys = ['subj','task','pattern','RTs','sums_stim_slow','sums_resp_slow','sums_stim_fast','sums_resp_fast','means_stim_slow','means_resp_slow', 'means_stim_fast','means_resp_fast','maxes_stim_slow', 'maxes_resp_slow','maxes_resp_fast','maxes_stim_fast','lats_stim_slow','lats_resp_slow','lats_stim_fast','lats_resp_fast','stds_stim_slow','stds_resp_slow','stds_stim_fast', 'stds_resp_fast', 'lats_pro_stim_slow','lats_pro_resp_slow','lats_pro_stim_fast','lats_pro_resp_fast']
            values = [subj, task, pattern, RTs, sums_stim_slow, sums_resp_slow, sums_stim_fast, sums_resp_fast, means_stim_slow, means_resp_slow, means_stim_fast, means_resp_fast, maxes_stim_slow, maxes_resp_slow, maxes_resp_fast, maxes_stim_fast, lats_stim_slow, lats_resp_slow, lats_stim_fast, lats_resp_fast, stds_stim_slow, stds_resp_slow, stds_stim_fast, stds_resp_fast, lats_pro_stim_slow, lats_pro_resp_slow, lats_pro_stim_fast, lats_pro_resp_fast]
            data_dict = dict(zip(keys, values))

            with open(filename, 'w') as f:
                pickle.dump(data_dict, f)
                f.close()


        if pattern == 'D':
            sums, means, stds, maxes, lats_stim, lats_resp, lats_pro_stim, lats_pro_resp, RTs = [data_dict.get(k) for k in ['sums', 'means', 'stds', 'maxes', 'lats_stim', 'lats_resp','lats_pro_stim','lats_pro_resp', 'RTs']]

            #calculate median RT to split trials
            slowidx = np.where(RTs > np.median(RTs))[0]
            fastidx = np.where(RTs < np.median(RTs))[0]

            if slowidx.shape > fastidx.shape:
                slowidx = slowidx[0:len(fastidx)]
            if slowidx.shape < fastidx.shape:
                fastidx = fastidx[0:len(slowidx)]

            #calculate stats (single trials)
            means_slow = means[slowidx]
            maxes_slow = maxes[slowidx]
            lats_stim_slow = lats_stim[slowidx]
            lats_resp_slow = lats_resp[slowidx]
            sums_slow = sums[slowidx]        
            stds_slow = stds[slowidx]
            lats_pro_stim_slow = lats_pro_stim[slowidx]
            lats_pro_resp_slow = lats_pro_resp[slowidx]

            means_fast = means[fastidx]
            maxes_fast = maxes[fastidx]
            lats_stim_fast = lats_stim[fastidx]
            lats_resp_fast = lats_resp[fastidx]
            sums_fast = sums[fastidx]
            stds_fast = stds[fastidx]
            lats_pro_stim_fast = lats_pro_stim[fastidx]
            lats_pro_resp_fast = lats_pro_resp[fastidx]

            #calculate ttests
            mean_p_stim.append(stats.ttest_ind(means_slow, means_fast)[1])
            std_p_stim.append(stats.ttest_ind(stds_slow, stds_fast)[1])
            max_p_stim.append(stats.ttest_ind(maxes_slow, maxes_fast)[1])
            lat_p_stim.append(stats.ttest_ind(lats_stim_slow, lats_stim_fast)[1])
            lat_p_resp.append(stats.ttest_ind(lats_resp_slow, lats_resp_fast)[1])
            sum_p_stim.append(stats.ttest_ind(sums_slow, sums_fast)[1])
            lat_pro_p_stim.append(stats.ttest_ind(lats_pro_stim_slow, lats_pro_stim_fast)[1])
            lat_pro_p_resp.append(stats.ttest_ind(lats_pro_resp_slow, lats_pro_resp_fast)[1])

            mean_p_resp.append(np.nan)
            std_p_resp.append(np.nan)
            max_p_resp.append(np.nan)
            sum_p_resp.append(np.nan)

            #calculate mean values for fast vs slow
            mean_slow_mean_stim.append(means_slow.mean())
            mean_fast_mean_stim.append(means_fast.mean())
            std_fast_mean_stim.append(stds_fast.mean())
            std_slow_mean_stim.append(stds_slow.mean())
            max_slow_mean_stim.append(maxes_slow.mean())
            max_fast_mean_stim.append(maxes_fast.mean())
            lat_slow_mean_stim.append(lats_stim_slow.mean())
            lat_fast_mean_stim.append(lats_stim_fast.mean())
            sum_slow_mean_stim.append(sums_slow.mean())
            sum_fast_mean_stim.append(sums_fast.mean())
            lat_pro_slow_mean_stim.append(lats_pro_stim_slow.mean())
            lat_pro_fast_mean_stim.append(lats_pro_stim_fast.mean())

            lat_slow_mean_resp.append(lats_resp_slow.mean())
            lat_fast_mean_resp.append(lats_resp_fast.mean())
            lat_pro_slow_mean_resp.append(lats_pro_resp_slow.mean())
            lat_pro_fast_mean_resp.append(lats_pro_resp_fast.mean())
            
            lat_pro_slow_std_resp.append(lats_pro_resp_slow.std())
            lat_pro_fast_std_resp.append(lats_pro_resp_fast.std())
            lat_pro_slow_std_stim.append(lats_pro_stim_slow.std())
            lat_pro_fast_std_stim.append(lats_pro_stim_fast.std())

            lat_slow_std_resp.append(lats_resp_slow.std())
            lat_fast_std_resp.append(lats_resp_fast.std())
            lat_slow_std_stim.append(lats_stim_slow.std())
            lat_fast_std_stim.append(lats_stim_fast.std())

            mean_slow_mean_resp.append(np.nan)
            mean_fast_mean_resp.append(np.nan)
            std_fast_mean_resp.append(np.nan)
            std_slow_mean_resp.append(np.nan)
            max_slow_mean_resp.append(np.nan)
            max_fast_mean_resp.append(np.nan)
            sum_slow_mean_resp.append(np.nan)
            sum_fast_mean_resp.append(np.nan)

            subjs.append(subj)
            tasks.append(task)
            clusts.append(cluster)
            patterns.append(pattern)

            #save stats (single trials)
            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'significance_windows', 'RTsplit', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            keys = ['subj','task','pattern','RTs','sums_slow','sums_fast','means_slow','means_fast','maxes_slow','maxes_fast','lats_stim_slow','lats_resp_slow','lats_stim_fast','lats_resp_fast','stds_slow','stds_fast','lats_pro_stim_slow','lats_pro_resp_slow','lats_pro_stim_fast','lats_pro_resp_fast',]
            values = [subj,task, pattern, RTs, sums_slow, sums_fast, means_slow, means_fast, maxes_slow, maxes_fast, lats_stim_slow, lats_resp_slow, lats_stim_fast, lats_resp_fast, stds_slow, stds_fast, lats_pro_stim_slow, lats_pro_resp_slow, lats_pro_stim_fast, lats_pro_resp_fast,]
            data_dict = dict(zip(keys, values))

            with open(filename, 'w') as f:
                pickle.dump(data_dict, f)
                f.close()


    keys = ['subj','task','cluster','pattern','mean_slow_mean_stim','mean_fast_mean_stim','mean_p_stim','mean_slow_mean_resp','mean_fast_mean_resp','mean_p_resp','std_slow_mean_stim','std_fast_mean_stim','std_p_stim','std_slow_mean_resp','std_fast_mean_resp','std_p_resp','max_slow_mean_stim','max_fast_mean_stim','max_p_stim','max_slow_mean_resp','max_fast_mean_resp','max_p_resp','sums_slow_mean_stim','sums_fast_mean_stim','sum_p_stim','sums_slow_mean_resp','sums_fast_mean_resp','sum_p_resp','lats_slow_mean_stim','lats_fast_mean_stim','lat_p_stim','lat_p_resp','lats_slow_mean_resp','lats_fast_mean_resp', 'lats_pro_slow_mean_stim','lats_pro_fast_mean_stim','lat_pro_p_stim','lats_pro_slow_mean_resp','lats_pro_fast_mean_resp','lat_pro_p_resp', 'lats_pro_slow_std_stim','lats_pro_fast_std_stim', 'lats_pro_slow_std_resp','lats_pro_fast_std_resp', 'lats_slow_std_stim','lats_fast_std_stim', 'lats_slow_std_resp','lats_fast_std_resp']

    values = [subjs, tasks, clusts, patterns, mean_slow_mean_stim, mean_fast_mean_stim, mean_p_stim, mean_slow_mean_resp, mean_fast_mean_resp, mean_p_resp, std_slow_mean_stim, std_fast_mean_stim, std_p_stim, std_slow_mean_resp, std_fast_mean_resp, std_p_resp, max_slow_mean_stim, max_fast_mean_stim, max_p_stim, max_slow_mean_resp, max_fast_mean_resp, max_p_resp, sum_slow_mean_stim, sum_fast_mean_stim, sum_p_stim, sum_slow_mean_resp, sum_fast_mean_resp, sum_p_resp, lat_slow_mean_stim, lat_fast_mean_stim, lat_p_stim, lat_p_resp, lat_slow_mean_resp, lat_fast_mean_resp, lat_pro_slow_mean_stim, lat_pro_fast_mean_stim, lat_pro_p_stim, lat_pro_slow_mean_resp, lat_pro_fast_mean_resp, lat_pro_p_resp, lat_pro_slow_std_stim, lat_pro_fast_std_stim, lat_pro_slow_std_resp, lat_pro_fast_std_resp,  lat_slow_std_stim, lat_fast_std_stim, lat_slow_std_resp, lat_fast_std_resp]

    activity_stats = dict(zip(keys, values))

    filename = os.path.join(SJdir,'PCA','ShadePlots_hclust', 'significance_windows', 'RTsplit', 'significance_windows_stats.p')
    with open(filename, 'w') as f:
        pickle.dump(activity_stats, f)
        f.close()

    df_stats = pd.DataFrame(activity_stats)
    df_stats = df_stats[keys]

    filename = os.path.join(SJdir,'PCA', 'Stats', 'significance_windows_stats_RTsplit.csv')
    df_stats.to_csv(filename)


if __name__ == '__main__':
    shadeplots_clusters_stats_RTsplit()
