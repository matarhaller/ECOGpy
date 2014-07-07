from __future__ import division
import pandas as pd
import os
import numpy as np
import sys
import cPickle as pickle

def shadeplots_clusters_stats():
    """ 
    calculates mean, peak, latency, and std per trial
    saves pickle file with numbers per trial in ShadePlots_hclust/significance_windows
    """

    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'

    filename = os.path.join(SJdir,'PCA','duration_dict_500_FINAL', 'stim_resp_cluster_windows_withduration_EDITED.xlsx')
    df = pd.ExcelFile(filename)
    df = df.parse('stim_resp_cluster_windows_withd')

    subjs = list(); tasks = list()
    clusts = list();  patterns = list()
    mean_activity_stim = list(); mean_activity_resp = list(); 
    std_activity_stim = list(); std_activity_resp = list()
    max_activity_stim = list(); max_activity_resp = list(); 
    sum_activity_stim = list(); sum_activity_resp = list()
    lat_activity_stim = list(); lat_activity_resp = list()
    lat_pro_activity_stim = list(); lat_pro_activity_resp = list()


    for row in df.itertuples():
        i, subj, task, cluster, pattern, start_idx, end_idx, start_idx_resp, end_idx_resp, _ = row


        if any([(pattern == 'S'), (pattern == 'sustained'), (pattern == 'S+sustained')]):

            #load data
            filename = os.path.join(SJdir, 'PCA','ShadePlots_hclust', 'data', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            with open(filename, 'r') as x:
                data_dict = pickle.load(x)
                x.close()

            cdata, bl_st, srate, RTs = [data_dict.get(k) for k in ['cdata','bl_st', 'srate', 'RTs']]

            start_idx = start_idx + abs(bl_st)
            end_idx = end_idx + abs(bl_st)

            #calculate stats (single trials)
            means = cdata[:,start_idx:end_idx].mean(axis = 1)
            stds = cdata[:,start_idx:end_idx].std(axis = 1)
            maxes = cdata[:,start_idx:end_idx].max(axis = 1)
            lats = cdata[:,start_idx:end_idx].argmax(axis = 1)
            sums = cdata[:, start_idx:end_idx].sum(axis = 1)
            lats_pro = lats / len(np.arange(start_idx, end_idx))

            #calculate mean stats
            mean_activity_stim.append(means.mean())
            std_activity_stim.append(stds.mean())
            max_activity_stim.append(maxes.mean())
            lat_activity_stim.append(lats.mean())
            sum_activity_stim.append(sums.mean())
            lat_pro_activity_stim.append(lats_pro.mean())

            mean_activity_resp.append(np.nan)
            std_activity_resp.append(np.nan)
            max_activity_resp.append(np.nan)
            lat_activity_resp.append(np.nan)
            sum_activity_resp.append(np.nan)
            lat_pro_activity_resp.append(np.nan)

            subjs.append(subj)
            tasks.append(task)
            clusts.append(cluster)
            patterns.append(pattern)

            #save stats (single trials)
            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'significance_windows', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            data_dict = {'pattern':pattern, 'lats_pro': lats_pro, 'sums':sums, 'means':means, 'stds':stds, 'maxes':maxes, 'lats':lats, 'cdata': cdata, 'start_idx': start_idx, 'end_idx':end_idx, 'srate': srate, 'bl_st':bl_st,'RTs':RTs}

            with open(filename, 'w') as f:
                pickle.dump(data_dict, f)
                f.close()

        if pattern == 'R':

            #load data
            filename = os.path.join(SJdir, 'PCA','ShadePlots_hclust', 'resplocked_all', 'data', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            with open(filename, 'r') as x:
                data_dict = pickle.load(x)
                x.close()

            cdata_resp, bl_st, srate, st_resp, en_resp, RTs = [data_dict.get(k) for k in ['cdata_resp','bl_st', 'srate', 'st_resp', 'en_resp', 'RTs']]
            start_idx_resp = start_idx_resp+abs(st_resp)
            end_idx_resp = end_idx_resp+abs(st_resp)

            #calculate stats (single trials)
            means = cdata_resp[:,start_idx_resp:end_idx_resp].mean(axis = 1)
            stds = cdata_resp[:,start_idx_resp:end_idx_resp].std(axis = 1)
            maxes = cdata_resp[:,start_idx_resp:end_idx_resp].max(axis = 1)
            lats = cdata_resp[:,start_idx_resp:end_idx_resp].argmax(axis = 1)
            sums = cdata_resp[:, start_idx_resp:end_idx_resp].sum(axis = 1)
            lats_pro = lats / len(np.arange(start_idx_resp, end_idx_resp))

            #calculate mean stats
            mean_activity_resp.append(means.mean())
            std_activity_resp.append(stds.mean())
            max_activity_resp.append(maxes.mean())
            lat_activity_resp.append(lats.mean())
            sum_activity_resp.append(sums.mean())
            lat_pro_activity_resp.append(lats_pro.mean())

            mean_activity_stim.append(np.nan)
            std_activity_stim.append(np.nan)
            max_activity_stim.append(np.nan)
            lat_activity_stim.append(np.nan)
            sum_activity_stim.append(np.nan)
            lat_pro_activity_stim.append(np.nan)

            subjs.append(subj)
            tasks.append(task)
            clusts.append(cluster)
            patterns.append(pattern)

            #save stats (single trials)
            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'significance_windows', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            data_dict = {'pattern':pattern, 'lats_pro':lats_pro, 'sums':sums, 'means':means, 'stds':stds, 'maxes':maxes, 'lats':lats, 'cdata_resp': cdata_resp, 'start_idx_resp': start_idx_resp, 'end_idx_resp':end_idx_resp, 'st_resp': st_resp, 'en_resp':en_resp, 'RTs':RTs, 'srate': srate, 'bl_st':bl_st}

            with open(filename, 'w') as f:
                pickle.dump(data_dict, f)
                f.close()

        if pattern == 'SR':
            #load data (stim)
            filename = os.path.join(SJdir, 'PCA','ShadePlots_hclust', 'data', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            with open(filename, 'r') as x:
                data_dict = pickle.load(x)
                x.close()

            cdata, bl_st, srate = [data_dict.get(k) for k in ['cdata','bl_st', 'srate']]
            start_idx = start_idx + abs(bl_st)
            end_idx = end_idx+abs(bl_st)

            #load data (resp)
            filename = os.path.join(SJdir, 'PCA','ShadePlots_hclust', 'resplocked_all', 'data', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            with open(filename, 'r') as x:
                data_dict = pickle.load(x)
                x.close()

            cdata_resp, bl_st, srate, st_resp, en_resp, RTs = [data_dict.get(k) for k in ['cdata_resp','bl_st', 'srate', 'st_resp', 'en_resp', 'RTs']]
            end_idx_resp = end_idx_resp+abs(st_resp)
            start_idx_resp = start_idx_resp+abs(st_resp)

            #calculate stats (single trials)
            means_stim = cdata[:,start_idx:end_idx].mean(axis = 1)
            stds_stim = cdata[:,start_idx:end_idx].std(axis = 1)
            maxes_stim = cdata[:,start_idx:end_idx].max(axis = 1)
            lats_stim = cdata[:,start_idx:end_idx].argmax(axis = 1)
            sums_stim = cdata[:, start_idx:end_idx].sum(axis = 1)
            lats_pro_stim = lats_stim / len(np.arange(start_idx, end_idx))

            means_resp = cdata_resp[:,start_idx_resp:end_idx_resp].mean(axis = 1)
            stds_resp = cdata_resp[:,start_idx_resp:end_idx_resp].std(axis = 1)
            maxes_resp = cdata_resp[:,start_idx_resp:end_idx_resp].max(axis = 1)
            lats_resp = cdata_resp[:,start_idx_resp:end_idx_resp].argmax(axis = 1)
            sums_resp = cdata_resp[:, start_idx_resp:end_idx_resp].sum(axis = 1)
            lats_pro_resp = lats_resp / len(np.arange(start_idx_resp, end_idx_resp))

            #calculate mean stats
            mean_activity_stim.append(means_stim.mean())
            std_activity_stim.append(stds_stim.mean())
            max_activity_stim.append(maxes_stim.mean())
            lat_activity_stim.append(lats_stim.mean())
            sum_activity_stim.append(sums_stim.mean())
            lat_pro_activity_stim.append(lats_pro_stim.mean())

            mean_activity_resp.append(means_resp.mean())
            std_activity_resp.append(stds_resp.mean())
            max_activity_resp.append(maxes_resp.mean())
            lat_activity_resp.append(lats_resp.mean())
            sum_activity_resp.append(sums_resp.mean())
            lat_pro_activity_resp.append(lats_pro_resp.mean())

            subjs.append(subj)
            tasks.append(task)
            clusts.append(cluster)
            patterns.append(pattern)

            #save stats (single trials)
            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'significance_windows', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            data_dict = {'pattern':pattern,'lats_pro_stim':lats_pro_stim, 'lats_pro_resp':lats_pro_resp, 'sums_stim':sums_stim, 'sums_resp':sums_resp, 'means_stim':means_stim, 'means_resp':means_resp, 'stds_stim':stds_stim,'stds_resp':stds_resp, 'maxes_stim':maxes_stim, 'maxes_resp':maxes_resp, 'lats_stim':lats_stim, 'lats_resp':lats_resp, 'cdata_resp': cdata_resp, 'start_idx_resp': start_idx_resp, 'end_idx_resp':end_idx_resp, 'st_resp': st_resp, 'en_resp':en_resp, 'RTs':RTs, 'srate': srate, 'bl_st':bl_st, 'start_idx':start_idx, 'end_idx':end_idx}

            with open(filename, 'w') as f:
                pickle.dump(data_dict, f)
                f.close()

        if pattern == 'D':
            #load data (stim)
            filename = os.path.join(SJdir, 'PCA','ShadePlots_hclust', 'data', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            with open(filename, 'r') as x:
                data_dict = pickle.load(x)
                x.close()

            cdata, bl_st, srate = [data_dict.get(k) for k in ['cdata','bl_st', 'srate']]
            start_idx = start_idx + abs(bl_st)

            #load data (resp)
            filename = os.path.join(SJdir, 'PCA','ShadePlots_hclust', 'resplocked_all', 'data', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            with open(filename, 'r') as x:
                data_dict = pickle.load(x)
                x.close()

            cdata_resp, bl_st, srate, st_resp, en_resp, RTs = [data_dict.get(k) for k in ['cdata_resp','bl_st', 'srate', 'st_resp', 'en_resp', 'RTs']]
            end_idx_resp = end_idx_resp+abs(st_resp)

            #create data matrices
            cdata_dur_stim = np.empty(cdata.shape)
            for j, r in enumerate(RTs):
                tmp = cdata[j, start_idx : r + end_idx_resp]
                tmp = np.pad(tmp, (0, cdata.shape[1]-len(tmp)), 'constant', constant_values = -999)
                cdata_dur_stim[j,:] = tmp
            cdata_dur_stim[cdata_dur_stim == -999] = np.nan

            cdata_dur_resp = np.empty(cdata.shape)
            for j, r in enumerate(RTs):
                tmp = cdata[j, start_idx : r+ end_idx_resp]
                tmp = np.pad(tmp, (cdata.shape[1]-len(tmp),0), 'constant', constant_values = -999)
                cdata_dur_resp[j,:] = tmp
            cdata_dur_resp[cdata_dur_resp == -999] = np.nan

            #calculate stats (single trials)
            means = np.nanmean(cdata_dur_stim, axis = 1)
            stds = np.nanstd(cdata_dur_stim, axis = 1)
            maxes = np.nanmax(cdata_dur_stim, axis = 1)
            sums = np.nansum(cdata_dur_stim, axis = 1)
            lats_stim = np.nanargmax(cdata_dur_stim, axis = 1)
            lats_resp = np.nanargmax(cdata_dur_resp, axis = 1)
            lats_pro_stim = np.nanargmax(cdata_dur_stim, axis = 1) / np.sum(~np.isnan(cdata_dur_stim), axis = 1)
            lats_pro_resp = np.nanargmax(cdata_dur_resp, axis = 1) / np.sum(~np.isnan(cdata_dur_resp), axis = 1)

            #calculate mean stats
            mean_activity_stim.append(np.nanmean(means))
            std_activity_stim.append(np.nanmean(stds))
            max_activity_stim.append(np.nanmean(maxes))
            lat_activity_stim.append(np.nanmean(lats_stim))
            lat_activity_resp.append(np.nanmean(lats_resp))
            sum_activity_stim.append(np.nanmean(sums))
            lat_pro_activity_stim.append(np.nanmean(lats_pro_stim))
            lat_pro_activity_resp.append(np.nanmean(lats_pro_resp))

            mean_activity_resp.append(np.nan)
            std_activity_resp.append(np.nan)
            max_activity_resp.append(np.nan)
            sum_activity_resp.append(np.nan)

            subjs.append(subj)
            tasks.append(task)
            clusts.append(cluster)
            patterns.append(pattern)

            #save stats (single trials)
            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'significance_windows', ''.join([subj, '_', task, '_c', str(cluster), '.p']))
            data_dict = {'lats_pro_stim': lats_pro_stim, 'lats_pro_resp':lats_pro_resp, 'pattern':pattern, 'sums':sums, 'means':means, 'stds':stds, 'maxes':maxes, 'lats_stim':lats_stim, 'lats_resp':lats_resp, 'cdata_dur_resp': cdata_dur_resp, 'end_idx_resp':end_idx_resp, 'st_resp': st_resp, 'en_resp':en_resp, 'RTs':RTs, 'srate': srate, 'bl_st':bl_st,'start_idx':start_idx, 'cdata_dur_stim':cdata_dur_stim}
            with open(filename, 'w') as f:
                pickle.dump(data_dict, f)
                f.close()

    keys = ['subj','task','cluster','pattern','mean_activity_stim','mean_activity_resp','std_activity_stim','std_activity_resp','max_activity_stim','max_activity_resp','sum_activity_stim','sum_activity_resp','lat_activity_stim','lat_activity_resp', 'lat_pro_activity_stim', 'lat_pro_activity_resp']
    values = [subjs, tasks, clusts, patterns, mean_activity_stim, mean_activity_resp, std_activity_stim, std_activity_resp, max_activity_stim, max_activity_resp, sum_activity_stim, sum_activity_resp, lat_activity_stim, lat_activity_resp, lat_pro_activity_stim, lat_pro_activity_resp]
    activity_stats = dict(zip(keys, values))

    filename = os.path.join(SJdir,'PCA','ShadePlots_hclust', 'significance_windows', 'significance_windows_stats.p')
    with open(filename, 'w') as f:
        pickle.dump(activity_stats, f)
        f.close()

    df_stats = pd.DataFrame(activity_stats)
    df_stats = df_stats[keys]

    filename = os.path.join(SJdir,'PCA', 'Stats', 'significance_windows_stats.csv')
    df_stats.to_csv(filename)


if __name__ == '__main__':
    shadeplots_clusters_stats()
