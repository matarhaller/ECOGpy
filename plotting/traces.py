from __future__ import division
from utils.get_HGdata import get_HGdata
from utils.loadmat import loadmat
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_trace(subj, task, elec_list, color = 'darkslateblue', fig = None, ax = None, **kwargs):
    """
    plot trace with SEM
    input:
        subj, task as string
        elec - list of electrodes to plot
        ax = axis object where to plot (optional, default None, plot to gca)
    """

    if ax is None: #if didn't specify where to plot it
        ax = plt.gca()    
        # f, ax = plt.subplots(figsize = (10,10))

    #get data
    var_list = ['srate','data_zscore','active_elecs', 'RTs'] #vars of interest
    srate, alldata, active_elecs, RTs = get_HGdata(subj, task, var_list)
    bl_st = -500/1000*srate
    
    #get data for elecs of choice
    idx = np.in1d(active_elecs, elec_list) #indices of these elecs in the HG dataframe
    data = np.vstack(alldata[idx,:,:]) #trials x time for all elecs in cluster
    RTs = np.tile(RTs, len(elec_list))
   
    #plot
    ax.plot(np.arange(bl_st, data.shape[1]+bl_st), data.mean(axis = 0), zorder = 1, linewidth = 3, color = color)
    sem = np.std(data, axis = 0)/np.sqrt(data.shape[0])
    ax.fill_between(np.arange(bl_st, data.shape[1]+bl_st), data.mean(axis = 0)+sem, data.mean(axis=0)-sem, alpha = 0.5, zorder = 0, edgecolor = 'None', facecolor = color)
    ax.plot(np.arange(bl_st, data.shape[1]+bl_st), np.zeros(data.shape[1]), color = 'k', linewidth = 3) #xaxis
    ax.axvline(0, color = 'k', linewidth = 3)
    ax.set_ylabel('HG (zscore)', fontsize = 14)
    ax.set_xlabel('time (ms)', fontsize = 14)
    ax.autoscale(tight=True)
    ax.set_xticks(np.arange(0, 3000, 1000))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(labelsize = 14, width = 0)
    ax.yaxis.set_tick_params(labelsize = 14, width = 0)
       
    return ax


def plot_resp_trace(subj, task, elec_list, color = 'darkslateblue', fig = None, ax = None, **kwargs):
    """
    plot trace with SEM
    input:
        subj, task as string
        elec - list of electrodes to plot
        ax = axis object where to plot (optional, default None, plot to gca)
    """

    if ax is None: #if didn't specify where to plot it
        ax = plt.gca()    
        # f, ax = plt.subplots(figsize = (10,10))

    #get data
    var_list = ['srate','data_zscore','active_elecs', 'RTs'] #vars of interest
    srate, alldata, active_elecs, RTs = get_HGdata(subj, task, var_list)
    bl_st = -500/1000*srate
    
    #get data for elecs of choice
    idx = np.in1d(active_elecs, elec_list) #indices of these elecs in the HG dataframe
    data = np.vstack(alldata[idx,:,:]) #trials x time for all elecs in cluster
    RTs = np.tile(RTs, len(elec_list))
    
   
    #plot
    ax.plot(np.arange(bl_st, data.shape[1]+bl_st), data.mean(axis = 0), zorder = 1, linewidth = 3, color = color)
    sem = np.std(data, axis = 0)/np.sqrt(data.shape[0])
    ax.fill_between(np.arange(bl_st, data.shape[1]+bl_st), data.mean(axis = 0)+sem, data.mean(axis=0)-sem, alpha = 0.5, zorder = 0, edgecolor = 'None', facecolor = color)
    ax.plot(np.arange(bl_st, data.shape[1]+bl_st), np.zeros(data.shape[1]), color = 'k', linewidth = 3) #xaxis
    ax.axvline(0, color = 'k', linewidth = 3)
    ax.set_ylabel('HG (zscore)', fontsize = 14)
    ax.set_xlabel('time (ms)', fontsize = 14)
    ax.autoscale(tight=True)
    ax.set_xticks(np.arange(0, 3000, 1000))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(labelsize = 14, width = 0)
    ax.yaxis.set_tick_params(labelsize = 14, width = 0)

        
    return ax
    

def plot_shadetrace(subj, task, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'):
        """
        plot shade plots with activity window in red (from ShadePlots_allelecs.py)
        """
        #files = glob.glob(os.path.join(SJdir, 'PCA', 'ShadePlots_allelecs', 'data', ''.join([subj, '_', task, '*empty.p'])))
        files = glob.glob(os.path.join(SJdir, 'PCA', 'ShadePlots_allelecs', 'data', ''.join([subj, '_', task, '*.p'])))

        for f in files:
            print f
            #elecname = f.split('_')[-2].split('.p')[0] #before empty was -1
            elecname = f.split('_')[-1].split('.p')[0] 

            with open(f, 'r') as x:
                data_dict = pickle.load(x)
                x.close()

            #map dictionary to variables
            edata, bl_st, srate, start_idx, end_idx, chunksize, black_chunksize, thresh = [data_dict.get(k) for k in ['edata','bl_st', 'srate', 'start_idx', 'end_idx', 'chunksize', 'black_chunksize', 'thresh']]

            #plot
            f, ax = plt.subplots(figsize = (10,10))
            scale_min = edata.mean(axis = 0).min() - 10
            scale_max = edata.mean(axis = 0).max() + 10
            tmp = (np.arange(scale_min, scale_max))

            ax.plot(np.arange(bl_st, edata.shape[1]+bl_st), edata.mean(axis = 0), zorder = 1, linewidth = 3)
            sem = np.std(edata, axis = 0)/np.sqrt(edata.shape[0])
            ax.fill_between(np.arange(bl_st, edata.shape[1]+bl_st), edata.mean(axis = 0)+sem, edata.mean(axis=0)-sem, alpha = 0.5, zorder = 0, edgecolor = 'None', facecolor = 'slateblue')
            ax.plot(np.arange(bl_st, edata.shape[1]+bl_st), np.zeros(edata.shape[1]), color = 'k', linewidth = 3) #xaxis
            ax.plot(np.zeros(tmp.size), tmp, color = 'k', linewidth = 3) #yaxis
            ax.set_ylabel('% change HG')
            ax.set_xlabel('time (ms)')
            ax.autoscale(tight=True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            if start_idx.size>0:
                for i, s in enumerate(start_idx):
                    tmp = np.arange(s, end_idx[i])
                    start = s
                    finish = end_idx[i]
                    ax.plot(tmp, np.zeros(tmp.size), color = 'r', linewidth = 3.5, label = (start, finish))
                    ax.legend()

            #ax.set_title(' '.join(['empty trials', subj, task, ':', elecname, 'chunksize', str(chunksize), 'smoothing', str(black_chunksize),'thresh', str(thresh)]))
            ax.set_title(' '.join([subj, task, ':', elecname, 'chunksize', str(chunksize), 'smoothing', str(black_chunksize),'thresh', str(thresh)]))
            #plt.savefig(os.path.join(SJdir, 'PCA', 'ShadePlots_allelecs','images', ''.join([subj, '_', task, '_', elecname, '_empty'])))
            plt.savefig(os.path.join(SJdir, 'PCA', 'ShadePlots_allelecs','images', ''.join([subj, '_', task, '_', elecname,])))
            plt.close()

