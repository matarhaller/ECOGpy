from __future__ import division
import pandas as pd
import os
import scipy.stats as stats
from utils import fdr_correct
import numpy as np
import sys
from utils.loadmat import loadmat
from utils.get_HGdata import get_HGdata
import matplotlib.pyplot as plt


def calc_shadeplot(subj, task, thresh = 10, chunk_size = 100, baseline = -500, merge_chunk_size = 0):
    """ 
    calculate onset and offset window for given electrode
    input:
        subj, task
        thresh = minimum value that electrode has to reach for at least chunksize (default 10%)
        chunk_size = minimum window size for significance (default 100ms)
        baseline = length of prestimulus baseline (default -500ms)
        merge_chunk = length of chunk to ignore if between significant segments (ie to smooth over) (default 0ms)
        
    output: 
        dataframe with onset/offset per electrode
    """
    #get data
    var_list = ['srate','data_percent','active_elecs'] #vars of interest
    srate, data, active_elecs = get_HGdata(subj, task, var_list, type = 'percent')
        
    #convert to srate
    bl_st = baseline/1000*srate
    chunksize = chunk_size/1000*srate
    merge_chunk_size = merge_chunk_size/1000*srate

    #account for cue in Decision tasks (different start point for data)
    if task in ['DecisionAud']:
        st_tp = 600/1000*srate
    elif task in ['DecisionVis']:
        st_tp = 500/1000*srate
    else:
        st_tp = 0

    subjs = list(); tasks = list(); pthr = list(); elecs = list(); starts = list(); ends = list(); 

    for i, e in enumerate(active_elecs): #for each active electrode

        pvals = list();
        edata = data[i,:,:]
        nozero = np.copy(edata)
        nozero[:,nozero.mean(axis=0)<0] = 0 #zero out negative values in mean

        for j in np.arange(abs(bl_st)+st_tp, edata.shape[1]): #ttest at every time point
            (t, p) = stats.ttest_1samp(nozero[:,j], 0)
            pvals.append(p)
        thr = fdr_correct.fdr2(pvals, q = 0.05) #fdr correct p values
        H = np.array(np.array(pvals<thr)).astype('int')

        if (thr>0): #a corrected pvalue is possible (>0)

            #find elecs with window that > chunksize and > threshold (10%)
            passed_thresh = edata[:, abs(bl_st)+st_tp::].mean(axis=0)>thresh
            sig_and_thresh = H * passed_thresh #significant windows
            start_idx, end_idx = make_windows(sig_and_thresh, st_tp, bl_st, edata) #calculate significant windows
            chunk = (end_idx - start_idx) >= chunksize
            
            if sum(chunk) > 0: #passed the 10% threshold
                #significant windows on elecs that passed threshold (10%) (ignoring threshold and chunksize)
                start_idx, end_idx = make_windows(sig_and_thresh, st_tp, bl_st, edata)

                merge_chunk = (start_idx[1:] - end_idx[:-1])> merge_chunk_size #combine window separated by <100ms

                tmp = np.append(1,merge_chunk).astype('bool')
                end_idx = end_idx[np.append(np.where(np.in1d(start_idx, start_idx[tmp]))[0][1:]-1, -1)]
                start_idx = start_idx[tmp]           

                #drop chunks that <100ms
                chunk = (end_idx - start_idx) >= chunksize
                start_idx = start_idx[chunk]
                end_idx = end_idx[chunk]
                
            else: #no significant chunks
                start_idx = np.zeros((1,))
                end_idx = np.zeros((1,))
                
        else: #thr<0
            start_idx = np.zeros((1,))
            end_idx = np.zeros((1,))

        subjs.extend([subj] * len(start_idx))
        tasks.extend([task] * len(end_idx))
        elecs.extend([e] * len(start_idx))
        pthr.extend([thr] * len(end_idx))
        starts.extend(start_idx)
        ends.extend(end_idx)

    sig_windows = pd.DataFrame({'subj':subjs, 'task':tasks, 'elec':elecs, 'pthreshold':pthr, 'start_idx':starts, 'end_idx':ends})
    sig_windows = sig_windows[['subj','task','elec', 'start_idx','end_idx','pthreshold']]
    sig_windows = sig_windows.set_index('elec')
    
    params = pd.DataFrame({'bl_st' : bl_st, 'srate' : srate, 'chunksize' : chunksize, 'merge_chunk_size' : merge_chunk_size, 'thresh' : thresh}, index = range(1))
    return sig_windows, params

def make_windows(sig_and_thresh, st_tp, bl_st, edata):
    """
    calculates window size given a vector of 1/0s for significant timepoints
    input: sig_and_thresh = vector of 1/0 for passing pvalue cutoff
    st_tp = data offset (in srate)
    """
    difference = np.diff(sig_and_thresh, n = 1, axis = 0)
    start_idx = np.where(difference==1)[0]+1
    end_idx = np.where(difference == -1)[0]

    if start_idx.size > end_idx.size: #last chunk goes until end
        end_idx = np.append(end_idx, int(edata.shape[1]-abs(bl_st)-st_tp))

    elif start_idx.size < end_idx.size:
        start_idx = np.append(0, start_idx) #starts immediately significant

    if (start_idx.size!=0):
        if (start_idx[0] > end_idx[0]): #starts immediately significant
            start_idx = np.append(0, start_idx)
    if (start_idx.size!=0):
        if (end_idx[-1] < start_idx[-1]):#significant until end
            end_idx = np.append(end_idx, int(edata.shape[1]-abs(bl_st)-st_tp))

    start_idx = start_idx + st_tp #shift by st_tp
    end_idx = end_idx + st_tp
    return start_idx, end_idx

def plot_shadeplot(edata, sig_window, params, ax = None):
    """
    plots shade plot with significance window in in red
    input:
        edata - electrode time series (trials x time)
        sig_window - dataframe with significance windows for that electrode
        params - parameters used to calculate the window
        ax - (optional) where to plot. default None to plot in gca
    """
    if ax is None: #if didn't specify where to plot it
        ax = plt.gca()

    #get parameters
    bl_st = params.bl_st.values[0]
    start_idx = sig_window.start_idx
    end_idx = sig_window.end_idx
    
    #format into list
    if start_idx.size > 1:
        start_idx = start_idx.values
        end_idx = end_idx.values
    else:
        start_idx = [start_idx]
        end_idx = [end_idx]
    
    #plot
    scale_min = edata.mean(axis = 0).min() - 10
    scale_max = edata.mean(axis = 0).max() + 10
    tmp = (np.arange(scale_min, scale_max))

    ax.plot(np.arange(bl_st, edata.shape[1]+bl_st), edata.mean(axis = 0), zorder = 1, linewidth = 3)
    sem = np.std(edata, axis = 0)/np.sqrt(edata.shape[0])
    ax.fill_between(np.arange(bl_st, edata.shape[1]+bl_st), edata.mean(axis = 0)+sem, edata.mean(axis=0)-sem, alpha = 0.5, zorder = 0, edgecolor = 'None', facecolor = 'slateblue')

    #axes
    ax.axhline(y=0, color = 'k', lw = 3) #xaxis
    ax.axvline(x = 0, color = 'k', lw = 3)
    
    #format fig
    ax.set_ylabel('% change HG', fontsize = 14)
    ax.set_xlabel('time (ms)', fontsize = 14)
    ax.autoscale(tight=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    #color significant window in red
    if len(start_idx)>0:
        for i, s in enumerate(start_idx):
            tmp = np.arange(s, end_idx[i])
            start = s
            finish = end_idx[i]
            ax.plot(tmp, np.zeros(tmp.size), color = 'r', linewidth = 3.5, label = (start, finish))
            ax.legend()
    return ax



def shadeplots_allelecs_2conditions(DATASET, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta', chunk_size = 100, baseline = -500):
    """ 
    calculate onset and offset window for difference between 2 conditions (real and empty)
    saves csv for each sub/task for easy plotting later
    #only relevant for EmoGen (not adjusted for my data start times)

    """

    subj, task = DATASET.split('_')

    filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent.mat')
    data = loadmat.loadmat(filename)
    srate = data['srate']
    active_elecs = data['active_elecs']
    data = data['data_percent']

    filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent_empty.mat')
    data_empty = loadmat.loadmat(filename)
    data_empty = data_empty['data_percent']

    #convert to srate
    bl_st = baseline/1000*srate
    chunksize = chunk_size/1000*srate
    st_tp = 0

    filename = os.path.join(SJdir, 'PCA', 'ShadePlots_allelecs', ''.join([subj, '_', task, '_real_vs_empty.csv']))
    subjs = list(); tasks = list(); pthr = list(); elecs = list(); starts = list(); ends = list(); 

    for i, e in enumerate(active_elecs):

        pvals = list();
        edata = data[i,:]
        edata_empty = data_empty[i,:]

        #ttest between conditions for every time point
        for j in np.arange(abs(bl_st)+st_tp, edata.shape[1]):
            (t, p) = stats.ttest_ind(edata[:,j], edata_empty[:,j], equal_var = True)
            pvals.append(p)
        thr = fdr_correct.fdr2(pvals, q = 0.05)
        H = np.array(np.array(pvals)<thr).astype('int')

        #significance windows
        difference = np.diff(H, n = 1, axis = 0)
        start_idx = np.where(difference==1)[0]+1
        end_idx = np.where(difference == -1)[0]

        if start_idx.size > end_idx.size: #last chunk goes until end
            end_idx = np.append(end_idx, int(edata.shape[1]-abs(bl_st)-st_tp))

        elif start_idx.size < end_idx.size:
            start_idx = np.append(0, start_idx) #starts immediately significant

        if (start_idx.size!=0):
            if (start_idx[0] > end_idx[0]): #starts immediately significant
                start_idx = np.append(0, start_idx)
        if (start_idx.size!=0):
            if (end_idx[-1] < start_idx[-1]):#significant until end
                end_idx = np.append(end_idx, int(edata.shape[1]-abs(bl_st)-st_tp))

        #drop chunks that < chunk_size
        chunk = (end_idx - start_idx) >= chunksize
        start_idx = start_idx[chunk]
        end_idx = end_idx[chunk]
 
        
        subjs.extend([subj] * len(start_idx))
        tasks.extend([task] * len(end_idx))
        elecs.extend([e] * len(start_idx))
        pthr.extend([thr] * len(end_idx))
        starts.extend(start_idx)
        ends.extend(end_idx)

        data_dict = {'edata':edata, 'edata_empty':edata_empty, 'bl_st':bl_st, 'start_idx':start_idx, 'end_idx':end_idx, 'srate':srate,'chunksize':chunksize}
        data_path = os.path.join(SJdir, 'PCA','ShadePlots_allelecs', 'data',''.join([subj, '_', task, '_e', str(e), '_real_vs_empty.p']))
        with open(data_path, 'w') as f:
            pickle.dump(data_dict, f)
            f.close()

    sig_windows = pd.DataFrame({'subj':subjs, 'task':tasks, 'elec':elecs, 'pthreshold':pthr, 'start_idx':starts, 'end_idx':ends})
    sig_windows = sig_windows[['subj','task','elec', 'start_idx','end_idx','pthreshold']]
    sig_windows.to_csv(filename)
