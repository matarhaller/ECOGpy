from __future__ import division
import pandas as pd
import os
import loadmat
import scipy.stats as stats
import fdr_correct
import matplotlib.pyplot as plt
import numpy as np
import sys
import cPickle as pickle

def shadeplots_allelecs(DATASET, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta', thresh = 10, chunk_size = 100, baseline = -500, black_chunk_size = 0):
    """ 
    calculate onset and offset window for every active electrode (ignoring clusters)
    saves csv for each sub/task for easy plotting later

    EDITED FOR EMPTY TRIALS 2-2-15
    """

    subj, task = DATASET.split('_')

    filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent_empty.mat')
    data = loadmat.loadmat(filename)
    srate = data['srate']
    active_elecs = data['active_elecs']
    data = data['data_percent']

    #convert to srate
    bl_st = baseline/1000*srate
    chunksize = chunk_size/1000*srate
    black_chunksize = black_chunk_size/1000*srate

    if task in ['DecisionAud']:
        st_tp = 600/1000*srate
    elif task in ['DecisionVis']:
        st_tp = 500/1000*srate
    else:
        st_tp = 0

    filename = os.path.join(SJdir, 'PCA', 'ShadePlots_allelecs', ''.join([subj, '_', task, '_empty.csv']))
    subjs = list(); tasks = list(); pthr = list(); elecs = list(); starts = list(); ends = list(); 

    for i, e in enumerate(active_elecs):

        pvals = list();
        edata = data[i,:]
        nozero = np.copy(edata)
        nozero[:,nozero.mean(axis=0)<0] = 0 #zero out negative values in mean

        for j in np.arange(abs(bl_st)+st_tp, edata.shape[1]):
            (t, p) = stats.ttest_1samp(nozero[:,j], 0)
            pvals.append(p)
        thr = fdr_correct.fdr2(pvals, q = 0.05)
        H = np.array(np.array(pvals<thr)).astype('int')

        if (thr>0):

            #find elecs with window that > chunksize and > threshold (10%)
            passed_thresh = edata[:, abs(bl_st)+st_tp::].mean(axis=0)>thresh
            sig_and_thresh = H * passed_thresh
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
            chunk = (end_idx - start_idx) >= chunksize
            if sum(chunk) > 0:
                #significant windows on elecs that passed threshold (10%) (ignoring threshold and chunksize)
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

                start_idx = start_idx + st_tp #shift by st_tp
                end_idx = end_idx + st_tp

                black_chunk = (start_idx[1:] - end_idx[:-1])> black_chunksize #combine window separated by <200ms

                tmp = np.append(1,black_chunk).astype('bool')
                end_idx = end_idx[np.append(np.where(np.in1d(start_idx, start_idx[tmp]))[0][1:]-1, -1)]
                start_idx = start_idx[tmp]           

                #drop chunks that <100ms
                chunk = (end_idx - start_idx) >= chunksize
                start_idx = start_idx[chunk]
                end_idx = end_idx[chunk]
            else: #no chunks
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

        data_dict = {'edata':edata, 'bl_st':bl_st, 'start_idx':start_idx, 'end_idx':end_idx, 'srate':srate,'thresh':thresh, 'chunksize':chunksize, 'black_chunksize':black_chunksize}
        data_path = os.path.join(SJdir, 'PCA','ShadePlots_allelecs', 'data',''.join([subj, '_', task, '_e', str(e), '_empty.p']))
        with open(data_path, 'w') as f:
            pickle.dump(data_dict, f)
            f.close()

    sig_windows = pd.DataFrame({'subj':subjs, 'task':tasks, 'elec':elecs, 'pthreshold':pthr, 'start_idx':starts, 'end_idx':ends})
    sig_windows = sig_windows[['subj','task','elec', 'start_idx','end_idx','pthreshold']]
    sig_windows.to_csv(filename)



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
