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
import glob

def shadeplots_clusters(DATASET, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta', thresh = 15, chunk_size = 100, baseline = -500, black_chunk_size = 0):
    """ 
    calculate onset and offset window for every active electrode (ignoring clusters)
    saves csv for each sub/task for easy plotting later
    """

    subj, task = DATASET.split('_')

    filenames = glob.glob(os.path.join(SJdir, 'PCA', 'SingleTrials_hclust', '_'.join([subj, task, 'c*mat'])))

    subjs = list(); tasks = list(); pthr = list(); clusts = list(); starts = list(); ends = list(); 
    for filename in filenames:

        data = loadmat.loadmat(filename)
        srate = data['srate']
        cdata = data['cdata']
        RTs = data['RTs_all'] #new

        cluster = int(filename.split('_')[-1].split('.')[0][1:])

        #convert to srate
        bl_st = baseline/1000*srate
        chunksize = chunk_size/1000*srate
        black_chunksize = black_chunk_size/1000*srate

        RTs = RTs+abs(bl_st) #new

        if task in ['DecisionAud']:
            st_tp = 600/1000*srate
        elif task in ['DecisionVis']:
            st_tp = 500/1000*srate
        else:
            st_tp = 0

        pvals = list();
        nozero = np.copy(cdata)
        nozero[:,nozero.mean(axis=0)<0] = 0 #zero out negative values

        for t in np.arange(abs(bl_st)+st_tp, cdata.shape[1]):
            (t, p) = stats.ttest_1samp(nozero[:,t], 0)
            pvals.append(p)

        thr = fdr_correct.fdr2(pvals, q = 0.05)
        H = np.array((pvals<thr)).astype('int')

        if (thr>0):

            #find elecs with window that > chunksize and > threshold (10%)
            passed_thresh = cdata[:, abs(bl_st)+st_tp::].mean(axis=0)>thresh
            sig_and_thresh = H * passed_thresh
            difference = np.diff(sig_and_thresh, n = 1, axis = 0)
            start_idx = np.where(difference==1)[0]+1
            end_idx = np.where(difference == -1)[0]

            if start_idx.size > end_idx.size: #last chunk goes until end
                end_idx = np.append(end_idx, int(cdata.shape[1]-abs(bl_st)-st_tp))

            elif start_idx.size < end_idx.size:
                start_idx = np.append(0, start_idx) #starts immediately significant

            if (start_idx.size!=0):
                if (start_idx[0] > end_idx[0]): #starts immediately significant
                    start_idx = np.append(0, start_idx)
            if (start_idx.size!=0):
                if (end_idx[-1] < start_idx[-1]):#significant until end
                    end_idx = np.append(end_idx, int(cdata.shape[1]-abs(bl_st)-st_tp))

            start_idx = start_idx + st_tp #shift by st_tp
            end_idx = end_idx + st_tp
            chunk = (end_idx - start_idx) >= chunksize
            if sum(chunk) > 0:
                #significant windows on those that passed threshold (10%) (ignoring threshold and chunksize)
                difference = np.diff(H, n = 1, axis = 0)
                start_idx = np.where(difference==1)[0]+1
                end_idx = np.where(difference == -1)[0]

                if start_idx.size > end_idx.size: #last chunk goes until end
                    end_idx = np.append(end_idx, int(cdata.shape[1]-abs(bl_st)-st_tp))

                elif start_idx.size < end_idx.size:
                    start_idx = np.append(0, start_idx) #starts immediately significant

                if (start_idx.size!=0):
                    if (start_idx[0] > end_idx[0]): #starts immediately significant
                        start_idx = np.append(0, start_idx)
                if (start_idx.size!=0):
                    if (end_idx[-1] < start_idx[-1]):#significant until end
                        end_idx = np.append(end_idx, int(cdata.shape[1]-abs(bl_st)-st_tp))

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
        clusts.extend([cluster] * len(start_idx))
        pthr.extend([thr] * len(end_idx))
        starts.extend(start_idx)
        ends.extend(end_idx)
        
        data_dict = {'cdata':cdata, 'bl_st':bl_st, 'start_idx':start_idx, 'end_idx':end_idx, 'srate':srate, 'chunksize': chunksize, 'black_chunksize':black_chunksize, 'cluster':cluster, 'thresh': thresh, 'RTs':RTs}
        data_path = os.path.join(SJdir, 'PCA','ShadePlots_hclust', 'data',''.join([subj, '_', task, '_c', str(cluster), '.p']))
        
        with open(data_path, 'w') as f:
            pickle.dump(data_dict, f)
            f.close()
    
    fname = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', ''.join([subj, '_', task, '.csv']))
    sig_windows = pd.DataFrame({'subj':subjs, 'task':tasks, 'cluster':clusts, 'pthreshold':pthr, 'start_idx':starts, 'end_idx':ends})
    sig_windows = sig_windows[['subj','task','cluster', 'start_idx','end_idx','pthreshold']]
    sig_windows.to_csv(fname)

if __name__ == '__main__':
    DATASET = sys.argv[1]
    shadeplots_clusters(DATASET)
