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

def shadeplots_median_split(subj, task, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta', thresh = 0, chunk_size = 0, baseline = -500, black_chunk_size = 0):
    """ 
    takes median split of RTs and calculates difference between them (short vs long RT trials)
    only runs on elecs that are easy/difficult from overlap csv
    calculate onset and offset window for given electrode.
    Compares short vs long RT trials for each electrode for the unique/overlap tasks
    saves csv for each sub/task for easy plotting later

    """

    filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent.mat')
    data = loadmat.loadmat(filename)
    srate = data['srate']
    elecs = data['active_elecs']
    RTs = data['RTs']
    data = data['data_percent']

    median_value = np.median(RTs)
    shortdata = data[:, RTs<median_value, :]
    longdata = data[:, RTs>median_value, :]
   

    #convert to srate
    bl_st = baseline/1000*srate
    chunksize = chunk_size/1000*srate
    black_chunksize = black_chunk_size/1000*srate

    subjs = list();  pthr = list(); elecs = list(); starts = list(); ends = list(); 

    overlapfile = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'smoothed', 'mean_traces', 'csv_files', subj+'_ovelapped_dur_elecs.csv')
    df = pd.read_csv(overlapfile)
    elecs_list = np.unique((df.easy.fillna(0) + df.difficult.fillna(0)).values)

    for i, e in enumerate(elecs_list):

        idx = np.in1d(elecs_list, e)
    
        edataShort = shortdata[idx,:,:].squeeze()
        edataLong = longdata[idx,:,:].squeeze()

        pvals = list();

        for j in np.arange(abs(bl_st), edataShort.shape[1]):
            (t, p) = stats.ttest_ind(edataShort[:,j], edataLong[:,j])
            pvals.append(p)
        thr = fdr_correct.fdr2(pvals, q = 0.05)
        H = np.array(np.array(pvals<thr)).astype('int')

        if (thr>0):

            #find elecs with window that > chunksize and > threshold (10%)
            passed_thresh = abs(edataShort[:, abs(bl_st)::].mean(axis=0) - edataLong[:, abs(bl_st)::].mean(axis = 0)) >thresh #difference between blocks is > 10% threshold
            sig_and_thresh = H * passed_thresh
            difference = np.diff(sig_and_thresh, n = 1, axis = 0)
            start_idx = np.where(difference==1)[0]+1
            end_idx = np.where(difference == -1)[0]

            if start_idx.size > end_idx.size: #last chunk goes until end
                end_idx = np.append(end_idx, int(edataShort.shape[1]-abs(bl_st)))

            elif start_idx.size < end_idx.size:
                start_idx = np.append(0, start_idx) #starts immediately significant

            if (start_idx.size!=0):
                if (start_idx[0] > end_idx[0]): #starts immediately significant
                    start_idx = np.append(0, start_idx)

            if (start_idx.size!=0):
                if (end_idx[-1] < start_idx[-1]):#significant until end
                    end_idx = np.append(end_idx, int(edataShort.shape[1]-abs(bl_st)))

            chunk = (end_idx - start_idx) >= chunksize

            if sum(chunk) > 0:
                #significant windows on elecs that passed threshold (10%) (ignoring threshold and chunksize)
                difference = np.diff(H, n = 1, axis = 0)
                start_idx = np.where(difference==1)[0]+1
                end_idx = np.where(difference == -1)[0]

                if start_idx.size > end_idx.size: #last chunk goes until end
                    end_idx = np.append(end_idx, int(edataShort.shape[1]-abs(bl_st)))

                elif start_idx.size < end_idx.size:
                    start_idx = np.append(0, start_idx) #starts immediately significant

                if (start_idx.size!=0):
                    if (start_idx[0] > end_idx[0]): #starts immediately significant
                        start_idx = np.append(0, start_idx)

                if (start_idx.size!=0):
                    if (end_idx[-1] < start_idx[-1]):#significant until end
                        end_idx = np.append(end_idx, int(edataShort.shape[1]-abs(bl_st)))


                black_chunk = (start_idx[1:] - end_idx[:-1]) > black_chunksize #combine window separated by <200ms

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
        elecs.extend([e] * len(start_idx))
        pthr.extend([thr] * len(end_idx))
        starts.extend(start_idx)
        ends.extend(end_idx)

        data_dict = {'edataShort':edataShort, 'edataLong':edataLong, 'bl_st':bl_st, 'start_idx':start_idx, 'end_idx':end_idx, 'srate':srate,'thresh':thresh, 'chunksize':chunksize, 'black_chunksize':black_chunksize}
        data_path = os.path.join(SJdir, 'PCA','ShadePlots_hclust', 'elecs', 'significance_windows', 'smoothed', 'mean_traces', 'csv_files', ''.join([subj,task, '_', 'Long_vs_Short', '_e', str(int(e)), '.p']))
       
        with open(data_path, 'w') as f:
            pickle.dump(data_dict, f)
            f.close()

    filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'smoothed','mean_traces', 'csv_files', '_'.join([subj, task, 'long_vs_short_RTs']) +'.csv')
    sig_windows = pd.DataFrame({'subj':subjs, 'elec':elecs, 'pthreshold':pthr, 'start_idx':starts, 'end_idx':ends})
    sig_windows = sig_windows[['subj', 'elec', 'start_idx','end_idx','pthreshold']]
    sig_windows.to_csv(filename)
    
    return sig_windows


def plot_shadeplot(subj,task, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'):
        """
        plot shade plots with activity window in red (from ShadePlots_allelecs_plot.py)
        """
        files = glob.glob(os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs','significance_windows', 'smoothed', 'mean_traces', 'csv_files', ''.join([subj,task, '_', 'Long_vs_Short','*.p']))) #electrodes
        #files = filter(lambda f: not(f[-6:-2] == 'resp'), files) #drop resp files

        for f in files:
            print f
            elecname = f.split('_')[-1].split('.p')[0] 

            with open(f, 'r') as x:
                data_dict = pickle.load(x)
                x.close()

            #map dictionary to variables
            edataShort, edataLong, bl_st, srate, start_idx, end_idx, chunksize, black_chunksize, thresh = [data_dict.get(k) for k in ['edataShort','edataLong', 'bl_st', 'srate', 'start_idx', 'end_idx', 'chunksize', 'black_chunksize', 'thresh']]

            #plot
            f, ax = plt.subplots(figsize = (10,10))

            if edataShort.shape[0] == 0: #if not active
                edataShort = np.zeros_like(edataLong)
            elif edataLong.shape[0] == 0:
                edataLong = np.zeros_like(edataShort)

            scale_min = min(edataLong.mean(axis = 0).min(), edataShort.mean(axis = 0).min()) - 10
            scale_max = max(edataLong.mean(axis = 0).max(), edataShort.mean(axis = 0).max()) + 10
            tmp = (np.arange(scale_min, scale_max))
            
            ax.plot(np.arange(bl_st, edataShort.shape[1]+bl_st), edataShort.mean(axis = 0), zorder = 1, linewidth = 3, color = 'green', label = 'Short RTs')
            sem = np.std(edataShort, axis = 0)/np.sqrt(edataShort.shape[0])
            ax.fill_between(np.arange(bl_st, edataShort.shape[1]+bl_st), edataShort.mean(axis = 0)+sem, edataShort.mean(axis=0)-sem, alpha = 0.5, zorder = 0, edgecolor = 'None', facecolor = 'green', label = None)
            
            ax.plot(np.arange(bl_st, edataLong.shape[1]+bl_st), edataLong.mean(axis = 0), zorder = 1, linewidth = 3, color = 'blue', label = 'Long RTs')
            sem = np.std(edataLong, axis = 0)/np.sqrt(edataLong.shape[0])
            ax.fill_between(np.arange(bl_st, edataLong.shape[1]+bl_st), edataLong.mean(axis = 0)+sem, edataLong.mean(axis=0)-sem, alpha = 0.5, zorder = 0, edgecolor = 'None', facecolor = 'slateblue', label = None)

            ax.plot(np.arange(bl_st, edataShort.shape[1]+bl_st), np.zeros(edataShort.shape[1]), color = 'k', linewidth = 3, label = None) #xaxis
            ax.plot(np.zeros(tmp.size), tmp, color = 'k', linewidth = 3, label = None) #yaxis

            ax.set_ylabel('% change HG')
            ax.set_xlabel('time (ms)')
            ax.autoscale(tight=True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            legend1 = ax.legend(loc = 1)

            if start_idx.size>0:
                for i, s in enumerate(start_idx):
                    tmp = np.arange(s, end_idx[i])
                    start = s
                    finish = end_idx[i]
                    ax.plot(tmp, np.zeros(tmp.size), color = 'r', linewidth = 3.5, label = (start, finish))
                    ax.legend()

            
            ax.set_title(' '.join(['Long vs Short', subj, ':', elecname, 'chunksize', str(chunksize), 'smoothing', str(black_chunksize),'thresh', str(thresh)]))
            plt.savefig(os.path.join(SJdir, 'PCA', 'ShadePlots_hclust','elecs','significance_windows', 'smoothed', 'mean_traces', 'images','_'.join([subj,task, 'long_vs_short', elecname])))
            plt.close()


def plot_average_overlap(subj, task, resplocked = False, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'):
        """
        plots of traces (not a shade plot bc no significance window calculated)
        average for
        1. easy task overlap elecs
        2. diff task overlap elecs
        3. diff task unique elcs
        """
        
        filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent.mat')
        data = loadmat.loadmat(filename)
        srate = data['srate']
        elecs = data['active_elecs']
        RTs = data['RTs']
        bl_st = data['Params']['bl_st']/1000*srate
        data = data['data_percent']
        
        RTs = RTs+abs(bl_st)
        bl_st = int(bl_st)

        overlapfile = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'smoothed', 'mean_traces', 'csv_files', subj+'_ovelapped_dur_elecs.csv')
        df = pd.read_csv(overlapfile)
        easy_overlap = df.easy.dropna()[np.in1d(df.overlapped_elecs.dropna(), df.easy.dropna())]
        diff_overlap = df.difficult.dropna()[np.in1d(df.overlapped_elecs.dropna(), df.difficult.dropna())]
        diff_unique = df.unique_to_diff.dropna()

        elec_dict = {'easy_overlap':easy_overlap, 'diff_overlap':diff_overlap, 'diff_unique':diff_unique}
        data_dict = dict()

        #average data per grouping
        for k in elec_dict.keys():
            elec_list = elec_dict[k]
            eidx = np.in1d(elecs, elec_list)

            if resplocked:
                tmp = np.empty((data.shape[0], data.shape[1], len(np.arange(bl_st,abs(bl_st))))) #elecs x trials x time
                for j, e in enumerate(eidx): #elecs
                    tmp2 = np.empty((data.shape[1], len(np.arange(bl_st, abs(bl_st))))) #per elec, trials x time
                    for i, r in enumerate(RTs): #trials
                        tmp2[i,:] = data[e,i,(r-abs(bl_st)):(r+abs(bl_st))]
                    tmp[j, :, :] = tmp2
                data_dict[k] = tmp.mean(axis = 1).mean(axis = 0)    
            else:
                data_dict[k] = data[eidx,:,:].mean(axis = 1).mean(axis = 0)

        #plot
        f, ax = plt.subplots(1, 1, figsize = (30,10))
        scale_min = min([min(data_dict[x]) for x in data_dict.keys()])
        scale_max = max([max(data_dict[x]) for x in data_dict.keys()])

        tmp = (np.arange(scale_min, scale_max))

        for i, k in enumerate(data_dict.keys()):
            data = data_dict[k]
            ax.plot(np.arange(bl_st, data.shape[0]+bl_st), data, zorder = 1, linewidth = 3, label = k)

        ax.set_ylim([scale_min, scale_max])

        ax.axhline(y = 0, color = 'k', lw = 3, label = None) #xaxis
        ax.axvline(x = 0, color = 'k', lw = 3, label = None)

        ax.set_ylabel('% change HG')
        ax.set_xlabel('time (ms)')
        ax.autoscale(tight=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        legend1 = ax.legend(loc = 'best')

        ax.set_title(' '.join([subj, task]))

        filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust','elecs','significance_windows', 'smoothed', 'mean_traces', 'images', 'median_split', '_'.join([subj,task, 'easy_diff_overlap_unique']))
        if resplocked:
            filename = filename + '_resplocked'
 
        plt.savefig(filename+'.png')
        plt.close()

