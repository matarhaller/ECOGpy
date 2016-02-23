from __future__ import division
import pandas as pd
import os
import loadmat
import scipy.stats as stats
import fdr_correct
import glob
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

def shadeplots_allelecs_plot(SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/', subj_task = ''):
    '''
    edited for empty 2/2/15
    '''
    
    reconlist = os.path.join(SJdir, 'PCA', 'reconlist.csv')
    reconlist = pd.read_csv(reconlist)

    if len(subj_task) == 0: #none specified, then loop on all
        for x in reconlist.values:
            subj_task, _, _,_, _ = x
            subj, task = subj_task.split('_')
            plot_shadeplot(subj, task)
        else:
            plot_shadeplot(subj, task)



def plot_shadeplot(subj, task, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'):
        """
        plot shade plots with activity window in red (from ShadePlots_allelecs.py)
        """
        files = glob.glob(os.path.join(SJdir, 'PCA', 'ShadePlots_allelecs', 'data', ''.join([subj, '_', task, '*empty.p'])))

        for f in files:
            print f
            elecname = f.split('_')[-2].split('.p')[0] #before empty was -1

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

            ax.set_title(' '.join(['empty trials', subj, task, ':', elecname, 'chunksize', str(chunksize), 'smoothing', str(black_chunksize),'thresh', str(thresh)]))
            plt.savefig(os.path.join(SJdir, 'PCA', 'ShadePlots_allelecs','images', ''.join([subj, '_', task, '_', elecname, '_empty'])))
            plt.close()
        


def plot_shadeplot_2conditions(subj, task, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'):
        """
        plot shade plots for real vs empty  with activity window in red (from ShadePlots_allelecs.py) 
        """
        files = glob.glob(os.path.join(SJdir, 'PCA', 'ShadePlots_allelecs', 'data', ''.join([subj, '_', task, '*real_vs_empty.p'])))

        for f in files:
            print f
            elecname = f.split('_')[-4].split('.p')[0] 

            with open(f, 'r') as x:
                data_dict = pickle.load(x)
                x.close()

            #map dictionary to variables
            edata, edata_empty, bl_st, srate, start_idx, end_idx, chunksize = [data_dict.get(k) for k in ['edata','edata_empty', 'bl_st', 'srate', 'start_idx', 'end_idx', 'chunksize']]

            #plot
            f, ax = plt.subplots(figsize = (10,10))
            scale_min = edata.mean(axis = 0).min() - 10
            scale_max = edata.mean(axis = 0).max() + 10
            tmp = (np.arange(scale_min, scale_max))

            ax.plot(np.arange(bl_st, edata.shape[1]+bl_st), edata.mean(axis = 0), zorder = 1, linewidth = 3, label = 'real')
            sem = np.std(edata, axis = 0)/np.sqrt(edata.shape[0])
            ax.fill_between(np.arange(bl_st, edata.shape[1]+bl_st), edata.mean(axis = 0)+sem, edata.mean(axis=0)-sem, alpha = 0.5, zorder = 0, edgecolor = 'None', facecolor = 'slateblue')

            ax.plot(np.arange(bl_st, edata_empty.shape[1]+bl_st), edata_empty.mean(axis = 0), zorder = 1, linewidth = 3, label = 'empty')
            sem = np.std(edata_empty, axis = 0)/np.sqrt(edata_empty.shape[0])
            ax.fill_between(np.arange(bl_st, edata_empty.shape[1]+bl_st), edata_empty.mean(axis = 0)+sem, edata_empty.mean(axis=0)-sem, alpha = 0.5, zorder = 0, edgecolor = 'None', facecolor = 'darkgoldenrod')

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
            else:
                ax.legend()

            ax.set_title(' '.join(['real vs empty:', subj, task, ':', elecname, 'chunksize', str(chunksize)]))
            plt.savefig(os.path.join(SJdir, 'PCA', 'ShadePlots_allelecs','images', ''.join([subj, '_', task, '_', elecname, '_real_vs_empty'])))
            plt.close()
