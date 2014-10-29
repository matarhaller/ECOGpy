from __future__ import division
import pandas as pd
import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import scipy.stats as stats


def plot_latencies(static = True):
    """
    plots histograms for lats_pro for each electrode for data within its significance window
    also plots single trial latencies (with RT, not normalized)
    saves pngs in PCA/Stats/latencies
    saves csv with skew, skew test pvalue, median and max for each duration electrode.
    """
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'

    #only take duration electrodes
    filename = os.path.join(SJdir,'PCA', 'Stats', 'single_electrode_windows_withdesignation_EDITED.csv')
    df = pd.read_csv(filename)

    df = df.query("pattern == 'D'")
    df = df.sort(['subj','task','elec'])

    skew_list, pval_list, mean_list, median_list, elec_list, subj_list, task_list = [[] for i in range(7)]

    for s_t in df.groupby(['subj','task']):
        subj, task = s_t[0]
        elecs = s_t[1].elec #duration electrodes

        #pickle file with latencies per subj/task
        if static:
            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'static', 'data', ''.join([subj, '_', task, '.p']))
        else:
            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'data', ''.join([subj, '_', task, '.p']))
        
        data_dict = pickle.load( open(filename, "rb" ) )

        #RTs, latencies, and normalized latencies
       
        RTs_dict, lats_dict, lats_pro_dict, bl_st = [data_dict[k] for k in ['RTs','lats','lats_pro','bl_st']] #static also has lats_pro, but not useful

        for elec in elecs:
            onset = df[(df.subj == subj) & (df.task == task) & (df.elec == elec)].start_idx.values[0]
            offset = df[(df.subj == subj) & (df.task == task) & (df.elec == elec)].end_idx_resp.values[0]

            ridx = RTs_dict[elec].argsort(axis=0) 
            RTs = RTs_dict[elec][ridx]
            RTs = RTs - abs(bl_st) #remove bl from data (start from stim onset)
            lats = lats_dict[elec][ridx]+onset #trials sorted by RTs, add HG onset back on so calculated from stim onset
            lats_pro = lats_pro_dict[elec]

            #plot
            f,axs = plt.subplots(1,2, figsize = (10,5))
            f.suptitle(' '.join([subj, task, '-', 'e'+str(elec)]), fontsize = 14)
            axs[0].set_title(' : '.join([str(onset)+'ms', str(offset)+'ms']))
            #axs[1].set_title('latencies (normalized)')

            for i in np.arange(2):
                axs[i].autoscale(enable = True, tight = True)

            for j in np.arange(len(RTs)):
                axs[0].plot((RTs[j], RTs[j]), (j-0.5, j+0.5), 'k', linewidth = 3,zorder = 1)
                axs[0].plot((lats[j], lats[j]), (j-0.5, j + 0.5), 'blue', linewidth = 3, zorder = 1)

            #axs[1].hist(lats_pro[~np.isnan(lats_pro)], bins = np.arange(0, 1, 0.1), color = 'blue')
         
            #format data for kernel density estimation
            #y = lats_pro.squeeze()
            y = lats.squeeze() #now doesn't plot the kde pdf (only works for lats_pro)
            y = y[~np.isnan(y)] #drop nan trials
            x_grid = np.linspace(0, 1, 1000)

            #skew calculation
            n, min_max, mn, var, skw, kurt = stats.describe(y)
            pval = stats.skewtest(y)[1]
            med = np.median(y)

            #cv for bw parameter
            #grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv = 20)
            #grid.fit(y[:,None])
            #bw = grid.best_params_['bandwidth']

            #kde = KernelDensity(bandwidth = bw)
            #kde.fit(y[:,np.newaxis])
            #log_pdf = kde.score_samples(x_grid[:,np.newaxis])
            #pdf = np.exp(log_pdf)

            #plot
            #axs[1].plot(x_grid, pdf, color = 'blue', lw = 3)
            #axs[1].hist(y, 10, color = 'gray', alpha = 0.4, normed = True)
            axs[1].hist(y, 10, color = 'gray', alpha = 0.4, normed = False)
            axs[1].set_title('skew %.2f, pval %.3f' %(skw, pval))
            axs[1].axvline(med, color = 'r', lw = 3, label = 'median')
            axs[1].axvline(mn, color = 'g', lw = 3, label = 'mean')
            axs[1].legend(loc = 'lower right')

            
            if static:
                filename = os.path.join(SJdir,'PCA','Stats', 'latencies', 'static', '_'.join([subj, task, 'e'+str(elec)+'.png']))
            else:
                filename = os.path.join(SJdir,'PCA','Stats', 'latencies', '_'.join([subj, task, 'e'+str(elec)+'.png']))

            plt.savefig(filename, dpi = 100)
            plt.close()

            #add to csv
            skew_list.append(skw)
            pval_list.append(pval)
            mean_list.append(mn)
            median_list.append(med)
            elec_list.append(elec)
            subj_list.append(subj)
            task_list.append(task)

    df = pd.DataFrame({'subj':subj_list, 'task':task_list, 'elec': elec_list, 'skew':skew_list, 'pval': pval_list,'mean': mean_list, 'median': median_list})
    if static:
        filename = os.path.join(SJdir,'PCA', 'Stats','latencies', 'static','skew_stats.csv')
    else:
        filename = os.path.join(SJdir,'PCA', 'Stats','latencies', 'skew_stats.csv')
    df.to_csv(filename, index = False)
    

    
if __name__ == '__main__':
    plot_latencies()
