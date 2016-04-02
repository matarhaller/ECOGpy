from __future__ import division
import pandas as pd
import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy import stats
import loadmat
import sys

def RT_median_split(DATASET, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/', numiter = 1000):
    
    filename = os.path.join(SJdir,'PCA', 'Stats', 'single_electrode_windows_withdesignation_EDITED_dropped_withROI.csv')
    df = pd.read_csv(filename)

    subj, task = DATASET.split('_')

    #load data
    filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent_unsmoothed.mat')
    data_dict = loadmat.loadmat(filename)
    Params, srate, data_percent, active_elecs, RT = [data_dict.get(k) for k in ['Params', 'srate', 'data_percent', 'active_elecs', 'RTs']]
    bl_st = Params['bl_st']
    bl_st = bl_st/1000*srate

    #load RTs csv file   
    filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'csv_files', '_'.join([subj, task, 'RTs']) + '.csv')
    data = pd.read_csv(filename)
    RTs = np.round(np.array(data)[:,0])
    #don't remove baseline. want RT to include baseline so can index properly (here they already include baseline from Shadeplots_elecs_stats.py)        

    #sort trials by RTs
    idx = np.argsort(RTs)
    data_percent = data_percent[:, idx, :]
    RTs = RTs[idx]

    median_idx = np.floor(data_percent.shape[1]/2) #index of median split for this subject
    df_subj = df[(df.subj == subj) & (df.task == task)][['elec','start_idx','end_idx','start_idx_resp','end_idx_resp', 'pattern']]

    #iterate on electrodes
    for row in df_subj.itertuples():
        
        _, elec, start_idx, end_idx, start_idx_resp, end_idx_resp, pattern = row
        
        print('%s %s e%i, %s' %(subj, task, elec, pattern))

        eidx = np.where(elec == active_elecs)[0][0]

        skews, kurts, means, medians, means_l, means_s, medians_s, medians_l, skews_s, skews_l, kurts_s, kurts_l = [[] for i in range(12)]    
        skews_surr, kurts_surr, means_surr, medians_surr, means_l_surr, means_s_surr = [[] for i in range(6)]

        if (pattern == 'S') | (pattern == 'SR'):
            start_idx = start_idx + abs(bl_st)
            end_idx = end_idx + abs(bl_st)

            shorttrials, longtrials, trial_lengths = [[] for i in range(3)]
            for i, r in enumerate(RTs):
                if i < median_idx:
                    shorttrials.extend(data_percent[eidx, i, start_idx:end_idx])
                    trial_lengths.append(int(end_idx-start_idx)) #length of each short trial so can use for long trial indexing
                elif i > median_idx: #might only work with odd num of trials
                    longtrials.extend(data_percent[eidx, i, start_idx:end_idx])

        if (pattern == 'R'):
            start_idx = start_idx_resp
            end_idx = end_idx_resp

            shorttrials, longtrials, trial_lengths = [[] for i in range(3)]
            for i, r in enumerate(RTs):
                if i < median_idx:
                    shorttrials.extend(data_percent[eidx, i, int(r)+start_idx:int(r)+end_idx])
                    trial_lengths.append(int(end_idx-start_idx+1)) #length of each short trial so can use for long trial indexing
                elif i > median_idx: #might only work with odd num of trials
                    longtrials.extend(data_percent[eidx, i, int(r)+start_idx:int(r)+end_idx])

        if pattern == 'D':
            start_idx = start_idx + abs(bl_st)
            end_idx = end_idx_resp

            #create data vectors for long and short trials
            shorttrials, longtrials, trial_lengths = [[] for i in range(3)]
            for i, r in enumerate(RTs):
                if i < median_idx:
                    shorttrials.extend(data_percent[eidx, i, start_idx:int(r)+end_idx])
                    trial_lengths.append(int(r+end_idx-start_idx+1)) #length of each short trial so can use for long trial indexing
                elif i > median_idx: #might only work with odd num of trials
                    longtrials.extend(data_percent[eidx, i, start_idx:int(r)+end_idx])

            #bootstrap from long distribution
            print('\tbootstrapping from long distribution')
            for j in range(numiter):
                randidx = np.random.permutation(len(longtrials))[0:len(shorttrials)]
                longsample = np.array(longtrials)[randidx]

                #calculate stats for duration sample
                skews.append(stats.skew(longsample) - stats.skew(shorttrials))
                kurts.append(stats.kurtosis(longsample) - stats.kurtosis(shorttrials))
                means.append(np.mean(longsample) - np.mean(shorttrials))
                medians.append(np.median(longsample) - np.median(shorttrials))
                means_l.append(np.mean(longsample))
                skews_l.append(stats.skew(longsample))
                kurts_l.append(stats.kurtosis(longsample))
                medians_l.append(np.median(longsample))

        else: #calculate stats for for nonduration no need to subsample long sample
            longsample = longtrials
            skews.append(stats.skew(longsample) - stats.skew(shorttrials))
            kurts.append(stats.kurtosis(longsample) - stats.kurtosis(shorttrials))
            means.append(np.mean(longsample) - np.mean(shorttrials))
            medians.append(np.median(longsample) - np.median(shorttrials)) 
            means_l.append(np.mean(longsample))
            skews_l.append(stats.skew(longsample))
            kurts_l.append(stats.kurtosis(longsample))
            medians_l.append(np.median(longsample))

        #calculate values for short trials (same for duration and nonduration)
        medians_s.append(np.median(shorttrials))
        means_s.append(np.mean(shorttrials))
        kurts_s.append(stats.kurtosis(shorttrials))
        skews_s.append(stats.skew(shorttrials))

        #create permuted difference distribution
        print ('\tcalculating surrogate stats...')
        for j in range(numiter):
            randidx = np.random.permutation(len(shorttrials)*2) #no overlap between 'short' and 'long' datapoints
            randidx_short = randidx[0:len(randidx)/2] 
            randidx_long = randidx[len(randidx)/2+1::]
            shorttrials_surr = data_percent[eidx,:,:].flatten()[randidx_short]
            longsample_surr = data_percent[eidx,:,:].flatten()[randidx_long]

            #calculate stats
            skews_surr.append(stats.skew(longsample_surr) - stats.skew(shorttrials_surr))
            kurts_surr.append(stats.kurtosis(longsample_surr) - stats.kurtosis(shorttrials_surr))
            means_surr.append(np.mean(longsample_surr) - np.mean(shorttrials_surr))
            medians_surr.append(np.median(longsample_surr) - np.median(shorttrials_surr))
            means_l_surr.append(np.mean(longsample_surr))
            means_s_surr.append(np.mean(shorttrials_surr)) 

        #calculate pvalue
        if np.mean(means) <= np.mean(means_surr):
            p_mean = sum(means_surr<np.mean(means))/len(means_surr)
        else:
            p_mean = sum(means_surr>np.mean(means))/len(means_surr)
           
        if np.mean(medians) <= np.mean(medians_surr):
            p_median = sum(medians_surr<np.mean(medians))/len(medians_surr)
        else:
            p_median = sum(medians_surr>np.mean(medians))/len(medians_surr)
        
        if np.mean(skews) <= np.mean(skews_surr):
            p_skew = sum(skews_surr<np.mean(skews))/len(skews_surr)
        else:
            p_skew = sum(skews_surr>np.mean(skews))/len(skews_surr)
       
        if np.mean(kurts) <= np.mean(kurts_surr):
            p_kurt = sum(kurts_surr<np.mean(kurts))/len(kurts_surr)
        else:
            p_kurt = sum(kurts_surr>np.mean(kurts))/len(kurts_surr)
            
        #save
        print('\tsaving')
        data_dict = {'p_mean' : p_mean, 'p_median' : p_median, 'p_skew' : p_skew, 'p_kurt' : p_kurt, 'pattern':pattern, 'skews':skews, 'kurts':kurts, 'means':means, 'medians':medians, 'means_s':means_s, 'means_l':means_l, 'medians_l':medians_l, 'medians_s':medians_s, 'skews_l':skews_l, 'skews_s':skews_s,'kurts_l':kurts_l, 'kurts_s':kurts_s, 'shorttrials':shorttrials, 'longtrials':longtrials, 'longsample':longsample, 'skew_surr':skews_surr, 'kurtosis_surr':kurts_surr, 'mean_surr':means_surr, 'median_surr':medians_surr}
        filename = os.path.join(SJdir, 'PCA', 'Stats', 'RT_median_split', '%s_%s_e%i_distributions.p' %(subj, task, elec))
        pickle.dump(data_dict, open(filename, "wb"))


def RT_median_split_plot(subj, task, elec, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'):

    filename = os.path.join(SJdir, 'PCA', 'Stats', 'RT_median_split', '%s_%s_e%i_distributions.p' %(subj, task, elec))
    data_dict = pickle.load(open(filename, "rb" ) )        
    kurts_surr, kurts, kurts_l, kurts_s, longsample, longtrials, mean_surr, means, means_l, means_s, median_surr, medians, medians_l, medians_s, p_kurt, p_mean, p_median, p_skew, pattern, shorttrials, skew_surr, skews, skews_l, skews_s = [data_dict[k] for k in np.sort(data_dict.keys())]
    
    if len(shorttrials) == 0: #if start_idx == end_idx
        print('skipping %s %s e%i - no data' %(subj, task, elec))
        sys.stdout.flush()
        return
        
    if pattern == 'D':
        #stats distribution plot with surrogates (differences)
        f,ax = plt.subplots(2,2, figsize = (10, 7))
        plt.suptitle('%s %s e%i %s, long - short' %(subj, task, elec, pattern), fontsize = 14)

        ax = ax.flatten()
        ax[0].hist(means, bins = 20, alpha = 0.5)
        ax[0].hist(mean_surr, bins = 20, label = 'surr', alpha = 0.5)
        ax[0].set_title('means, p=%.3f' %(p_mean))
        ax[0].legend()

        ax[1].hist(medians, bins = 20, alpha = 0.5)
        ax[1].hist(median_surr, bins = 20, label = 'surr', alpha = 0.5)
        ax[1].set_title('medians, p=%.3f' %(p_median))
        ax[1].legend()

        ax[2].hist(skews, bins = 20, alpha = 0.5)
        ax[2].hist(skew_surr, bins = 20, label = 'surr', alpha = 0.5)
        ax[2].set_title('skews, p=%.3f' %(p_skew))
        ax[2].legend()

        ax[3].hist(kurts, bins = 20, alpha = 0.5)
        ax[3].hist(kurts_surr, bins = 20, label = 'surr', alpha = 0.5)
        ax[3].set_title('kurtosises, p=%.3f' %(p_kurt))
        ax[3].legend()

        for i, t in enumerate(ax):
            ax[i].patch.set_facecolor('white')
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].get_xaxis().tick_bottom()
            ax[i].get_yaxis().tick_left()

        filename = os.path.join(SJdir, 'PCA', 'Stats', 'RT_median_split', '%s_%s_e%i_%s_difference_hists.png' %(subj, task, elec, pattern))
        plt.savefig(filename)
        plt.close()

        #dist of means, skews, kurtosis, medians for long and short (not difference)
        f, ax = plt.subplots(2,2, figsize = (10, 7))
        plt.suptitle('%s %s e%i %s - values of long samples versus short' %(subj, task, elec, pattern), fontsize = 14)

        ax = ax.flatten()
        ax[0].hist(means_l, label = 'long', bins = 20)
        ax[0].axvline(x= means_s, label = 'short', lw = 2, color = 'red')
        ax[0].set_title('means')
        ax[0].legend(loc = 'best')

        ax[1].hist(medians_l, label = 'long', bins = 20)
        ax[1].axvline(x = medians_s, label = 'short', lw = 2, color = 'red')
        ax[1].set_title('medians')
        ax[1].legend(loc = 'best')

        ax[2].hist(skews_l, label = 'long', bins = 20)
        ax[2].axvline(x = skews_s, label = 'short', lw = 2, color = 'red')
        ax[2].set_title('skews')
        ax[2].legend(loc = 'best')

        ax[3].hist(kurts_l, label = 'long', bins = 20)
        ax[3].axvline(x = kurts_s, label = 'short', lw = 2, color = 'red')
        ax[3].set_title('kurtosis')
        ax[3].legend(loc = 'best')

        for i, t in enumerate(ax):
            ax[i].patch.set_facecolor('white')
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].get_xaxis().tick_bottom()
            ax[i].get_yaxis().tick_left()

        filename = os.path.join(SJdir, 'PCA', 'Stats', 'RT_median_split', '%s_%s_e%i_%s_values_hists.png' %(subj, task, elec, pattern))
        plt.savefig(filename)
        plt.close()

    else: #plot single value relative to surrogate distribution
        means, skews, kurts, medians = [np.float64(i[0]) for i in [means, skews, kurts, medians]]
        f,ax = plt.subplots(2,2, figsize = (10, 7))
        plt.suptitle('%s %s e%i %s, long - short' %(subj, task, elec, pattern), fontsize = 14)

        ax = ax.flatten()
        ax[0].hist(mean_surr, bins = 20, label = 'surr')
        ax[0].axvline(x = means, lw = 2, color = 'red')
        ax[0].set_title('means, p=%.2f' %(p_mean))
        ax[0].legend()

        ax[1].hist(median_surr, bins = 20, label = 'surr')
        ax[1].axvline(x = medians, lw = 2, color = 'red')
        ax[1].set_title('medians, p=%.2f' %(p_median))
        ax[1].legend()

        ax[2].hist(skew_surr, bins = 20, label = 'surr')
        ax[2].axvline(x = skews, lw = 2, color = 'red')
        ax[2].set_title('skews, p=%.2f' %(p_skew))
        ax[2].legend()

        ax[3].hist(kurts_surr, bins = 20, label = 'surr')
        ax[3].axvline(x = kurts, lw = 2, color = 'red')
        ax[3].set_title('kurtosises, p=%.2f' %(p_kurt))
        ax[3].legend()

        for i, t in enumerate(ax):
            ax[i].patch.set_facecolor('white')
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].get_xaxis().tick_bottom()
            ax[i].get_yaxis().tick_left()

        filename = os.path.join(SJdir, 'PCA', 'Stats', 'RT_median_split', '%s_%s_e%i_%s_difference_hists.png' %(subj, task, elec, pattern))
        plt.savefig(filename)
        plt.close()

    #data plot
    f,ax = plt.subplots(2,1, figsize = (7, 7))
    plt.suptitle('%s %s e%i %s - data' %(subj, task, elec, pattern), fontsize = 14)
    mybins = np.arange(min(shorttrials), max(shorttrials), (max(shorttrials)-min(shorttrials))/20)
    ax[0].hist(longtrials, bins = mybins, alpha = 0.5, label = 'long (full)\nmean = %.2f\n%i datapoints' %(np.mean(longtrials), len(longtrials)))
    ax[0].hist(shorttrials, bins = mybins, alpha = 0.5, label = 'short\nmean = %.2f\n%i datapoints' %(np.mean(shorttrials), len(shorttrials)))
    ax[0].legend()

    mybins = np.arange(min(shorttrials), max(shorttrials), (max(shorttrials)-min(shorttrials))/20)
    ax[1].hist(longsample, bins = mybins, alpha = 0.5, label = 'long (sample)\nmean = %.2f\n%i datapoints' %(np.mean(longsample), len(longsample)))
    ax[1].hist(shorttrials, bins = mybins, alpha = 0.5, label = 'short\nmean = %.2f\n%i datapoints' %(np.mean(shorttrials), len(shorttrials)))
    ax[1].legend()

    for i, t in enumerate(ax):
        ax[i].patch.set_facecolor('white')
        ax[i].autoscale(tight=True)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].get_xaxis().tick_bottom()
        ax[i].get_yaxis().tick_left()

    filename = os.path.join(SJdir, 'PCA', 'Stats', 'RT_median_split', '%s_%s_e%i_%s_data.png' %(subj, task, elec, pattern))
    plt.savefig(filename)
    plt.close()
 
def tail_comparison(SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/', dplot = True):
    '''
    compare tails of distributions (maximum tails)
    '''

    filename = os.path.join(SJdir,'PCA', 'Stats', 'single_electrode_windows_withdesignation_EDITED_dropped_withROI.csv')
    df = pd.read_csv(filename)
    
    subjs, tasks, elecs, patterns, p_kurts, p_means, p_medians, p_skews, kurt_all, skew_all, mean_all, median_all = [[] for i in range(13)]

    #iterate on electrodes
    for row in df.itertuples():

        _, subj, task, cluster, pattern, elec, start_idx, end_idx, start_idx_resp, end_idx_resp, dropped,  pattern = row

        filename = os.path.join(SJdir, 'PCA', 'Stats', 'RT_median_split', '%s_%s_e%i_distributions.p' %(subj, task, elec))
        data_dict = pickle.load(open(filename, "rb" ) )        

        #kurts_surr, kurts, kurts_l, kurts_s, longsample, longtrials, mean_surr, means, means_l, means_s, median_surr, medians, medians_l, medians_s, p_kurt, p_mean, p_median, p_skew, pattern, shorttrials, skew_surr, skews, skews_l, skews_s = [data_dict[k] for k in np.sort(data_dict.keys())]

        shorttrials, longtrials = [data_dict[k] for k in ['shorttrials','longtrials']]

        if len(shorttrials) == 0: #if start_idx == end_idx
            print('skipping %s %s e%i - no data' %(subj, task, elec))
            sys.stdout.flush()
            continue

        maxlong = np.sort(longtrials)[-100::]
        maxshort = np.sort(shorttrials)[-100::]
        
        mean_short = np.mean(maxshort)
        mean_long = np.mean(maxlong)
        median_short = np.median(maxshort)
        median_long = np.median(maxlong)
        skew_short = stats.skew(maxshort)
        skew_long = stats.skew(maxlong)
        kurt_short = stats.kurtosis(maxshort)
        kurt_long = stats.kurtosis(maxlong)

        skews_diff = (skew_long - skew_short)
        kurts_diff = (kurt_long - kurt_short)
        means_diff = (mean_long - mean_short)
        medians_diff = (median_long - median_short)

        means_l.append(mean_long)
        skews_l.append(skew_long)
        kurts_l.append(kurt_long)
        medians_l.append(median_long)

        medians_s.append(median_short)
        means_s.append(mean_short)
        kurts_s.append(kurt_short)
        skews_s.append(skew_short)


        #create surrogate dist
        numiter = 1000
        alltrials = []
        diff_surr = []

        for i in zip(maxlong, maxshort):
            alltrials.extend(i)

        for i in range(numiter):
            randidx = np.random.permutation(len(alltrials))
            short_surr = np.array(alltrials)[randidx[0:100]]
            long_surr = np.array(alltrials)[randidx[-100::]]

            skews_surr.append(stats.skew(long_surr) - stats.skew(short_surr))
            kurts_surr.append(stats.kurtosis(long_surr) - stats.kurtosis(short_surr))
            means_surr.append(np.mean(long_surr) - np.mean(short_surr))
            medians_surr.append(np.median(long_surr) - np.median(short_surr))
            means_l_surr.append(np.mean(long_surr))
            means_s_surr.append(np.mean(short_surr)) 

        #calculate pvalue
        if np.mean(means_diff) <= np.mean(means_surr):
            p_mean = sum(means_surr<np.mean(means))/len(means_surr)
        else:
            p_mean = sum(means_surr>np.mean(means))/len(means_surr)
           
        if np.mean(medians_diff) <= np.mean(medians_surr):
            p_median = sum(medians_surr<np.mean(medians))/len(medians_surr)
        else:
            p_median = sum(medians_surr>np.mean(medians))/len(medians_surr)
        
        if np.mean(skews_diff) <= np.mean(skews_surr):
            p_skew = sum(skews_surr<np.mean(skews))/len(skews_surr)
        else:
            p_skew = sum(skews_surr>np.mean(skews))/len(skews_surr)
       
        if np.mean(kurts_diff) <= np.mean(kurts_surr):
            p_kurt = sum(kurts_surr<np.mean(kurts))/len(kurts_surr)
        else:
            p_kurt = sum(kurts_surr>np.mean(kurts))/len(kurts_surr)

        #plot
        if dplot:
            f, ax = plt.subplots()
            ax.hist(means_surr)
            ax.axvline(x = means_diff, color = 'red', linewidth = 2)
            ax.set_title('difference in means\n%s %s e%i %s - pval %.3f' %(subj, task, elec, pattern, pval))

            ax.patch.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            filename = os.path.join(SJdir, 'PCA', 'Stats', 'RT_median_split', '%s_%s_e%i_%s_tails.png' %(subj, task, elec, pattern))
            plt.savefig(filename)
            plt.close()

        subjs.append(subj)
        tasks.append(task)
        elecs.append(elec)
        patterns.append(pattern)
        p_kurts.append(p_kurt)
        p_means.append(p_mean)
        p_medians.append(p_median)
        p_skews.append(p_skew)
        kurt_diff_all.append(np.mean(kurts_diff))
        skew_diff_all.append(np.mean(skews_diff))
        mean_diff_all.append(np.mean(means_diff))
        median_diff_all.append(np.mean(medians_diff))
        kurts_l_all.append(np.mean(kurts_l))
        kurts_s_all.append(np.mean(kurts_s))
        skews_s_all.append(np.mean(skews_s))
        skews_l_all.append(np.mean(skews_l))
        means_l_all.append(np.mean(means_l))
        means_s_all.append(np.mean(means_s))
        medians_l_all.append(np.mean(medians_l))
        medians_s_all.append(np.mean(medians_s))
        
        #pickle file with data for curr subj/task/elec
        data_dict = {'tail_long':maxlong, 'tail_short':maxshort, 'diff_surr':diff_surr,  'mean_short':mean_short, 'mean_long':mean_long, 'p_mean':p_mean, 'median_short':median_short, 'median_long':median_long, 'p_median':p_median, 'skew_short':skew_short, 'skew_long':skew_long, 'p_skew':p_skew, 'kurt_short':kurt_short, 'kurt_long':kurt_long, 'p_kurt':p_kurt}
        filename = os.path.join(SJdir, 'PCA','Stats','RT_median_split', '%s_%s_e%i_tail.p' %(subj, task, elec))
        pickle.dump(data_dict, open(filename, "wb"))
        
    #save pickle file
    data_dict = {'subj':subjs, 'task':tasks, 'elec':elecs, 'pattern':patterns, 'kurt_all':kurt_all, 'p_kurt':p_kurts, 'mean_all':mean_all, 'p_mean': p_means, 'median_all': median_all, 'p_median': p_medians, 'skew_all':skew_all, 'p_skew': p_skews}
    filename = os.path.join(SJdir, 'PCA', 'Stats', 'RT_median_split', 'median_split_dist_tails.p')
    pickle.dump(data_dict, open(filename, "wb"))


if __name__ == '__main__':
    DATASET = sys.argv[1]
    RT_median_split(DATASET)
