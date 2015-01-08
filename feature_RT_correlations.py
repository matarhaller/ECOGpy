from __future__ import division
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import scale


def plot_correlations(to_scale = False, static = False, to_plot = True, surrogate = False, id_num = None):
    '''
    This  plots the correlation between all the features and RTs
    as a scatterplot matrix and saves the r values to a csv
    it uses non-outlier rejected data
    In PCA/Stats/Correlations
    to_scale argument determines if to normalize the variables or not (including RT)
    *** edited to use unsmoothed 12/11/14. not for static ***
    *** surrogate option is for random surrogate data - with id number ***
    '''

    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'

    df_designations = pd.read_csv(os.path.join(SJdir, 'PCA','Stats', 'single_electrode_windows_csvs', 'single_electrode_windows_withdesignation_EDITED_dropped.csv'))

    subj_task = df_designations[['subj', 'task']].drop_duplicates()

    features =  ['stds','maxes_rel','medians']
    features.sort()
    features.append('RTs')

    #features = ['means','maxes','lats_pro','stds']
    if static:
        features = ['stds','maxes_rel','lats','medians','mins'] #use real latencies with static window

        features.sort()
        features.append('RTs') 

    #calcualte correlations for each subj, task, elec
    for s_t in subj_task.itertuples():
        _, subj, task = s_t
        #if '_'.join([subj, task]) != 'GP15_EmoGen':
        #    continue
        big_dict = dict()
        for f in features:
            dataDir = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows')
            if static:
                dataDir = os.path.join(dataDir, 'smoothed','csv_files','static')
            elif surrogate:
                dataDir = os.path.join(dataDir, 'unsmoothed', 'csv_files', 'surr_rand_' + str(id_num))
            else:
                dataDir = os.path.join(dataDir, 'unsmoothed', 'csv_files')

            filename = os.path.join(dataDir, '_'.join([subj, task, f]))
            if static:
                filename = filename + '_static'
            elif surrogate:
                filename = filename + '_surr_rand'
            df = pd.read_csv(filename + '.csv')

            elecs = df.columns.values
            big_dict[f] = df


        corr_dict = dict()

        #make directories
        saveDir = os.path.join(SJdir, 'PCA','Stats','Correlations', 'unsmoothed', 'feature_corrs')
        if static:
            saveDir = os.path.join(SJdir, 'PCA','Stats','Correlations', 'smoothed', 'feature_corrs', 'static')
        elif surrogate:
            saveDir = os.path.join(saveDir, 'surr_rand_' + str(id_num))

        if not(os.path.exists(saveDir)):
            os.makedirs(saveDir)
            print('making:\n%s' %(saveDir))

        for e in elecs:
            #if e == '27':
            #    print ba
            if to_plot:
                #plot scatterplot matrix
                if to_scale == True: #if to normalize features (easier to spot outliers)
                    filename = os.path.join(saveDir, '_'.join([subj, task, 'e'+str(e)+'_scaled']))
                    if static:
                        stds, maxes_rel, lats, medians, mins, RTs = [scale(big_dict[f][e].astype(float)) for f in features]
                    else:
                        stds, maxes_rel, medians, RTs = [scale(big_dict[f][e].astype(float)) for f in features]
                else: #don't scale
                    filename = os.path.join(saveDir, '_'.join([subj, task, 'e'+str(e)]))
                    if static:
                        stds, maxes_rel, lats, medians, mins, RTs = [big_dict[f][e] for f in features]
                    else:
                        stds, maxes_rel,  medians,  RTs = [big_dict[f][e] for f in features]

                if static:
                    tmp = pd.DataFrame({'stds' : stds, 'maxes_rel': maxes_rel, 'lats' : lats , 'medians' : medians, 'RTs': RTs, 'mins': mins})
                else:
                    tmp = pd.DataFrame({'stds' : stds, 'maxes_rel': maxes_rel, 'medians' : medians, 'RTs': RTs})

                f, ax = plt.subplots(figsize = (10, 10))
                scatter_matrix(tmp, ax = ax, grid = True, hist_kwds={'alpha': 0.5})

                plt.savefig(filename + '.png')
                plt.close()


            #create csv - correlation each feature with RT and with features for each elec
            corr_list = [subj, task]
            flist = []
            for j, f1 in enumerate(features):
                for k, f2 in enumerate(features):
                    if j < k:
                        idx = np.where(~np.isnan(big_dict[f1][e]))[0]
                        corr_list.append(np.corrcoef(big_dict[f1][e][idx], big_dict[f2][e][idx])[0,1])
                        corr_dict[e] = corr_list
                        flist.append('_'.join([f1, f2]))
            
        #csv with all elecs for subj/task
        df = pd.DataFrame(corr_dict).T
        df = df.reset_index()

        cols = ['elec','subj','task']
        cols.extend(flist)
        df.columns = cols

        if to_scale == True:
            filename = os.path.join(saveDir,  '_'.join([subj,task, 'feature_corrs_scaled']))
        else:
            filename = os.path.join(saveDir,  '_'.join([subj,task, 'feature_corrs']))
       
        df.to_csv(filename+'.csv', index = False)
