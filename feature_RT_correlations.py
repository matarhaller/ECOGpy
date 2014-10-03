from __future__ import division
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import scale


def plot_correlations(to_scale = False):

    '''
    This  plots the correlation between all the features and RTs
    as a scatterplot matrix and saves the r values to a csv
    In PCA/Stats/Correlations
    to_scale argument determines if to normalize the variables or not (including RT)
    '''

    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'

    df_designations = pd.read_csv(os.path.join(SJdir, 'PCA','Stats', 'single_electrode_windows_withdesignation_EDITED_dropped.csv'))

    subj_task = df_designations[['subj', 'task']].drop_duplicates()

    #calcualte correlations for each subj, task, elec
    for s_t in subj_task.itertuples():
        _, subj, task = s_t

        #features = ['means','maxes','lats_pro','stds']
        features = ['stds','maxes_rel','lats_pro','medians']
        features.append('RTs')

        #filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'data', ''.join([subj, '_', task, '.p']))
        #data_dict = pickle.load(open(filename, 'rb'))

        #elecs = data_dict['means'].keys()
        big_dict = dict()
        for f in features:
            filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'csv_files', '_'.join([subj, , task, f + '.csv']))
            df = pd.read_csv(filename)
            elecs = df.columns.values
            big_dict[f] = df
        #big_dict['maxes_rel'] = big_dict['maxes'] - big_dict['means']
        #features.append('maxes_rel')

        #pairs of comparisons (feature pairs)
        flist = []
        for f1 in features:
            for f2 in features:
                flist.append('_'.join([f1, f2]))

        corr_dict = dict()

        for e in elecs:

            #plot scatterplot matrix
            if to_scale == True: #if to normalize features (easier to spot outliers)
                #means, maxes, lats_pro, stds, RTs = [scale(data_dict[k][e].astype(float)) for k in features]
                #means, maxes, lats_pro, stds, RTs, maxes_rel = [scale(big_dict[f][e].astype(float)) for f in features]
                stds, maxes_rel, lats_pro, medians = [scale(big_dict[f][e].astype(float)) for f in features]
                filename = os.path.join(SJdir, 'PCA','Stats', 'Correlations', '_'.join([subj, task, 'e'+str(e)+'_scaled.png']))
            else:
                #means, maxes, lats_pro, stds, RTs = [data_dict[k][e] for k in features]
                #means, maxes, lats_pro, stds, RTs, maxes_rel = [big_dict[f][e] for f in features]
                stds, maxes_rel, lats_pro, medians = [big_dict[f][e] for f in features]
                filename = os.path.join(SJdir, 'PCA','Stats', 'Correlations', '_'.join([subj, task, 'e'+str(e)+'.png']))

            #tmp = pd.DataFrame({'means':means, 'maxes':maxes, 'maxes_rel': maxes_rel, 'lats_pro':lats_pro, 'stds':stds, 'RTs':RTs})
                tmp = pd.DataFrame({'stds' : stds, 'maxes_rel': maxes_rel, 'lats_pro' : lats_pro , 'medians' : medians})

            f, ax = plt.subplots(figsize = (10, 10))
            scatter_matrix(tmp, ax = ax, grid = True, hist_kwds={'alpha': 0.5})

            plt.savefig(filename)
            plt.close()

            #create csv - correlation each feature with RT and with features for each elec
            corr_list = [subj, task]
            for f1 in features:
                for f2 in features:
                    idx = np.where(~np.isnan(big_dict[f1][e]))[0]
                    corr_list.append(np.corrcoef(big_dict[f1][e][idx], big_dict[f2][e][idx])[0,1])
                    corr_dict[e] = corr_list

        df = pd.DataFrame(corr_dict).T
        df = df.reset_index()

        cols = ['elec','subj','task']
        cols.extend(flist)
        df.columns = cols

        df_elec = pd.merge(df, df_designations[(df_designations.subj==subj) & (df_designations.task==task)][['subj','task','elec','pattern']], how = 'outer', on = ['subj','task','elec'])

        if to_scale == True:
            filename = os.path.join(SJdir, 'PCA','Stats', 'Correlations', '_'.join([subj,task, 'feature_corrs_scaled.csv']))
        else:
            filename = os.path.join(SJdir, 'PCA','Stats', 'Correlations', '_'.join([subj,task, 'feature_corrs.csv']))
        df_elec.to_csv(filename)
