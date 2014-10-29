from __future__ import division
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import scale


def plot_correlations(to_scale = False, static = False, features =  ['stds','maxes_rel','lats_pro','medians'], to_plot = True):
    '''
    This  plots the correlation between all the features and RTs
    as a scatterplot matrix and saves the r values to a csv
    In PCA/Stats/Correlations
    to_scale argument determines if to normalize the variables or not (including RT)
    static argument determines if to use static duration windows or RT-adjusted
    '''

    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'

    df_designations = pd.read_csv(os.path.join(SJdir, 'PCA','Stats', 'single_electrode_windows_withdesignation_EDITED_dropped.csv'))

    subj_task = df_designations[['subj', 'task']].drop_duplicates()

    #calcualte correlations for each subj, task, elec
    for s_t in subj_task.itertuples():
        _, subj, task = s_t

        #features = ['means','maxes','lats_pro','stds']
        if static:
            features = ['stds','maxes_rel','lats','medians','mins'] #use real latencies with static window
        #else:
            #features = ['stds','maxes_rel','lats_pro','medians','mins','maxes','lats']

        features.sort()
        features.append('RTs')

        #filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'data', ''.join([subj, '_', task, '.p']))
        #data_dict = pickle.load(open(filename, 'rb'))

        #elecs = data_dict['means'].keys()
        big_dict = dict()
        for f in features:

            if static:
                filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'static', 'csv_files', '_'.join([subj, task, f + '.csv']))
            else:
                filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'csv_files', '_'.join([subj, task, f + '.csv']))

            df = pd.read_csv(filename)

            elecs = df.columns.values
            big_dict[f] = df

        #pairs of comparisons (feature pairs)
        #flist = []
        #for f1 in features:
        #    for f2 in features:
        #        flist.append('_'.join([f1, f2]))

        corr_dict = dict()

        for e in elecs:
            if to_plot:
                #plot scatterplot matrix
                if to_scale == True: #if to normalize features (easier to spot outliers)
                    if static:
                        filename = os.path.join(SJdir, 'PCA','Stats', 'Correlations', 'static', 'feature_corrs', '_'.join([subj, task, 'e'+str(e)+'_scaled.png']))
                        stds, maxes_rel, lats, medians, mins, RTs = [scale(big_dict[f][e].astype(float)) for f in features]
                    else:
                        filename = os.path.join(SJdir, 'PCA','Stats', 'Correlations', 'feature_corrs', '_'.join([subj, task, 'e'+str(e)+'_scaled.png']))
                        stds, maxes_rel, lats_pro, medians, mins, RTs = [scale(big_dict[f][e].astype(float)) for f in features]

                else:
                    if static:
                        filename = os.path.join(SJdir, 'PCA','Stats', 'Correlations', 'static', 'feature_corrs', '_'.join([subj, task, 'e'+str(e)+'.png']))
                        stds, maxes_rel, lats, medians, mins, RTs = [big_dict[f][e] for f in features]
                    else:
                        filename = os.path.join(SJdir, 'PCA','Stats', 'Correlations', 'feature_corrs', '_'.join([subj, task, 'e'+str(e)+'.png']))
                        stds, maxes_rel, lats_pro, medians, mins, RTs = [big_dict[f][e] for f in features]


                if static:
                    tmp = pd.DataFrame({'stds' : stds, 'maxes_rel': maxes_rel, 'lats' : lats , 'medians' : medians, 'RTs': RTs, 'mins': mins})
                else:
                    tmp = pd.DataFrame({'stds' : stds, 'maxes_rel': maxes_rel, 'lats_pro' : lats_pro , 'medians' : medians, 'RTs': RTs, 'mins': mins})

                f, ax = plt.subplots(figsize = (10, 10))
                scatter_matrix(tmp, ax = ax, grid = True, hist_kwds={'alpha': 0.5})

                plt.savefig(filename)
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

        df = pd.DataFrame(corr_dict).T
        df = df.reset_index()

        cols = ['elec','subj','task']
        cols.extend(flist)
        df.columns = cols

        #df_elec = pd.merge(df, df_designations[(df_designations.subj==subj) & (df_designations.task==task)][['subj','task','elec','pattern']], how = 'outer', on = ['subj','task','elec'])
        df_elec = df #problem with adding pattern for some reason

        if to_scale == True:
            if static:
                filename = os.path.join(SJdir, 'PCA','Stats', 'Correlations', 'static', 'feature_corrs', '_'.join([subj,task, 'feature_corrs_scaled.csv']))
            else:
                filename = os.path.join(SJdir, 'PCA','Stats', 'Correlations', 'feature_corrs', '_'.join([subj,task, 'feature_corrs_scaled.csv']))
        else:
            if static:
                filename = os.path.join(SJdir, 'PCA','Stats', 'Correlations', 'static', 'feature_corrs', '_'.join([subj,task, 'feature_corrs.csv']))
            else:
                filename = os.path.join(SJdir, 'PCA','Stats', 'Correlations', 'feature_corrs', '_'.join([subj,task, 'feature_corrs.csv']))
        df_elec.to_csv(filename)
