from __future__ import division, print_function
import pandas as pd
import os
import numpy as np
import sys
import cPickle as pickle
import loadmat
from scipy import stats
from sklearn import cross_validation, grid_search
from sklearn.linear_model import Ridge
from sklearn.preprocessing import scale
from collections import Counter
from scipy import stats

def HG_regression_surr_random_SGE(DATASET, numiter = 1000):
    '''
    creates random surrogate data numiter times
    calculate regression on each surrogate data set
    saves out distribution of regression parameters for surrogate data
    only runs on duration electrodes
    '''
    
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'
    subj, task = DATASET.split('_')
    print (DATASET)
    
    folder = 'maxes_medians_stds_lats'
    features = ['maxes_rel','medians', 'stds', 'lats']

    filename = os.path.join(SJdir, 'Subjs', subj, task, 'subj_globals.mat')
    data_dict = loadmat.loadmat(filename)
    srate = [data_dict.get(k) for k in ['srate']][0]
    srate = float(srate)
    
    filename = os.path.join(SJdir,'PCA', 'Stats', 'single_electrode_windows_csvs', 'single_electrode_windows_withdesignation_EDITED.csv')
    df_pattern = pd.read_csv(filename)

    bad_df = pd.DataFrame({'GP44_DecisionAud':233, 'GP15_SelfVis':1, 'JH2_FaceEmo':113, 'GP35_FaceEmo':60}, index = range(1)).T
    bad_df = bad_df.reset_index()
    bad_df.columns = ['subj_task','elec']

    #get data
    print ('get data')
    data_dict, start_idx, end_idx, start_idx_resp, end_idx_resp = stats_static250(subj, task, df_pattern)

    ##reject outliers
    print ('\nreject outliers')
    data_dict_clean = reject_outliers(DATASET, data_dict, start_idx, end_idx, start_idx_resp, end_idx_resp)

    #run regression for stim and resp
    scores, coefs, alphas, pvals = [[] for i in range(4)]
    for lock in ['resp', 'stim']:
        print ('run regression on %s\n' %(lock))
        coef, score, alpha, pval, nulls = run_regression(DATASET,  data_dict_clean[lock], numiter = numiter)

        #save out dataframes
        saveDir = os.path.join(SJdir, 'PCA', 'Stats', 'Regression', 'unsmoothed', folder, 'static_250windows', lock)
        if not(os.path.exists(saveDir)):
            os.makedirs(saveDir)
 
        df = pd.DataFrame({'score':score, 'coef':coef, 'pval':pval, 'alpha':alpha})
        df = df[['score','pval','alpha','coef']]
        
        filename = os.path.join(saveDir, '_'.join([DATASET, 'regression_values_%s.csv' %(lock)]))
        df.to_csv(filename)
        print('saving %s\n' %(filename))
        sys.stdout.flush()


def run_regression(DATASET, data_dict, numiter = 1000):
    """
    runs ridge regression without pvalue (surrogate distribution) - based on HG_regression_revised
    Loops on each electrode
    Splits data into train and test
    Runs 10 fold CV to get best alpha on training set
    Get best model, best coefficients, best score
    Returns dictionary with alpha, score, and coefficients per electrode

    runs on 10 test/training sets and takes median predictions score (for stability). not enough for pvalue
    so no confidence interval on prediction score or pvalue for prediction score
    """
    
    coefs_mean, score_median, alpha_median, pvals, nulls = [dict() for i in range(5)]

    elecs = data_dict.keys()
    colnames = list(data_dict[elecs[0]].columns)
    predictor = colnames.pop(colnames.index('RTs'))
    features = colnames
   
    for elec in elecs:

        #define data (NaNs already dropped)
        X = np.array(data_dict[elec][features])
        Y = np.array(data_dict[elec][predictor])

        #split data into training and test sets for the number of CV folds
        cvs = cross_validation.ShuffleSplit(len(Y), n_iter = numiter, test_size = 0.2) #100 models (for pvalue)

        alphas, models, scores, coefs, scores_null = [[] for i in range(5)]
        
        for train, test in cvs:
            #scale training and test data
            X_train = scale(X[train].astype(float))
            X_test = scale(X[test].astype(float))
            y_train = scale(Y[train].astype(float))
            y_test = scale(Y[test].astype(float))

            #define model (search over 10 alphas)
            model = Ridge(solver = 'lsqr', normalize=False, fit_intercept=False)
            params_grid = {'alpha': np.logspace(-4, 4, 10)}
            ridge_grid = grid_search.GridSearchCV(model, params_grid, cv = 10)

            #fit and find best model
            ridge_grid.fit(X_train, y_train)
            mod = ridge_grid.best_estimator_
            score = np.corrcoef(mod.predict(X_test), y_test)[1,0]
            coef = mod.coef_
            a = mod.alpha

            #calculate permuted score
            idx = np.random.permutation(len(y_test))
            null_score = np.corrcoef(mod.predict(X_test), y_test[idx])[1,0] #predict on shuffled test set

            #add to list
            alphas.append(a)
            models.append(mod)
            scores.append(score)
            coefs.append(coef)
            scores_null.append(null_score)

        if np.median(scores)>0:
            pval = 1 - (sum(np.median(scores) > scores_null)/ len(scores_null))
        elif np.median(scores)<0:
            pval = 1 - (sum(np.median(scores) < scores_null) / len(scores_null))
        else:
            pval = 1
        
        coefs_mean[elec] = np.mean(coefs, axis = 0)
        alpha_median[elec] = np.median(alphas)
        score_median[elec] = np.median(scores)
        pvals[elec] = pval
        nulls[elec] = scores_null
        
        print (elec, np.median(scores), pval)
        sys.stdout.flush()

    return coefs_mean, score_median, alpha_median, pvals, nulls


def reject_outliers(DATASET, data_dict, start_idx, end_idx, start_idx_resp, end_idx_resp, bad_df = pd.DataFrame({'subj_task':[]}), std_thresh = 4, features = ['maxes_rel','medians', 'stds', 'lats'], predictor = 'RTs'):
    """
    based on code in elec_values.ipynb - except doesn't save out all the iterations
    drops specific bad elecs, rejects at 4 stds, drops windows where RT-start_idx is < 500 (baseline)
    returns dictionary with clean dataframe per electrode
    outliers done for regression - each electrode can have different number of trials, 
    but if trial is bad for 1 feature, gets dropped from all features for that electrode
    """

    subj, task = DATASET.split('_')

    df_dict, data_dict_clean = [{'stim':dict(), 'resp':dict()} for i in range(2)]
    
    features_plus_predictor = features + [predictor]

    for lock in ['stim','resp']:
        #add RTs to data frame with dropped trials
        df_RT = pd.DataFrame(data_dict['RTs'][lock])
        df_RT.columns = [int(x) for x in df_RT.columns]
        df_dict[lock]['RTs'] = df_RT

        #reject outliers 
        for f in features:
            df = pd.DataFrame(data_dict[f][lock])

            #drop specific bad elecs elecs
            if DATASET in bad_df.subj_task.values:
                elec_to_drop = bad_df[bad_df.subj_task == '_'.join([subj, task])].elec
                df.pop(elec_to_drop.values[0])

            #drop 4 stds
            trials_to_drop = df.apply(lambda x: ((x > (x.mean() + x.std(ddof = 1)*std_thresh)) | (x < (x.mean() - x.std(ddof = 1)* std_thresh))))
            masked_values = np.where(~trials_to_drop, df.values, np.nan) #good trials     
            tmp = pd.DataFrame(masked_values, **df._construct_axes_dict()) #new dataframe with dropped trials for 1 feat
            tmp.columns = [int(x) for x in tmp.columns]
            df_dict[lock][f] = tmp

        #drop nans for all features for an electrode (based on 4std outlier rejection)   
        for elec in df_dict[lock][f].columns:
 
            data_array = np.array([df_dict[lock][x][elec] for x in features_plus_predictor]) #features x trials

            trials_to_drop = np.any(np.isnan(data_array), axis = 0)

            if lock == 'stim':
                nanidx = end_idx > data_dict['RTs'][lock][elec]
            else: #resp
                nanidx = (data_dict['RTs'][lock][elec] + start_idx_resp) <start_idx


            trials_to_drop = trials_to_drop | nanidx #combine 4std outliers with timing outliers
            
            data_array = data_array[:,~trials_to_drop]

            data_dict_clean[lock][elec] = pd.DataFrame(data_array.T, columns = features_plus_predictor)

    return data_dict_clean #dictionary with dataframe per electrode


def stats_static250(subj, task, df_pattern, start = 0, end = 250, start_idx_resp = -250, end_idx_resp = 0):

    """ 
    calculates params per electrode on for stim:stim+250 and resp-250:resp windows.

    drops trials that are <250 ms

    uses windows for individual electrodes from df_pattern (PCA/Stats/single_electrode_windows_csvs/single_electrode_windows_withdesignation_EDITED.csv)
    
    Uses unsmoothed data

    hardcoded params - medians, maxes_rel, stds, latencies, maxes, means

    returns dictionary with features. each feature is dictionary of elecs
    """

    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'

    #load data
    filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent_unsmoothed.mat')
    data_dict = loadmat.loadmat(filename)

    active_elecs, Params, srate, RT, data_all = [data_dict.get(k) for k in ['active_elecs','Params','srate','RTs','data_percent']]

    bl_st = 500/1000*srate #for my data, remove cue from baseline - start/end_idx are relative to cue onset) - change 12/24 - okay with RT 12/25

    RT = RT + abs(bl_st) #RTs are calculated from stim (my data cue) onset, need to account for bl in HG_elecMTX_percent (for 500, not 1000 baseline 12/25)
    
    #define start and end windows (stim locked)
    start = np.round((start / 1000 * srate) + abs(bl_st))
    end = np.round((end / 1000 * srate))

    start_idx_resp = np.round(start_idx_resp / 1000 * srate)
    end_idx_resp = np.round(end_idx_resp / 1000 * srate)

    RTs, medians, maxes_rel, means, stds, maxes, lats = [{'stim':dict(), 'resp':dict()} for i in range(7)]
    
    s_t = df_pattern[((df_pattern.subj == subj) & (df_pattern.task == task))]

    for e in s_t.elec.values:

        _, subj, task, cluster, pattern, elec, start_idx, end_idx, _, _, _, _ = s_t[s_t.elec == e].values[0]
        
        if (end_idx - start_idx) < (end- start): #HG duration is less than window size (250 or 500)
            print ('skipping %s %s %i' %(subj, task, e))
            sys.stdout.flush()
            continue

        print('%i...' %(elec), end = "")
        sys.stdout.flush()

        eidx = np.in1d(active_elecs, elec)
        data = data_all[eidx,:,:].squeeze()
    
        start_idx = start_idx + start #start and end relative to HG onset
        end_idx = start_idx + end
        
        #calculate values (single trials)
        means['stim'][elec] = np.nanmean(data[:,start_idx:end_idx], axis =1)
        stds['stim'][elec] = np.nanstd(data[:,start_idx:end_idx], axis = 1)
        maxes['stim'][elec] = np.nanmax(data[:,start_idx:end_idx], axis = 1)
        medians['stim'][elec] = stats.nanmedian(data[:,start_idx:end_idx], axis = 1)
        maxes_rel['stim'][elec] = maxes['stim'][elec] - means['stim'][elec]
        lats['stim'][elec] = np.argmax(data[:,start_idx:end_idx], axis = 1)

        data_resp = np.empty((len(RT), len(np.arange(start_idx_resp, end_idx_resp))))
        for j, r in enumerate(RT):
            data_resp[j,:] = data[j, r + start_idx_resp : r + end_idx_resp]

        means['resp'][elec] = np.nanmean(data_resp, axis = 1)
        stds['resp'][elec] = np.nanstd(data_resp, axis = 1)
        maxes['resp'][elec] = np.nanmax(data_resp, axis = 1)
        medians['resp'][elec] = stats.nanmedian(data_resp, axis = 1)
        maxes_rel['resp'][elec] = maxes['resp'][elec]-means['resp'][elec]
        lats['resp'][elec] = np.argmax(data_resp, axis = 1)
        
        RTs['stim'][elec] = RT
        RTs['resp'][elec] = RT

    #output dictionary of params per elec
    data_dict = {'RTs' : RTs, 'maxes_rel' : maxes_rel, 'medians' : medians, 'stds': stds, 'lats' : lats, 'means' : means, 'maxes' : maxes}

    return data_dict, start_idx, end_idx, start_idx_resp, end_idx_resp


if __name__ == '__main__':
    DATASET = sys.argv[1]
    HG_regression_surr_random_SGE(DATASET)
