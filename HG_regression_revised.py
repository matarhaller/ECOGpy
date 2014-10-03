from __future__ import division
import os
import numpy as np
import cPickle as pickle
import sys
import pandas as pd
from sklearn import cross_validation, grid_search
from sklearn.linear_model import Ridge
from sklearn.preprocessing import scale
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt

def HG_regression_allelecs_SGE(DATASET):
    """
    feeds in subj/task to HG_regression_all_elecs and saves output to file
    """
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'
    subj, task = DATASET.split('_')
    
    reg_dict = HG_regression_allelecs(subj, task)

    filename = os.path.join(SJdir, 'PCA', 'Stats', 'Regression', '_'.join([subj, task]))

    pickle.dump(reg_dict, open(filename + '.p', 'wb'))
    
    elecs, alphas, scores, zcoefs, pval, features  = [reg_dict[key] for key in ['elecs','alphas','scores', 'zcoefs', 'pval', 'features']]
    score = np.median(scores, axis = 1)
    alpha = np.median(alphas, axis = 1)

    data_array = np.hstack([np.asarray(zcoefs), np.reshape(score, (len(score), 1)), np.reshape(np.asarray(pval), (len(pval),1)), np.reshape(alpha, (len(alpha), 1))])

    features.append('pred_score')
    features.append('pval_predscore')
    features.append('alpha')

    df = pd.DataFrame(data_array, columns = features, index = elecs)

    df.to_csv(filename + '_coefs.csv')

    #plot_figures(subj, task, reg_dict)
    

def HG_regression_allelecs(subj, task): 
    '''
    Runs ridge regression on maxes, means, stds, sums, latency (proportion) data for a subj/task
    Loops on each electrode
    Splits data into training and test sets
    Runs 10 fold CV to get best alpha on training set
    Gets best model, best coefficients, best score
    Calculates a null prediction score by predicting shuffled test set
    Outputs dictionary of lists of coefficients, prediction scores, models, null prediction scores for each duration electrode
    '''
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'
    reg_dict = dict()

    #set parameters
    #features = ['maxes', 'means', 'stds', 'sums', 'lats_pro']
    #features = ['maxes', 'means', 'stds', 'lats_pro']
    features = ['maxes_rel','medians','stds','lats_pro']
    predictor = 'RTs'

    # load data
    #filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'data', ''.join([subj, '_', task, '.p']))
    #data_dict = pickle.load(open(filename, 'rb'))
    data_dict = dict()
    for j, f in enumerate(features):
        filename = os.path.join(SJdir, 'PCA', 'Stats', 'outliers', '_'.join([subj, task, f]) + '.csv') #run on cleaned trials (outliers dropped)
        df = pd.read_csv(filename)
        df.columns = [int(x) for x in df.columns]
        data_dict[f] = dict(df)

    filename = os.path.join(SJdir, 'PCA', 'Stats', 'outliers', '_'.join([subj, task, predictor]) + '.csv')
    df = pd.read_csv(filename)
    df.columns = [int(x) for x in df.columns]
    data_dict[predictor] = dict(df)    

    all_alphas, all_models, all_scores, all_coefs, all_scores_null, all_pval, all_zcoefs = [[] for i in range(7)]
    elecs = data_dict['medians'].keys()

    for elec in elecs:

        #define data
        X = np.array([data_dict[x][elec] for x in features]).T
        Y = data_dict[predictor][elec]

        #drop nans from data
        mask = np.all(np.isnan(X), axis = 1)
        X = X[~mask]
        mask = np.isnan(Y)
        Y = Y[~mask]        

        #split data into training and test sets for the number of CV folds
        cvs = cross_validation.ShuffleSplit(len(Y), n_iter = 1000, test_size = 0.2)

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

        #zscore coefficients, take mean
        zcoefs = stats.zscore(coefs, axis = 0)
        zcoefs = np.mean(coefs, axis = 0)

        all_alphas.append(alphas)
        all_models.append(models)
        all_scores.append(scores)
        all_coefs.append(coefs)
        all_scores_null.append(scores_null)
        all_pval.append(pval)
        all_zcoefs.append(zcoefs)

        print (elec, pval, np.median(scores))
        sys.stdout.flush()

    reg_dict['elecs'] = elecs
    reg_dict['features'] = features
    reg_dict['alphas'] = all_alphas
    reg_dict['models'] = all_models
    reg_dict['scores'] = all_scores
    reg_dict['coefs'] = all_coefs
    reg_dict['scores_null'] = all_scores_null
    reg_dict['pval'] = all_pval
    reg_dict['zcoefs'] = all_zcoefs

    return reg_dict

def plot_figures(subj, task, reg_dict):
    
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'

    elecs, alphas, scores, zcoefs, pval, scores_null, coefs, features  = [reg_dict[key] for key in ['elecs','alphas','scores', 'zcoefs', 'pval', 'scores_null', 'coefs', 'features']]

    for elec in elecs:
        idx = np.where(np.in1d(elecs, elec))[0][0]

        #plot alpha distribution
        count_dict = Counter(alphas[idx])
        i = np.argsort(count_dict.keys())
        sorted_values = np.array(count_dict.values())[i]
        sorted_keys = np.array(count_dict.keys())[i]

        f, ax = plt.subplots()
        ax.plot(sorted_values)
        ax.set_xticks(range(len(sorted_keys)))
        ax.set_xticklabels(['%.2f' %x for x in sorted_keys])
        ax.set_title('%s %s - e%i - distribution of alphas' %(subj, task, elec))

        plotname = os.path.join(SJdir, 'PCA','Stats','Regression', '_'.join([subj, task, str(elec), 'alpha_distribution.png']))
        plt.savefig(plotname)
        plt.close()

        #plot histogram of scores vs null distribution
        f, ax = plt.subplots()
        ax.hist(scores[idx])
        ax.hist(scores_null[idx], color = 'r', alpha = 0.5)
        ax.set_title('%s %s - e%i - distribution of scores, p =  %.3f' %(subj, task, elec, pval[idx]))

        plotname = os.path.join(SJdir, 'PCA','Stats','Regression', '_'.join([subj, task, str(elec), 'score_distribution.png']))
        plt.savefig(plotname)
        plt.close()

        #plot histogram of coefficients
        f,ax = plt.subplots(2,2, figsize = (10,7))
        for i, x in enumerate(features):
            j = np.unravel_index(i, (2,2))
            ax[j].hist(np.array(coefs[idx])[:,i])
            ax[j].set_title(x)
        f.suptitle('%s %s - e%i - coefficients' %(subj, task, elec))

        plotname = os.path.join(SJdir, 'PCA', 'Stats', 'Regression', '_'.join([subj, task, str(elec), 'coefficients.png']))
        plt.savefig(plotname)
        plt.close()


if __name__ == '__main__':
    DATASET = sys.argv[1]
    HG_regression_allelecs_SGE(DATASET)
