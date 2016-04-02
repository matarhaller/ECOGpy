from __future__ import division
import os
import numpy as np
import cPickle as pickle
import sys
import pandas as pd
from sklearn import cross_validation, grid_search
from sklearn.linear_model import Ridge
from sklearn.preprocessing import scale

def HG_regression_allelecs_SGE(DATASET):
    """
    feeds in subj/task to HG_regression_all_elecs and saves output to file
    """
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'
    subj, task = DATASET.split('_')
    
    betas, predscores, elecs, features, all_alphas, pvals = HG_regression_allelecs(subj, task)

    filename = os.path.join(SJdir, 'PCA', 'Stats', 'Regression', '_'.join([subj, task]))

    pickle.dump({'pvals': pvals, 'betas': betas, 'predscores': predscores, 'elecs': elecs, 'features': features, 'alphas':all_alphas}, open(filename + '.p', 'wb'))
    
    data_array =  np.hstack((np.asarray(betas), np.reshape(predscores, (len(predscores), 1)), np.reshape(pvals, (len(pvals), 1)), np.reshape(all_alphas, (len(all_alphas), 1))))
    features.append('pred_score')
    features.append('pval_predscore')
    features.append('alpha')

    df = pd.DataFrame(data_array, columns = features, index = elecs)
    df.to_csv(filename + '_betas.csv')
    

def HG_regression_allelecs(subj, task): 
    '''
    Runs ridge regression on maxes, means, stds, sums, latency (proportion) data for a subj/task
    Loops on each electrode
    Splits data into training and test sets
    Runs 100 fold CV to get best alpha on training set (calls fit_model on X_train)
    Calcualates best alpha from fit_model alphas (takes mean of fit_model output)
    Trains model with best alpha on full training set (X_train)
    Calculates prediction score on held out test set (X_test)
    Outputs coefficients and prediction scores for each duration electrode
    '''
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'

    # load data
    filename = os.path.join(SJdir, 'PCA', 'ShadePlots_hclust', 'elecs', 'significance_windows', 'data', ''.join([subj, '_', task, '.p']))
    data_dict = pickle.load(open(filename, 'rb'))
   
    all_alphas, betas, predscores, pvals = [[] for i in range(4)]

    #set parameters
    #features = ['maxes', 'means', 'stds', 'sums', 'lats_pro']
    features = ['maxes', 'means', 'stds', 'lats_pro']
    predictor = 'RTs'

    elecs = data_dict['means'].keys()

    for elec in elecs:

        #define data
        X = np.array([data_dict[x][elec] for x in features]).T
        Y = data_dict[predictor][elec]

        #drop nans from data
        mask = np.all(np.isnan(X), axis = 1)
        X = X[~mask]
        mask = np.isnan(Y)
        Y = Y[~mask]
        
        #external cv for prediction score stability
        nfolds = 100
        pval, scores, coefs, alphas_cv = [[] for x in range(4)]
        for c in range(nfolds):
            #define training and test set
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size = 0.2)

            #normalize
            X_train = scale(X_train.astype(float))
            X_test = scale(X_test.astype(float))
            y_train = scale(y_train.astype(float))
            y_test = scale(y_test.astype(float))

            #fit ridge regression with cross validation on training set
            alphas, models, scores, coefs = fit_model(X_train, y_train)

            #fit final model using mean hyperparam on all training data
            alpha = np.median(alphas) #alphas are not normally distributed, more stable if take median
            model = Ridge(solver = 'lsqr', alpha = alpha, normalize = False, fit_intercept = False)
            model.fit(X_train, y_train)

            p = bootstrap_predscore(X_train, y_train, X_test, y_test, model, niter = 1000)

            #calculate prediction accuracy
            scores.append(np.corrcoef(model.predict(X_test), y_test)[0,1])
            coefs.append(model.coef_)
            alphas_cv.append(alpha)
            pval.append(p)
        
        print (elec, np.median(scores), np.median(pval))
        sys.stdout.flush()

        betas.append(np.median(coefs, axis = 0))
        predscores.append(np.median(scores))
        all_alphas.append(np.median(alphas_cv))
        pvals.append(np.median(pval))
               
    return betas, predscores, elecs, features, all_alphas, pvals


def fit_model(X_train, y_train, cv = 100, test_size = .2):
    """
    takes in an array of features x observations and an array of predictors
    runs ridge with cv, grid search on alphas
    outputs the models
    """

    #set up cv folds
    cvs = cross_validation.ShuffleSplit(len(y_train), n_iter = cv, test_size = 0.2)

    #define model
    model = Ridge(solver = 'lsqr', normalize = False, fit_intercept = False)
    params_grid = {'alpha': np.logspace(-3, 3, 10)}

    #grid search for hyperparameter
    ridge_grid = grid_search.GridSearchCV(model, params_grid, cv = 10)

    alphas, models, scores, coefs = [[] for i in range(4)]

    for train, test in cvs:
        ridge_grid.fit(X_train[train], y_train[train])
        a = ridge_grid.best_params_.values()[0]
        mod = ridge_grid.best_estimator_
        score = np.corrcoef(mod.predict(X_train[test]), y_train[test])[1,0]
        coef = mod.coef_

        alphas.append(a)
        models.append(mod)
        scores.append(score)
        coefs.append(coef)

        return alphas, models, scores, coefs

def bootstrap_predscore(X_train, y_train, X_test, y_test, model, niter = 1000):
    """ calculate pvalue for prediction score (to be done on each cv fold)
    """
    real_score = np.corrcoef(model.predict(X_test), y_test)[0,1]

    surr = list()
    for i in range(niter):
        idx = np.random.permutation(len(y_train))
        model.fit(X_train, y_train[idx])
        score = np.corrcoef(model.predict(X_test), y_test)[0,1]
        surr.append(score)
    
    if (real_score > 0):
        pval = 1 - (np.sum(real_score > surr) / niter)
    elif (real_score < 0):
        pval = 1 - (np.sum(real_score < surr) / niter)
    else:
        pval = 1
    
    return pval


if __name__ == '__main__':
    DATASET = sys.argv[1]
    HG_regression_allelecs_SGE(DATASET)
