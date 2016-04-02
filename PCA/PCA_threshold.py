import os
import pandas as pd
import sys
import numpy as np

''' 
This function thresholds the PCA matrix, 
for each component, looks at the electrodes that passed the threshold 
as a network. Uses those electrodes to go back to correlation matrix and
consider the correlation between the electrodes in that network.
'''

def PCA_threshold_SGE(DATASET):
    """
    feeds in subj/task to PCA_threshold and saves output to file
    """
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'
    subj, task = DATASET.split('_')

    #for p in ['medians', 'means', 'stds', 'maxes','maxes_rel']:
    for p in ['means']: 
        saveDir = os.path.join(SJdir, 'PCA', 'Stats', 'Networks', 'zscore', '_'.join(['PCA',p]), 'networks')
        if not(os.path.exists(saveDir)):
            os.mkdir(saveDir)

        df_thresh = PCA_thresh(subj, task, p)
        
        filename = os.path.join(saveDir, '_'.join([subj, task, 'thresh.csv']))
        df_thresh.to_csv(filename, index = False)

        PCA_corr(subj, task, p, df_thresh) 

def PCA_thresh(subj, task, param):
    '''
    calculates the threshold on a PCA loading matrix (from PYTHON/PCA_elecs.R)
    '''
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'
    #load in ROIs, patterns per elec
    #df_details = pd.read_csv(os.path.join(SJdir, 'PCA','Stats','Regression', 'unsmoothed', folder, 'no_short_windows', 'all_coefs_withpatterns_withROIs.csv'))
    df_details = pd.read_csv(os.path.join(SJdir, 'PCA', 'csvs_FINAL', 'mean_traces_all_subjs_dropSR.csv'))

    #load in PCA loadings
    filename = os.path.join(SJdir, 'PCA', 'Stats', 'Networks', 'zscore', '_'.join(['PCA', param]), '_'.join([subj, task, 'loadings.csv']))
    df = pd.read_csv(filename)

    #format dataframe
    colnames = list(df.columns)
    colnames[0] = 'elec'
    df.columns = colnames
    df = df.set_index('elec')

    #calculate threshold
    thresh = np.min(np.max([df.max(axis = 1), abs(df.min(axis = 1))], axis = 0))

    #threshold PCA matrix
    df_thresh = df>thresh
    df_thresh = df[df_thresh] 

    #change elecs to ints
    df_thresh = df_thresh.reset_index()
    df_thresh.elec = [int(x[1:]) for x in df_thresh.elec]

    #combine details with thresholded dataframe
    df_details_subj = df_details[(df_details.subj == subj) & (df_details.task == task)].reset_index()[['elec','ROI','pattern']]
    df_thresh = pd.merge(df_thresh, df_details_subj, on = 'elec')
    colnames = list(df_thresh.columns)
    colnames.insert(1, colnames.pop(colnames.index('pattern')))
    colnames.insert(2, colnames.pop(colnames.index('ROI')))
    df_thresh = df_thresh[colnames]

    #add column with maximum PC
    b = pd.DataFrame(df_thresh.iloc[:,3:].idxmax(axis = 1),columns = ['max_pc'])
    df_thresh = df_thresh.join(b)

    #add list of PCs in order (no nan)
    c = (df_thresh.iloc[:,3:-1])
    pc_list = list()
    for row in c.itertuples():
        tmp = np.array(row[1:])
        idx = ~np.isnan(tmp)
        tmp = tmp[idx]
        cols = c.columns[idx]
        idx = tmp.argsort()
        pc_list.append(list(cols[idx][::-1]))
        #tmp[idx][::-1]

    df_thresh['pc_list'] = pc_list
    df_thresh['subj'] = subj
    df_thresh['task'] = task

    colnames = list(df_thresh.columns)
    colnames.insert(0, colnames.pop(colnames.index('subj')))
    colnames.insert(1, colnames.pop(colnames.index('task')))
    colnames.insert(3, colnames.pop(colnames.index('max_pc')))
    colnames.insert(4, colnames.pop(colnames.index('pc_list')))
    df_thresh = df_thresh[colnames]

    return df_thresh

def PCA_corr(subj, task, param, df_thresh):
    '''
    this function takes in a thresholded PCA matrix and creates
    correlation matrices for the electrodes that are relevant for 
    each principal component.
    '''
    
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'

    #get original trial parameter data
    filename = os.path.join(SJdir, 'PCA','Stats','outliers','for_PCA', 'zscore', '_'.join([subj, task, param]) + '.csv')
    df_data = pd.read_csv(filename)
    
    #create pairwise correlation matrix
    df_corr = df_data.corr()
    
    #mask correlation matrix by thresholded pc
    pc_idx = [x[:2]== 'PC' for x in df_thresh.columns]
    for pc in df_thresh.columns[pc_idx]:    

        #find indices of elecs that passed PC threshold (PC network)
        idx = df_thresh[~np.isnan(df_thresh[pc])].elec
        idx = [str(x) for x in idx]
        df_pc = df_corr.loc[idx, idx] #pull corr values for those elecs
        
        filename = os.path.join(SJdir, 'PCA', 'Stats', 'Networks', 'zscore', '_'.join(['PCA',param]), 'networks', '_'.join([subj, task, pc + '.csv']))
        df_pc.to_csv(filename)


if __name__ == '__main__':
    DATASET = sys.argv[1]
    PCA_threshold_SGE(DATASET)
