from __future__ import division
import os
import scipy.io as spio
import scipy.stats as stats
import numpy as np
import sys
import cPickle as pickle
import pandas as pd

def duration(DATASET, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/', start_time = 0, stop_time_pre = -1000, stop_time_post = 100, step = 25, numiter = 1000):
    """
    calculate R and pvalue for duration correlation for each cluster for a subj/task
    creates a dictionary of design matrices- onset is always onset of significant HG 
    (from ShadePlots_hclust_thresh10) - offset varies from start to stop in stepsize
    returns tuple of lists (Rvals, pvals)
    made to run on SGE (can be called from the command line)
    """
    subj, task = DATASET.split('_')

    #load TvD stim locked
    #(to know which clusters signficant, and their onsets)
    filename = os.path.join(SJdir, 'PCA','ShadePlots_hclust_thresh15', '_'.join([subj, task, 'TvD.mat']))
    data = spio.loadmat(filename)

    params = data['Params']
    TvD = data['TvD']
    srate = data['srate']

    clusters = list()
    for i, c  in enumerate(TvD[:,0]):
        clusters.append((str(c[0]), TvD[i,3].squeeze()))
    
    #TvD resplocked
    filename = os.path.join(SJdir, 'PCA','ShadePlots_hclust_thresh15', '_'.join([subj, task, 'TvD_resp.mat']))
    data = spio.loadmat(filename)
    TvD_resp = data['TvD']

    for i, c  in enumerate(np.setdiff1d(TvD_resp[:,0], TvD[:,0])): #purely resp locked, not stim locked
        clusters.append((str(c[0]), np.array(round(250/1000*srate)).reshape(1,1)))
    

    Rvals = list()
    pvals = list()
    tstats = list()
    pvals_ttest = list()
    bests = list()
    means = list()
    onsets = list()

    for y, x in enumerate(clusters):
        print x
        #load cluster single trial data 
        filename = os.path.join(SJdir, 'PCA','SingleTrials_hclust', x[0]) 
        data = spio.loadmat(filename)
        cdata = data['cdata']
        RTs = data['RTs_all'].squeeze()
        srate = data['srate']

        bl_st = round(int(params['bl_st'])/1000*srate) #baseline offset
        onset = x[1].flatten()[0] #start of significant activity

        #sliding window from onset:resp-stop in steps of stepsize
        st_tp = round(start_time/1000*srate)#start at st_tp + start
        stepsize = round(step/1000*srate) #sliding window size
        
        #stopB =  round(stop_time_post/1000*srate) #end sliding window at RT-stop
        #stopA = round(stop_time_pre/1000*srate)

        #create dictionary of design matrices (RTmtx)
        RTdict = dict()
        for s in np.arange(stop_time_pre/1000*srate, stop_time_post/1000*srate, step/1000*srate):
            RTmtx = np.zeros_like(cdata)
            for i in np.arange(RTmtx.shape[0]):
                RTmtx[i, abs(bl_st)+onset+st_tp : abs(bl_st)+RTs[i]+s] = 1
            if RTmtx.any():
                RTdict[str(int(s/srate*1000))] = RTmtx
        
        #find R matrix with highest correlation
        Rs = list()
        for r in np.sort(RTdict.keys()):
            RTmtx = RTdict[r]
            Rs.append(stats.pearsonr(RTmtx.flatten(), cdata.flatten())[0])
        best = np.sort(RTdict.keys())[np.argmax(Rs)] #design matrix with highest correlation
        RTmtx = RTdict[best]
        R = max(Rs)

        #calculate 1 sample ttest for data
        [tstat, ttest_p] = stats.ttest_1samp(cdata[np.logical_not(RTmtx)], 0)

        #calculate mean activity for zeroed out part of design matrix
        means.append(cdata[np.logical_not(RTmtx)].mean())

        #calculate pvalue for the best design matrix
        surr = np.zeros(numiter)
        for j in np.arange(numiter):
            idx = np.random.choice(np.arange(RTmtx.shape[0]), size = RTmtx.shape[0], replace = False)
            surr[j] = stats.pearsonr(RTmtx[idx,:].flatten(), cdata.flatten())[0]
            if not(j % 50):
                print j

        if R>0:
            p = sum(surr>R)/numiter
        else:
            p = sum(surr<R)/numiter
        if p==0:
            p = 1/numiter
        

        Rvals.append(R), pvals.append(p), tstats.append(tstat), pvals_ttest.append(ttest_p)
        bests.append(best), onsets.append(onset)

    dur_dict = {'R' : Rvals, 'p' : pvals, 'clusters' : [x[0] for x in clusters], 'tstat' : tstats, 'ttest_p': pvals_ttest, 'best_onset': bests, 'mean' : means, 'onset':onsets}
    df = pd.DataFrame(dur_dict)
    print df

    filename = os.path.join(SJdir, 'PCA','duration_dict', ''.join(['_'.join([subj,task]), '.csv']))
    print filename
    df.to_csv(filename)

if __name__ == '__main__':
    DATASET = sys.argv[1]
    duration(DATASET)

