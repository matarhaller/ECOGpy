import pandas as pd
import os
import loadmat
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt 

def crossover_by_time(subj, task):
    #PLV by time - 1 value per trial
    
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'
    
    filename= os.path.join(SJdir, 'Subjs', subj, task, 'lf_hg_phase')
    data_dict = loadmat.loadmat(filename)
    hg_phase, lf_phase, srate = [data_dict.get(key) for key in ['hg_phase','lf_phase', 'srate']]
    filename = os.path.join(SJdir, 'Subjs', subj, task, 'subj_globals.mat')
    
    filename = os.path.join(SJdir, 'Subjs', subj, task, 'subj_globals.mat')
    data_dict = loadmat.loadmat(filename)
    original_srate = data_dict['srate'] #1017.3 or 1k

    #get crossovers
    filename = os.path.join(SJdir, 'PCA', 'csvs_FINAL','Bin_Stats_v1_D+R.csv')
    df_val = pd.read_csv(filename)
    df_val[['RT','t2 (last cross)']].head()

    #get electrodes
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'
    filename = os.path.join(SJdir, 'PCA', 'csvs_FINAL', 'mean_traces_all_subjs_dropSR.csv')
    df = pd.read_csv(filename)
    df = df[(df.subj == subj) & (df.task == task)]
    elecs = df[(df.pattern == 'D') | (df.pattern == 'R')].elec.values
    
    #get bins
    p = 'D'
    filename= os.path.join(SJdir, 'SingleTrials','alltrials','data', 'singletrials_allelecs_smooth_nodecision_' + p + '_dropSR.mat')
    data_dict = loadmat.loadmat(filename)
    allRTs = data_dict['allRTs']
    bins = np.arange(allRTs.min(), allRTs.max(), 50)

    #add bin start and stop to df_vals
    starts, stops = [[] for i in range(2)]
    for i in range(df_val.shape[0]):
        tmp = bins - df_val.iloc[i].RT
        idx = np.where(tmp<0)[0][-1]
        start, stop = bins[idx:idx+2]
        starts.append(start)
        stops.append(stop)
        
    df_val['start'] = starts
    df_val['stop'] = stops

    #load in onsets, offsets
    filename= os.path.join(SJdir, 'Subjs', subj, task, 'onsets_word_corr_g')
    data_dict = loadmat.loadmat(filename)
    onsets_stim = data_dict['onsets_word_corr_g']/original_srate*1000

    filename= os.path.join(SJdir, 'Subjs', subj, task, 'onsets_resp_corr_g')
    data_dict = loadmat.loadmat(filename)
    onsets_resp = data_dict['onsets_resp_corr_g']/original_srate*1000

    #calculate RT, drop outliers
    RTs = onsets_resp - onsets_stim
    goodidx = (RTs > np.mean(RTs) - 3* np.std(RTs)) * (RTs < np.mean(RTs) + 3 * np.std(RTs))
    RTs = RTs[goodidx]
    onsets_stim = onsets_stim[goodidx]
    onsets_resp = onsets_resp[goodidx]

    #for each trial, calculate which RT bin it falls in, calculate crossover point (make array of crossoverpoints)
    bin_idx = np.digitize(RTs, starts)-1
    crossovers = df_val.iloc[bin_idx]['t2 (last cross)']

    #take relevant window across crossover point - build matrix
    lf_c_dict, hg_c_dict, lf_baseline_dict, hg_baseline_dict, lf_pre_c_dict, lf_post_c_dict, hg_pre_c_dict, hg_post_c_dict = [dict() for i in range(8)]
    for e in elecs:    
        lf_c_trials, hg_c_trials, lf_pre_c_trials, hg_pre_c_trials, lf_post_c_trials, hg_post_c_trials, lf_baseline_trials, hg_baseline_trials = [np.empty((len(onsets_stim),250)) for i in range(8)]

        for i, c in enumerate(crossovers.values):
            c_point = int((onsets_stim[i]+c))
            lf_c_trials[i,:] = lf_phase[e, c_point - 125 : c_point + 125]
            hg_c_trials[i,:] = hg_phase[e, c_point-125 : c_point + 125]

            lf_pre_c_trials[i,:] = lf_phase[e, c_point - 250 : c_point]
            hg_pre_c_trials[i,:] = hg_phase[e, c_point - 250 : c_point]

            lf_post_c_trials[i,:] = lf_phase[e, c_point : c_point + 250]
            hg_post_c_trials[i,:] = hg_phase[e, c_point : c_point + 250]

            lf_baseline_trials[i,:] = lf_phase[e, onsets_stim[i] - 250 : onsets_stim[i]]
            hg_baseline_trials[i,:] = hg_phase[e, onsets_stim[i] - 250 : onsets_stim[i]]

        lf_c_dict[e] = lf_c_trials
        hg_c_dict[e] = hg_c_trials

        lf_pre_c_dict[e] = lf_pre_c_trials
        hg_pre_c_dict[e] = hg_pre_c_trials

        lf_post_c_dict[e] = lf_post_c_trials
        hg_post_c_dict[e] = hg_post_c_trials

        lf_baseline_dict[e] = lf_baseline_trials
        hg_baseline_dict[e] = hg_baseline_trials

    #calculate plv - one value per trial (mean across time)
    tmp = pd.DataFrame()

    for e1 in elecs:
        for e2 in elecs:
            pattern = '%s-%s' %(df[df.elec == e1].pattern.values[0], df[df.elec == e2].pattern.values[0])
            print(e1, e2, pattern)
            f, ax = plt.subplots(2, 2, subplot_kw = dict(projection = 'polar'), figsize = (10,10))
            plt.suptitle('%s %s PLV : e%i (LF) - e%i (HG) : %s' %(subj, task, e1, e2, pattern))
            ax = ax.flatten()

            plv_c, plv_pre_c, plv_post_c, plv_baseline = [[] for i in range(4)]
            for i in range(len(onsets_stim)): #per trial
                plv_c.append(lf_c_dict[e1][i,:] - hg_c_dict[e2][i,:])
                plv_pre_c.append(lf_pre_c_dict[e1][i,:] - hg_pre_c_dict[e2][i,:])
                plv_post_c.append(lf_post_c_dict[e1][i,:] - hg_post_c_dict[e2][i,:])
                plv_baseline.append(lf_baseline_dict[e1][i,:] - hg_baseline_dict[e2][i,:])

            plv_c = np.mean(np.exp(1j * np.array(plv_c)), 1) #1 complex value per trial
            plv_pre_c = np.mean(np.exp(1j * np.array(plv_pre_c)), 1)
            plv_post_c = np.mean(np.exp(1j * np.array(plv_post_c)), 1)
            plv_baseline = np.mean(np.exp(1j * np.array(plv_baseline)), 1)

            for i in plv_baseline:
                ax[0].plot([0, i.real], [0, i.imag], color = 'b', alpha = 0.5)
            ax[0].plot([0, np.mean(np.angle(plv_baseline))], [0, np.mean(abs(plv_baseline))], color = 'k', lw = 4)
            ax[0].set_title("baseline = %.2f" %(np.mean(abs(plv_baseline))))

            for i in plv_pre_c:
                ax[1].plot([0, i.real], [0, i.imag], color = 'b', alpha = 0.5)
            ax[1].plot([0, (np.mean(np.angle(plv_pre_c)))], [0, np.mean(abs(plv_pre_c))], color = 'k', lw = 4)
            ax[1].set_title("pre cross = %.2f" %(np.mean(abs(plv_pre_c))))

            for i in plv_c: #per trial
                ax[2].plot([0, i.real], [0, i.imag], color = 'b', alpha = 0.5)
            ax[2].plot([0, (np.mean(np.angle(plv_c)))], [0, np.mean(abs(plv_c))], color = 'k', lw = 4)
            ax[2].set_title("crossover = %.2f" %(np.mean(abs(plv_c))))

            for i in plv_post_c:
                ax[3].plot([0, i.real], [0, i.imag], color = 'b', alpha = 0.5)
            ax[3].plot([0, np.mean(np.angle(plv_post_c))], [0, np.mean(abs(plv_post_c))], color = 'k', lw = 4)
            ax[3].set_title("post cross = %.2f" %(np.mean(abs(plv_post_c))))

            filename = '/home/knight/matar/MATLAB/DATA/Avgusta/PCA/PLV/figures/trials/%s_%s_e%i_e%i_trials.png' %(subj, task, e1, e2)
            f.savefig(filename, bbox_inches='tight')
            plt.close()

            tmp = tmp.append({'LF_elec': e1, 'HG_elec' : e2, 'window': 'baseline', 'plv' : np.mean(abs(plv_baseline)), 'p': 1}, ignore_index = True)

            _, pval = stats.ttest_ind(np.arctanh(abs(plv_baseline)), np.arctanh(abs(plv_pre_c)))
            tmp = tmp.append({'LF_elec': e1, 'HG_elec' : e2, 'window': 'pre', 'plv' : np.mean(abs(plv_pre_c)), 'p': pval}, ignore_index = True)

            _, pval = stats.ttest_ind(np.arctanh(abs(plv_baseline)), np.arctanh(abs(plv_c)))
            tmp = tmp.append({'LF_elec': e1, 'HG_elec' : e2, 'window': 'cross', 'plv' : np.mean(abs(plv_c)), 'p': pval}, ignore_index = True)

            _, pval = stats.ttest_ind(np.arctanh(abs(plv_baseline)), np.arctanh(abs(plv_post_c)))
            tmp = tmp.append({'LF_elec': e1, 'HG_elec' : e2, 'window': 'post', 'plv' : np.mean(abs(plv_post_c)), 'p': pval}, ignore_index = True)

    filename = '/home/knight/matar/MATLAB/DATA/Avgusta/PCA/PLV/%s_%s_plv_trials.csv' %(subj, task)
    tmp.to_csv(filename, index = False)

    
def crossover_by_trials(subj, task):
    #1 value per time point
    
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'
    
    filename= os.path.join(SJdir, 'Subjs', subj, task, 'lf_hg_phase')
    data_dict = loadmat.loadmat(filename)
    hg_phase, lf_phase, srate = [data_dict.get(key) for key in ['hg_phase','lf_phase', 'srate']]
    filename = os.path.join(SJdir, 'Subjs', subj, task, 'subj_globals.mat')
    
    filename = os.path.join(SJdir, 'Subjs', subj, task, 'subj_globals.mat')
    data_dict = loadmat.loadmat(filename)
    original_srate = data_dict['srate'] #1017.3 or 1k

    #get crossovers
    filename = os.path.join(SJdir, 'PCA', 'csvs_FINAL','Bin_Stats_v1_D+R.csv')
    df_val = pd.read_csv(filename)
    df_val[['RT','t2 (last cross)']].head()

    #get electrodes
    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'
    filename = os.path.join(SJdir, 'PCA', 'csvs_FINAL', 'mean_traces_all_subjs_dropSR.csv')
    df = pd.read_csv(filename)
    df = df[(df.subj == subj) & (df.task == task)]
    elecs = df[(df.pattern == 'D') | (df.pattern == 'R')].elec.values
    
    #get bins
    p = 'D'
    filename= os.path.join(SJdir, 'SingleTrials','alltrials','data', 'singletrials_allelecs_smooth_nodecision_' + p + '_dropSR.mat')
    data_dict = loadmat.loadmat(filename)
    allRTs = data_dict['allRTs']
    bins = np.arange(allRTs.min(), allRTs.max(), 50)

    #add bin start and stop to df_vals
    starts, stops = [[] for i in range(2)]
    for i in range(df_val.shape[0]):
        tmp = bins - df_val.iloc[i].RT
        idx = np.where(tmp<0)[0][-1]
        start, stop = bins[idx:idx+2]
        starts.append(start)
        stops.append(stop)
        
    df_val['start'] = starts
    df_val['stop'] = stops  
    
    #load in onsets, offsets
    if task in ['EmoGen', 'EmoRep']:
        filename= os.path.join(SJdir, 'Subjs', subj, task, 'onsets_word_corr_g')
        data_dict = loadmat.loadmat(filename)
        onsets_stim = data_dict['onsets_word_corr_g']/original_srate*1000
        
        filename= os.path.join(SJdir, 'Subjs', subj, task, 'onsets_resp_corr_g')
        data_dict = loadmat.loadmat(filename)
        onsets_resp = data_dict['onsets_resp_corr_g']/original_srate*1000

    elif task in ['SelfAud', 'SelfVis']:
        filename= os.path.join(SJdir, 'Subjs', subj, task, 'onsets_word_g')
        data_dict = loadmat.loadmat(filename)
        onsets_stim = data_dict['onsets_word_g']/original_srate*1000
        
        filename= os.path.join(SJdir, 'Subjs', subj, task, 'onsets_resp_g')
        data_dict = loadmat.loadmat(filename)
        onsets_resp = data_dict['onsets_resp_g']/original_srate*1000

    elif task in ['FaceEmo','FaceGen']:
        filename= os.path.join(SJdir, 'Subjs', subj, task, 'onsets_face_corr_g')
        data_dict = loadmat.loadmat(filename)
        onsets_stim = data_dict['onsets_face_corr_g']/original_srate*1000
        
        filename= os.path.join(SJdir, 'Subjs', subj, task, 'onsets_resp_corr_g')
        data_dict = loadmat.loadmat(filename)
        onsets_resp = data_dict['onsets_resp_corr_g']/original_srate*1000
        
    else:
        print 'task not found'
    

    #calculate RT, drop outliers
    RTs = onsets_resp - onsets_stim
    goodidx = (RTs > np.mean(RTs) - 3* np.std(RTs)) * (RTs < np.mean(RTs) + 3 * np.std(RTs))
    RTs = RTs[goodidx]
    onsets_stim = onsets_stim[goodidx]
    onsets_resp = onsets_resp[goodidx]

    #for each trial, calculate which RT bin it falls in, calculate crossover point (make array of crossoverpoints)
    bin_idx = np.digitize(RTs, starts)-1
    crossovers = df_val.iloc[bin_idx]['t2 (last cross)']

    #take relevant window across crossover point - build matrix
    lf_c_dict, hg_c_dict, lf_baseline_dict, hg_baseline_dict, lf_pre_c_dict, lf_post_c_dict, hg_pre_c_dict, hg_post_c_dict = [dict() for i in range(8)]
    for e in elecs:    
        lf_c_trials, hg_c_trials = [np.empty((len(onsets_stim),1000)) for i in range(2)]

        for i, c in enumerate(crossovers.values):
            c_point = int((onsets_stim[i]+c))
            lf_c_trials[i,:] = lf_phase[e, c_point - 500 : c_point + 500]
            hg_c_trials[i,:] = hg_phase[e, c_point - 500 : c_point + 500]

        lf_c_dict[e] = lf_c_trials
        hg_c_dict[e] = hg_c_trials
    
    #calculate plv - one value per timepoint (mean across trials, 1000 points)
    tmp = pd.DataFrame()

    for e1 in elecs:
        for e2 in elecs:
            pattern = '%s-%s' %(df[df.elec == e1].pattern.values[0], df[df.elec == e2].pattern.values[0])

            phase_diff = list()
            for i in range(len(onsets_stim)): #per trial
                phase_diff.append(lf_c_dict[e1][i,:] - hg_c_dict[e2][i,:])

            plv_c = np.mean(np.exp(1j * np.array(phase_diff)), 0) #average across trials, get time series

            plv_c_z = (abs(plv_c) - np.mean(abs(plv_c))) / np.std(abs(plv_c))

            tmp = tmp.append(pd.DataFrame({'LF_elec': [e1] *1000, 'HG_elec' : [e2] * 1000, 'pattern': [pattern] * 1000, 'plv' : pd.Series(plv_c_z, name = 'plv', index = np.arange(-500,500))}))

    df_zscore = tmp.reset_index()
    df_zscore.columns = ['time','HG_elec','LF_elec','pattern','plv'] #zscored
    
    #stats on 300ms flanking crossover point
    dftmp = pd.DataFrame()
    for e1 in elecs:
        for e2 in elecs:
            df_plv = df_zscore[(df_zscore.HG_elec == e1) & (df_zscore.LF_elec == e2)]
            pattern = np.unique(df_plv.pattern)[0]
            pre = df_plv[(df_plv.time<=0) & (df_plv.time>-300)].plv
            post = df_plv[(df_plv.time<300) & (df_plv.time>=0)].plv
            t, pval = stats.ttest_ind(pre, post)
            dftmp = dftmp.append({'LF_elec': e1, 'HG_elec' : e2, 'pattern': pattern, 'mean_pre' : pre.mean(), 'mean_post': post.mean(), 'p' : pval, 't': t}, ignore_index = True)

    filename = '/home/knight/matar/MATLAB/DATA/Avgusta/PCA/PLV/flanking_300ms/plv_signifiance_crossover_300ms_zscore_%s_%s.csv' %(subj, task)
    dftmp.to_csv(filename)
    
    #stats on mean traces
    dftmp = pd.DataFrame()
    for pattern in ['D-D','D-R','R-R','R-D']:
        pre, post = [[] for i in range(2)]
        for e1 in elecs:
            for e2 in elecs:
                df_plv = df_zscore[(df_zscore.HG_elec == e1) & (df_zscore.LF_elec == e2) & (df_zscore.pattern == pattern)]
                pre.append(df_plv[(df_plv.time<=0) & (df_plv.time>-300)].plv.mean())
                post.append(df_plv[(df_plv.time<300) & (df_plv.time>=0)].plv.mean())

                pre = [x for x in pre if str(x) != 'nan']    
                post = [x for x in post if str(x) != 'nan'] 

        t, pval = stats.ttest_ind(pre, post)
        dftmp = dftmp.append({'pattern': pattern, 'mean_pre' : np.nanmean(pre), 'mean_post': np.nanmean(post), 'p' : pval, 't': t}, ignore_index = True)
    
    filename = '/home/knight/matar/MATLAB/DATA/Avgusta/PCA/PLV/mean_traces/plv_signifiance_crossover_300_time_mean_%s_%s.csv' %(subj, task)
    dftmp.to_csv(filename)

if __name__ == '__main__':
    DATASET = sys.argv[1]
    subj, task = '_'.split(DATASET)
    crossover_by_time(subj, task)
