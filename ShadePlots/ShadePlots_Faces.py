from __future__ import division
import pandas as pd
import os
import loadmat
import scipy.stats as stats
import fdr_correct
import matplotlib.pyplot as plt
import numpy as np
import sys
import cPickle as pickle
import glob

def shadeplots_faces(subj, elecs_list, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta', thresh = 0, chunk_size = 0, baseline = -500, black_chunk_size = 0):
    """ 
    calculate onset and offset window for given electrode.
    compares face emo to face gen
    saves csv for each sub/task for easy plotting later

    """

    filename = os.path.join(SJdir, 'Subjs', subj, 'FaceEmo', 'HG_elecMTX_percent_eleclist.mat')
    data = loadmat.loadmat(filename)
    srate = data['srate']
    elecsEmo = data['elec_list']
    dataEmo = data['data_percent']

    filename = os.path.join(SJdir, 'Subjs', subj, 'FaceGen', 'HG_elecMTX_percent_eleclist.mat')
    data = loadmat.loadmat(filename)
    srate = data['srate']
    elecsGen = data['elec_list']
    dataGen = data['data_percent']

    #convert to srate
    bl_st = baseline/1000*srate
    chunksize = chunk_size/1000*srate
    black_chunksize = black_chunk_size/1000*srate

    filename = os.path.join(SJdir, 'Anat', 'ShadePlots_Faces', '_'.join([subj, 'Emo', 'vs', 'Gen']) +'.csv')
    subjs = list();  pthr = list(); elecs = list(); starts = list(); ends = list(); 

    for i, e in enumerate(elecs_list):

        #idx_Emo = np.in1d(activeEmo, e)
        #idx_Gen = np.in1d(activeGen, e)

        #if (sum(idx_Emo) == 0) | (sum(idx_Gen) == 0): #elec not active in one of the blocks
        #    subjs.extend([subj])
        #    elecs.extend([e])
        #    pthr.extend([np.nan])
        #    starts.extend([np.nan])
        #    ends.extend([np.nan])
        #    continue

        
        idx_Emo, idx_Gen = (i, i)
        edataEmo = dataEmo[idx_Emo,:].squeeze()
        edataGen = dataGen[idx_Gen,:].squeeze()

        if edataEmo.shape[1]>edataGen.shape[1]:
            edataEmo = edataEmo[:,:edataGen.shape[1]]
        else:
            edataGen = edataGen[:,:edataEmo.shape[1]]

        pvals = list();

        for j in np.arange(abs(bl_st), edataEmo.shape[1]):
            (t, p) = stats.ttest_ind(edataEmo[:,j], edataGen[:,j])
            pvals.append(p)
        thr = fdr_correct.fdr2(pvals, q = 0.05)
        H = np.array(np.array(pvals<thr)).astype('int')

        if (thr>0):

            #find elecs with window that > chunksize and > threshold (10%)
            passed_thresh = abs(edataEmo[:, abs(bl_st)::].mean(axis=0) - edataGen[:, abs(bl_st)::].mean(axis = 0)) >thresh #difference between blocks is > 10% threshold
            sig_and_thresh = H * passed_thresh
            difference = np.diff(sig_and_thresh, n = 1, axis = 0)
            start_idx = np.where(difference==1)[0]+1
            end_idx = np.where(difference == -1)[0]

            if start_idx.size > end_idx.size: #last chunk goes until end
                end_idx = np.append(end_idx, int(edataEmo.shape[1]-abs(bl_st)))

            elif start_idx.size < end_idx.size:
                start_idx = np.append(0, start_idx) #starts immediately significant

            if (start_idx.size!=0):
                if (start_idx[0] > end_idx[0]): #starts immediately significant
                    start_idx = np.append(0, start_idx)

            if (start_idx.size!=0):
                if (end_idx[-1] < start_idx[-1]):#significant until end
                    end_idx = np.append(end_idx, int(edataEmo.shape[1]-abs(bl_st)))

            chunk = (end_idx - start_idx) >= chunksize

            if sum(chunk) > 0:
                #significant windows on elecs that passed threshold (10%) (ignoring threshold and chunksize)
                difference = np.diff(H, n = 1, axis = 0)
                start_idx = np.where(difference==1)[0]+1
                end_idx = np.where(difference == -1)[0]

                if start_idx.size > end_idx.size: #last chunk goes until end
                    end_idx = np.append(end_idx, int(edataEmo.shape[1]-abs(bl_st)))

                elif start_idx.size < end_idx.size:
                    start_idx = np.append(0, start_idx) #starts immediately significant

                if (start_idx.size!=0):
                    if (start_idx[0] > end_idx[0]): #starts immediately significant
                        start_idx = np.append(0, start_idx)

                if (start_idx.size!=0):
                    if (end_idx[-1] < start_idx[-1]):#significant until end
                        end_idx = np.append(end_idx, int(edataEmo.shape[1]-abs(bl_st)))


                black_chunk = (start_idx[1:] - end_idx[:-1]) > black_chunksize #combine window separated by <200ms

                tmp = np.append(1,black_chunk).astype('bool')
                end_idx = end_idx[np.append(np.where(np.in1d(start_idx, start_idx[tmp]))[0][1:]-1, -1)]
                start_idx = start_idx[tmp]           

                #drop chunks that <100ms
                chunk = (end_idx - start_idx) >= chunksize
                start_idx = start_idx[chunk]
                end_idx = end_idx[chunk]

            else: #no chunks
                start_idx = np.zeros((1,))
                end_idx = np.zeros((1,))
                
        else: #thr<0
            start_idx = np.zeros((1,))
            end_idx = np.zeros((1,))

        subjs.extend([subj] * len(start_idx))
        elecs.extend([e] * len(start_idx))
        pthr.extend([thr] * len(end_idx))
        starts.extend(start_idx)
        ends.extend(end_idx)

        data_dict = {'edataEmo':edataEmo, 'edataGen':edataGen, 'bl_st':bl_st, 'start_idx':start_idx, 'end_idx':end_idx, 'srate':srate,'thresh':thresh, 'chunksize':chunksize, 'black_chunksize':black_chunksize}
        data_path = os.path.join(SJdir, 'Anat','ShadePlots_Faces', 'data',''.join([subj, '_', 'Emo_vs_Gen', '_e', str(e), '.p']))
       
        with open(data_path, 'w') as f:
            pickle.dump(data_dict, f)
            f.close()

    sig_windows = pd.DataFrame({'subj':subjs, 'elec':elecs, 'pthreshold':pthr, 'start_idx':starts, 'end_idx':ends})
    sig_windows = sig_windows[['subj', 'elec', 'start_idx','end_idx','pthreshold']]
    sig_windows.to_csv(filename)
    
    return sig_windows

def shadeplots_faces_resp(subj, elecs_list, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta', thresh = 0, chunk_size = 0, baseline = -500, black_chunk_size = 0):
    """ 
    calculate onset and offset window for given electrode.
    compares face emo to face gen
    saves csv for each sub/task for easy plotting later

    """

    filename = os.path.join(SJdir, 'Subjs', subj, 'FaceEmo', 'HG_elecMTX_percent_eleclist.mat')
    data = loadmat.loadmat(filename)
    srate = data['srate']
    elecsEmo = data['elec_list']
    dataEmo = data['data_percent_resp']

    filename = os.path.join(SJdir, 'Subjs', subj, 'FaceGen', 'HG_elecMTX_percent_eleclist.mat')
    data = loadmat.loadmat(filename)
    srate = data['srate']
    elecsGen = data['elec_list']
    dataGen = data['data_percent_resp']

    #convert to srate
    chunksize = chunk_size/1000*srate
    black_chunksize = black_chunk_size/1000*srate

    filename = os.path.join(SJdir, 'Anat', 'ShadePlots_Faces', '_'.join([subj, 'Emo', 'vs', 'Gen']) +'_resp.csv')
    subjs = list();  pthr = list(); elecs = list(); starts = list(); ends = list(); 

    for i, e in enumerate(elecs_list):

        idx_Emo, idx_Gen = (i, i)
        edataEmo = dataEmo[idx_Emo,:].squeeze()
        edataGen = dataGen[idx_Gen,:].squeeze()

        if edataEmo.shape[1]>edataGen.shape[1]:
            edataEmo = edataEmo[:,:edataGen.shape[1]]
        else:
            edataGen = edataGen[:,:edataEmo.shape[1]]

        pvals = list();

        for j in np.arange(0, edataEmo.shape[1]):
            (t, p) = stats.ttest_ind(edataEmo[:,j], edataGen[:,j])
            pvals.append(p)
        thr = fdr_correct.fdr2(pvals, q = 0.05)
        H = np.array(np.array(pvals<thr)).astype('int')

        if (thr>0):

            #find elecs with window that > chunksize and > threshold (10%)
            passed_thresh = abs(edataEmo[:, 0::].mean(axis=0) - edataGen[:, 0::].mean(axis = 0)) >thresh #difference between blocks is > 10% threshold
            sig_and_thresh = H * passed_thresh
            difference = np.diff(sig_and_thresh, n = 1, axis = 0)
            start_idx = np.where(difference==1)[0]+1
            end_idx = np.where(difference == -1)[0]

            if start_idx.size > end_idx.size: #last chunk goes until end
                end_idx = np.append(end_idx, int(edataEmo.shape[1]))

            elif start_idx.size < end_idx.size:
                start_idx = np.append(0, start_idx) #starts immediately significant

            if (start_idx.size!=0):
                if (start_idx[0] > end_idx[0]): #starts immediately significant
                    start_idx = np.append(0, start_idx)

            if (start_idx.size!=0):
                if (end_idx[-1] < start_idx[-1]):#significant until end
                    end_idx = np.append(end_idx, int(edataEmo.shape[1]))

            chunk = (end_idx - start_idx) >= chunksize

            if sum(chunk) > 0:
                #significant windows on elecs that passed threshold (10%) (ignoring threshold and chunksize)
                difference = np.diff(H, n = 1, axis = 0)
                start_idx = np.where(difference==1)[0]+1
                end_idx = np.where(difference == -1)[0]

                if start_idx.size > end_idx.size: #last chunk goes until end
                    end_idx = np.append(end_idx, int(edataEmo.shape[1]))

                elif start_idx.size < end_idx.size:
                    start_idx = np.append(0, start_idx) #starts immediately significant

                if (start_idx.size!=0):
                    if (start_idx[0] > end_idx[0]): #starts immediately significant
                        start_idx = np.append(0, start_idx)

                if (start_idx.size!=0):
                    if (end_idx[-1] < start_idx[-1]):#significant until end
                        end_idx = np.append(end_idx, int(edataEmo.shape[1]))


                black_chunk = (start_idx[1:] - end_idx[:-1]) > black_chunksize #combine window separated by <200ms

                tmp = np.append(1,black_chunk).astype('bool')
                end_idx = end_idx[np.append(np.where(np.in1d(start_idx, start_idx[tmp]))[0][1:]-1, -1)]
                start_idx = start_idx[tmp]           

                #drop chunks that <100ms
                chunk = (end_idx - start_idx) >= chunksize
                start_idx = start_idx[chunk]
                end_idx = end_idx[chunk]

            else: #no chunks
                start_idx = np.zeros((1,))
                end_idx = np.zeros((1,))
                
        else: #thr<0
            start_idx = np.zeros((1,))
            end_idx = np.zeros((1,))

        start_idx = start_idx - np.round(500/1000*srate) #check, should shift it back to be -500 to 500 window
        end_idx = end_idx - np.round(500/1000*srate)

        subjs.extend([subj] * len(start_idx))
        elecs.extend([e] * len(start_idx))
        pthr.extend([thr] * len(end_idx))
        starts.extend(start_idx)
        ends.extend(end_idx)

        data_dict = {'edataEmo':edataEmo, 'edataGen':edataGen, 'start_idx':start_idx, 'end_idx':end_idx, 'srate':srate,'thresh':thresh, 'chunksize':chunksize, 'black_chunksize':black_chunksize}
        data_path = os.path.join(SJdir, 'Anat','ShadePlots_Faces', 'data',''.join([subj, '_', 'Emo_vs_Gen', '_e', str(e), '_resp.p']))
       
        with open(data_path, 'w') as f:
            pickle.dump(data_dict, f)
            f.close()

    sig_windows = pd.DataFrame({'subj':subjs, 'elec':elecs, 'pthreshold':pthr, 'start_idx':starts, 'end_idx':ends})
    sig_windows = sig_windows[['subj', 'elec', 'start_idx','end_idx','pthreshold']]
    sig_windows.to_csv(filename)
    
    return sig_windows


def plot_shadeplot(subj, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'):
        """
        plot shade plots with activity window in red (from ShadePlots_allelecs_plot.py)
        """
        files = glob.glob(os.path.join(SJdir, 'Anat', 'ShadePlots_Faces', 'data', ''.join([subj, '*.p']))) #electrodes
        files = filter(lambda f: not(f[-6:-2] == 'resp'), files) #drop resp files

        for f in files:
            print f
            elecname = f.split('_')[-1].split('.p')[0] 

            with open(f, 'r') as x:
                data_dict = pickle.load(x)
                x.close()

            #map dictionary to variables
            edataEmo, edataGen, bl_st, srate, start_idx, end_idx, chunksize, black_chunksize, thresh = [data_dict.get(k) for k in ['edataEmo','edataGen', 'bl_st', 'srate', 'start_idx', 'end_idx', 'chunksize', 'black_chunksize', 'thresh']]

            #plot
            f, ax = plt.subplots(figsize = (10,10))

            if edataEmo.shape[0] == 0: #if not active
                edataEmo = np.zeros_like(edataGen)
            elif edataGen.shape[0] == 0:
                edataGen = np.zeros_like(edataEmo)

            scale_min = min(edataGen.mean(axis = 0).min(), edataEmo.mean(axis = 0).min()) - 10
            scale_max = max(edataGen.mean(axis = 0).max(), edataEmo.mean(axis = 0).max()) + 10
            tmp = (np.arange(scale_min, scale_max))
            
            ax.plot(np.arange(bl_st, edataEmo.shape[1]+bl_st), edataEmo.mean(axis = 0), zorder = 1, linewidth = 3, color = 'green', label = 'FaceEmo')
            sem = np.std(edataEmo, axis = 0)/np.sqrt(edataEmo.shape[0])
            ax.fill_between(np.arange(bl_st, edataEmo.shape[1]+bl_st), edataEmo.mean(axis = 0)+sem, edataEmo.mean(axis=0)-sem, alpha = 0.5, zorder = 0, edgecolor = 'None', facecolor = 'green', label = None)
            
            ax.plot(np.arange(bl_st, edataGen.shape[1]+bl_st), edataGen.mean(axis = 0), zorder = 1, linewidth = 3, color = 'blue', label = 'FaceGen')
            sem = np.std(edataGen, axis = 0)/np.sqrt(edataGen.shape[0])
            ax.fill_between(np.arange(bl_st, edataGen.shape[1]+bl_st), edataGen.mean(axis = 0)+sem, edataGen.mean(axis=0)-sem, alpha = 0.5, zorder = 0, edgecolor = 'None', facecolor = 'slateblue', label = None)

            ax.plot(np.arange(bl_st, edataEmo.shape[1]+bl_st), np.zeros(edataEmo.shape[1]), color = 'k', linewidth = 3, label = None) #xaxis
            ax.plot(np.zeros(tmp.size), tmp, color = 'k', linewidth = 3, label = None) #yaxis

            ax.set_ylabel('% change HG')
            ax.set_xlabel('time (ms)')
            ax.autoscale(tight=True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            legend1 = ax.legend(loc = 1)

            if start_idx.size>0:
                for i, s in enumerate(start_idx):
                    tmp = np.arange(s, end_idx[i])
                    start = s
                    finish = end_idx[i]
                    ax.plot(tmp, np.zeros(tmp.size), color = 'r', linewidth = 3.5, label = (start, finish))
                    ax.legend()

            
            ax.set_title(' '.join(['Face Emo vs Gen', subj, ':', elecname, 'chunksize', str(chunksize), 'smoothing', str(black_chunksize),'thresh', str(thresh)]))
            plt.savefig(os.path.join(SJdir, 'Anat', 'ShadePlots_Faces','images', ''.join([subj, '_', elecname])))
            plt.close()
        

def plot_shadeplot_resp(subj, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta'):
        """
        plot shade plots with activity window in red (from ShadePlots_allelecs_plot.py)
        """
        files = glob.glob(os.path.join(SJdir, 'Anat', 'ShadePlots_Faces', 'data', ''.join([subj, '*_resp.p']))) #electrodes

        for f in files:
            print f
            elecname = f.split('_')[-2]

            with open(f, 'r') as x:
                data_dict = pickle.load(x)
                x.close()

            #map dictionary to variables
            edataEmo, edataGen, srate, start_idx, end_idx, chunksize, black_chunksize, thresh = [data_dict.get(k) for k in ['edataEmo','edataGen', 'srate', 'start_idx', 'end_idx', 'chunksize', 'black_chunksize', 'thresh']]

            #plot
            f, ax = plt.subplots(figsize = (10,10))

            if edataEmo.shape[0] == 0: #if not active
                edataEmo = np.zeros_like(edataGen)
            elif edataGen.shape[0] == 0:
                edataGen = np.zeros_like(edataEmo)

            scale_min = min(edataGen.mean(axis = 0).min(), edataEmo.mean(axis = 0).min()) - 10
            scale_max = max(edataGen.mean(axis = 0).max(), edataEmo.mean(axis = 0).max()) + 10
            tmp = (np.arange(scale_min, scale_max))
            
            st_tp = -500/1000*srate
            en_tp = 500/1000*srate+1

            ax.plot(np.arange(st_tp, en_tp), edataEmo.mean(axis = 0), zorder = 1, linewidth = 3, color = 'green', label = 'FaceEmo')
            sem = np.std(edataEmo, axis = 0)/np.sqrt(edataEmo.shape[0])
            ax.fill_between(np.arange(st_tp, en_tp), edataEmo.mean(axis = 0)+sem, edataEmo.mean(axis=0)-sem, alpha = 0.5, zorder = 0, edgecolor = 'None', facecolor = 'green', label = None)
            
            ax.plot(np.arange(st_tp, en_tp), edataGen.mean(axis = 0), zorder = 1, linewidth = 3, color = 'blue', label = 'FaceGen')
            sem = np.std(edataGen, axis = 0)/np.sqrt(edataGen.shape[0])
            ax.fill_between(np.arange(st_tp, en_tp), edataGen.mean(axis = 0)+sem, edataGen.mean(axis=0)-sem, alpha = 0.5, zorder = 0, edgecolor = 'None', facecolor = 'slateblue', label = None)

            ax.plot(np.arange(st_tp, en_tp), np.zeros(edataEmo.shape[1]), color = 'k', linewidth = 3, label = None) #xaxis
            ax.plot(np.zeros(tmp.size), tmp, color = 'k', linewidth = 3, label = None) #yaxis

            ax.set_ylabel('% change HG')
            ax.set_xlabel('time (ms)')
            ax.autoscale(tight=True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            legend1 = ax.legend(loc = 1)

            if start_idx.size>0:
                for i, s in enumerate(start_idx):
                    tmp = np.arange(s, end_idx[i])
                    start = s
                    finish = end_idx[i]
                    ax.plot(tmp, np.zeros(tmp.size), color = 'r', linewidth = 3.5, label = (start, finish))
                    ax.legend()

            
            ax.set_title(' '.join(['Face Emo vs Gen', subj, ':', elecname, 'chunksize', str(chunksize), 'smoothing', str(black_chunksize),'thresh', str(thresh)]))
            plt.savefig(os.path.join(SJdir, 'Anat', 'ShadePlots_Faces','images', ''.join([subj, '_', elecname, '_resp'])))
            plt.close()


def shadeplots_faces_stats(subj, task, elecs_list, SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta', baseline = -500):

    #get data
    filename = os.path.join(SJdir, 'Subjs', subj, task, 'HG_elecMTX_percent_eleclist.mat')
    data_dict = loadmat.loadmat(filename)
    srate, elecs, data, RTs, onsets_stim, onsets_resp, data_resp = [data_dict.get(k) for k in ['srate','elecs','data_percent', 'RTs', 'onsets_stim', 'onsets_resp', 'data_percent_resp']]

    bl_st = baseline/1000*srate

    filename = os.path.join(SJdir, 'Anat', 'ShadePlots_Faces', '_'.join([subj, task, 'maxes']) +'.csv')
    peaks, lats, peaks_resp, lats_resp, peaks_maxRT, lats_maxRT, peaks_mean, lats_mean, peaks_mean_resp, lats_mean_resp = [ dict() for x in range(10)]

    for i, e in enumerate(elecs_list):
        edata = data[i, :, :].squeeze()
        edata_resp = data_resp[i,:,:].squeeze()

        #get maxes from stim onset to resp + 300ms
        p, l = [list() for x in range(2)]
        for m in range(edata.shape[0]): #per trial
            
            p.append(edata[m,abs(bl_st) : abs(bl_st) + RTs[m] + (300/1000*srate)].max())
            l.append(edata[m,abs(bl_st) : abs(bl_st) + RTs[m] + (300/1000*srate)].argmax())
        
        peaks[e] = p
        lats[e] = l
        
        peaks_resp[e] = edata_resp.max(axis = 1)
        lats_resp[e] = edata_resp.argmax(axis = 1)
        

        #get maxes in a single window (stim onset to max RT + 500)
        peaks_maxRT[e] = edata[:, abs(bl_st) : abs(bl_st) + RTs.max() + (500/1000*srate)].max(axis = 1)
        lats_maxRT[e] = edata[:, abs(bl_st) : abs(bl_st) + RTs.max() + (500/1000*srate)].argmax(axis = 1)

        #get maxes and latencies on the mean trace
        peaks_mean[e] = edata[:, abs(bl_st) : abs(bl_st) + RTs.max() + (500/1000*srate)].mean(axis = 0).max()
        lats_mean[e] = edata[:, abs(bl_st) : abs(bl_st) + RTs.max() + (500/1000*srate)].mean(axis = 0).argmax()

        peaks_mean_resp[e] = edata_resp.mean(axis = 0).max()
        lats_mean_resp[e] = edata_resp.mean(axis = 0).argmax()

    #save stats (single trials)
    filename = os.path.join(SJdir, 'Anat', 'ShadePlots_Faces', 'SingleTrials', 'data', 'RT_300ms_pertrial' ''.join([subj, '_', task, '.p']))
    data_dict = {'peaks':peaks, 'lats':lats, 'peaks_resp' : peaks_resp, 'lats_resp' : lats_resp, 'peaks_maxRT' : peaks_maxRT, 'lats_maxRT' : lats_maxRT, 'peaks_mean' : peaks_mean, 'lats_mean' : lats_mean, 'lats_mean_resp' : lats_mean_resp, 'peaks_mean_resp' : peaks_mean_resp}

    with open(filename, 'w') as f:
        pickle.dump(data_dict, f)
        f.close()
    return data_dict
