from __future__ import division
import loadmat
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_singletrials_cluster(subj, task, cluster):

    SJdir = '/home/knight/matar/MATLAB/DATA/Avgusta/'
    datadir = os.path.join(SJdir, 'Subjs')
    
    filename = os.path.join(SJdir, 'PCA', 'SingleTrials_hclust', ''.join([subj, '_', task, '_c', str(cluster)]))

    data_dict = loadmat.loadmat(filename)
    cdata, srate, RTs_all = [data_dict.get(k) for k in ['cdata', 'srate', 'RTs_all']]

    bl_st = np.round(-500/1000*srate) #doesn't work for my data
    RTs_all = RTs_all

    #cut data
    off = 4000/1000*srate
    cdata = cdata[:, 0:off]
    
    idx = RTs_all<cdata.shape[1]+bl_st
    RTs_all = RTs_all[idx]

    cdata = cdata[idx,:]

    #plot
    f,ax = plt.subplots()
    ax.set_title(' '.join([subj, task, str(cluster)]))
    ax.autoscale(enable = True, tight = True)
    cax = ax.pcolormesh(np.arange(bl_st, cdata.shape[1]+bl_st), np.arange(0, len(RTs_all)), cdata, zorder = 0)
    cbar = f.colorbar(cax, ticks = [-150, 0 , 150], orientation = 'vertical')
    cax.set_clim(vmin=-150,vmax=150)

    for j in np.arange(len(RTs_all)):
        ax.plot((RTs_all[j], RTs_all[j]), (j-0.5, j+0.5), 'k', linewidth = 3,zorder = 1)

    filename = os.path.join(SJdir,'PCA','figs_for_Bob', ''.join([subj, '_', task, '_c', str(cluster), '300.png']))
    plt.savefig(filename, format = 'png', dpi = 300)
   
