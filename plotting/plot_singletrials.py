from __future__ import division
from utils.get_HGdata import get_HGdata
from utils.loadmat import loadmat
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

def plot_singletrials(subj, task, elec_list, fig = None, ax = None, smooth = True, cm = plt.get_cmap('RdGy_r'), cbar = True, **kwargs):
    """
    plot single trial plot
    input:
        subj, task as string
        elec - number 
        ax = axis object where to plot (optional, default None, plot to gca)
        smooth - whether or not to do 1d gaussian smoothing (optional, default True)
        cm = colormat to use (default RdGy_r)
    """
    if fig is None:
        fig = plt.gcf()
        
    if ax is None: #if didn't specify where to plot it
        ax = plt.gca()
    
        
    #get data
    var_list = ['srate','data_zscore','active_elecs', 'RTs'] #vars of interest
    srate, alldata, active_elecs, RTs = get_HGdata(subj, task, var_list)

    #get data for elecs of choice
    # data = alldata[active_elecs == elec,:,:].squeeze()
    idx = np.in1d(active_elecs, elec_list) #indices of these elecs in the HG dataframe
    data = np.vstack(alldata[idx,:,:]) #trials x time for all elecs in cluster
    RTs = np.tile(RTs, len(elec_list))
    
    #sort trials by reaction time 
    idx = np.argsort(RTs)
    RTs = RTs[idx]
    data =  data[idx,:]

    #1d gaussian smoothing
    if smooth:
        img = ndimage.gaussian_filter1d(data, axis = -1, sigma = 10, order=0) #gaussian smooth trials
    else:
        img = data
    
    cax = ax.imshow(img, aspect = 'auto', origin = 'lower', extent = (-500/1000*srate, img.shape[1]-500/1000*srate, 0, len(RTs)), cmap = cm)
    cax.set_clim(vmin=-3,vmax=3)

    #reaction times
    for j in np.arange(len(RTs)):
        ax.plot((RTs[j], RTs[j]), (j-0.5, j+0.5), 'black', linewidth = 3,zorder = 1)

    #colorbar
    if cbar:
        cbar = fig.colorbar(cax, ticks = [-3, 0 , 3], orientation = 'vertical', label = 'zscore')
        cbar.set_label(label='zscore',size=14,weight='bold')

    #format
    ax.autoscale(enable = True, tight = True)
    ax.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(labelsize = 14, width = 3)
    ax.yaxis.set_tick_params(labelsize = 14, width = 0)
    ax.axvline(x = 0, lw = 3, color = 'black')
    ax.set_xlabel('Time (ms)', fontsize = 20, weight = 'bold')
    ax.set_ylabel('Trials', fontsize = 20, weight = 'bold')
    ax.set_title('{0} {1} e{2}'.format(subj, task, elec_list), fontsize = 20, weight = 'bold')
        
    return ax