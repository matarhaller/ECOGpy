from __future__ import division
import pandas as pd
import os

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import brewer2mpl
import scipy.stats as stats
import cPickle as pickle

from utils.get_HGdata import get_HGdata
from plotting.plot_singletrials import plot_singletrials

def plot_clusters(subj, task, recon_path, xycoord_path, groupidx):
    
    """
    Plot mean traces of cluster and single trials with color coded brain. Plots all clusters.
    Create custom colormaps so colors will match withing same subj, different task, diff num of clusters
    Drops clusters of 1 electrode
    
    input: 
    subj, task = string
    recon_path = filepath to brain recon png file
    xycoord_path = filepath to pickle of xy coordinates per electrode
    groupidx = dataframe elecs as index and cluster designations per electrode
    stim_or_resp = whether to plot stim or response locked traces (optional, default = stim)
    
    """
    #get data
    var_list = ['srate','data_percent','active_elecs', 'RTs'] #vars of interest
    srate, alldata, active_elecs, RTs = get_HGdata(subj, task, var_list, type = 'percent')

    #color stuff
    colors = ['#1f78b4', '#33a02c','#e31a1c','#ff7f00', '#6a3d9a','gold','darkturquoise', '#cf00cf', 'saddlebrown','#b2df8a']
    custom_cmap = matplotlib.colors.ListedColormap(colors, name = 'custom_cmap')
    cmap = matplotlib.colors.ListedColormap(colors)

    #load xy coordinates for electrodes, format as dataframe
    with open(xycoord_path, 'r') as f:
        xycoords = pickle.load(f)
        f.close()
    xycoords = pd.DataFrame(np.array(xycoords.values()), columns=['x_2d', 'y_2d'], index=np.array(xycoords.keys())+1)
    
    #format data - mean, sems 
    sems, clusters, data_dict = [dict() for i in range(3)]
    for c in groupidx.groupby('cluster'): #active elecs per active cluster
        df = c[1]
        idx = np.in1d(active_elecs, df.index) #indices of these elecs in the HG dataframe

        #calculate means and stds across elecs for the cluster
        cdata = np.vstack(alldata[idx,:,:])
        #clusters[c[0]] = cdata.mean(axis = 1).mean(axis = 0) #mean across all elecs in cluster
        #sems[c[0]] = stats.sem(cdata, axis = 1).mean(axis = 0) #sem between elecs in cluster
        clusters[c[0]] = cdata.mean(axis = 0)
        sems[c[0]] = stats.sem(cdata, axis = 0)
        
    clusters = pd.DataFrame(clusters)
    sems = pd.DataFrame(sems)
    
    #figure properties
    f, ax1 = plt.subplots(figsize = (55,50))
    n = int(np.ceil(np.sqrt(len(clusters.columns)))) #number of rows/cols for single trials
    gs = gridspec.GridSpec(2+n, 50)
    gs.update(wspace=20)
    ax1 = plt.subplot(gs[0, :25]) #traces
    ax3 = plt.subplot(gs[:n+1, 26:]) #brain
  
    #SINGLE TRIALS
    for c in clusters.columns:
        #where to place the single trial figure
        [x,y] = np.unravel_index(c,(n,n)) #x, y coordinate of single trial (ie (2,1))
        span = int(np.ceil(25/n))
        ax4 = f.add_subplot(gs[1+x, y*span:(y+1)*span])
        
        #plot single trial figure for cluster
        elec_list = list(groupidx[groupidx.cluster == c].index)
        ax = plot_singletrials(subj, task, elec_list, fig = f, ax = ax4, cbar = False)
        ax.set_title('c{0}'.format(c), fontsize = 36)
        
        #outline in cluster color (to match brain electrodes)
        plt.setp(ax.spines.values(), color=colors[c], linewidth = 15)
     
    #TRACES
    for c in clusters.columns:
        x = clusters[c]
        sem = sems[c]
        ax1.fill_between(np.arange(len(x)), x - sem, x + sem, alpha = 0.5, color = colors[c])
    
    #BRAIN
    #create list of colors for scatter on brain
    c = [colors[i] for i in groupidx.cluster]
    
    #plot brain with color-coded electrodes
    plot_xy_map(groupidx[['cluster']], locs = xycoords.loc[groupidx.index], ax = ax3, colors = c, szmult=1000, cmap = cmap, im_path = recon_path)    

    #format figure
    ax3.set_title(' '.join([subj, task]), fontsize = 36)
    ax1.autoscale(tight=True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_ylabel('% change bl', fontsize = 36)
    ax1.set_xlabel('time (ms)', fontsize = 36)
    ax1.xaxis.set_tick_params(labelsize = 36)
    ax1.yaxis.set_tick_params(labelsize = 36)



def plot_xy_map(weights, locs=None, im_path=None, ecog=None,  szmult=2000, colors = None, cmap=plt.cm.Reds, pltkey=None, ax=None, cbar=False, vaxes = False, **kwargs):
    '''
    This plots a 2-D map of electrodes along with some weight that we specify

    Weights must be a dataframe with a single column, and rows corresponding to electrodes

    Inputs are all dataframes with indices = electrode numbers.
    Locs has two columns, 'x', and 'y'
    Weights is a dataframe with features you wish to plot.

    Note that this uses dataframes so that we make sure indexing is correct with the electrodes.
    '''
    # Make sure we've specified a thing to plot
    assert weights.index.nlevels == 1, 'Index must have a single level (electrodes)'
    if locs is None and im_path is None:
        try:
            locs = ecog.colmetadata[['x_2d', 'y_2d']]
            im_path = ecog.params.im_2d
        except:
            raise AssertionError, 'you need to specify image location and xy poisitions \
            or a df with this info.'
    else:
        pass

    pltkey='impts'
    weights = pd.DataFrame(weights)
    weights.columns = [pltkey]
    locs.columns = ['x', 'y']
    impts = locs.join(weights)
    im = plt.imread(im_path)
    if ax==None:
        f, ax = plt.subplots(figsize=(15, 15))

    if colors ==None:
        colors = impts[pltkey]

    ax.imshow(im)
    if vaxes == False:
        vaxes = (impts[pltkey].min(), impts[pltkey].max()) #tuple of vmin, vmax

    ax.scatter(impts.x, impts.y, c=colors, s=szmult, cmap=cmap, vmin = vaxes[0], vmax = vaxes[1]) 

    ax.set_axis_off()
    if cbar==True: 
        ax.figure.colorbar(ax.collections[0], shrink=.75)
    return ax
