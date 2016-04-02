from __future__ import division
import cPickle as pickle
import pandas as pd
import os
import scipy.io as spio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import brewer2mpl
import scipy.stats as stats
import matplotlib

def plot_cluster_brain_duration(subj, task, reconpath, stim_or_resp = 'stim', xycoords = 'xycoords.p', datadir = '/home/knight/matar/MATLAB/DATA/Avgusta/', groupidx = '/home/knight/matar/MATLAB/DATA/Avgusta/PCA/duration_dict/groupidx_activeclusters_hclust_withduration_thresh15_maxRTlocked_withcriteria.csv'):
    
    """
    Plot mean traces of cluster with color coded brain. Only plots active clusters. outlines the electrodes with Rval>01 AND pval<0.05
    Taken from plot_cluster_brain_duration.ipynb
    Create custom colormaps so colors will match withing same subj, different task, diff num of clusters
    """

    #color stuff
    colors = ['#1f78b4', '#33a02c','#e31a1c','#ff7f00', '#6a3d9a','gold','darkturquoise', '#cf00cf', 'saddlebrown','#b2df8a']
    custom_cmap = matplotlib.colors.ListedColormap(colors, name = 'custom_cmap')
    cmap = matplotlib.colors.ListedColormap(colors)

    filename = os.path.join(datadir, 'Subjs', subj, xycoords)
    with open(filename, 'r') as f:
        xycoords = pickle.load(f)
        f.close()

    #format as dataframe (for use in chris's plotting function)
    xycoords = pd.DataFrame(np.array(xycoords.values()), columns=['x_2d', 'y_2d'], index=np.array(xycoords.keys())+1)

    #get subject/task duration R and pvalues - as weights (from groupidx_activeclusters_duration); format as dataframe
    df = pd.DataFrame.from_csv(groupidx).reset_index()
    subj_task = df[(df.subj.isin([subj])) & (df.task.isin([task]))]
    subj_task = subj_task.sort('active_elecs')


    if stim_or_resp == 'stim':

        weights = subj_task[['group','active_elecs','all_criteria_passed']].loc[(subj_task.active_cluster_stim.isin([True]))].set_index('active_elecs')
        dur_clust = weights.group.loc[weights.all_criteria_passed]
        dur_clust = np.unique(dur_clust.values)

        sems = dict()
        clusters = dict()
        for c in subj_task[['active_cluster_stim','active_cluster_resp', 'active_elecs','group']].groupby(subj_task['group']): #active elecs per active cluster
            c = c[1]
            if not(all(c.active_cluster_stim)):
                continue
            filename = os.path.join(datadir, 'PCA','SingleTrials_hclust', '_'.join([subj, task, ''.join(['c', str(int(c.group.iloc[0]))])]))
            data = spio.loadmat(filename)
            cdata = data['cdata']
            sems[str(int(c.group.iloc[0]))] = stats.sem(cdata, axis = 0)
            clusters[str(int(c.group.iloc[0]))] = cdata.mean(axis = 0)
        clusters = pd.DataFrame(clusters)
        clusters_sem = pd.DataFrame(sems)

        #sort column indices (important if have >=10 clusters)
        cols = map(str, np.sort(map(int, clusters.columns)))
        clusters = clusters[cols]
        clusters_sem = clusters_sem[cols]

        #append c
        cols2 = [''.join(['c', x]) for x in cols]
        clusters.columns = cols2
        clusters_sem.columns = cols2

    elif stim_or_resp == 'resp':
 
        weights = subj_task[['group','active_elecs','all_criteria_passed']].loc[(subj_task.active_cluster_resp.isin([True]))].set_index('active_elecs')
        dur_clust = weights.group.loc[weights.all_criteria_passed]
        dur_clust = np.unique(dur_clust.values)

        filename = os.path.join(datadir, 'PCA','ShadePlots_hclust_thresh15', '_'.join([subj, task, 'cdata_resp.mat']))
        data = spio.loadmat(filename, struct_as_record = True)
        params = data['Params'].flatten()
        srate = data['srate']
        data = data['cdata_resp_all']
        st_tp = params['st'][0]/1000*srate
        en_tp = params['en'][0]/1000*srate

        c = list()
        [c.append(str(x[0])) for x in data[:,0]]
        c = [x.split('_')[-1].split('.')[0] for x in c]
        cdict = dict(zip(c, [x.mean(axis = 0) for x in data[:,1]]))
        clusters = pd.DataFrame(cdict)
        cdict_sem = dict(zip(c, [stats.sem(x, axis = 0) for x in data[:,1]]))
        clusters_sem = pd.DataFrame(cdict_sem)

        #sort column indices (important if have >=10 clusters)
        cols = [x.split('c') for x in clusters.columns]
        cols = [x[-1] for x in cols]
        cols = map(str, np.sort(map(int, cols)))

        #append c
        cols = [''.join(['c', x]) for x in cols]

        #reorder columsn
        clusters = clusters[cols]
        clusters_sem = clusters_sem[cols]
    else:
        raise AssertionError, [stim_or_resp + ' is not a valid argument. you need to specify stim or resp']


    #figure properties
    n = int(np.ceil(np.sqrt(len(clusters.columns)))) #number of rows/cols for single trials

    f, ax1 = plt.subplots(figsize = (55,50))
    gs = gridspec.GridSpec(2+n, 50)

    ax1 = plt.subplot(gs[0, :25])
    ax3 = plt.subplot(gs[:n+1, 26:])

    singletrial_pngs = map(lambda x: ''.join(['_'.join([subj, task]),'_',x,'.png']), clusters.columns)

    """
    #plots significant stim locked traces
    if stim_or_resp == 'stim':
        cplot = clusters.plot(ax = ax1, colormap = custom_cmap, grid = 'off', linewidth = 3)
    else:
        cplot = clusters.plot(np.arange(st_tp, en_tp+1),ax = ax1, colormap = custom_cmap, grid = 'off', linewidth = 3) #resp locked
    
    #pull line colors for shading
    colors = list()
    clines = cplot.get_children() #all lines in plot
    [colors.append(x.get_color()) for x in clines if hasattr(x,'get_color')]
    colors = filter(lambda c: c != "k", colors) #remove black
    cmap = matplotlib.colors.ListedColormap(colors)

    """

    #single trials
    for i, fname in enumerate(singletrial_pngs):
        arr = plt.imread(os.path.join(datadir, 'PCA','SingleTrials_hclust', fname))
        [x,y] = np.unravel_index(i,(n,n))
        span = int(np.ceil(25/n))
        ax4 = f.add_subplot(gs[1+x, y*span:(y+1)*span])
        ax4.imshow(arr, aspect = 'equal')        

        plt.setp(ax4.spines.values(), color=colors[i], linewidth = 3.5)

        clust = int(fname.split('_')[-1].split('.')[0][1:])
        if (clust in dur_clust):
            plt.setp(ax4, title = ''.join(['c', str(clust), ' duration']))
        else:
            plt.setp(ax4, title = ''.join(['c', str(clust)]))

        ax4.xaxis.set_ticklabels([])#hide labels
        ax4.xaxis.set_ticks([])#hide gridlines
        ax4.yaxis.set_ticklabels([])
        ax4.yaxis.set_ticks([])

    """ 
    #SEMS
    for q, i in enumerate(clusters.columns):
        x = clusters[i]
        sem = clusters_sem[i]
        if stim_or_resp == 'stim':
            ax1.fill_between(np.arange(len(x)), x - sem, x + sem, alpha = 0.7, color = colors[q])
        else:
            ax1.fill_between(np.arange(st_tp, en_tp+1), x - sem, x + sem, alpha = 0.7, color = colors[q])
    """
        
    #create list of colors for scatter
    c = list()
    u = np.unique(weights.group)
    for i in weights.group:
        idx = np.where(u == i)
        c.append(colors[idx[0]])
    
    #plot recon
    plot_xy_map(weights[['group']], locs = xycoords.loc[weights.index], ax = ax3, colors = c, szmult=400, cmap = cmap, im_path = reconpath)    

    #highlight duration
    idx = weights.all_criteria_passed
    
    x = xycoords.loc[weights.index]['x_2d'][idx]
    y = xycoords.loc[weights.index]['y_2d'][idx]
    ax3.scatter(x, y, facecolors = 'None', edgecolor = 'black', s = 500, linewidth = 4.5)
    #ax3.scatter(x, y, facecolors = 'None', edgecolor = '#FFFF99', s = 500, linewidth = 2.5)
    ax3.scatter(x, y, facecolors = 'None', edgecolor = 'black', s = 500, linewidth = 2.5)
    ax3.set_title(' '.join([subj, task, stim_or_resp.upper()]), fontsize = 36)

    ax1.autoscale(tight=True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.legend(loc = 'upper right')

    return f


def plot_xy_map(weights, locs=None, im_path=None, ecog=None,  szmult=2000, colors = None, cmap=plt.cm.Reds, pltkey=None, ax=None, cbar=False, vaxes = False, **kwargs):
    '''
    (plotting func from chris - with colormap and scaling edits)

    This plots a 2-D map of electrodes along with some weight that we specify

    Weights must be a dataframe with a single column, and rows corresponding to electrodes

    Inputs are all dataframes with indices = electrode numbers.
    Locs has two columns, 'x', and 'y'
    Weights is a dataframe with features you wish to plot.

    Note that this uses dataframes so that we make sure indexing is correct with the electrodes.
    '''
    # Make sure we've specified a thing to plot
    #assert len(weights.squeeze().shape) == 1, 'Weights have more than one column'
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
    #ax.scatter(impts.x, impts.y, c=impts[pltkey], s=szmult*np.abs(impts[pltkey]), cmap=cmap, vmin=impts[pltkey].min(), vmax=impts[pltkey].max())
    #ax.scatter(impts.x, impts.y, c=impts[pltkey], s=szmult, cmap=cmap, vmin=1, vmax=cmax)
    if vaxes == False:
        vaxes = (impts[pltkey].min(), impts[pltkey].max()) #tuple of vmin, vmax

    ax.scatter(impts.x, impts.y, c=colors, s=szmult, cmap=cmap, vmin = vaxes[0], vmax = vaxes[1]) 

    ax.set_axis_off()
    if cbar==True: ax.figure.colorbar(ax.collections[0], shrink=.75)
    return ax
