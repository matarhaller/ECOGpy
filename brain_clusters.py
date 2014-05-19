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

def plot_cluster_brain(subj, task, reconpath, xycoords = 'xycoords.p', datadir = '/home/knight/matar/MATLAB/DATA/Avgusta/', groupidx = '/home/knight/matar/MATLAB/DATA/Avgusta/PCA/plots/groupidx_activeclusters.csv'):
    """
    Plot mean traces of cluster with color coded brain. Only plots active clusters
    Taken from plot_cluster_brain.ipynb
    Create custom colormaps so colors will match withing same subj, different task, diff num of clusters
    """
    #open xycoords dictionary
    filename = os.path.join(datadir, 'Subjs', subj, xycoords)
    with open(filename, 'r') as f:
        xycoords = pickle.load(f)
        f.close()

    #format as dataframe (for use in chris's plotting function)
    xycoords = pd.DataFrame(np.array(xycoords.values()), columns=['x_2d', 'y_2d'], index=np.array(xycoords.keys())+1)

    #get subject/task cluster designations - as weights (from groupidx); format as dataframe
    df = pd.DataFrame.from_csv(groupidx)
    subj_task = df.loc[df['subj_task'] == '_'.join([subj, task])]

    weights = subj_task[['group', 'active_elecs']].loc[subj_task.active_cluster==True].set_index('active_elecs')
    #1 column of cluster designations (index is active_elecs)

    filename = os.path.join(datadir, 'Subjs', subj, task, 'HG_elecMTX_percent.mat')
    data = spio.loadmat(filename, struct_as_record = True)
    HGdata = data['data_percent']
    srate = data['srate']

    #calculate mean trace per cluster; format as dataframe
    clusters = dict()
    sems = list()
    for c in subj_task[['active_cluster', 'active_elecs','group']].groupby(subj_task['group']): #active elecs per active cluster
        c = c[1]
        if not(all(c.active_cluster)):
            continue
        cidx = np.in1d(subj_task.active_elecs, c.active_elecs)
        cdata = HGdata[cidx,:,:]
        cdata = np.vstack([item for item in cdata]) #same thing as cdata.reshape([-1,2783]); gives you trials x time
        sems.append(stats.sem(cdata, axis = 0))
        clusters[''.join(['c', str(c.group.iloc[0])])] = cdata.mean(axis = 0)
    clusters = pd.DataFrame(clusters)

    #resp locked
    filename = os.path.join(datadir, 'PCA', 'ShadePlots_thresh10', '_'.join([subj, task, 'cdata_resp.mat']))
    data = spio.loadmat(filename, struct_as_record = True)
    params = data['Params'].flatten()
    data = data['cdata_resp_all']
    st_tp = params['st'][0]/1000*srate
    en_tp = params['en'][0]/1000*srate

    c = list()
    [c.append(str(x[0])) for x in data[:,0]]
    c = [x.split('_')[-1].split('.')[0] for x in c]
    cdict = dict(zip(c, [x.mean(axis = 0) for x in data[:,1]]))
    clusters_resp= pd.DataFrame(cdict)

    #plot  - colors
    ncolors = len(clusters.keys())
    if ncolors<3:
        ncolors = 3
    bmap = brewer2mpl.get_map('Spectral', 'diverging', ncolors)  #max num of clusters
    cmap = bmap.get_mpl_colormap()
    colors = bmap.mpl_colors

    #plot - set up grid
    singletrial_pngs = map(lambda x: ''.join(['_'.join([subj, task]),'_',x,'.png']), clusters.columns)
    n = int(np.ceil(np.sqrt(len(clusters.columns)))) #number of rows/cols for single trials

    gs = gridspec.GridSpec(2+n, 50)
    f, ax1 = plt.subplots(figsize = (35,20))

    ax1 = plt.subplot(gs[0, :25])
    ax2 = plt.subplot(gs[1, :25])
    ax3 = plt.subplot(gs[:, 26:])

    #plot significant stimlocked traces
    #xspan = np.arange(-500, clusters.shape[0]-500)
    cplot = clusters.plot(ax = ax1, colormap = cmap, grid = 'off', linewidth = 3)

    #pull line colors for shading (so sems and lines have same)
    colors = list()
    clines = cplot.get_children() #all lines in plot
    [colors.append(x.get_color()) for x in clines if hasattr(x,'get_color')]
    colors = filter(lambda c: c != "k", colors) #remove black
    cmap = mpl.colors.ListedColormap(colors) #create colormap from list

    #SEMs
    for i, j in enumerate(clusters.keys()):
        x = clusters.values[:,i];
        sem = sems[i]
        ax1.fill_between(np.arange(len(x)), x - sem, x + sem, alpha=0.7, color = colors[i])

    #plot significant resp locked traces
    cplot = clusters_resp.plot(np.arange(st_tp, en_tp+1),ax = ax2, colormap = plt.cm.Spectral, grid = 'off', linewidth = 3)

    #single trials
    for i, fname in enumerate(singletrial_pngs):
        arr = plt.imread(os.path.join(datadir, 'PCA','SingleTrials', fname))
        [x,y] = np.unravel_index(i,(n,n))
        span = int(np.ceil(25/n))
        ax4 = f.add_subplot(gs[2+x, y*span:(y+1)*span])
        ax4.imshow(arr, aspect = 'equal')
        plt.setp(ax4.spines.values(), color=colors[i], linewidth = 2.5)
        ax4.xaxis.set_ticklabels([])#hide labels
        ax4.xaxis.set_ticks([])#hide gridlines
        ax4.yaxis.set_ticklabels([])
        ax4.yaxis.set_ticks([])

    #create list of colors for scatter
    c = list()
    u = np.unique(weights.group)
    for i in weights.group:
        idx = np.where(u == i)
        c.append(colors[idx[0]])

    #recons
    plot_xy_map(weights, locs = xycoords.loc[weights.index], ax = ax3, szmult=250, colors = c, cmap = cmap, im_path = reconpath)
    plt.title(' '.join([subj, task]))

    ax1.autoscale(tight=True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    ax2.autoscale(tight=True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()

    #change trace colors
    #for line, klass in zip(ax1.lines, clusters):
    #   line.set_color(colors[klass])
    return f, (ax1, ax2)


def plot_cluster_brain_withinactive(subj, task, reconpath, xycoords = 'xycoords.p', datadir = '/Users/matar/Documents/MATLAB/DATA/Avgusta/', groupidx = '/Users/matar/Dropbox/PCA_elecs/groupidx.csv'):
    """
    Plot mean traces of cluster with color coded brain. Plots both active and inactive
    Taken from plot_cluster_brain.ipynb
    Create custom colormaps so colors will match withing same subj, different task, diff num of clusters
    """
    #open xycoords dictionary
    filename = os.path.join(datadir, 'Subjs', subj, xycoords)
    with open(filename, 'r') as f:
        xycoords = pickle.load(f)
        f.close()

    #format as dataframe (for use in chris's plotting function)
    xycoords = pd.DataFrame(np.array(xycoords.values()), columns=['x_2d', 'y_2d'], index=np.array(xycoords.keys())+1)

    #get subject/task cluster designations - as weights (from groupidx); format as dataframe
    df = pd.DataFrame.from_csv(groupidx,index_col = [0,1])
    subj_task = df.loc[subj].loc[task]

    weights = subj_task.set_index('active elec')

    filename = os.path.join(datadir, 'Subjs', subj, task, 'HG_elecMTX_percent.mat')
    data = spio.loadmat(filename, struct_as_record = True)
    HGdata = data['data_percent']

    #calculate mean trace per cluster; format as dataframe
    clusters = list()
    sems = list()
    for x, c in enumerate(subj_task['active elec'].groupby(subj_task['group'])): #active elecs per cluster
        c = c[1].values
        cidx = np.in1d(subj_task['active elec'], c)
        cdata = HGdata[cidx,:,:]
        cdata = np.vstack([item for item in cdata]) #trials x time
        sems.append(stats.sem(cdata, axis = 0))
        clusters.append(cdata.mean(axis = 0))
    clusters = pd.DataFrame(clusters).transpose()

    #plot
    bmap = brewer2mpl.get_map('Spectral', 'diverging',len(clusters.keys()))  #max num of clusters
    cmap = bmap.get_mpl_colormap()
    #colors = bmap.mpl_colors
    #cmap  = plt.cm.Spectral

    #gs = gridspec.GridSpec(1, 2,height_ratios=[1,2])
    #f, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize = (25,25))
    gs = gridspec.GridSpec(2, 1, height_ratios = [1,2])
    f, (ax1, ax2) = plt.subplots(nrows=2,ncols=1, figsize = (25,25))
    plt.subplots_adjust(wspace=.001)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    plot_xy_map(weights, locs = xycoords.loc[subj_task['active elec']], ax = ax2, szmult=250, cmap = cmap, im_path = reconpath)
    cplot = clusters.plot(ax = ax1, colormap = cmap, grid = 'off', linewidth = 3)

    #pull line colors for shading (so sems and lines have same)
    colors = list()
    clines = cplot.get_children() #all lines in plot
    [colors.append(c.get_color()) for c in clines if hasattr(c,'get_color')]


    #SEMs
    for j in clusters.keys():
        x = clusters.values[:,j];
        sem = sems[j]
        #ax1.plot(x, linewidth = 3, color = colors[j])
        ax1.fill_between(np.arange(len(x)), x - sem, x + sem, alpha=0.7, color = colors[j])

    ax1.autoscale(tight=True)
    plt.title(' '.join([subj, task]))

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    #change trace colors
    #for line, klass in zip(ax1.lines, clusters):
    #   line.set_color(colors[klass])

    return f, (ax1, ax2)

#plotting func from chris - with colormap and scaling edits
def plot_xy_map(weights, locs=None, im_path=None, ecog=None,  szmult=2000, colors = None, cmap=plt.cm.Reds, pltkey=None, ax=None, cbar=False, **kwargs):
    '''This plots a 2-D map of electrodes along with some weight that we specify

    Weights must be a dataframe with a single column, and rows corresponding to electrodes

    Inputs are all dataframes with indices = electrode numbers.
    Locs has two columns, 'x', and 'y'
    Weights is a dataframe with features you wish to plot.

    Note that this uses dataframes so that we make sure indexing is correct with the electrodes.
    '''
    # Make sure we've specified a thing to plot
    assert len(weights.squeeze().shape) == 1, 'Weights have more than one column'
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
    ax.scatter(impts.x, impts.y, c=colors, s=szmult, cmap=cmap, vmin=impts[pltkey].min(), vmax=impts[pltkey].max())
    #ax.scatter(impts.x, impts.y, c=impts[pltkey], s=szmult, cmap=cmap, vmin=1, vmax=cmax)

    ax.set_axis_off()
    if cbar==True: ax.figure.colorbar(ax.collections[0], shrink=.75)
    return ax
