import cPickle as pickle
import tables
import pandas as pd
import os
import scipy.io as spio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def plot_cluster_brain(subj, task, reconpath, datadir = '/Users/matar/Documents/MATLAB/DATA/Avgusta/', groupidx = '/Users/matar/Dropbox/PCA_elecs/groupidx.csv'):
    """
    Plot mean traces of cluster with color coded brain.
    Taken from plot_cluster_brain.ipynb
    """
    #open xycoords dictionary
    filename = os.path.join(datadir, 'Subjs', subj, 'xycoords.p')
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
    srate = data['srate']
    HGdata = data['data_percent']

    #calculate mean trace per cluster; format as dataframe
    clusters = list()
    for x, c in enumerate(subj_task['active elec'].groupby(subj_task['group'])): #active elecs per cluster
        c = c[1].values
        cidx = np.in1d(subj_task['active elec'], c)
        cdata = HGdata[cidx,:,:]
        cdata = np.vstack([item for item in cdata]) #trials x time
        clusters.append(cdata.mean(axis = 0))
    clusters = pd.DataFrame(clusters).transpose()

    #plot
    gs = gridspec.GridSpec(1, 2,height_ratios=[1,2])

    f, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize = (25,25))
    plt.subplots_adjust(wspace=.001)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    plot_xy_map(weights, locs = xycoords.loc[subj_task['active elec']], ax = ax2, szmult=100, cmap = plt.cm.Spectral, im_path = reconpath)
    clusters.plot(ax = ax1, colormap = 'Spectral', grid = 'off', linewidth = 3)


#plotting func from chris
def plot_xy_map(weights, locs=None, im_path=None, ecog=None, szmult=2000, cmap=plt.cm.Reds, pltkey=None, ax=None, cbar=False, **kwargs):
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

    ax.imshow(im)
#    ax.scatter(impts.x, impts.y, c=impts[pltkey], s=szmult*np.abs(impts[pltkey]), cmap=cmap, vmin=impts[pltkey].min(), vmax=impts[pltkey].max())
    ax.scatter(impts.x, impts.y, c=impts[pltkey], s=szmult, cmap=cmap, vmin=impts[pltkey].min(), vmax=impts[pltkey].max())
    ax.set_axis_off()
    if cbar==True: ax.figure.colorbar(ax.collections[0], shrink=.75)
    return ax
