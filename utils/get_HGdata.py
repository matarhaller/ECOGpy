from __future__ import division
import scipy.io as spio
import numpy as np
import os
from utils.loadmat import loadmat

def get_HGdata(subj, task, var_list, type = 'zscore', base = '/home/knight/matar/MATLAB/DATA/Avgusta/'):
    """
    loads a matrix of zscored HG data plus variables
    input:
        subj = string with subj name (ex: 'GP15')
        task = string with task name (ex: 'EmoGen')
        base = (optional) base directory where Subjs folder with data is
        var_list = list of variables names. possible values are:
                onsets_stim, onsets_resp = onset time per trial
                data = HG data matrix - elecs x trials x time
                srate = sampling rate
                active_elecs = electrode indices
    output:
        list of variable values from var_list
    """
    if type == 'zscore':
        filename = os.path.join(base, 'Subjs', subj, task, 'HG_elecMTX_zscore.mat') #matrix of elecs x trials x time plus params
    elif type == 'percent':
        filename = os.path.join(base, 'Subjs', subj, task, 'HG_elecMTX_percent.mat') #matrix of elecs x trials x time plus params
    else:
        raise ValueError('data type not recognized. must be zscore or percent')
        
    data_dict = loadmat(filename) #load the data dictionary
    variables = [data_dict[k] for k in var_list]
    return variables