from __future__ import division
import scipy.io as spio
import numpy as np
import os

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


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    see: http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
