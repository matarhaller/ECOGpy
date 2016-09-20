import numpy as np
from mne.filter import band_pass_filter
from mne.utils import _time_mask
from mne.parallel import parallel_func
from mne.time_frequency import cwt_morlet
from mne.preprocessing import peak_finder
from mne.utils import ProgressBar
from mne.baseline import rescale

from connectivity import  _pre_filter_ph_am, _array_raw_to_epochs

# Supported PAC functions
_pac_funcs = ['plv', 'glm', 'mi_tort', 'mi_canolty', 'ozkurt', 'otc']
# Calculate the phase of the amplitude signal for these PAC funcs
_hi_phase_funcs = ['plv']

def _phase_amplitude_coupling_edited(data, sfreq, f_phase, f_amp, ixs,
                              pac_func='plv', ev=None, ev_grouping=None,
                              tmin=None, tmax=None,
                              baseline=None, baseline_kind='mean',
                              scale_amp_func=None, use_times=None, npad='auto',
                              return_data=False, concat_epochs=True, n_jobs=1,
                              verbose=None, to_filter = False, ev2 = None):
    """ Compute phase-amplitude coupling using pacpy.
    Parameters
    ----------
    data : array, shape ([n_epochs], n_channels, n_times)
        The data used to calculate PAC
    sfreq : float
        The sampling frequency of the data
    f_phase : array, dtype float, shape (2,)
        The frequency range to use for low-frequency phase carrier.
    f_amp : array, dtype float, shape (2,)
        The frequency range to use for high-frequency amplitude modulation.
    ixs : array-like, shape (n_pairs x 2)
        The indices for low/high frequency channels. PAC will be estimated
        between n_pairs of channels. Indices correspond to rows of `data`.
    pac_func : string, ['plv', 'glm', 'mi_canolty', 'mi_tort', 'ozkurt']
        The function for estimating PAC. Corresponds to functions in pacpy.pac
    ev : array-like, shape (n_events,) | None
        Indices for events. To be supplied if data is 2D and output should be
        split by events. In this case, tmin and tmax must be provided
    ev_grouping : array-like, shape (n_events,) | None
        Calculate PAC in each group separately, the output will then be of
        length unique(ev)
    tmin : float | None
        If ev is not provided, it is the start time to use in inst. If ev
        is provided, it is the time (in seconds) to include before each
        event index.
    tmax : float | None
        If ev is not provided, it is the stop time to use in inst. If ev
        is provided, it is the time (in seconds) to include after each
        event index.
    baseline : array, shape (2,) | None
        If ev is provided, it is the min/max time (in seconds) to include in
        the amplitude baseline. If None, no baseline is applied.
    baseline_kind : str
        What kind of baseline to use. See mne.baseline.rescale for options.
    scale_amp_func : None | function
        If not None, will be called on each amplitude signal in order to scale
        the values. Function must accept an N-D input and will operate on the
        last dimension. E.g., skl.preprocessing.scale
    use_times : array, shape (2,) | None
        If ev is provided, it is the min/max time (in seconds) to include in
        the PAC analysis. If None, the whole window (tmin to tmax) is used.
    npad : int | 'auto'
        The amount to pad each signal by before calculating phase/amplitude if
        the input signal is type Raw. If 'auto' the signal will be padded to
        the next power of 2 in length.
    return_data : bool
        If True, return the phase and amplitude data along with the PAC values.
    concat_epochs : bool
        If True, epochs will be concatenated before calculating PAC values. If
        epochs are relatively short, this is a good idea in order to improve
        stability of the PAC metric.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    to_filter : False
        If True, function expects pre-filtered data (argument is a list of (data, idx) tuples from _pre_filter_ph_am)
    ev2 : None
        If exists, then it's the random onsets to use for amplitude data
    Returns
    -------
    pac_out : array, dtype float, shape (n_pairs, [n_events])
        The computed phase-amplitude coupling between each pair of data sources
        given in ixs.
    """
    from pacpy import pac as ppac
    if pac_func not in _pac_funcs:
        raise ValueError("PAC function {0} is not supported".format(pac_func))
    func = getattr(ppac, pac_func)
    ixs = np.array(ixs, ndmin=2)
    f_phase = np.atleast_2d(f_phase)
    #print f_phase
    f_amp = np.atleast_2d(f_amp)
    #print f_amp
    #print (f_phase.shape, f_phase[0], f_amp.shape, f_amp[0])
    if to_filter:
        if data.ndim != 2:
            raise ValueError('Data must be shape (n_channels, n_times)')
        if ixs.shape[1] != 2:
            raise ValueError('Indices must have have a 2nd dimension of length 2')
        for ifreqs in [f_phase, f_amp]:
            if ifreqs.ndim > 2:
                raise ValueError('frequencies must be of shape (n_freq, 2)')
            if ifreqs.shape[1] != 2:
                raise ValueError('Phase frequencies must be of length 2')
        print('Pre-filtering data and extracting phase/amplitude...')
        hi_phase = pac_func in _hi_phase_funcs
        data_ph, data_am, ix_map_ph, ix_map_am = _pre_filter_ph_am(
            data, sfreq, ixs, f_phase, f_amp, npad=npad, hi_phase=hi_phase)
    else:
        ph, am = data
        data_ph, ix_map_ph = ph
        data_am, ix_map_am = am
    #print(data_ph[0], data_am[0]) #diverge!
    ixs_new = [(ix_map_ph[i], ix_map_am[j]) for i, j in ixs]
    # print ixs_new[0]

    if ev is not None: 
        use_times = [tmin, tmax] if use_times is None else use_times
        ev_grouping = np.ones_like(ev) if ev_grouping is None else ev_grouping
        data_ph, times, msk_ev = _array_raw_to_epochs(
            data_ph, sfreq, ev, tmin, tmax)
        ev, ev_grouping = [i[msk_ev] for i in [ev, ev_grouping]]
        if ev2 is not None:
            print 'random onsets'
            data_am, times2, msk_ev = _array_raw_to_epochs(
                data_am, sfreq, ev2, tmin, tmax) #EDIT FOR RANDOM ONSETS OF AMPLITUDE
            # In case we cut off any events
            ev2, ev_grouping = [i[msk_ev] for i in [ev2, ev_grouping]]
        else:
            data_am, times, msk_ev = _array_raw_to_epochs(
                data_am, sfreq, ev, tmin, tmax)
            # In case we cut off any events
            ev, ev_grouping = [i[msk_ev] for i in [ev, ev_grouping]]

        # Baselining before returning
        rescale(data_am, times, baseline, baseline_kind, copy=False)
        msk_time = _time_mask(times, *use_times)
        data_am, data_ph = [i[..., msk_time] for i in [data_am, data_ph]]
        # Stack epochs to a single trace if specified
        if concat_epochs is True:
            ev_unique = np.unique(ev_grouping)
            concat_data = []
            for i_ev in ev_unique:
                msk_events = ev_grouping == i_ev
                concat_data.append([np.hstack(i[msk_events])
                                    for i in [data_am, data_ph]])
            data_am, data_ph = zip(*concat_data)
    else:
        data_ph = np.array([data_ph])
        data_am = np.array([data_am])
    data_ph = list(data_ph)
    data_am = list(data_am)
    if scale_amp_func is not None:
        for i in range(len(data_am)):
            data_am[i] = scale_amp_func(data_am[i], axis=-1)

    n_ep = len(data_ph)
    pac = np.zeros([n_ep, len(ixs_new)])
    pbar = ProgressBar(n_ep)
    print(len(data_ph), len(data_am))
    for iep, (ep_ph, ep_am) in enumerate(zip(data_ph, data_am)):
        for iix, (i_ix_ph, i_ix_am) in enumerate(ixs_new):
            # f_phase and f_amp won't be used in this case
            pac[iep, iix] = func(ep_ph[i_ix_ph], ep_am[i_ix_am],
                                 f_phase, f_amp, filterfn=False)
        pbar.update_with_increment_value(1)
    if return_data:
        return pac, data_ph, data_am
    else:
        return pac
