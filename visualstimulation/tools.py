import numpy as np
import quantities as pq
import warnings
from visualstimulation.helper import *


def compute_circular_variance(rates, orients):
    """
    calculates circular variance (see Ringach 2002)
    Parameters
    ----------
    rates : quantity array
        array of firing rates
    orients : quantity array
        array of orientations
    Returns
    -------
    out : float
        circular variance
    """
    orients = orients.rescale(pq.rad)
    R = np.sum(rates * np.exp(1j*2*orients.magnitude)) / np.sum(rates)
    return 1 - np.absolute(R)


def compute_dsi(rates, orients):
    """
    calculates direction selectivity index
    Parameters
    ----------
    rates : quantity array
        array of firing rates
    orients : quantity array
        array of orientations
    Returns
    -------
    out : float
        direction selectivity index
    """
    from visualstimulation.tools import wrap_angle

    orients = orients.rescale(pq.deg)
    pref_orient = orients[np.argmax(rates)]
    opposite_orient = wrap_angle(pref_orient.rescale(pq.deg).magnitude + 180,
                                 wrap_range=360.)*pq.deg

    opposite_id, nearest_opposite = find_nearest(array=orients, value=opposite_orient)
    if opposite_orient != nearest_opposite:
        warnings.warn("opposite angle ({}) wrt pref orient ({}) not find in orients, using nearest angle ({}) in orient".format(opposite_orient, pref_orient, nearest_opposite))

    R_pref = rates.max()
    R_opposite = rates[opposite_id]

    return (R_pref - R_opposite) / (R_pref + R_opposite)


def compute_osi(rates, orients):
    """
    calculates orientation selectivity index
    Parameters
    ----------
    rates : quantity array
        array of firing rates
    orients : quantity array
        array of orientations
    Returns
    -------
    out : float
        orientation selectivity index
    """
    from visualstimulation.tools import wrap_angle

    orients = orients.rescale(pq.deg)
    pref_orient = orients[np.argmax(rates)]
    ortho = wrap_angle(pref_orient.rescale(pq.deg).magnitude + 90, wrap_range=360.)*pq.deg

    ortho_id, nearest_ortho = find_nearest(array=orients, value=ortho)
    if ortho != nearest_ortho:
        warnings.warn("ortho angle ({}) wrt pref orient ({}) not find in orients, using nearest angle ({}) in orient".format(ortho, pref_orient, nearest_ortho))

    R_pref = rates.max()
    R_ortho = rates[ortho_id]

    return (R_pref - R_ortho) / (R_pref + R_ortho)


def fit_orient_tuning_curve(rates, orients, func, guess, bounds, binsize=1*pq.deg):
    """
    Use non-linear least squares to fit a function to
    orientation tuning data.
    Parameters
    ----------
    rates : quantity array
        Array of firing rates
    orients : quantity array
        Array of orientations
    func : callable
        The model function
    guess :  None, scalar, or N-length sequence, optional
        Initial guess for the parameters.
    bounds: None, scalar, or N-length sequence, optional
        bounds for intervals
    binsize: float/quantity scalar
        Resolution of fitted curve.
    Returns
    -------
    data_fitted : array
        Array with fitted data
    new_orients : array
        Orientation array for fitted tuning curve
    params : tuple
        Returns optimal values and the estimated covariance
        for the parameters.
    """

    from scipy import optimize
    params, params_cov = optimize.curve_fit(f=func,
                                            xdata=orients.rescale(pq.deg).magnitude,
                                            ydata=rates.magnitude,
                                            p0=guess,
                                            bounds=bounds)

    new_orients = np.arange(0, 360, binsize.magnitude)*pq.deg
    data_fitted = func(new_orients.magnitude, *params)*rates.units

    return data_fitted, new_orients, (params, params_cov)


def compute_spontan_rate(chxs, stim_off_epoch):
    # TODO: test
    '''
    Calculates spontaneous firing rate

    Parameters
    ----------
    chxs : list
        list of neo.core.ChannelIndex
    stim_off_epoch : neo.core.Epoch
        stimulus epoch

    Returns
    -------
    out : defaultdict(dict)
        rates[channel_index_name][unit_id] = spontaneous rate
    '''
    from collections import defaultdict
    from elephant.statistics import mean_firing_rate

    rates = defaultdict(dict)
    unit_rates = pq.Hz

    for chx in chxs:
        for un in chx.units:
            cluster_group = un.annotations.get('cluster_group') or 'noise'
            if cluster_group.lower() != "noise":
                sptr = un.spiketrains[0]
                unit_id = un.annotations["cluster_id"]
                trials = make_spiketrain_trials(epoch=stim_off_epoch,
                                                t_start=0 * pq.s,
                                                t_stop=stim_off_epoch.durations,
                                                spike_train=sptr)
                rate = 0 * unit_rates
                for trial in trials:
                    rate += mean_firing_rate(trial, trial.t_start, trial.t_stop)

                rates[chx.name][unit_id] = rate / len(trials)

    return rates


def compute_orientation_tuning(orient_trials):
    from visualstimulation.tools import (make_orientation_trials,
                                         convert_string_to_quantity_scalar)
    '''
    Calculates the mean firing rate for each orientation

    Parameters
    ----------
    trials : collections.OrderedDict
        OrderedDict with orients as keys and trials as values.

    Returns
    -------
    rates : quantity array
        average rates
    orients : quantity array
        sorted stimulus orientations
    '''
    from elephant.statistics import mean_firing_rate

    unit_orients = pq.deg
    unit_rates = pq.Hz
    orient_count = len(orient_trials)

    rates = np.zeros((orient_count)) * unit_rates
    orients = np.zeros((orient_count)) * unit_orients

    for i, (orient, trials) in enumerate(orient_trials.items()):
        orient = convert_string_to_quantity_scalar(orient)
        rate = 0 * unit_rates

        for trial in trials:
            rate += mean_firing_rate(trial, trial.t_start, trial.t_stop)

        rates[i] = rate / len(trials)
        orients[i] = orient.rescale(unit_orients)

    return rates, orients


def rate_latency(trials=None, epo=None, unit=None, t_start=None, t_stop=None,
                 kernel=None, search_stop=None, sampling_period=None):
    assert trials != unit
    import neo
    import elephant
    if trials is None:
        trials = make_spiketrain_trials(epo=epo, unit=unit, t_start=t_start,
                                        t_stop=t_stop)
    else:
        t_start = trials[0].t_start
        t_stop = trials[0].t_stop
        if search_stop is None:
            search_stop = t_stop
    trial = neo.SpikeTrain(times=np.array([st for trial in trials
                                           for st in trial.times.rescale('s')])*pq.s,
                           t_start=t_start, t_stop=t_stop)
    rate = elephant.statistics.instantaneous_rate(trial, sampling_period,
                                                  kernel=kernel, trim=True)/len(trials)
    rate_mag = rate.rescale('Hz').magnitude.reshape(len(rate))
    if not any(rate_mag):
        return np.nan, rate
    else:
        mask = (rate.times > 0*pq.ms) & (rate.times < 250*pq.ms)
        spont_mask = (rate.times > -250*pq.ms) & (rate.times < 0*pq.ms)
        # spk, ind = find_max_peak(rate_mag[mask])
        krit1 = rate_mag[mask].mean() + rate_mag[mask].std() > rate_mag[spont_mask].mean() + rate_mag[spont_mask].std()
        spike_mask = (trial.times > 0*pq.ms) & (trial.times < search_stop)
        krit2 = len(trial.times[spike_mask])/search_stop.rescale('s') > 1.*pq.Hz
        if not krit1 and krit2:
            return np.nan, rate
        t0 = 0*pq.ms
        while t0 < search_stop:
            mask = (rate.times > t0) & (rate.times < search_stop)
            pk, ind = find_first_peak(rate_mag[mask])
            if len(pk) == 0:
                break
            krit3 = pk > rate_mag[mask].mean() + rate_mag[mask].std()
            krit4 = pk > 1.*pq.Hz
            krit5 = pk != 0
            lat_time = rate.times[mask][ind]
            assert len(lat_time) == 1
            if krit3 and krit4 and krit5:
                return lat_time, rate
            else:
                t0 = lat_time
        return np.nan, rate


###############################################################################
#                      functions for organizing data
###############################################################################
def add_orientation_to_trials(trials, orients):
    """
    Adds annotation 'orient' to trials

    Parameters
    ----------
    trials : list of neo.SpikeTrains
    orients : quantity array
        orientation array
    """
    assert len(trials) == len(orients)
    for trial, orient in zip(trials, orients):
        trial.annotations["orient"] = orient


def make_stimulus_trials(chxs, stim_epoch):
    '''
    makes stimulus trials for every units (good) in each channel

    Parameters
    ----------
    chxs : list
        list of neo.core.ChannelIndex
    stim_epoch : neo.core.Epoch
        stimulus epoch

    Returns
    -------
    out : defaultdict(dict)
        trials[channel_index_name][unit_id] = list of spike_train trials.
    '''
    from collections import defaultdict
    stim_trials = defaultdict(dict)

    for chx in chxs:
        for un in chx.units:
            cluster_group = un.annotations.get('cluster_group') or 'noise'
            if cluster_group.lower() != "noise":
                sptr = un.spiketrains[0]
                trials = make_spiketrain_trials(epoch=stim_epoch,
                                                t_start=0 * pq.s,
                                                t_stop=stim_epoch.durations,
                                                spike_train=sptr)
                unit_id = un.annotations["cluster_id"]
                stim_trials[chx.name][unit_id] = trials

    # Add orientation value to each trial as annotation
    for chx in stim_trials.values():
        for trials in chx.values():
            add_orientation_to_trials(trials, stim_epoch.labels)

    return stim_trials


def make_orientation_trials(trials, unit=pq.deg):
    """
    Makes trials based on stimulus orientation

    Parameters
    ----------
    trials : neo.SpikeTrains
        list of spike trains where orientation is given as
        annotation 'orient' (quantity scalar) on each spike train.
    unit : Quantity, optional
        scaling unit (default is degree) used for orients
        used as keys in dictionary.

    Returns
    -------
    trials : collections.OrderedDict
        OrderedDict with orients as keys and trials as values.
    """
    from collections import defaultdict, OrderedDict
    sorted_trials = defaultdict(list)
    rescale_orients(trials, unit)

    for trial in trials:
        orient = trial.annotations["orient"]
        key = convert_quantity_scalar_to_string(orient)
        sorted_trials[key].append(trial)

    return OrderedDict(sorted(sorted_trials.items(),
                              key=lambda x: convert_string_to_quantity_scalar(x[0]).magnitude))


def make_stimulus_off_epoch(epo, include_boundary=False):
    '''
    Creates a neo.Epoch of off periods.
    Parameters
    ----------
    epo : neo.Epoch
        stimulus epoch
    include_boundary :
        add 0 to be first off period
    Returns
    ------
    out : neo.Epoch
    '''

    from neo.core import Epoch
    times = epo.times[:-1] + epo.durations[:-1]
    durations = epo.times[1:] - times
    if(include_boundary):
        times = np.append([0], times)*pq.s
        durations = np.append(epo.times[0], durations)*pq.s

    off_epoch = Epoch(labels=[None]*len(times),
                      durations=durations,
                      times=times)

    return off_epoch
