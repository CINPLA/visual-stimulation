import numpy as np
import quantities as pq
import warnings


from visualstimulation.utils import minmax_scale

def compute_circular_variance(rates, orients, normalise=False):
    """
    calculates circular variance (see Ringach 2002)
    Parameters
    ----------
    rates : quantity array
        array of firing rates
    orients : quantity array
        array of orientations
    normalise : bool
        Feature scaling; true or false
    Returns
    -------
    out : float
        circular variance
    """

    if normalise is True:
        rates = minmax_scale(rates)

    orients = orients.rescale(pq.rad)
    R = np.sum(rates * np.exp(1j*2*orients.magnitude)) / np.sum(rates)
    return 1 - np.absolute(R)


def compute_dsi(rates, orients, normalise=False):
    """
    calculates direction selectivity index
    Parameters
    ----------
    rates : quantity array
        array of firing rates
    orients : quantity array
        array of orientations
    normalise : bool
        Feature scaling; true or false
    Returns
    -------
    out : float
        direction selectivity index
    """
    from visualstimulation.helper import wrap_angle, find_nearest

    if normalise is True:
        rates = minmax_scale(rates)

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


def compute_osi(rates, orients, normalise=False):
    """
    calculates orientation selectivity index
    Parameters
    ----------
    rates : quantity array
        array of firing rates
    orients : quantity array
        array of orientations
    normalise : bool
        Feature scaling; true or false
    Returns
    -------
    out : float
        orientation selectivity index
    """
    from visualstimulation.helper import wrap_angle, find_nearest

    if normalise is True:
        rates = minmax_scale(rates)

    orients = orients.rescale(pq.deg)
    pref_orient = orients[np.argmax(rates)]
    ortho = wrap_angle(pref_orient.rescale(pq.deg).magnitude + 90, wrap_range=360.)*pq.deg

    ortho_id, nearest_ortho = find_nearest(array=orients, value=ortho)
    if ortho != nearest_ortho:
        warnings.warn("ortho angle ({}) wrt pref orient ({}) not found in orients, using nearest angle ({}) in orient".format(ortho, pref_orient, nearest_ortho))

    R_pref = rates.max()
    R_ortho = rates[ortho_id]

    return (R_pref - R_ortho) / (R_pref + R_ortho)


def compute_orientation_tuning(orient_trials, weigh=False, weights=(1, 0.6)):
    """
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
    weigh: bool; default: False
        Use gradiently weighed data
    weights: tuple, list; default: (1, 0.6)
        (initial weight, last weight) ===(generate graident)==> (initial weight, ..., last weight)
    """
    from elephant.statistics import mean_firing_rate
    from visualstimulation.helper import convert_string_to_quantity_scalar
    from visualstimulation.utils import generate_gradiently_weighed_data as ggwd

    unit_orients = pq.deg
    unit_rates = pq.Hz
    orient_count = len(orient_trials)

    rates = np.zeros((orient_count)) * unit_rates
    orients = np.zeros((orient_count)) * unit_orients

    for i, (orient, trials) in enumerate(orient_trials.items()):
        orient = convert_string_to_quantity_scalar(orient)
        rate = 0 * unit_rates

        if weigh is True:
            for trial in trials:
                weighed_trial = ggwd(trial, weight_start=weights[0], weight_end=weights[1])
                rate += mean_firing_rate(weighed_trial, trial.t_start, trial.t_stop)
        else:
            for trial in trials:
                rate += mean_firing_rate(trial, trial.t_start, trial.t_stop)

        rates[i] = rate / len(trials)
        orients[i] = orient.rescale(unit_orients)

    return rates, orients


def fit_orient_tuning_curve(rates, orients, func, guess, bounds, normalise=False, binsize=1*pq.deg, ):
    # TODO: write tests
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
    normalise : bool
        Feature scaling; true or false
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

    if normalise is True:
        rates = minmax_scale(rates)

    params, params_cov = optimize.curve_fit(f=func,
                                            xdata=orients.rescale(pq.deg).magnitude,
                                            ydata=rates.magnitude,
                                            p0=guess,
                                            bounds=bounds)

    new_orients = np.arange(0, 360, binsize.magnitude)*pq.deg
    data_fitted = func(new_orients.magnitude, *params)*rates.units

    return data_fitted, new_orients, (params, params_cov)


def compute_spontan_rate(chxs, stim_off_epoch):
    # TODO: write tests
    """
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
    """
    from collections import defaultdict
    from elephant.statistics import mean_firing_rate
    from visualstimulation.utils import make_spiketrain_trials

    rates = defaultdict(dict)
    unit_rates = pq.Hz

    for chx in chxs:
        for un in chx.units:
            cluster_group = un.annotations.get("cluster_group") or "noise"
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


def rate_latency(trials=None, epo=None, unit=None, t_start=None, t_stop=None,
                 kernel=None, search_stop=None, sampling_period=None):
    # TODO: write tests
    from visualstimulation.utils import make_spiketrain_trials
    import neo
    import elephant

    warnings.warn("This function is not tested")
    assert trials != unit

    if trials is None:
        trials = make_spiketrain_trials(epo=epo, unit=unit, t_start=t_start,
                                        t_stop=t_stop)
    else:
        t_start = trials[0].t_start
        t_stop = trials[0].t_stop
        if search_stop is None:
            search_stop = t_stop
    trial = neo.SpikeTrain(times=np.array([st for trial in trials
                                           for st in trial.times.rescale("s")])*pq.s,
                           t_start=t_start, t_stop=t_stop)
    rate = elephant.statistics.instantaneous_rate(trial, sampling_period,
                                                  kernel=kernel, trim=True)/len(trials)
    rate_mag = rate.rescale("Hz").magnitude.reshape(len(rate))
    if not any(rate_mag):
        return np.nan, rate
    else:
        mask = (rate.times > 0*pq.ms) & (rate.times < 250*pq.ms)
        spont_mask = (rate.times > -250*pq.ms) & (rate.times < 0*pq.ms)
        # spk, ind = find_max_peak(rate_mag[mask])
        krit1 = rate_mag[mask].mean() + rate_mag[mask].std() > rate_mag[spont_mask].mean() + rate_mag[spont_mask].std()
        spike_mask = (trial.times > 0*pq.ms) & (trial.times < search_stop)
        krit2 = len(trial.times[spike_mask])/search_stop.rescale("s") > 1.*pq.Hz
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


def calculate_psth(st, epoch,
                   lags=[-2, 10]*pq.s,
                   bin_size=0.01*pq.s,
                   unit='rate',
                   return_bins=False):
    """
    calculate peristimulus time histogram for given spike train and epoch

    Parameters
    ----------
    st : neo.SpikeTrain
    epoch : neo.Epoch
    lags : Quantity list/array
    bin_size : Quantity scalar
    unit : None/string, default 'rate'
       desired output unit, 'count' or 'rate'
    return_bins : bool
       whether psth bins should be returned

    Returns
    -------
    psth : Quantitiy array
    bins : Quantity array
        optional
    """

    # check if lags is array like
    assert hasattr(lags, "__len__")
    lags = list(lags)

    # rescale to seconds and drop units
    lags[0] = lags[0].rescale(pq.s).magnitude
    lags[1] = lags[1].rescale(pq.s).magnitude
    bin_size = bin_size.rescale(pq.s).magnitude

    # define bins of histogram
    l0 = lags[0]
    l1 = lags[1]
    # make sure right edge is included
    if np.mod(l1-l0, bin_size) == 0:
        l1 += 10e-10
    bins = np.arange(l0, l1, bin_size)
    
    # count spikes
    psth = np.zeros(len(bins)-1, dtype=int)
    for t_epo in epoch.times:
        st_ = st.time_slice(t_start=t_epo + lags[0]*pq.s,
                            t_stop=t_epo + lags[1]*pq.s)
        ts = st_.times
        ts -= t_epo
        ts = ts.rescale(pq.s).magnitude

        hist, _ = np.histogram(ts, bins)
        psth += hist
    
    # rescale to desired unit
    assert unit in ['count', 'rate'] 
    if unit == 'rate':
        # find number of events in epoch
        n_epoch = len(epoch.times)
        psth = psth.astype(float)
        # divide by number of events
        psth *= 1./n_epoch
        # scale by binwidth
        psth *= 1./bin_size
    elif unit == 'count':
        pass
    
    if return_bins:
        return psth, bins[:-1]
    else:
        return psth

    
def calculate_psth_from_trials(trials,
                   bin_size=0.01*pq.s,
                   unit='rate',
                   return_bins=False):
    """
    calculate psth based on trials

    """
    l0 = np.unique([st.t_start for st in trials])
    assert len(l0) == 1
    l0 = l0[0]
    
    l1 = np.unique([st.t_stop for st in trials])
    assert len(l1) == 1
    l1 = l1[0]

    bin_size = bin_size.rescale(pq.s).magnitude
    
    if np.mod(l1-l0, bin_size) == 0:
        l1 += 10e-10
    bins = np.arange(l0, l1, bin_size)

    n_trials = len(trials)
    assert n_trials > 1

    # merge spikes
    spks = np.concatenate([st.times for st in trials])

    #  create histogram
    psth, bins_hist = np.histogram(spks, bins)
    bins = bins_hist[:-1]

    # rescale to desired unit
    assert unit in ['count', 'rate'] 
    if unit == 'rate':
        # find number of events in epoch
        psth = psth.astype(float)
        # divide by number of events
        psth *= 1./n_trials
        # scale by binwidth
        psth *= 1./bin_size
        # assign unit
        psth = psth*pq.Hz
    elif unit == 'count':
        pass
    
    if return_bins:
        # assign unit to bins
        bins = bins*pq.s
        return psth, bins
    else:
        return psth

