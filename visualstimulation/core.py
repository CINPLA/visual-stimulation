def wrap_angle(angle, wrap_range=360.):
    '''
    wraps angle in to the interval [0, wrap_range]

    Parameters
    ----------
    angle : numpy.array/float
        input array/float
    wrap_range : float
        wrap range (eg. 360 or 2pi)

    Returns
    -------
    out : numpy.array/float
        angle in interval [0, wrap_range]
    '''
    return angle - wrap_range * np.floor(angle/float(wrap_range))


def compute_osi(rates, orients):
    # TODO: write tests
    '''
    calculates orientation selectivity index

    Parameters
    ----------
    rates : quantity array
        array of mean firing rates
    orients : quantity array
        array of orientations

    Returns
    -------
    out : quantity scalar
        preferred orientation
    out : float
        selectivity index
    '''

    orients = orients.rescale(pq.deg)
    preferred = np.where(rates == rates.max())
    null_angle = wrap_angle(orients[preferred] + 180*pq.deg, wrap_range=360.)

    null = np.where(orients == null_angle)
    if len(null[0]) == 0:
        raise Exception("orientation not found: "+str(null_angle))

    orth_angle_p = wrap_angle(orients[preferred] + 90*pq.deg, wrap_range=360.)
    orth_angle_n = wrap_angle(orients[preferred] - 90*pq.deg, wrap_range=360.)
    orth_p = np.where(orients == orth_angle_p)
    orth_n = np.where(orients == orth_angle_n)

    if len(orth_p[0]) == 0:
        raise Exception("orientation not found: " + str(orth_angle_p))
    if len(orth_n[0]) == 0:
        raise Exception("orientation not found: " + str(orth_angle_n))

    index = 1. - (rates[orth_p] + rates[orth_n]) / (rates[preferred]+rates[null])

    return float(orients[preferred])*orients.units, float(index)


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
    from exana.stimulus.tools import (make_orientation_trials,
                                      _convert_string_to_quantity_scalar)
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
        orient = _convert_string_to_quantity_scalar(orient)
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
