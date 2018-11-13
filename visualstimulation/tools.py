###############################################################################
#                      functions for organizing data
###############################################################################
def _rescale_orients(trials, unit=pq.deg):
    """
    Rescales all orient annotations to the same unit

    Parameters
    ----------
    trials : neo.SpikeTrains
        list of spike trains where orientation is given as
        annotation 'orient' (quantity scalar) on each spike train.
    unit : Quantity, optional
        scaling unit. Default is degree.
    """
    if unit not in [pq.deg, pq.rad]:
        raise ValueError("unit can only be deg or rad, ", str(unit))

    for trial in trials:
        orient = trial.annotations["orient"]
        trial.annotations["orient"] = orient.rescale(unit)


def _convert_quantity_scalar_to_string(value):
    """
    converts quantity scalar to string

    Parameters
    ----------
    value : quantity scalar

    Returns
    -------
    out : str
        magnitude and unit are separated with space.
    """
    return str(value.magnitude)+" "+value.dimensionality.string


def _convert_string_to_quantity_scalar(value):
    """
    converts string to quantity scalar

    Parameters
    ----------
    value : str
        magnitude and unit are assumed to be separated with space.

    Returns
    -------
    out : quantity scalar
    """
    v = value.split(" ")
    return pq.Quantity(float(v[0]), v[1])


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
    _rescale_orients(trials, unit)

    for trial in trials:
        orient = trial.annotations["orient"]
        key = _convert_quantity_scalar_to_string(orient)
        sorted_trials[key].append(trial)

    return OrderedDict(sorted(sorted_trials.items(),
                              key=lambda x: _convert_string_to_quantity_scalar(x[0]).magnitude))
