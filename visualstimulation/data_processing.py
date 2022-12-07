import numpy as np
import quantities as pq
import neo
import exdir
import spikeextractors as se


def generate_grating_stimulus_group(exdir_path, data, timestamps, mode="None"):
    """
    Generates grating exdir group with timestamp dataset and
    data (eg. orientation) dataset.
    Parameters
    ----------
    exdir_path : string
            Path to exdir file
    data : array_like
        array with grating data (eg. orientation)
    timestamps : array_like
    mode: string, optional
        describes grating data
    """
    exdir_file = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])
    stimulus = exdir_file.require_group("stimulus")
    presentation = stimulus.require_group("presentation")
    visual = presentation.require_group("visual")

    grating = visual.require_group("grating")
    grating.require_dataset("timestamps", data=timestamps)
    dset = grating.require_dataset("data", data=data)
    dset.attrs["mode"] = mode


def generate_blank_group(exdir_path, timestamps):
    """
    Generates blank exdir group with timestamp dataset
    Parameters
    ----------
    exdir_path : string
            Path to exdir file
    timestamps : array_like
    """
    exdir_file = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])
    stimulus = exdir_file.require_group("stimulus")
    presentation = stimulus.require_group("presentation")
    visual = presentation.require_group("visual")

    blank = visual.require_group("blank")
    blank.require_dataset("timestamps", data=timestamps)


def generate_key_event_group(exdir_path, keys, timestamps):
    """
    Generates key press exdir group with timestamp
    dataset and key dataset.
    Parameters
    ----------
    exdir_path : string
            Path to exdir file
    keys : array_like
        array with pressed keys
    timestamps : array_like
    """
    exdir_file = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])
    stimulus = exdir_file.require_group("stimulus")
    presentation = stimulus.require_group("presentation")
    key_press = presentation.require_group("key_press")

    key_press.require_dataset("timestamps", data=timestamps)
    key_press.require_dataset("keys", data=keys)


def generate_grating_stimulus_epoch(exdir_path, timestamps, durations, data):
    """
    Generates visual stimulus epoch exdir group with timestamps
    and duration.
    Parameters
    ----------
    exdir_path : string
            Path to exdir file
    timestamps : array_like
    durations : array_like
    """
    exdir_file = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])
    epochs = exdir_file.require_group("epochs")
    stim_epoch = epochs.require_group("visual_stimulus")
    stim_epoch.attrs["provenance"] = "psychstim"
    times = stim_epoch.require_dataset("timestamps", data=timestamps)
    times.attrs["num_samples"] = len(timestamps)
    durations = stim_epoch.require_dataset("durations", data=durations)
    durations.attrs["num_samples"] = len(durations)
    data = stim_epoch.require_dataset("data", data=data)
    data.attrs["num_samples"] = len(data)

# Epochs #################################################################################
def load_epochs(data_path):
    # TODO: add test
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    epochs_group = f['epochs']
    epochs = []
    for group in epochs_group.values():
        if 'timestamps' in group.keys():
            epo = _read_epoch(f, group.name)
            epochs.append(epo)
        else:
            for g in group.values():
                if 'timestamps' in g.keys():
                    epo = _read_epoch(f, g.name)
                    epochs.append(epo)
    return epochs

def _read_epoch(exdir_file, path, lazy=False):
    # TODO: add test
    group = exdir_file[path]
    if lazy:
        times = []
    else:
        times = pq.Quantity(group['timestamps'].data,
                            group['timestamps'].attrs['unit'])

    if "durations" in group and not lazy:
        durations = pq.Quantity(group['durations'].data, group['durations'].attrs['unit'])
    elif "durations" in group and lazy:
        durations = []
    else:
        durations = None

    if 'data' in group and not lazy:
        if 'unit' not in group['data'].attrs:
            labels = group['data'].data
        else:
            labels = pq.Quantity(group['data'].data,
                                 group['data'].attrs['unit'])
    elif 'data' in group and lazy:
        labels = []
    else:
        labels = None
    annotations = {'exdir_path': path}
    annotations.update(group.attrs.to_dict())

    if lazy:
        lazy_shape = (group.attrs['num_samples'],)
    else:
        lazy_shape = None
    epo = neo.Epoch(times=times, durations=durations, labels=labels,
                lazy_shape=lazy_shape, **annotations)

    return epo

# Spiketrains #################################################################################
def load_spiketrains(data_path, channel_group=None, load_waveforms=False, lim=None):
    '''

    Parameters
    ----------
    data_path
    channel_group
    load_waveforms
    remove_label

    Returns
    -------
    '''
    sample_rate = _get_sample_rate(data_path)
    sorting = se.ExdirSortingExtractor(
        data_path, sampling_frequency=sample_rate,
        channel_group=channel_group, load_waveforms=load_waveforms)
    sptr = []
    # build neo pbjects
    for u in sorting.get_unit_ids():
        times = sorting.get_unit_spike_train(u) / sample_rate
        if lim is None:
            t_stop = _get_duration(data_path)
            t_start = 0 * pq.s
        else:
            t_start = pq.Quantity(lim[0], 's')
            t_stop = pq.Quantity(lim[1], 's')
        mask = (times >= t_start) & (times <= t_stop)
        times = times[mask]
        if load_waveforms and 'waveforms' in sorting.get_unit_spike_feature_names(u):
            wf = sorting.get_unit_spike_features(u, 'waveforms')
            wf = wf[mask] * pq.uV
        else:
            wf = None
        st = neo.SpikeTrain(times=times, t_stop=t_stop, waveforms=wf, sampling_rate=sample_rate)
        for p in sorting.get_unit_property_names(u):
            st.annotations.update({p: sorting.get_unit_property(u, p)})
        sptr.append(st)

    return sptr


def _get_channel_groups(data_path):
    '''
    Returns channel groups of processing/electrophysiology

    Parameters
    ----------
    data_path: Path
        The action data path

    Returns
    -------
    channel groups: list
        The channel groups
    '''
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    channel_groups = []
    if 'processing' in f.keys():
        processing = f['processing']
        if 'electrophysiology' in processing.keys():
            ephys = processing['electrophysiology']
            for chname, ch in ephys.items():
                if 'channel' in chname:
                    channel_groups.append(int(chname.split('_')[-1]))
    return channel_groups

def _get_sample_rate(data_path, default_sample_rate=30000*pq.Hz):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    sr = default_sample_rate
    if 'processing' in f.keys():
        processing = f['processing']
        if 'electrophysiology' in processing.keys():
            ephys = processing['electrophysiology']
            if 'sample_rate' in ephys.attrs.keys():
                sr = ephys.attrs['sample_rate']
    return sr


def _get_duration(data_path):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])

    return f.attrs['session_duration'].rescale('s')

# Trails #################################################################################
def get_stimulus_trials(exdir_path, stimulus_epoch):
    """
    Returns stimulus trials of the action
    Parameters
    ----------
    action : expipe.core.Action

    Returns
    -------
        stim_trials : defaultdict(dict)
            trials[channel_index_name][unit_id] = list of spike_train trials.
    """
    f = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])
    ch_grps = _get_channel_groups(exdir_path)
    ch_grps_sptrs = [load_spiketrains(exdir_path, ch_grp) for ch_grp in ch_grps]
    stim_trials = _make_stimulus_trials(ch_grps_sptrs, stimulus_epoch)

    return stim_trials

def _make_stimulus_trials(ch_grps_sptrs, stim_epoch):
    """
    Makes stimulus trials for every units in each channel

    Parameters
    ----------
    ch_grps_sptrs : list
        list of neo.core.SpikeTrains for each of the channel groups
    stim_epoch : neo.core.Epoch
        stimulus epoch

    Returns
    -------
    out : defaultdict(dict)
        trials[ch_grp_index][unit_id] = list of spike_train trials.
    """
    from collections import defaultdict
    stim_trials = defaultdict(dict)

    for ch_id, ch_grp_sptrs in enumerate(ch_grps_sptrs):
        for unit_id, sptr in enumerate(ch_grp_sptrs):
            trials = make_spiketrain_trials(epoch=stim_epoch,
                                            t_start=0 * pq.s,
                                            t_stop=stim_epoch.durations,
                                            spike_train=sptr)
            stim_trials[ch_id][unit_id] = trials

    # Add orientation value to each trial as annotation
    for ch in stim_trials.values():
        for trials in ch.values():
            _add_orientation_to_trials(trials, stim_epoch.labels)

    return stim_trials


def _add_orientation_to_trials(trials, orients):
    """
    Adds annotation "orient" to trials
    Parameters
    ----------
    trials : list of neo.SpikeTrains
    orients : quantity array
        orientation array
    """
    assert len(trials) == len(orients)
    for trial, orient in zip(trials, orients):
        trial.annotations["orient"] = orient


def make_spiketrain_trials(spike_train, epoch, t_start=None, t_stop=None,
                           dim=None):
    """
    Makes trials based on an Epoch and given temporal bound
    Parameters
    ----------
    spike_train : neo.SpikeTrain, neo.Unit, numpy.array, quantities.Quantity
    epoch : neo.Epoch
    t_start : quantities.Quantity
        time before epochs, default is 0 s
    t_stop : quantities.Quantity
        time after epochs default is duration of epoch
    dim : str
        if spike_train is numpy.array, the unit must be provided, e.g. "s"

    Returns
    -------
    out : list of neo.SpikeTrains
    """

    if isinstance(spike_train, neo.Unit):
        sptr = []
        dim = unit.spiketrains[0].times.dimensionality
        unit = unit.spiketrains[0].times.units
        for st in unit.spiketrains:
            sptr.append(spike_train.rescale(dim).magnitude)
        sptr = np.sort(sptr) * unit
    elif isinstance(spike_train, neo.SpikeTrain):
        sptr = spike_train.times
        dim = sptr.dimensionality
        unit = sptr.units
    elif isinstance(spike_train, pq.Quantity):
        assert is_quantities(spike_train, "vector")
        sptr = spike_train
        dim = sptr.dimensionality
        unit = sptr.units
    elif isinstance(spike_train, np.array):
        sptr = spike_train * pq.Quantity(1, unit)
        dim = sptr.dimensionality
        unit = sptr.units
    else:
        raise TypeError("Expected (neo.Unit, neo.SpikeTrain, " +
                        "quantities.Quantity, numpy.array), got" +
                        str(type(spike_train)))

    from neo.core import SpikeTrain
    if t_start is None:
        t_start = 0 * unit
    if t_start.ndim == 0:
        t_starts = t_start * np.ones(len(epoch.times))
    else:
        t_starts = t_start
        assert len(epoch.times) == len(t_starts), "epoch.times and t_starts have different size"
    if t_stop is None:
        t_stop = epoch.durations
    if t_stop.ndim == 0:
        t_stops = t_stop * np.ones(len(epoch.times))
    else:
        t_stops = t_stop
        assert len(epoch.times) == len(t_stops), "epoch.times and t_stops have different size"

    if not isinstance(epoch, neo.Epoch):
        raise TypeError("Expected {} got {}".format(neo.Epoch, str(type(epoch))))

    trials = []
    for j, t in enumerate(epoch.times.rescale(dim)):
        t_start = t_starts[j].rescale(dim)
        t_stop = t_stops[j].rescale(dim)
        spikes = []
        for spike in sptr[(t+t_start < sptr) & (sptr < t+t_stop)]:
            spikes.append(spike-t)
        trials.append(SpikeTrain(times=spikes * unit,
                                 t_start=t_start,
                                 t_stop=t_stop))
    return trials


def make_orientation_trials(trials, unit=pq.deg):
    """
    Makes trials based on stimulus orientation
    Parameters
    ----------
    trials : neo.SpikeTrains
        list of spike trains where orientation is given as
        annotation "orient" (quantity scalar) on each spike train.
    unit : Quantity, optional
        scaling unit (default is degree) used for orients
        used as keys in dictionary.
    Returns
    -------
    trials : collections.OrderedDict
        OrderedDict with orients as keys and trials as values.
    """
    from collections import defaultdict, OrderedDict
    from visualstimulation.helper import convert_quantity_scalar_to_string, rescale_orients, convert_string_to_quantity_scalar

    sorted_trials = defaultdict(list)
    rescale_orients(trials, unit)

    for trial in trials:
        orient = trial.annotations["orient"]
        key = convert_quantity_scalar_to_string(orient)
        sorted_trials[key].append(trial)

    return OrderedDict(sorted(sorted_trials.items(),
                              key=lambda x: convert_string_to_quantity_scalar(x[0]).magnitude))


def make_stimulus_off_epoch(epo, include_boundary=False):
    """
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
    """

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

def find_epoch_difference(ep0, ep1):
    # create intervaltree for each epoch
    tree = [it(), it()]
    ep = [ep0, ep1]
    for i, t in enumerate(tree):
        ts = ep[i].times
        durs = ep[i].durations
        for t, d in zip(ts, durs):
            t.addi(t, t+d, _)
    diff = tree[0] - tree[1]

def _return_index_in_target_epoch(ep0, ep1):
    """
    Return indices of start and stops of query epoch ep0
    in relation to target epoch ep1.
    Start and stops of events in ep1 are merged into one
    long, sorted target array, trgt.
    Any timepoint is inside/outside an interval if index
    is odd/even.

    Parameters
    ----------
    ep0 : neo.Epoch
        Query epoch
    ep1 : neo.Epoch
        Target epoch

    Returns
    ----------
    query_starts : Array
        indices of epoch starts in target array
    query_stops : Array
        indices of epoch starts in target array
    """

    # get start and stop of query epoch
    ep0_starts = ep0.times.rescale(pq.s).magnitude      # [0t1. 0t2, ..., 0tn]
    ep0_durs = ep0.durations.rescale(pq.s).magnitude    # [0dur1, 0dur2, ..., 0durn]
    ep0_stops = ep0_starts + ep0_durs                   # [0t1+0dur1, 0t2+0dur2, ..., 0tn+0durn]

    #
    ep1_starts = ep1.times.rescale(pq.s).magnitude      # [1t1, 1t2, ..., 1tn]
    ep1_durs = ep1.durations.rescale(pq.s).magnitude    # [1dur1, 1dur2, ..., 1durn]
    ep1_stops = ep1_starts + ep1_durs                   # [1t1+1dur1, 1t2+1dur2, ..., 1tn+1durn]

    # make sure start values are sorted
    assert np.allclose(np.sort(ep0_starts), ep0_starts)
    assert np.allclose(np.sort(ep1_starts), ep1_starts)

    # we create a target array to contain
    # (start0, stop0, start1, stop1, ...)
    trgt = np.zeros(len(ep1_starts)*2)                  # [0, 0, ..., 0] (1, 2n)
    trgt[0::2] = ep1_starts                             # [1t1, 0, 1t2, 0, ...]
    trgt[1::2] = ep1_stops                              # [1t1, 1t1+1dur1, 1t2, 1t2+1dur1, ...]

    query_starts = np.searchsorted(trgt, ep0_starts)    # [idx0t1, idx0t2, ..., idx0tn]
    query_stops = np.searchsorted(trgt, ep0_stops)      # [idx(0t1+0dur1), idx(0t2+0dur2), ...]

    return query_starts, query_stops


def find_epoch_difference(ep0, ep1):
    """
    Return elements of query epoch ep0 that are not in elements of target epoch ep1.

    Parameters
    ----------
    ep0 : neo.Epoch
    ep1 : neo.Epoch

    Returns
    ----------
    out : neo.Epoch
        Copy of ep0 that contains only elements outside elements of ep1

    """
    query_starts, query_stops = _return_index_in_target_epoch(ep0, ep1)

    def is_odd(num):
        return num & 0x1

    # determine whether both start and stop are in same interval
    is_even_starts = ~is_odd(query_starts).astype(bool) # False: the index is odd, True: the index is even 
    is_even_stops = ~is_odd(query_stops).astype(bool)   # False: the index is odd, True: the index is even 

    # trgt indices of start and stop have to be the same
    # otherwise start and stop could be inside two different intervals
    is_same = query_starts == query_stops

    is_outside_trgt = np.logical_and.reduce((is_even_starts, is_even_stops, is_same))

    # create new epoch | select only times when the epoch_0 is outside the target 
    starts_new = ep0.times[is_outside_trgt] 
    durs_new = ep0.durations[is_outside_trgt]
    labels_new = ep0.labels[is_outside_trgt]
    ep0_new = ep0.duplicate_with_new_data(starts_new, durs_new, labels_new)
    ep0_new._copy_annotations(ep0)

    return ep0_new


def find_epoch_intersection(ep0, ep1):
    """
    Return elements of query epoch ep0 that are in elements of target epoch ep1.

    Parameters
    ----------
    ep0 : neo.Epoch
    ep1 : neo.Epoch

    Returns
    ----------
    out : neo.Epoch
        Copy of ep0 that contains only elements inside elements of ep1

    """
    query_starts, query_stops = _return_index_in_target_epoch(ep0, ep1)

    def is_odd(num):
        return num & 0x1

    # determine whether both start and stop are in same interval
    is_odd_starts = is_odd(query_starts).astype(bool)
    is_odd_stops = is_odd(query_stops).astype(bool)
    # trgt indices of start and stop have to be the same
    # otherwise start and stop could be inside two different intervals
    is_same = query_starts == query_stops

    is_inside_trgt = np.logical_and.reduce((is_odd_starts, is_odd_stops, is_same))

    # create new epoch
    starts_new = ep0.times[is_inside_trgt]
    labels_new = ep0.labels[is_inside_trgt]
    durs_new = ep0.durations[is_inside_trgt]
    ep0_new = ep0.duplicate_with_new_data(starts_new, durs_new, labels_new)
    ep0_new._copy_annotations(ep0)

    return ep0_new

    
