import os
import quantities as pq
import exdir
import numpy as np
import neo


################################################################################
# From old expipe_plugin_cinpla
################################################################################
def generate_stim_group_and_epoch(action):
    exdir_path = os.path.join(str(action._backend.path), "data/main.exdir")
    exdir_object = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])

    grating = get_grating_stimulus_events(exdir_object["epochs/axona_inp"])
    keys = get_key_press_events(exdir_object["epochs/axona_inp"])
    durations = grating["blank"]["timestamps"][1:] - grating["grating"]["timestamps"]

    # generate stimulus groups
    generate_blank_group(exdir_path, grating["blank"]["timestamps"])
    generate_key_event_group(exdir_path, keys=keys["keys"], timestamps=keys["timestamps"])
    generate_grating_stimulus_group(exdir_path,
                                    data=grating["grating"]["data"],
                                    timestamps=grating["grating"]["timestamps"],
                                    mode=grating["grating"]["mode"])

    # generate stimulus epoch
    generate_grating_stimulus_epoch(exdir_path,
                                    timestamps=grating["grating"]["timestamps"],
                                    durations=durations,
                                    data=grating["grating"]["data"])

    print("successfully created stimulus groups and epoch.")


# Core parser functions for stimulus files-------------------------------------
def get_raw_inp_data(inp_group):
    # TODO: check tests
    '''
    Return raw data from axona inp data
    Parameters
    ----------
    inp_group : exdir.Group
        exdir group containing the inp data
    Returns
    -------
    event_types : array of stings
        event type, I, O, or K
    timestamps : array
        event timestamps
    values : array
            value of the event (bytes)
    '''
    event_types = inp_group["event_types"].data
    timestamps = pq.Quantity(inp_group["timestamps"].data,
                             inp_group["timestamps"].attrs["unit"])
    values = inp_group["values"].data

    return event_types, timestamps, values


def convert_inp_values_to_keys(values):
    # TODO: check tests
    '''
    Converts inp values to keys (strings)
    Parameters
    ----------
    values : array_like
        event values, byte 6 and 7 (see DacqUSB doc)
    Returns
    -------
    keys : array_like
         pressed keys (strings)
    '''
    keys = [None] * len(values)
    for i in range(len(values)):
        if(values[i, 0].astype(int) != 0):
            raise ValueError("Cannot map a functional key event:", values[i, 0])
        else:
            key = str(chr(values[i, 1]))
            if(key == " "):
                keys[i] = "space"
            else:
                keys[i] = key

    return np.array(keys)


def get_key_press_events(inp_group):
    # TODO: check tests
    '''
    Parameters
    ----------
    inp_group : exdir.Group
        exdir group containing the inp data
    Returns
    -------
    data : dict
           dict with pressed keys and corrosponding timestamps
    '''
    event_types, timestamps, values = get_raw_inp_data(inp_group)

    data = {}
    event_ids = np.where(event_types == "K")[0]
    keys = convert_inp_values_to_keys(values[event_ids])
    data = {"timestamps": timestamps[event_ids],
            "keys": keys}

    return data


def _find_identical_trialing_elements(values):
    # TODO: check tests
    '''
    Finds indices of the first elements when there are two or more
    trialing elements with the same value
    Parameters
    ----------
    values : array_like
        event values
    Returns
    -------
    ids : list
             list of indices
    '''
    ids = []
    value_id = 0
    samples_count = len(values)
    while value_id < samples_count - 1:
        current_id = value_id
        current_value = values[value_id]
        rep_count = 1
        next_id = value_id + 1
        for i in range(next_id, samples_count):
            if values[i] == current_value:
                rep_count += 1
            else:
                value_id = i
                break
        if len(ids) != 0 and ids[-1] == current_id:
                break
        if rep_count > 1:
            ids.append(current_id)
    return ids


def get_synced_orientation_data(timestamps, values):
    # TODO: check tests
    '''
    Converts inp values to degrees
    Parameters
    ----------
    timestamps : array_like
        inp file timestamps
    values : array_like
        event values, byte 6 and 7 (see DacqUSB doc)
    Returns
    -------
    orientations : array of quantities
                    orientation in degrees
    t_stim : array of quantities
               stimulus onset times
    t_blank : array of quantities
                stimulus offset times
    '''
    orientations = None
    t_stim = None
    t_blank = None

    value = values[:, 1]  # Only the last byte is used to carry information
    ids = _find_identical_trialing_elements(value)  # ids confirmed to carry data
    if not ids:
        raise AssertionError("Could not find identical trialing elements, ids: ", ids)

    offset = value[ids[0]]  # first input is value for blank screen

    # If the last index is single and is a blank, add it to the ids
    if value[-1] != value[-2] and value[-1] == offset:
        id_last_element = len(value) - 1
        ids.append(id_last_element)

    times = timestamps[ids]
    offset_values = value[ids] - offset

    if (offset_values < 0).any():
        raise AssertionError("Negative numbers in offset values, offset_values: ", offset_values)

    # Find the corresponding orientations
    stim_ids = np.where(offset_values > 0)[0]  # 0 > corrospond to stimulus
    blank_ids = np.where(offset_values == 0)[0]  # 0 corrospond to blank

    # orientations are given in range [0, 360>.
    orientation_count = max(offset_values)
    orientations = (offset_values[stim_ids] - 1) * 360. / orientation_count

    t_stim = times[stim_ids]
    t_blank = times[blank_ids]

    return orientations * pq.deg, t_stim, t_blank


def get_grating_stimulus_events(inp_group, mode="orientation"):
    # TODO: check tests
    # TODO: add more read modes if necessary
    '''
    Parameters
    ----------
    inp_group : exdir.Group
        exdir group containing the inp data
    Returns
    -------
    data : dict
           dict with grating data and blank times.
           grating data includes timestamps and
           grating parameters (e.g. orientation)
    '''
    t_blank, t_stim, grating_param = None, None, None
    event_types, timestamps, values = get_raw_inp_data(inp_group)

    data = {}
    event_ids = np.where(event_types == "I")[0]

    if(mode == "orientation"):
        grating_param, t_stim, t_blank = get_synced_orientation_data(timestamps[event_ids], values[event_ids])
    else:
        raise NameError("unknown mode: ", mode)

    data["grating"] = {"timestamps": t_stim, "data": grating_param, "mode": mode}
    data["blank"] = {"timestamps": t_blank}

    return data


###############################################################################
#              stimulus group/epoch generate functions
###############################################################################
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
    from visualstimulation.helper import convert_quantity_scalar_to_string, rescale_orients, convert_string_to_quantity_scalar
    sorted_trials = defaultdict(list)
    rescale_orients(trials, unit)

    for trial in trials:
        orient = trial.annotations["orient"]
        key = convert_quantity_scalar_to_string(orient)
        sorted_trials[key].append(trial)

    return OrderedDict(sorted(sorted_trials.items(),
                              key=lambda x: convert_string_to_quantity_scalar(x[0]).magnitude))


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


def generate_grating_stimulus_group(exdir_path, data, timestamps, mode="None"):
    '''
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
    '''
    exdir_file = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])
    stimulus = exdir_file.require_group("stimulus")
    presentation = stimulus.require_group("presentation")
    visual = presentation.require_group("visual")

    grating = visual.require_group("grating")
    grating.require_dataset("timestamps", data=timestamps)
    dset = grating.require_dataset("data", data=data)
    dset.attrs["mode"] = mode


def generate_blank_group(exdir_path, timestamps):
    '''
    Generates blank exdir group with timestamp dataset
    Parameters
    ----------
    exdir_path : string
            Path to exdir file
    timestamps : array_like
    '''
    exdir_file = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])
    stimulus = exdir_file.require_group("stimulus")
    presentation = stimulus.require_group("presentation")
    visual = presentation.require_group("visual")

    blank = visual.require_group("blank")
    blank.require_dataset("timestamps", data=timestamps)


def generate_key_event_group(exdir_path, keys, timestamps):
    '''
    Generates key press exdir group with timestamp
    dataset and key dataset.
    Parameters
    ----------
    exdir_path : string
            Path to exdir file
    keys : array_like
        array with pressed keys
    timestamps : array_like
    '''
    exdir_file = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])
    stimulus = exdir_file.require_group("stimulus")
    presentation = stimulus.require_group("presentation")
    key_press = presentation.require_group("key_press")

    key_press.require_dataset("timestamps", data=timestamps)
    key_press.require_dataset("keys", data=keys)


def generate_grating_stimulus_epoch(exdir_path, timestamps, durations, data):
    '''
    Generates visual stimulus epoch exdir group with timestamps
    and duration.
    Parameters
    ----------
    exdir_path : string
            Path to exdir file
    timestamps : array_like
    durations : array_like
    '''
    exdir_file = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])
    epochs = exdir_file.require_group("epochs")
    stim_epoch = epochs.require_group("visual_stimulus")
    stim_epoch.attrs["type"] = "visual_stimulus"
    times = stim_epoch.require_dataset('timestamps', data=timestamps)
    times.attrs['num_samples'] = len(timestamps)
    durations = stim_epoch.require_dataset('durations', data=durations)
    durations.attrs['num_samples'] = len(durations)
    data = stim_epoch.require_dataset('data', data=data)
    data.attrs['num_samples'] = len(data)


################################################################################
# From mros and old exana
################################################################################
def make_spiketrain_trials(spike_train, epoch, t_start=None, t_stop=None,
                           dim=None):
    '''
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
    '''

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
        assert is_quantities(spike_train, 'vector')
        sptr = spike_train
        dim = sptr.dimensionality
        unit = sptr.units
    elif isinstance(spike_train, np.array):
        sptr = spike_train * pq.Quantity(1, unit)
        dim = sptr.dimensionality
        unit = sptr.units
    else:
        raise TypeError('Expected (neo.Unit, neo.SpikeTrain, ' +
                        'quantities.Quantity, numpy.array), got "' +
                        str(type(spike_train)) + '"')

    from neo.core import SpikeTrain
    if t_start is None:
        t_start = 0 * unit
    if t_start.ndim == 0:
        t_starts = t_start * np.ones(len(epoch.times))
    else:
        t_starts = t_start
        assert len(epoch.times) == len(t_starts), 'epoch.times and t_starts have different size'
    if t_stop is None:
        t_stop = epoch.durations
    if t_stop.ndim == 0:
        t_stops = t_stop * np.ones(len(epoch.times))
    else:
        t_stops = t_stop
        assert len(epoch.times) == len(t_stops), 'epoch.times and t_stops have different size'

    if not isinstance(epoch, neo.Epoch):
        raise TypeError('Expected "neo.Epoch" got "' + str(type(epoch)) + '"')

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


def make_stimulus_trials(chxs, stim_epoch):
    # NOTE: Based on old expipe
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


def get_epoch(epochs, epoch_type):
    # NOTE: Based on old expipe
    '''
    returns epoch with matching name
    Parameters
    ----------
    epochs : list
        list of neo.core.Epoch
    epoch_type : str
        epoch type (name)
    Returns
    -------
    out : neo.core.Epoch
    '''
    for epoch in epochs:
        if epoch_type == epoch.annotations.get("type", None):
            return epoch
    else:
        raise ValueError("epoch not found", epoch_type)


def get_lfp_signals(action):
    # NOTE: Based on old expipe
    """
    Returns list with LFPs (analogsignals)
    Parameters
    ----------
    action : expipe.core.Action
    Returns
    -------
        stim_trials : list
            list of neo.core.AnalogSignal
    """
    from neo.io.exdirio import ExdirIO

    exdir_path = os.path.join(str(action._backend.path), "data/main.exdir")
    f = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])

    io = ExdirIO(exdir_path)
    block = io.read_block()
    segment = block.segments[0]

    return segment.analogsignals


def get_stim_trials(action, time_offset=0*pq.ms):
    # NOTE: Based on old expipe
    """
    Returns stimulus trials of the action
    Parameters
    ----------
    action : expipe.core.Action
    time_offset : Quantity scalar
        Time offset with respect to stimulus
        onset and offset.
    Returns
    -------
        stim_trials : defaultdict(dict)
            trials[channel_index_name][unit_id] = list of spike_train trials.
    """
    from neo.io.exdirio import ExdirIO

    exdir_path = os.path.join(str(action._backend.path), "data/main.exdir")
    f = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])

    io = ExdirIO(exdir_path)
    block = io.read_block()
    segment = block.segments[0]

    stim_epoch = get_epoch(segment.epochs, "visual_stimulus")
    stim_trials = make_stimulus_trials(block.channel_indexes, stim_epoch)

    return stim_trials


def get_segment_sptrs(action):
    # NOTE: Based on old expipe
    """
    Returns segment spike trains
    Parameters
    ----------
    action : expipe.core.Action
    Returns
    -------
        seg_sptrs : defaultdict(dict)
            seg_sptrs[channel_index_name][unit_id] = neo spike trains
    """
    from collections import defaultdict
    from neo.io.exdirio import ExdirIO

    seg_sptrs = defaultdict(dict)

    exdir_path = os.path.join(str(action._backend.path), "data/main.exdir")
    f = exdir.File(exdir_path, plugins=[exdir.plugins.quantities])

    # Read in neo:
    io = ExdirIO(exdir_path)
    block = io.read_block()
    segment = block.segments[0]

    for sptr in segment.spiketrains:
        if "UnitTimes" in sptr.annotations["exdir_path"]:
            ch_id, unit_id = sptr.annotations["exdir_path"].split("channel_group_")[-1].split("/UnitTimes/")
            ch_name = ("Channel group {}".format(ch_id))
            seg_sptrs[ch_name][int(unit_id)] = sptr

    return seg_sptrs


def get_unit_sptr(action, unit_id):
    # NOTE: Based on old expipe
    """
    Returns unit spike train (neo.core.spiketrain)
    for unit with id unit_id in action.
    Parameters
    ----------
    actions : expipe.core.Action
    unit_id : str
        unit id
    Returns
    -------
        units_trials : neo.core.spiketrain.SpikeTrain
            Neo spike trains
    """
    channels = [{"ch": key, "units": value} for key, value in cell_module.items() if 'channel_group_' in key]

    seg_sptrs = get_segment_sptrs(action)
    sptr = []
    for ch in channels:
        ch_name = ("Channel group {}".format(ch["ch"].split("channel_group_")[-1]))

        for unit, unit_items in ch["units"].items():
            unit_name = int(unit.split("unit_")[-1])

            if unit_items["cell_id"] == unit_id:
                sptr.append(seg_sptrs[ch_name][unit_name])

    if not sptr:
        raise Exception("could not find unit {} in action {}".format(unit_id, action.id))
    elif len(sptr) > 1:
        warnings.warn("found multiple units with same unit id ({}) in action {}".format(unit_id,
                                                                                        action.id))

    return sptr[0]


def get_unit_trials(action, unit_id, time_offset=0*pq.ms):
    # NOTE: Based on old expipe
    """
    Returns unit trials (list of spike trains)
    for unit with id unit_id in action.
    Parameters
    ----------
    actions : expipe.core.Action
    unit_id : str
        unit id
    time_offset : Quantity scalar
        Time offset with respect to stimulus
        onset and offset.
    Returns
    -------
        units_trials : list
            list of Neo spike trains
    """
    stim_trials = get_stim_trials(action, time_offset)

    channels = [{"ch": key, "units": value} for key, value in cell_module.items() if 'channel_group_' in key]

    trials = []
    for ch in channels:
        ch_name = ("Channel group {}".format(ch["ch"].split("channel_group_")[-1]))

        for unit, unit_items in ch["units"].items():
            unit_name = int(unit.split("unit_")[-1])

            if unit_items["cell_id"] == unit_id:
                trials.append(stim_trials[ch_name][unit_name])

    if not trials:
        raise Exception("could not find unit {} in action {}".format(unit_id, action.id))
    elif len(trials) > 1:
        warnings.warn("found multiple units with same unit id ({}) in action {}".format(unit_id, action.id))

    return trials[0]


def get_all_units_trials(actions, time_offset=0*pq.ms):
    # NOTE: Based on old expipe
    """
    Organizes units in a dictionary with unit_id as key
    and a list with action trials as value.
    Parameters
    ----------
    actions : list
        list of expipe.core.Action
    Returns
    -------
        units_trials : dict
            dictionary units trials (see Notes)
    Notes
    -----
    The structure of the output is as follows:
        units_trials[<unit_id>][<action-id>] = list of spike_train trials.
    """
    units_trials = {}
    without_cell_module = []

    for action in actions:
        try:
            stim_trials = get_stim_trials(action, time_offset)
        except Exception:
            print("skipped action {}".format(action.id))
            continue
        channels = [{"ch": key, "units": value} for key, value in cell_module.items() if 'channel_group_' in key]

        for ch in channels:
            ch_name = ("Channel group {}".format(ch["ch"].split("channel_group_")[-1]))

            for unit, unit_items in ch["units"].items():
                unit_name = int(unit.split("unit_")[-1])
                unit_id = unit_items["cell_id"]

                try:
                    trials = stim_trials[ch_name][unit_name]
                except KeyError:
                    print("unit {} in channel {} with unit-id {} not found in action {}".format(unit_name,
                                                                                                ch_name,
                                                                                                unit_id,
                                                                                                action.id))
                    continue

                if unit_id not in units_trials:
                    units_trials[unit_id] = {}

                units_trials[unit_id][action.id] = trials

    if without_cell_module:
        warnings.warn("action without cell module were found: {}".format(without_cell_module))

    return units_trials
