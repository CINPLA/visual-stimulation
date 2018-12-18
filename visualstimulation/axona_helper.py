def get_lfp_signals(action):
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

    exdir_path = action.require_filerecord().local_path
    f = exdir.File(exdir_path)

    io = ExdirIO(exdir_path)
    block = io.read_block()
    segment = block.segments[0]

    return segment.analogsignals


def get_stim_trials(action, time_offset=0*pq.ms):
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
    import exana.stimulus.tools as exs

    exdir_path = action.require_filerecord().local_path
    f = exdir.File(exdir_path)

    io = ExdirIO(exdir_path)
    block = io.read_block()
    segment = block.segments[0]

    stim_epoch = exs.get_epoch(segment.epochs, "visual_stimulus")
    stim_trials = make_stimulus_trials(block.channel_indexes, stim_epoch, time_offset)

    return stim_trials


def get_epoch(action, epoch_name="visual_stimulus"):
    """
    Returns stimulus trials of the action
    Parameters
    ----------
    action : expipe.core.Action
    Returns
    -------
        epoch : neo.core.Epoch
    """

    from neo.io.exdirio import ExdirIO
    import exana.stimulus.tools as exs

    exdir_path = action.require_filerecord().local_path
    f = exdir.File(exdir_path)

    io = ExdirIO(exdir_path)
    block = io.read_block()
    segment = block.segments[0]

    epoch = exs.get_epoch(segment.epochs, epoch_name)

    return epoch


def get_segment_sptrs(action):
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

    exdir_path = action.require_filerecord().local_path
    f = exdir.File(exdir_path)

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
    try:
        cell_module = action.get_module("ida_mros_cell").to_dict()
    except NameError:
        print("action {} does not have 'ida_mros_cell' as a module".format(action.id))

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
    try:
        cell_module = action.get_module("ida_mros_cell").to_dict()
    except NameError:
        print("action {} does not have 'ida_mros_cell' as a module".format(action.id))

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
            cell_module = action.get_module("ida_mros_cell").to_dict()
        except NameError:
            without_cell_module.append(action.id)
            continue

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
