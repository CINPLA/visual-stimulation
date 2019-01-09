import os
import quantities as pq
import exdir
import numpy as np
from visualstimulation.utils import (generate_blank_group,
                                     generate_key_event_group,
                                     generate_grating_stimulus_group,
                                     generate_grating_stimulus_epoch
                                     )


def generate_stim_group_and_epoch(action):
    """
    Generates stimulus group and epoch. 
    """
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
