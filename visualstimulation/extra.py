import numpy as np 
import quantities as pq 
import exdir
import neo


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


def _read_epoch(exdir_file, path, cascade=True, lazy=False):
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


def load_epochs(data_path):
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
    # io = neo.ExdirIO(str(data_path), plugins=[exdir.plugins.quantities, exdir.plugins.git_lfs])
    # blk = io.read_block()
    # seg = blk.segments[0]
    # epochs = seg.epochs
    return epochs


def _return_query_timestamps(epoch_0, epoch_1):

    # extract epoch 0 data
    epoch_0_starts = epoch_0.times.rescale(pq.s).magnitude
    epoch_0_durs = epoch_0.durations.rescale(pq.s).magnitude
    epoch_0_stops = epoch_0_starts + epoch_0_durs


    # extract epoch 1 data    
    epoch_1_starts = epoch_1.times.rescale(pq.s).magnitude
    epoch_1_durs = epoch_1.durations.rescale(pq.s).magnitude
    epoch_1_stops = epoch_1_starts + epoch_1_durs

    # make sure start values are sorted
    assert np.allclose(np.sort(epoch_0_starts), epoch_0_starts)
    assert np.allclose(np.sort(epoch_1_starts), epoch_1_starts)

    # target array
    target = np.zeros(len(epoch_1_starts)*2)
    target[0::2] = epoch_1_starts 
    target[1::2] = epoch_1_stops

    query_starts = np.searchsorted(target, epoch_0_starts)
    query_stops = np.searchsorted(target, epoch_0_stops)
  
    return query_starts, query_stops


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


def _get_orientation_mask(query, epoch_orient):

    # mask matrix, with rows as orientations
    mask = np.zeros((8, len(query)))
    
    # make $query an index set, starting at 0
    query -= min(query)

    # possible orientations 
    angles = [float(x) for x in range(0, 360, 45)]
    orientations = epoch_orient.labels

    for i, ang in enumerate(angles):
        last_idx = 0
        for j, idx in enumerate(query):
            if last_idx != idx:
                last_idx = idx 

            mask[i, j] = epoch_orient.labels[idx] == ang 

    return mask 


def get_difference_orientations(epoch_0, epoch_1, epoch_orient):

    """
    compute the difference between epoch_0 and epoch_1 and superimpose the orientations in epoch_orient

    Parameters
    ----------
    epoch_0: neo.Epoch 
    epoch_1: neo.Epoch
    epoch_orient: neo.Epoch 
        with orientations based on epoch_1 intervals

    Returns
    -------
    list:
        list of neo.Epoch objects copy of epoch_0 containing only elements not inside epoch_1  
    """

    # queried timestamps
    query_starts, query_stops = _return_index_in_target_epoch(epoch_0, epoch_1)

    def is_odd(num):
        return num & 0x1

    # determine whether both start and stop are in same interval
    is_even_starts = ~is_odd(query_starts).astype(bool) # False: the index is odd, True: the index is even 
    is_even_stops = ~is_odd(query_stops).astype(bool)   # False: the index is odd, True: the index is even 

    # trgt indices of start and stop have to be the same
    # otherwise start and stop could be inside two different intervals
    is_same = query_starts == query_stops
    
    # check if start and stop are not in the same interval
    is_outside_trgt = np.logical_and.reduce((is_even_starts, is_even_stops, is_same))

    # orientation mask 
    orient_masks = _get_orientation_mask(query=query_starts, epoch_orient=epoch_orient)

    angles = [x for x in range(0, 360, 45)]

    # new epochs 
    epoch_matrix = []

    for i, angle_mask in enumerate(orient_masks):

        # create new epoch | select only times when the epoch_0 is outside the target 
        starts_new = epoch_0.times[is_outside_trgt] * angle_mask[is_outside_trgt]

        labels_new = epoch_0.labels[is_outside_trgt] * angle_mask[is_outside_trgt]
        
        durs_new = epoch_0.durations[is_outside_trgt] * angle_mask[is_outside_trgt]
        
        epoch_0_new = epoch_0.duplicate_with_new_data(starts_new, durs_new, labels_new)
        epoch_0_new._copy_annotations(epoch_0)

        # record 
        epoch_0_new.name = f"vis_epoch_{angles[i]}"
        epoch_matrix += [epoch_0_new]

    return epoch_matrix 


def get_intersection_orientations(epoch_0, epoch_1, epoch_orient):

    """
    compute the intersections of epoch_0 and epoch_1 and superimpose the orientations in epoch_orient

    Parameters
    ----------
    epoch_0: neo.Epoch 
    epoch_1: neo.Epoch
    epoch_orient: neo.Epoch 
        with orientations based on epoch_1 intervals

    Returns
    -------
    list:
        list of neo.Epoch objects copy of epoch_0 containing only elements inside epoch_1  
    """

    # queried timestamps
    query_starts, query_stops = _return_query_timestamps(epoch_0, epoch_1)

    def is_odd(num):
        return num & 0x1

    # True : odd intervals [intersections of epochs] - False : even intervals [difference of epochs]
    is_odd_starts = is_odd(query_starts).astype(bool)
    is_odd_stops = is_odd(query_stops).astype(bool)

    # equality evaluation
    is_same = query_starts == query_stops

    # check if start and stop are in the same interval 
    is_inside_target = np.logical_and.reduce((is_odd_starts, is_odd_stops, is_same))

    # orientation mask
    orient_masks = _get_orientation_mask(query=query_starts, epoch_orient=epoch_orient)

    angles = list(range(0, 360, 45))

    # new epochs 
    epoch_matrix = []

    for i, angle_mask in enumerate(orient_masks):

        starts_new = epoch_0.times[is_inside_target] * angle_mask[is_inside_target]
        #starts_new = starts_new[np.where(starts_new != 0)]

        labels_new = epoch_0.labels[is_inside_target] * angle_mask[is_inside_target]
        #labels_new = labels_new[np.where(labels_new != 0)]

        durs_new = epoch_0.durations[is_inside_target] * angle_mask[is_inside_target]
        #durs_new = durs_new[np.where(durs_new != 0)]

        epoch_0_new = epoch_0.duplicate_with_new_data(starts_new, durs_new, labels_new)
        epoch_0_new._copy_annotations(epoch_0)

        epoch_0_new.name = f'laser_epoch_vis_{angles[i]}°'  # edit name if necessary
        epoch_matrix += [epoch_0_new]

    return epoch_matrix 


def get_orientation_vis(epoch_0, epoch_1, epoch_or, from_laser_end=False):

    """
    compute the difference between epoch_0 [visual stimulation] and epoch_1 [laser epoch] and 
    superimpose orientations in epoch_or [orientations]. 
    In this function, the intervals of epoch_0 are calculated either all before or all after the
    laser session (start: start of first laser interval, end: end of the last laser interval).
    Then, the orientations for each visual stimulation inverval is calculated
    
    
    Example:
    visual stimulation interval: |---|
    laser interval: LLL
    only visual stimulation: +++
    orientation: d0

    |-------------------|  |----------------|  |----
    ++++LLL  LLL  LLL  LLL  LLL  LLL  LLL++++  +++++
    d3                                          d1

    Parameters
    ----------
    epoch_0: neo.Epoch 
    epoch_1: neo.Epoch 
    epoch_or: neo.Epoch

    Returns 
    -------
    list: 
        a neo.Epoch for each orientation 
    """

    # laser data
    laser_duration = epoch_1.durations[0].magnitude
    laser_times = iter(epoch_1.times)
    laser_start = min(epoch_1.times)
    laser_end = max(epoch_1.times)

    # visualstm data 
    vis_durations = epoch_0.durations

    # orientations 
    angles = list(range(0, 360, 45))
    orientations = epoch_or.labels 
    
    # dict for all orientations
    orient_dict = {}
    for angle in angles:
        orient_dict[angle] = {'name': f"vis_only_epoch_{angle}",
                              'times': np.array([]),
                              'durations': np.array([]),
                              'labels': [0],
                              }

   
    def record(orient_dict, orientation, onset, durations):

        """
        record the times and durations in the Epoch of a given orientation 
        """
    
        # retrieve orientation data 
        loc_times = orient_dict[orientation]['times']
        loc_durations = orient_dict[orientation]['durations']
        count = orient_dict[orientation]['labels'][-1]
        
        # preprocess new data 
        onset = np.array(onset).reshape(-1)
        durations = np.array(durations).reshape(-1)

        # merge 
        orient_dict[orientation]['times'] = np.concatenate((loc_times, onset))
        orient_dict[orientation]['durations'] = np.concatenate((loc_durations, durations))
        orient_dict[orientation]['labels'] += [count + 1]

        return orient_dict

    
    # loop over all the visual stimulation intervals until the laser interval starts 
    for i, vis_onset in enumerate(epoch_0):
    
        if from_laser_end:
            if vis_onset.magnitude + vis_durations[i].magnitude < laser_end:
                continue 
        else:
            if vis_onset.magnitude + vis_durations[i].magnitude > laser_start:
                break 
        
        # store data for the ith orientation 
        orient_dict = record(orient_dict=orient_dict, orientation=orientations[i], \
                onset=vis_onset.magnitude, durations=vis_durations[i].magnitude)
   

    # redefine orient_dict as dict of neo.Epochs 
    new_dict = {}
    for angle in angles:
        new_epoch = neo.Epoch(times=orient_dict[angle]['times'] * pq.s,
                              durations=orient_dict[angle]['durations'] * pq.s,
                              labels=np.array(orient_dict[angle]['labels'][1:]),
                              name=f"vis_only_epoch_{angle}") 
        new_dict[angle] = new_epoch 

    return new_dict


def get_diff_orientations(epoch_0, epoch_1, epoch_or):

    """
    compute the difference between epoch_0 [visual stimulation] and epoch_1 [laser epoch] and 
    superimpose orientations in epoch_or [orientations]. 
    In this function, the intervals of epoch_0 are calculated over its entire length and individuated
    as times in which there is no epoch_1 intervals. Then, the orientations are superimposed.
    
    Example:
    visual stimulation interval: |---|
    laser interval: LLL
    only visual stimulation: +++
    orientation: d0

    |-------------------|  |----------------|
    ++++LLL++LLL++LLL++LLL  LLL++LLL++LLL++++
    d3     d3   d3   d3        d1   d1   d1 

    Parameters
    ----------
    epoch_0: neo.Epoch 
    epoch_1: neo.Epoch 
    epoch_or: neo.Epoch

    Returns 
    -------
    list: 
        a neo.Epoch for each orientation 
    """

    # laser data
    laser_duration = epoch_1.durations[0].magnitude
    laser_times = iter(epoch_1.times)
    laser_start = min(epoch_1.times)
    laser_end = max(epoch_1.times)

    # visualstm data 
    vis_durations = epoch_0.durations

    # orientations 
    angles = list(range(0, 360, 45))
    orientations = epoch_or.labels 
    
    # dict for all orientations
    orient_dict = {}
    for angle in angles:
        orient_dict[angle] = {'name': f"vis_only_epoch_{angle}",
                              'times': np.array([]),
                              'durations': np.array([])
                              }

   
    def record(orient_dict, orientation, onset, durations):

        """
        record the times and durations in the Epoch of a given orientation 
        """
    
        # retrieve orientation data 
        loc_times = orient_dict[orientation]['times']
        loc_durations = orient_dict[orientation]['durations']
        
        # preprocess new data 
        onset = np.array(onset).reshape(-1)
        durations = np.array(durations).reshape(-1)

        # merge 
        orient_dict[orientation]['times'] = np.concatenate((loc_times, onset))
        orient_dict[orientation]['durations'] = np.concatenate((loc_durations, durations))

        return orient_dict

    # next laser onset 
    laser_onset = next(laser_times).magnitude 
    laser_end = laser_onset + laser_duration

    vis_onset_local = 0.
    
    # loop over all the visual stimulation intervals
    for i, vis_onset in enumerate(epoch_0):

        # previous visual stimulation onset before new actual visual stimulation onset 
        if vis_onset_local < vis_onset.magnitude:
            vis_onset_local = vis_onset.magnitude  
        
        # duration adjusted for the adjusted visual stimulation onset
        vis_duration_local = vis_durations[i].magnitude - (vis_onset_local - vis_onset.magnitude)
        
        # the visual stimulation interval has no laser stimulation in it 
        if vis_onset_local + vis_duration_local < laser_onset or laser_onset < 0:

            # store data for the ith orientation 
            orient_dict = record(orient_dict=orient_dict, orientation=orientations[i], \
                    onset=vis_onset_local, durations=vis_duration_local)
            continue 
 
        # the interval ends after the next laser stimulation
        # define the smaller intervals before the interval ends
        local_onsets = []
        local_durations = []

        vis_onset_local = vis_onset.magnitude
        # loop until the visual stimulation interval is over 
        while 1:
            
            # duration of the visual stimulation before laser onset 
            #print(laser_onset, vis_onset_local)
            vis_duration_local = laser_onset - vis_onset_local
            
            # append small onset and duration 
            local_onsets += [vis_onset_local]
            local_durations += [vis_duration_local]

            # new visual stimulation onset 
            vis_onset_local = laser_end 
           
            # next laser stimulation onset
            laser_onset = next(laser_times, None)
            # no more laser intervals
            if laser_onset is None:
                laser_onset = -1
                break 
            else:
                laser_onset = laser_onset.magnitude
            laser_end = laser_onset + laser_duration 

            # exit if the main visual stimulation is finished
            if vis_onset + vis_durations[i] < laser_onset:
                break

        # record data for the last intervals 
        orient_dict = record(orient_dict=orient_dict, orientation=orientations[i], \
                onset=vis_onset_local, durations=vis_duration_local)
    
    # redefine orient_dict as dict of neo.Epochs 
    new_dict = {}
    for angle in angles:
        new_epoch = neo.Epoch(times=orient_dict[angle]['times'] * pq.s,
                              durations=orient_dict[angle]['durations'] * pq.s,
                              name=f"vis_only_epoch_{angle}") 
        new_dict[angle] = new_epoch 

    return new_dict


def compute_response(spike_times, stim_times, times, kernel, e_percentile, i_percentile, limit=1e-3):

    """

    Parameters
    ----------
    spike_times : array-like
        1D for unimodal dstribution, or 2D for bimodal
    stim_times : array-like
        stimulus onset timestamps 
    times : array-like 
    kernel : 
    e_percentile : scalar 
    i_percentile : scalar 
    limit : scalar 
    """

    hist = kernel(times)
    p_e = hist > e_percentile
    p_i = hist < i_percentile
    idxs_e, _ = find_peaks(hist)
    
    # significant peaks
    idxs_e = idxs_e[p_e[idxs_e]]
    te_peak, pe_peak = np.nan, np.nan
    if len(idxs_e) > 0:
        # pick the largest
        idxs_e = idxs_e[np.argmax(hist[idxs_e])]
        te_peak = times[idxs_e]
        pe_cnt = spike_stim_count(spike_times, stim_times, te_peak, limit)
        pe_peak = sum(pe_cnt > 0) / len(stim_times)

    idxs_i, _ = find_peaks(- hist)
    # significant peaks
    idxs_i = idxs_i[p_i[idxs_i]]
    # only allow inhibition before excitation
    ti_peak, pi_peak = np.nan, np.nan
    if any(times[idxs_i] < te_peak):
        idxs_i = idxs_i[times[idxs_i] < te_peak]
        # pick the smallest
        idxs_i = idxs_i[np.argmin(hist[idxs_i])]
        ti_peak = times[idxs_i]
        pi_cnt = spike_stim_count(spike_times, stim_times, ti_peak, limit)
        pi_peak = sum(pi_cnt > 0) / len(stim_times)
   
    return te_peak, pe_peak, ti_peak, pi_peak


def stimulus_response_latency(spike_times, stim_times, window, std, t_start=0,\
        percentile=99, plot=False):

    """

    Parameters
    ----------
    spike_times : array-like 
        1D for unimodal dstribution, or 2D for bimodal
    stim_times : array-like
        stimulus onset timestamps 
    window : scalar
        half base width 
    std : str,, scalar or callable, optional
        [from numpy.gaussian_kde] bandwidth method, e.g. 'scott', 'silverman' 
    t_start : scalar 
    percentile : int 
        bounded to [0, 100]
    plot : bool 
        if True, it directly plots the kernel estimation 
    """
    
    from scipy.stats import gaussian_kde

    # scale/steps of the timesteps
    scale = 1e-4
    
    # conversion to array
    #spike_times = np.array(spike_times)
    #stim_times = np.array(stim_times)
    n_spikes, n_stim = len(spike_times), len(stim_times)
    
    # definition of trials batched by stimulus times 
    trials = [spike_times[(spike_times >= t - window) & (spike_times <= t + window)] - t for t in stim_times]

    # collection of all spikes from each trials 
    spikes = [s for t in trials for s in t]
    if len(spikes) == 0:
        return [np.nan] * 5
   
    # calculate the kernel estimation object
    kernel = gaussian_kde(spikes, std)
    print('-kernel computed')

    # we start 10 % away from -window due to edge effects
    pre_times = np.arange(- window + window * 0.1, 0, scale)
    i_percentile = np.percentile(kernel(pre_times), 100 - percentile, 0)
    e_percentile = np.percentile(kernel(pre_times), percentile, 0)
    print('-percentile quantities computed') 
    # window size | used to plot the percentiles
    times = np.arange(t_start, t_start + window, scale)

    if plot:
        import matplotlib.pyplot as plt
        
        print('-plotting')
        all_times = np.arange(-window, window, scale)
        
        # plot kernel
        plt.plot(all_times, kernel(all_times))
        plt.plot(pre_times, kernel(pre_times))
        
        # plot percentiles
        plt.plot(times, [i_percentile] * len(times))
        plt.plot(times, [e_percentile] * len(times))


    return times, spikes, kernel, e_percentile, i_percentile


# not used at the moment
class UnitsName:
    """
    class for handling unit names 
    """
    def __init__(self, units: list):
        
        """
        Parameters
        ----------
        
        units : list
        """
        
        self.epochs = {}
        self.names = []
        self.unit_info = {}

        self.units_name_dict = {}
        for i, unit in enumerate(units):
            self.units_name_dict[int(unit[6:])] = i
            self.names += [unit]
    
    
    def add_unit_info(self, unit_info: dict):

        """
        add the unit info 

        Parameters
        ----------

        unit_info : dict 

        Returns
        -------
        None 
        """

        self.unit_info = unit_info 

    def add_epochs(self, epoch: str, epochs_name: list):

        """
        add a list of epoch names

        Parameters
        ----------

        epoch : str
            name of the epoch 
        epochs_name : list 
            list of the epochs names of the added epoch 

        Returns 
        -------
        None 
        """

        self.epochs[epoch] = epochs_name 

        print(f"+{epoch} added")


    def get_indexes(self, queries: list):
        
        """
        query a list of unit names [numbers]
        
        Parameters
        ----------
        
        name_queries : list
            
        Returns
        -------
        indexes : list
            list of indexes corresponding to the unit names queried
        """
        
        return [self.units_name_dict[name] for name in queries]
    
    def print_names(self):
        
        """
        print the list of unit names
        """
        
        print('units:\n', self.names)


def plot_raster(trials, ylabel="Trials", id_start=0, ylim=None, ax=None, max_id=300, redbar=True, light_color=False, marker_size=None):
    """
    Raster plot of trials
    Parameters
    ----------
    trials : list of neo.SpikeTrains
    ylabel : str 
        default "Trials"
    id_start : int 
        default 0 
    ylim : tuple 
        default None
    ax : matplotlib axes
    max_id : int
        max id number for a trials, in case it rescales
    redbar : bool 
        if True, it displays the red bar of trials onset,
        default True
    light_color : bool
        if True, the spikes have a lighter color, usecase: background raster 
        plot; default False

    Returns
    -------
    out : axes
    """

    # default settings
    color = "black" if light_color else "#3498db"
    transparency = 0.6 if light_color else 1
    lw = 1
    marker = "|"
    marker_size = marker_size if marker_size is not None else 1

    if max_id is None:
        id_scale = len(trials)
    else:
        id_scale = max_id / len(trials)

    from matplotlib.ticker import MaxNLocator

    # get axis
    if ax is None:
        fig, ax = plt.subplots()
    
    # obtain spike-trains
    trial_id = []
    spikes = []
    try:
        dim = trials[0].times.dimensionality
    except AttributeError:
        print(type(trials[0]))
        raise AttributeError("trials[0] is not a neo.SpikeTrain")

    for n, trial in enumerate(trials):  # TODO what about empty trials?
        n += id_start 
        spikes.extend(trial.times.magnitude)
        trial_id.extend([n*id_scale]*len(trial.times))
    
    # marker setings
    if marker_size is None:
        heights = 6000./len(trials)
        if heights < 0.9:
            heights = 1.  # min size
    else:
        heights = marker_size

    #print('n ', max(trial_id))
    ax.scatter(spikes, trial_id, marker=marker, s=heights, lw=lw, color=color,
               edgecolors="face", alpha=transparency)
    if ylim is None:
        #ax.set_ylim(-0.5, len(trials)-0.5)
        pass
    elif ylim is True:
        #ax.set_ylim(ylim)
        pass
    else:
        pass

    if redbar:

        #y_ax = ax.axes.get_yaxis()  # Get X axis
        #y_ax.set_major_locator(MaxNLocator(integer=True))
        t_start = trials[0].t_start.rescale(dim)
        t_stop = trials[0].t_stop.rescale(dim)
        #ax.set_xlim([t_start, t_stop])
        ax.set_xlabel("Times [{}]".format(dim))
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    return ax, spikes


def compute_kde(trials: list, std: str, window: float, percentile=95, \
                t_start=0, plot=False, ax=None, plot_ax=False, plot_spikes=False, verbose=False):
    """
    compute the KDE over all trials and plot the curve   

    Paramters
    ---------
    trials : list
        list of class.SpikeTrain
    std : str, float or callable 
        [from numpy.gaussian_kde] bandwidth method, e.g. 'scott', 'silverman' 
    window : float 
        half width of the sample
    percentile : float
        defalt 95
    t_start : float 
        default 0
    plot : bool 
        default False 
    ax : matplotlib.axes
        default None 
    plot_ax : bool 
        if True return the matplotlib.axes, default False
    plot_spikes : bool
        if True and plot_ax is True, then spikes are plotted in the background,
        default False
    verbose : bool
        default False
    """

    scale = 1e-4 

    # collection of all spikes from each trial
    spikes = [s for trial in trials for s in trial.times.magnitude]
    if len(spikes) == 0:
        return [np.nan] * 5
    
    # calculate the kernel estimation object
    from scipy.stats import gaussian_kde
    kernel = gaussian_kde(spikes, std)
    if verbose:
        print('-kernel computed')

    # we start 10 % away from -window due to edge effects
    pre_times = np.arange(- window + window * 0.1, 0, scale) 
    times = np.arange(t_start, t_start + window, scale)
    all_times = np.arange(-window, window, scale)
    
    # percentiles
    i_percentile = np.percentile(kernel(pre_times), 100 - percentile, 0)
    e_percentile = np.percentile(kernel(pre_times), percentile, 0)
    if verbose:
        print('-percentile quantities computed') 
    
    if plot:

        import matplotlib.pyplot as plt
        
        if verbose:
            print('-plotting')
        
        # plot kernel
        plt.plot(all_times, kernel(all_times))
        plt.plot(pre_times, kernel(pre_times))
        
        # plot percentiles
        plt.plot(times, [i_percentile] * len(times))
        plt.plot(times, [e_percentile] * len(times))
        plt.show()
    
    if plot_ax:

        if verbose:
            print('-returning ax')

        if ax is None:
            from matplotlib.pyplot import subplots 
            fig, ax = subplots()

        if plot_spikes:
            ax.plot(spikes, np.ones(len(spikes))*10, '.k')

        # add kernel
        ker_all = kernel(all_times)
        ker_pre = kernel(pre_times)
        ax.plot(all_times, kernel(all_times))
        ax.plot(pre_times, kernel(pre_times))

        # add percentiles
        #ax.plot(times, [i_percentile] * len(times))
        #ax.plot(times, [e_percentile] * len(times))

        # descriptos
        ax.set_xlabel("Times [1/Hz]")
   

        ax.set_ylim((0, max((max(ker_all), max(ker_pre)))*1.1))

        return

    return times, spikes, kernel, e_percentile, i_percentile


def compute_kde_spikes(spikes: list, std: str, window: float, percentile=95, \
                       t_start=0, plot=False, ax=None, plot_ax=False, plot_spikes=False, verbose=False):

    """
    compute the KDE over all trials and plot the curve   

    Paramters
    ---------
    spikes : list
    std : str, float or callable 
        [from numpy.gaussian_kde] bandwidth method, e.g. 'scott', 'silverman' 
    window : float 
        half width of the sample
    percentile : float
        defalt 95
    t_start : float 
        default 0
    plot : bool 
        default False 
    ax : matplotlib.axes
        default None 
    plot_ax : bool 
        if True return the matplotlib.axes, default False
    plot_spikes : bool
        if True it plots spikes in the background, default False
    verbose : bool
        default False
    """
    
    scale = 1e-4

    # calculate the kernel estimation object
    from scipy.stats import gaussian_kde
    kernel = gaussian_kde(spikes, std)
    if verbose:
        print('-kernel computed')

    # we start 10 % away from -window due to edge effects
    pre_times = np.arange(- window + window * 0.1, 0, scale) 
    times = np.arange(t_start, t_start + window, scale)
    all_times = np.arange(-window, window, scale)
    
    # percentiles
    i_percentile = np.percentile(kernel(pre_times), 100 - percentile, 0)
    e_percentile = np.percentile(kernel(pre_times), percentile, 0)
    if verbose:
        print('-percentile quantities computed') 
    
    if plot:

        import matplotlib.pyplot as plt
        
        if verbose:
            print('-plotting')
        
        # plot kernel
        plt.plot(all_times, kernel(all_times))
        plt.plot(pre_times, kernel(pre_times))
        
        # plot percentiles
        plt.plot(times, [i_percentile] * len(times))
        plt.plot(times, [e_percentile] * len(times))
        plt.show()
   
    if plot_ax:

        if verbose:
            print('-returning ax')

        if ax is None:
            from matplotlib.pyplot import subplots 
            fig, ax = subplots()

        if plot_spikes:
            ax.plot(spikes, np.ones(len(spikes))*10, '.k')

        # add kernel
        ker_all = kernel(all_times)
        ker_pre = kernel(pre_times)
        ax.plot(all_times, ker_all)
        ax.plot(pre_times, ker_pre)

        # add percentiles
        ax.plot(times, [i_percentile] * len(times))
        ax.plot(times, [e_percentile] * len(times))

        # descriptos
        ax.set_xlabel("Times [1/Hz]")
        
        ax.set_ylabel("spikes/s")
        height = max((max(ker_all), max(ker_pre)))
        ax.set_ylim((0, height*1.1))
        return height

    return  times, spikes, kernel, e_percentile, i_percentile


class Data:
    def __init__(self, stim_mask=False, baseline_duration=None, project_path="", project=None, **kwargs):
        self.project_path = project_path
        self.params = kwargs
        self.project = project
        self.actions = self.project.actions
        self._spike_trains = {}
        self._templates = {}
        self._stim_times = {}
        self._unit_names = {}
        self._tracking = {}
        self._head_direction = {}
        self._lfp = {}
        self._occupancy = {}
        self._rate_maps = {}
        self._rate_maps_split = {}
        self._prob_dist = {}
        self._spatial_bins = None
        self.stim_mask = stim_mask
        self.baseline_duration = baseline_duration

    def data_path(self, action_id):
        return pathlib.Path(self.project_path) / "actions" / action_id / "data" / "main.exdir"

    def get_lim(self, action_id):
        stim_times = self.stim_times(action_id)
        if stim_times is None:
            if self.baseline_duration is None:
                return [0, get_duration(self.data_path(action_id))]
            else:
                return [0, self.baseline_duration]
        stim_times = np.array(stim_times)
        return [stim_times.min(), stim_times.max()]

    def duration(self, action_id):
        return get_duration(self.data_path(action_id))

    def tracking(self, action_id):
        if action_id not in self._tracking:
            x, y, t, speed = load_tracking(
                self.data_path(action_id),
                sampling_rate=self.params['position_sampling_rate'],
                low_pass_frequency=self.params['position_low_pass_frequency'],
                box_size=self.params['box_size'])
            if self.stim_mask:
                t1, t2 = self.get_lim(action_id)
                mask = (t >= t1) & (t <= t2)
                x = x[mask]
                y = y[mask]
                t = t[mask]
                speed = speed[mask]
            self._tracking[action_id] = {
                'x': x, 'y': y, 't': t, 'v': speed
            }
        return self._tracking[action_id]

    @property
    def spatial_bins(self):
        if self._spatial_bins is None:
            box_size_, bin_size_ = sp.maps._adjust_bin_size(
                box_size=self.params['box_size'],
                bin_size=self.params['bin_size'])
            xbins, ybins = sp.maps._make_bins(box_size_, bin_size_)
            self._spatial_bins = (xbins, ybins)
            self.box_size_, self.bin_size_ = box_size_, bin_size_
        return self._spatial_bins

    def occupancy(self, action_id):
        if action_id not in self._occupancy:
            xbins, ybins = self.spatial_bins

            occupancy_map = sp.maps._occupancy_map(
                self.tracking(action_id)['x'],
                self.tracking(action_id)['y'],
                self.tracking(action_id)['t'], xbins, ybins)
            threshold = self.params.get('occupancy_threshold')
            if threshold is not None:
                occupancy_map[occupancy_map <= threshold] = 0
            self._occupancy[action_id] = occupancy_map
        return self._occupancy[action_id]

    def prob_dist(self, action_id):
        if action_id not in self._prob_dist:
            xbins, ybins = xbins, ybins = self.spatial_bins
            prob_dist = sp.stats.prob_dist(
                self.tracking(action_id)['x'],
                self.tracking(action_id)['y'], bins=(xbins, ybins))
            self._prob_dist[action_id] = prob_dist
        return self._prob_dist[action_id]

    def rate_map_split(self, action_id, channel_group, unit_name, smoothing):
        make_rate_map = False
        if action_id not in self._rate_maps_split:
            self._rate_maps_split[action_id] = {}
        if channel_group not in self._rate_maps_split[action_id]:
            self._rate_maps_split[action_id][channel_group] = {}
        if unit_name not in self._rate_maps_split[action_id][channel_group]:
            self._rate_maps_split[action_id][channel_group][unit_name] = {}
        if smoothing not in self._rate_maps_split[action_id][channel_group][unit_name]:
            make_rate_map = True


        if make_rate_map:
            xbins, ybins = self.spatial_bins
            x, y, t = map(self.tracking(action_id).get, ['x', 'y', 't'])
            spikes = self.spike_train(action_id, channel_group, unit_name)
            t_split = t[-1] / 2
            mask_1 = t < t_split
            mask_2 = t >= t_split
            x_1, y_1, t_1 = x[mask_1], y[mask_1], t[mask_1]
            x_2, y_2, t_2 = x[mask_2], y[mask_2], t[mask_2]
            spikes_1 = spikes[spikes < t_split]
            spikes_2 = spikes[spikes >= t_split]
            occupancy_map_1 = sp.maps._occupancy_map(
                x_1, y_1, t_1, xbins, ybins)
            occupancy_map_2 = sp.maps._occupancy_map(
                x_2, y_2, t_2, xbins, ybins)

            spike_map_1 = sp.maps._spike_map(
                x_1, y_1, t_1, spikes_1, xbins, ybins)
            spike_map_2 = sp.maps._spike_map(
                x_2, y_2, t_2, spikes_2, xbins, ybins)

            smooth_spike_map_1 = sp.maps.smooth_map(
                spike_map_1, bin_size=self.bin_size_, smoothing=smoothing)
            smooth_spike_map_2 = sp.maps.smooth_map(
                spike_map_2, bin_size=self.bin_size_, smoothing=smoothing)
            smooth_occupancy_map_1 = sp.maps.smooth_map(
                occupancy_map_1, bin_size=self.bin_size_, smoothing=smoothing)
            smooth_occupancy_map_2 = sp.maps.smooth_map(
                occupancy_map_2, bin_size=self.bin_size_, smoothing=smoothing)

            rate_map_1 = smooth_spike_map_1 / smooth_occupancy_map_1
            rate_map_2 = smooth_spike_map_2 / smooth_occupancy_map_2
            self._rate_maps_split[action_id][channel_group][unit_name][smoothing] = [rate_map_1, rate_map_2]

        return self._rate_maps_split[action_id][channel_group][unit_name][smoothing]

    def rate_map(self, action_id, channel_group, unit_name, smoothing):
        make_rate_map = False
        if action_id not in self._rate_maps:
            self._rate_maps[action_id] = {}
        if channel_group not in self._rate_maps[action_id]:
            self._rate_maps[action_id][channel_group] = {}
        if unit_name not in self._rate_maps[action_id][channel_group]:
            self._rate_maps[action_id][channel_group][unit_name] = {}
        if smoothing not in self._rate_maps[action_id][channel_group][unit_name]:
            make_rate_map = True


        if make_rate_map:
            xbins, ybins = self.spatial_bins

            spike_map = sp.maps._spike_map(
                self.tracking(action_id)['x'],
                self.tracking(action_id)['y'],
                self.tracking(action_id)['t'],
                self.spike_train(action_id, channel_group, unit_name),
                xbins, ybins)

            smooth_spike_map = sp.maps.smooth_map(
                spike_map, bin_size=self.bin_size_, smoothing=smoothing)
            smooth_occupancy_map = sp.maps.smooth_map(
                self.occupancy(action_id), bin_size=self.bin_size_, smoothing=smoothing)
            rate_map = smooth_spike_map / smooth_occupancy_map
            self._rate_maps[action_id][channel_group][unit_name][smoothing] = rate_map

        return self._rate_maps[action_id][channel_group][unit_name][smoothing]

    def head_direction(self, action_id):
        if action_id not in self._head_direction:
            a, t = load_head_direction(
                self.data_path(action_id),
                sampling_rate=self.params['position_sampling_rate'],
                low_pass_frequency=self.params['position_low_pass_frequency'],
                box_size=self.params['box_size'])
            if self.stim_mask:
                t1, t2 = self.get_lim(action_id)
                mask = (t >= t1) & (t <= t2)
                a = a[mask]
                t = t[mask]
            self._head_direction[action_id] = {
                'a': a, 't': t
            }
        return self._head_direction[action_id]

    def lfp(self, action_id, channel_group, clean_memory=False):
        lim = self.get_lim(action_id) if self.stim_mask else None
        if clean_memory:
            return load_lfp(
            self.data_path(action_id), channel_group, lim)
        if action_id not in self._lfp:
            self._lfp[action_id] = {}
        if channel_group not in self._lfp[action_id]:
            self._lfp[action_id][channel_group] = load_lfp(
                self.data_path(action_id), channel_group, lim)
        return self._lfp[action_id][channel_group]

    def template(self, action_id, channel_group, unit_id):
        if action_id not in self._templates:
            self._templates[action_id] = {}
        if channel_group not in self._templates[action_id]:
            lim = self.get_lim(action_id) if self.stim_mask else None
            self._templates[action_id][channel_group] = {
                get_unit_id(st): Template(st)
                for st in load_spiketrains(
                    self.data_path(action_id), channel_group,
                    load_waveforms=True,
                    lim=lim)
            }
        return self._templates[action_id][channel_group][unit_id]

    def spike_train(self, action_id, channel_group, unit_id):
        self.spike_trains(action_id, channel_group)
        return self._spike_trains[action_id][channel_group][unit_id]

    def spike_trains(self, action_id, channel_group):
        if action_id not in self._spike_trains:
            self._spike_trains[action_id] = {}
        if channel_group not in self._spike_trains[action_id]:
            lim = self.get_lim(action_id) if self.stim_mask else None
            self._spike_trains[action_id][channel_group] = {
                get_unit_id(st): st
                for st in load_spiketrains(
                    self.data_path(action_id), channel_group,
                    load_waveforms=False,
                    lim=lim)
            }
        return self._spike_trains[action_id][channel_group]

    def unit_names(self, action_id, channel_group):
        if action_id not in self._unit_names:
            self._unit_names[action_id] = {}
        if channel_group not in self._unit_names[action_id]:
            self._unit_names[action_id][channel_group] = [
                get_unit_id(st)
                for st in load_unit_annotations(
                    self.data_path(action_id), channel_group)
            ]
        return self._unit_names[action_id][channel_group]

    def stim_times(self, action_id):
        if action_id not in self._stim_times:
            epochs = load_epochs(self.data_path(action_id))
            # Major hack (Malin), very different epochs in this script from my epochs,
            # none I need to relate to tracking atm so I changed "if len(epochs) == 0" to 3, because I have 3 epochs, and also 2 because might be 2..
            if len(epochs) == 3:
                self._stim_times[action_id] = None
            elif len(epochs) == 2:
                self._stim_times[action_id] = None
            elif len(epochs) == 1:
                stim_times = epochs[0]
                stim_times = np.sort(np.abs(stim_times))
                # there are some 0 times and inf times, remove those
                stim_times = stim_times[stim_times <= get_duration(self.data_path(action_id))]
                # stim_times = stim_times[stim_times >= 1e-20]
                self._stim_times[action_id] = stim_times
            else:
                raise ValueError('Found multiple epochs')
        return self._stim_times[action_id]


def load_tracking(data_path, sampling_rate, low_pass_frequency, box_size, velocity_threshold=5):
    x1, y1, t1, x2, y2, t2, stop_time = load_leds(data_path)
    x1, y1, t1 = rm_nans(x1, y1, t1)
    x2, y2, t2 = rm_nans(x2, y2, t2)

    x1, y1, t1 = filter_t_zero_duration(x1, y1, t1, stop_time.magnitude)
    x2, y2, t2 = filter_t_zero_duration(x2, y2, t2, stop_time.magnitude)

    # select data with least nan
    if len(x1) > len(x2):
        x, y, t = x1, y1, t1
    else:
        x, y, t = x2, y2, t2

    # OE saves 0.0 when signal is lost, these can be removed
    x, y, t = filter_xy_zero(x, y, t)

    # x, y, t = filter_xy_box_size(x, y, t, box_size)

    # remove velocity artifacts
    x, y, t = velocity_filter(x, y, t, velocity_threshold)

    x, y, t = interp_filt_position(
        x, y, t, fs=sampling_rate, f_cut=low_pass_frequency)

    check_valid_tracking(x, y, box_size)

    vel = np.gradient([x, y], axis=1) / np.gradient(t)
    speed = np.linalg.norm(vel, axis=0)

    return x, y, t, speed

