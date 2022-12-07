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


def get_intersection_orientations(epoch_0, epoch_1, epoch_orient):

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


    angles = [x for x in range(0, 360, 45)]

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

        epoch_0_new.name = f'laser_epoch_vis_{angles[i]}°'
        epoch_matrix += [epoch_0_new]

    return epoch_matrix 


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


def plot_raster(trials, ylabel="Trials", id_start=0, ylim=None, ax=None, max_id=300, redbar=True, light_color=False):
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
    color = "gray" if light_color else "#3498db"
    transparency = 0.5 if light_color else 1
    lw = 1
    marker = "|"
    marker_size = 45 
    id_scale = max_id / len(trials)

    from matplotlib.ticker import MaxNLocator
    if ax is None:
        fig, ax = plt.subplots()
    trial_id = []
    spikes = []
    dim = trials[0].times.dimensionality
    for n, trial in enumerate(trials):  # TODO what about empty trials?
        n += id_start 
        spikes.extend(trial.times.magnitude)
        trial_id.extend([n*id_scale]*len(trial.times))
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
        ax.set_ylim(-0.5, len(trials)-0.5)
    elif ylim is True:
        ax.set_ylim(ylim)
    else:
        pass

    if redbar:

        #y_ax = ax.axes.get_yaxis()  # Get X axis
        #y_ax.set_major_locator(MaxNLocator(integer=True))
        t_start = trials[0].t_start.rescale(dim)
        t_stop = trials[0].t_stop.rescale(dim)
        ax.set_xlim([t_start, t_stop])
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
        ax.plot(all_times, kernel(all_times))
        ax.plot(pre_times, kernel(pre_times))

        # add percentiles
        ax.plot(times, [i_percentile] * len(times))
        ax.plot(times, [e_percentile] * len(times))

        # descriptos
        ax.set_xlabel("Times [1/Hz]")
   
        ax.set_ylim((0, 50))

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
        ax.plot(all_times, kernel(all_times))
        ax.plot(pre_times, kernel(pre_times))

        # add percentiles
        ax.plot(times, [i_percentile] * len(times))
        ax.plot(times, [e_percentile] * len(times))

        # descriptos
        ax.set_xlabel("Times [1/Hz]")
    
        return

    return times, spikes, kernel, e_percentile, i_percentile


