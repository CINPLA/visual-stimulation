import numpy as np 
import matplotlib.pyplot as plt
import quantities as pq 
import exdir
import exdir.plugins.git_lfs
import neo
import pathlib



def check_valid_tracking(x, y, box_size):
    if np.isnan(x).any() and np.isnan(y).any():
        raise ValueError('nans found in  position, ' +
            'x nans = %i, y nans = %i' % (sum(np.isnan(x)), sum(np.isnan(y))))

    if (x.min() < 0 or x.max() > box_size[0] or y.min() < 0 or y.max() > box_size[1]):
        warnings.warn(
            "Invalid values found " +
            "outside box: min [x, y] = [{}, {}], ".format(x.min(), y.min()) +
            "max [x, y] = [{}, {}]".format(x.max(), y.max()))


def interp_filt_position(x, y, tm, fs=100 , f_cut=10 ):
    """
    rapid head movements will contribute to velocity artifacts,
    these can be removed by low-pass filtering
    see http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876586/
    code addapted from Espen Hagen
    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    tm : quantities.Quantity array in s
        1d vector of times at x, y positions
    fs : quantities scalar in Hz
        return radians
    Returns
    -------
    out : angles, resized t
    """
    import scipy.signal as ss
    assert len(x) == len(y) == len(tm), 'x, y, t must have same length'
    t = np.arange(tm.min(), tm.max() + 1. / fs, 1. / fs)
    x = np.interp(t, tm, x)
    y = np.interp(t, tm, y)
    # rapid head movements will contribute to velocity artifacts,
    # these can be removed by low-pass filteringpar
    # see http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876586/
    # code addapted from Espen Hagen
    b, a = ss.butter(N=1, Wn=f_cut * 2 / fs)
    # zero phase shift filter
    x = ss.filtfilt(b, a, x)
    y = ss.filtfilt(b, a, y)
    # we tolerate small interpolation errors
    x[(x > -1e-3) & (x < 0.0)] = 0.0
    y[(y > -1e-3) & (y < 0.0)] = 0.0

    return x, y, t


def velocity_filter(x, y, t, threshold):
    """
    Removes values above threshold
    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    t : quantities.Quantity array in s
        1d vector of times at x, y positions
    threshold : float
    """
    assert len(x) == len(y) == len(t), 'x, y, t must have same length'
    vel = np.gradient([x, y], axis=1) / np.gradient(t)
    speed = np.linalg.norm(vel, axis=0)
    speed_mask = (speed < threshold)
    speed_mask = np.append(speed_mask, 0)
    x = x[np.where(speed_mask)]
    y = y[np.where(speed_mask)]
    t = t[np.where(speed_mask)]
    return x, y, t


def filter_xy_zero(x, y, t):
    idxs, = np.where((x == 0) & (y == 0))
    return [np.delete(a, idxs) for a in [x, y, t]]


def filter_xy_box_size(x, y, t, box_size):
    idxs, = np.where((x > box_size[0]) | (x < 0) | (y > box_size[1]) | (y < 0))
    return [np.delete(a, idxs) for a in [x, y, t]]


def filter_t_zero_duration(x, y, t, duration):
    idxs, = np.where((t < 0) | (t > duration))
    return [np.delete(a, idxs) for a in [x, y, t]]


def rm_nans(*args):
    """
    Removes nan from all corresponding arrays
    Parameters
    ----------
    args : arrays, lists or quantities which should have removed nans in
           all the same indices
    Returns
    -------
    out : args with removed nans
    """

    nan_indices = []
    for arg in args:
        nan_indices.extend(np.where(np.isnan(arg))[0].tolist())
    nan_indices = np.unique(nan_indices)
    out = []
    for arg in args:
        # Mikkel, I hacked a fix here (JV)
        if nan_indices.size is not 0:
            out.append(np.delete(arg, nan_indices))
        else:
            out.append(arg)
    return out


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


def load_leds(data_path):
    root_group = exdir.File(
        data_path, "r",
        plugins=[exdir.plugins.quantities, exdir.plugins.git_lfs])

    # tracking data
    position_group = root_group['processing']['tracking']['camera_0']['Position']
    stop_time = position_group.attrs["stop_time"]
    x1, y1 = position_group['led_0']['data'].data.T
    t1 = position_group['led_0']['timestamps'].data
    x2, y2 = position_group['led_1']['data'].data.T
    t2 = position_group['led_1']['timestamps'].data

    return x1, y1, t1, x2, y2, t2, stop_time


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



### functions for path length and speed

def total_length(X:  np.ndarray, Y:  np.ndarray):

    """
    Compute the total length of a path.

    Parameters
    ----------
    X : np.ndarray
        x coordinates
    Y : np.ndarray
        y coordinates

    Returns
    -------
    length : float
    """
    
    n = len(X)
    length = 0
    for i in range(1, n):
        
        # sum of distances between successive points
        length += np.sqrt((X[i] - X[i-1])**2 + (Y[i] - Y[i-1])**2)
        
    return length

def binned_speed(X:  np.ndarray, Y:  np.ndarray, T:  np.ndarray, nbins: int):

    """
    Compute the speed of a path in bins.

    Parameters
    ----------
    X : np.ndarray
        x coordinates
    Y : np.ndarray
        y coordinates
    T : np.ndarray
        time
    nbins : int
        number of bins

    Returns
    -------
    bins_len : np.ndarray
    """
    
    #nbins -= 1 # to adjust for eventual remainders
    
    n = len(X)
    bsize, rem = n // nbins, n % nbins  # compute bin size and the remainder
    
    max_steps = nbins * bsize
    
    bins_len = np.zeros(nbins) 
    bins_tim = np.zeros(nbins)  # the first position's time is not counted
    
    length = 0
    for i in range(1, n):
        
        if i >= max_steps:
            break
        
        # sum of distances between successive points and add to the current bin        
        bins_len[i//bsize] += np.sqrt((X[i] - X[i-1])**2 + (Y[i] - Y[i-1])**2)
        
        # add times to the current bin
        bins_tim[i//bsize] += T[i] - T[i-1]
       
    bins_speed = bins_len / bins_tim
        
    return np.around(bins_speed, 2), np.around(bins_len), np.around(bins_tim), bsize*bins_tim[0]//1000
    

def plot_binned_speed(tracking_data: dict, nbins: int, title="", save=False, figpath=None, idx=0, return_data=False):

    """
    Plot the speed of a path in bins.

    Parameters
    ----------
    tracking_data : dict
        tracking data
    nbins : int
        number of bins
    title : str
    save : bool
        if true, save the figure
    figpath : str
        path to save the figure
    idx : int
        index of the figure

    Returns
    -------
    None 
    """

    # fixed box size, change if needed
    BOX_SIZE = 28

    x, y, t, _ = tracking_data.values()

    # total length ran by the subject
    tot_len = total_length(X=x*BOX_SIZE, Y=y*BOX_SIZE) 

    # average speed as total length divided by the session duration
    avg_speed_unif = tot_len / max(t)

    # binned length
    nbins = 10
    bspeeds, blens, btims, bsize = binned_speed(X=x*BOX_SIZE, Y=y*BOX_SIZE, T=t, nbins=nbins)

    # binned speed

    print(f"\ntotal length: {tot_len:.2f} cm")
    print(f"average uniform speed: {avg_speed_unif:.2f} cm/s")
    print()
    

    # plots

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{title} metrics for {nbins} bins")
    fig.set_tight_layout(tight={'h_pad': 1})

    xa = range(1, nbins+1)
    axs[0].plot(xa, bspeeds, label='average speed')
    axs[0].legend()
    axs[0].set_xticks(xa)
    #axs[0].grid()
    axs[0].set_ylabel('cm/s')

    axs[1].plot(xa, blens, label='distance')
    axs[1].grid()
    axs[1].set_ylabel('cm')
    #axs[1].set_ylim(0, max(blens)+100)
    axs[1].set_xticks(xa)
    axs[1].set_xticklabels([f"{s:.0f} s" for s in np.around(np.arange(0, (nbins)*bsize, bsize))], fontsize=13)
    axs[1].set_xlabel('bins and seconds')
    axs[1].legend()
    axs[1].grid()

    plt.show()

    print('- '*45)

    # save figure
    if save and figpath:
        name = f"/{title}_speed_{idx}.png"
        fig.savefig(figpath + name)

    if return_data:
        return avg_speed_unif, tot_len 
