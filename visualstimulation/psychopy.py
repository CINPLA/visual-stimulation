import matplotlib.pyplot as plt
import quantities as pq
import seaborn as sns
import expipe

from numpy import empty

from .openephys import *
from .utils import make_spiketrain_trials, add_orientation_to_trials
from .plot import orient_raster_plots, plot_tuning_overview


def psycho_plot(project_path, action_id, n_channel=8, rem_channel="all", skip_channels=None, raster_start=-0.5, raster_stop=1):
    """
    Plot raster, isi-mean-median, tuning (polar and linear) from visual data acquired through open-ephys and psychopy
    Parameters
    ----------
    project_path: os.path, or equivallent
        Path to project directory
    action_id: string
        The experiment/action id
    n_channel: default 8; int
        Number of channel groups in recordings
    rem_channel: default "all"; "all" or int
        Signify what channels are of interest(=rem)
    skip_channels: default None; int or (list, array, tupule)
        ID of channel groups to skip
    raster_start: default -0.5; int or float
        When the rasters start on the x-axis relative to trial start, which is 0
    raster_stop: default 1; int or float
        When the rasters stop on the x-axis relative to trial start, which is 0
    Returns
    -------
    savesfigures into exdir directory: main.exdir/figures
    """
    if not (rem_channel == "all" or (isinstance(rem_channel, int) and rem_channel < n_channel)):
        msg = "rem_channel must be either 'all' or integer between 0 and n_channel ({}); not {}".format(
            n_channel, rem_channel
        )
        raise AttributeError(msg)
    # Define project tree
    project = expipe.get_project(project_path)
    action = project.actions[action_id]
    data_path = get_data_path(action)
    epochs = load_epochs(data_path)

    # Get data of interest (orients vs rates vs channel)
    oe_epoch = epochs[0]       # openephys
    assert(oe_epoch.annotations['provenance'] == 'open-ephys')
    ps_epoch = epochs[1]       # psychopy
    assert(ps_epoch.annotations['provenance'] == 'psychopy')

    # Create directory for figures
    exdir_file = exdir.File(data_path, plugins=exdir.plugins.quantities)
    figures_group = exdir_file.require_group('figures')

    raster_start = raster_start * pq.s
    raster_stop = raster_stop * pq.s
    orients = ps_epoch.labels       # the labels are orrientations (135, 90, ...)

    def plot(channel_num, channel_path, spiketrains):
        # Create figures from spiketrains
        for spiketrain in spiketrains:
            try:
                if spiketrain.annotations["cluster_group"] == "noise":
                    continue
            except KeyError:
                msg = "Cluster/channel group {} seems to not have been sorted in phys".format(channel_num)
                raise KeyError(msg)
            
            figure_id = "{}_{}_".format(channel_num, spiketrain.annotations['cluster_id'])

            sns.set()
            sns.set_style("white")
            # Raster plot processing
            trials = make_spiketrain_trials(spiketrain, oe_epoch, t_start=raster_start, t_stop=raster_stop)
            add_orientation_to_trials(trials, orients)
            orf_path = os.path.join(channel_path, figure_id + "orrient_raster.svg")
            orient_raster_fig = orient_raster_plots(trials)
            orient_raster_fig.savefig(orf_path)

            # Orrientation vs spikefrequency plot (tuning curves) processing
            trials = make_spiketrain_trials(spiketrain, oe_epoch)
            add_orientation_to_trials(trials, orients)
            tf_path = os.path.join(channel_path, figure_id + "tuning.svg")
            tuning_fig = plot_tuning_overview(trials)
            tuning_fig.savefig(tf_path)

            # Reset before next loop to save memory
            plt.close(fig="all")

    if rem_channel == "all":
        channels = range(n_channel)
        if isinstance(skip_channels, int):
            channels = [x for x in channels]
            del channels[skip_channels]
        elif isinstance(skip_channels, (list, tuple, type(empty(0)))):
            channels = [x for x in channels]
            for channel in skip_channels:
                del channels[channel]
        for channel in channels:
            channel_name = "channel_{}".format(channel)
            channel_group = figures_group.require_group(channel_name)
            channel_path = os.path.join(str(data_path), "figures\\"  + channel_name)

            spiketrains = load_spiketrains(str(data_path), channel)

            plot(channel, channel_path, spiketrains)

    elif isinstance(rem_channel, int):
        channel_name = "channel_{}".format(rem_channel)
        channel_group = figures_group.require_group(channel_name)
        channel_path = os.path.join(str(data_path), "figures\\"  + channel_name)

        spiketrains = load_spiketrains(str(data_path), rem_channel)

        plot(rem_channel, channel_path, spiketrains)
