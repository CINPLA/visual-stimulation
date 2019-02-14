import matplotlib.pyplot as plt
import quantities as pq
import expipe

from .openephys import *
from .utils import make_spiketrain_trials, add_orientation_to_trials
from .plot import orient_raster_plots, plot_tuning_overview


def osi_analysis(project_path, action_id, n_channel=8, raster_start=0.5):
    # Define project tree
    project = expipe.get_project(project_path)
    action = project.actions[action_id]
    data_path = get_data_path(action)
    epochs = load_epochs(data_path)

    # Create branch for figures
    exdir_file = exdir.File(data_path, plugins=exdir.plugins.quantities)
    figures_group = exdir_file.require_group('figures')

    for channel in range(n_channel):
        channel_name = "channel_{}".format(channel)
        channel_group = figures_group.require_group(channel_name)
        channel_path = os.path.join(project_path, "figures\\"  + channel_name)

        # Define project dependencies
        spiketrains = load_spiketrains(str(data_path), channel)

        # Get orrientation data
        oe_epoch = epochs[0]       # openephys
        assert(oe_epoch['provenance'] == 'open_ephys')

        ps_epoch = epochs[1]       # psychopy
        assert(ps_epoch['provenance'] == 'psychopy')
    
        orients = ps_epoch.labels       # the labels are orrientations (135, 90, ...)

        # Create figures from spiketrains
        raster_start = raster_start * pq.s
        for s_id, spiketrain in enumerate(spiketrains):
            figure_id = "{}_{}_".format(channel, s_id)

            # Raster plot processing
            trials = make_spiketrain_trials(spiketrain, oe_epoch, t_start=raster_start)
            add_orientation_to_trials(trials, orients)
            orf_path = os.path.join(channel_path, figure_id + "orrient_raster.png")
            orient_raster_fig = orient_raster_plots(trials)
            orient_raster_fig.savefig(orf_path)

            # Orrientation vs spikefrequency plot (tuning curves) processing
            trials = make_spiketrain_trials(spiketrain, oe_epoch)
            add_orientation_to_trials(trials, orients)
            tf_path = os.path.join(channel_path, figure_id + "tuning.png")
            tuning_fig = plot_tuning_overview(trials)
            tuning_fig.savefig(tf_path)

            # Reset before next loop to save memory
            plt.close(fig="all")
