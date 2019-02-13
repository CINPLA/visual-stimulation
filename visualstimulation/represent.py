import matplotlib.pyplot as plt
import expipe
import os

from .openephys import *
from .utils import make_spiketrain_trials, add_orientation_to_trials
from .plot import orient_raster_plots, plot_tuning_overview


def os_plots(project_path, action_id, n_channel=8, t_start=0.5):
    project_path = os.path.realpath(project_path)
    
    # Define project tree
    project = expipe.get_project(project_path)
    action = project.actions[action_id]
    data_path = get_data_path(action)
    epochs = load_epochs(data_path)

    # Create branch for figures
    analysis_group = data_path.require_group('analysis')
    figures_group = analysis_group.require_group('figures')

    for channel in range(n_channel):
        channel_group = figures_group.require_group('channel_{}'.format(channel))
        channel_path = get_data_path(channel_group)

        # Define project dependencies
        spiketrains = load_spiketrains(str(data_path), channel)

        # Get orrientation data
        oe_epoch = epochs[0]            # openephys
        ps_epoch = epochs[1]            # psychopy
        orients = ps_epoch.labels       # the labels are orrientations (135, 90, ...)

        for s_id, spiketrain in enumerate(spiketrains):
            # Raster plot processing
            trials = make_spiketrain_trials(spiketrain, oe_epoch, t_start=t_start)
            add_orientation_to_trials(trials, orients)
            orf_path = os.path.join(channel_path, "{}_{}_orrient_raster.png".format(channel, s_id))
            orient_raster_fig = orient_raster_plots(trials)
            orient_raster_fig.savefig(orf_path)

            # Orrientation vs spikefrequency plot (tuning curves) processing
            trials = make_spiketrain_trials(spiketrain, oe_epoch)
            add_orientation_to_trials(trials, orients)
            tf_path = os.path.join(channel_path, "{}_{}_tuning.png".format(channel, s_id))
            tuning_fig = plot_tuning_overview(trials)
            tuning_fig.savefig(tf_path)

            plt.close(fig='all')
