import matplotlib.pyplot as plt
import exdir.plugins.quantities
import exdir.plugins.git_lfs
import pathlib
import expipe
import exdir
import neo
import os

from .openephys import *
from .utils import make_spiketrain_trials, add_orientation_to_trials
from .plot import orient_raster_plots, plot_tuning_overview


class visual_data:
    def __init__(self, project_path, action_id, n_channel=8, outpath="."):
        self.id_path = os.path.join(outpath, str(action_id))
        try:
            os.mkdir(self.id_path)
        except:
            print("Path already exists: {}".format(str(self.id_path)))
        
        self.action_id = action_id
        self.n_channel = n_channel

        self.project = expipe.get_project(project_path)
        self.action = self.project.actions[action_id]
        self.data_path = get_data_path(self.action)
        self.epochs = load_epochs(self.data_path)

    def get_plot():
        for channel in range(1, n_channel+1):
            channel_path = os.path.join(self.id_path, "Channel_{}".format(str(channel)))

            # Define project dependencies
            spiketrains = load_spiketrains(self.data_path, channel)

            # Get orrientation data
            oe_epoch = self.epochs[0]       # openephys
            ps_epoch = self.epochs[1]       # psychopy

            for s_id, spiketrain in enumerate(spiketrains):
                trials = make_spiketrain_trials(spiketrain, oe_epoch)
                add_orientation_to_trials(trials, orients)

                orf_path = os.path.join(channel_path, "{}_{}_orrient_raster.png".format(channel, s_id))
                orient_raster_fig = orient_raster_plots(trials)
                orient_raster_fig.savefig(orf_path)

                tf_path = os.path.join(channel_path, "{}_{}_tuning.png".format(channel, s_id))
                tuning_fig = plot_tuning_overview(trials)
                tuning_fig.savefig(tf_path)
