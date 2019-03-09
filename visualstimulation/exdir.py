import visualstimulation as vs
import exdir.plugins.git_lfs
import pathlib
import expipe
import exdir
import neo
import os


def get_data_path(action):
    action_path = action._backend.path
    project_path = action_path.parent.parent
    print(project_path)
    # data_path = action.data['main']
    data_path = str(pathlib.Path(pathlib.PureWindowsPath(action.data['main'])))
    print(data_path)
    return project_path / data_path


def load_epochs(data_path):
    io = neo.ExdirIO(str(data_path), plugins=[exdir.plugins.quantities, exdir.plugins.git_lfs])
    blk = io.read_block()
    seg = blk.segments[0]
    epochs = seg.epochs
    return epochs


def load_spiketrains(data_path, channel_idx):
    io = neo.ExdirIO(data_path, plugins=[exdir.plugins.quantities, exdir.plugins.git_lfs.Plugin(verbose=True)])
    blk = io.read_block()
    channels = blk.channel_indexes
    chx = channels[channel_idx]
    sptr = [u.spiketrains[0] for u in chx.units]
    return sptr
