def polar_tuning_curve(orients, rates, ax=None, params={}):
    """
    Direction polar tuning curve
    """
    import numpy as np
    import math
    import pretty_plotting

    assert len(orients) == len(rates)

    if ax is None:
        fig, ax = plt.subplots()
        ax = plt.subplot(111, projection='polar')

    ax.plot(orients, rates, '-', **params)
    ax.fill(orients, rates, alpha=1)
    ax.set_yticklabels([])
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    return ax


def plot_tuning_overview(trials, spontan_rate=None):
    """
    Makes orientation tuning plots (line and polar plot)
    for each stimulus orientation.

    Parameters
    ----------
    trials : list
        list of neo.SpikeTrain
    spontan_rates : defaultdict(dict), optional
        rates[channel_index_name][unit_id] = spontaneous firing rate trials.
    """
    import seaborn
    from visualstimulation.analysis import (make_orientation_trials,
                                            compute_orientation_tuning,
                                            compute_osi)

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    trials = make_orientation_trials(trials)
    rates, orients = compute_orientation_tuning(trials)
    preferred_orient, index = compute_osi(rates, orients)

    ax1.set_title("Preferred orientation={},\n OSI={}".format(preferred_orient,
                                                              round(index, 2)))
    ax1.plot(orients, rates, "-o", label="with bkg")
    ax1.set_xlabel("Orientation")
    ax1.set_ylabel("Rate (1/s)")

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    polar_tuning_curve(orients.rescale("rad"), rates, ax=ax2)

    if spontan_rate is not None:
        ax1.plot(orients, rates - spontan_rate, "-o", label="without bkg")
        ax1.legend()

    fig.tight_layout()

    return fig


def orient_raster_plots(trials):
    """
    Makes raster plot for each stimulus orientation

    Parameters
    ----------
    trials : list
        list of neo.SpikeTrain
    """
    import seaborn
    orient_trials = make_orientation_trials(trials)
    col_count = 4
    row_count = int(np.ceil(len(orient_trials))/col_count)
    fig = plt.figure(figsize=(2*col_count, 2*row_count))
    for i, (orient, trials) in enumerate(orient_trials.items()):
        ax = fig.add_subplot(row_count, col_count, i+1)
        ax = plot_raster(trials, ax=ax)
        ax.set_title(orient)
        ax.grid(False)
    fig.tight_layout()

    return fig
