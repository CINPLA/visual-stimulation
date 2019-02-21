import numpy as np
import matplotlib.pyplot as plt
from .utils import make_orientation_trials

def polar_tuning_curve(orients, rates, ax=None, params={}):
    """
    Direction polar tuning curve
    """
    import math

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


def plot_tuning_overview(trials, spontan_rate=None, weights=(1, 0.6)):
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
    from .analysis import (compute_orientation_tuning, compute_osi, compute_rosi, compute_dsi, compute_circular_variance)
    fig = plt.figure(figsize=(21, 9))

    ax1 = fig.add_subplot(1, 2, 1)
    trials = make_orientation_trials(trials)
    rates, orients = compute_orientation_tuning(trials)
    pref_or = orients[np.argmax(rates)]
    
    """ Analytical parameters """
    # Non-Weighed
    osi = compute_osi(rates, orients, weight=False)

    rosi = compute_rosi(rates, orients, weight=False)
    dsi = compute_dsi(rates, orients, weight=False)
    cv = compute_circular_variance(rates, orients, weight=False)
    # Weighed
    w_osi = compute_osi(rates, orients, weight=True, weights=weights)
    w_rosi = compute_rosi(rates, orients, weight=True, weights=weights)
    w_dsi = compute_dsi(rates, orients, weight=True, weights=weights)
    w_cv = compute_circular_variance(rates, orients, weight=True, weights=weights)

    title_1 = "PO={}\n".format(pref_or)
    title_2 = "Non-weighed: OSI={:.2f}  CV={:.2f}  DSI={:.2f}  rOSI={:.2f}\n".format(osi, cv, dsi, rosi)
    title_3 = "Weighed:     OSI={:.2f}  CV={:.2f}  DSI={:.2f}  rOSI={:.2f}".format(w_osi, w_cv, w_dsi, w_rosi)
    fig.suptitle(title_1 + title_2 + title_3, fontsize=17)
    ax1.plot(orients, rates, "-o", label="with bkg")
    ax1.set_xticks(orients.magnitude)
    ax1.set_xlabel("Orientation angle (deg)")
    ax1.set_ylabel("Rate (Hz)")

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    polar_tuning_curve(orients.rescale("rad"), rates, ax=ax2)

    if spontan_rate is not None:
        ax1.plot(orients, rates - spontan_rate, "-o", label="without bkg")
        ax1.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.87])

    return fig


def orient_raster_plots(trials):
    """
    Makes raster plot for each stimulus orientation

    Parameters
    ----------
    trials : list
        list of neo.SpikeTrain
    """

    orient_trials = make_orientation_trials(trials)
    col_count = 4
    row_count = int(np.ceil(len(orient_trials))/col_count)
    fig = plt.figure(figsize=(4*col_count, 2*row_count))
    for i, (orient, trials) in enumerate(orient_trials.items()):
        ax = fig.add_subplot(row_count, col_count, i+1)
        ax = plot_raster(trials, ax=ax)
        ax.set_title(orient)
        ax.grid(False)
    print(i)
    fig.tight_layout()

    return fig


def plot_raster(trials, color="#3498db", lw=1, ax=None, marker='|', marker_size=20,
                ylabel='Trials', id_start=0, ylim=None):
    """
    Raster plot of trials
    Parameters
    ----------
    trials : list of neo.SpikeTrains
    color : color of spikes
    lw : line width
    ax : matplotlib axes
    Returns
    -------
    out : axes
    """
    from matplotlib.ticker import MaxNLocator
    if ax is None:
        fig, ax = plt.subplots()
    trial_id = []
    spikes = []
    dim = trials[0].times.dimensionality
    for n, trial in enumerate(trials):  # TODO what about empty trials?
        n += id_start
        spikes.extend(trial.times.magnitude)
        trial_id.extend([n]*len(trial.times))
    if marker_size is None:
        heights = 6000./len(trials)
        if heights < 0.9:
            heights = 1.  # min size
    else:
        heights = marker_size
    ax.scatter(spikes, trial_id, marker=marker, s=heights, lw=lw, color=color,
               edgecolors='face')
    if ylim is None:
        ax.set_ylim(-0.5, len(trials)-0.5)
    elif ylim is True:
        ax.set_ylim(ylim)
    else:
        pass
    y_ax = ax.axes.get_yaxis()  # Get X axis
    y_ax.set_major_locator(MaxNLocator(integer=True))
    t_start = trials[0].t_start.rescale(dim)
    t_stop = trials[0].t_stop.rescale(dim)
    ax.set_xlim([t_start, t_stop])
    ax.set_xlabel("Times [{}]".format(dim))
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax
