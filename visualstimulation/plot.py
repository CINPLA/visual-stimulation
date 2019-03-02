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
    from .analysis import (compute_orientation_tuning, compute_osi, compute_dsi, compute_circular_variance)
    from elephant.statistics import isi
    import seaborn

    fig = plt.figure(figsize=(21, 9))
    trials = make_orientation_trials(trials)
    
    """ Analytical parameters """
    # Non-Weighed
    rates, orients = compute_orientation_tuning(trials)

    pref_or = orients[np.argmax(rates)]
    osi = compute_osi(rates, orients)
    rosi = compute_osi(rates, orients, relative=True)
    dsi = compute_dsi(rates, orients)
    cv = compute_circular_variance(rates, orients)

    # Weighed
    w_rates, orients = compute_orientation_tuning(trials, weigh=True, weights=weights)

    w_pref_or = orients[np.argmax(w_rates)]
    w_osi = compute_osi(w_rates, orients)
    w_rosi = compute_osi(w_rates, orients, relative=True)
    w_dsi = compute_dsi(w_rates, orients)
    w_cv = compute_circular_variance(w_rates, orients)

    title_1 = "Preferred orientation={}  Weighed PO={}\n".format(pref_or, w_pref_or)
    title_2 = "Non-weighed: OSI={:.2f}  CV={:.2f}  DSI={:.2f}  rOSI={:.2f}\n".format(osi, cv, dsi, rosi)
    title_3 = "Weighed:     OSI={:.2f}  CV={:.2f}  DSI={:.2f}  rOSI={:.2f}".format(w_osi, w_cv, w_dsi, w_rosi)
    fig.suptitle(title_1 + title_2 + title_3, fontsize=17)

    ax1 = fig.add_subplot(1, 2, 1)
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
    m_orient_trials = make_orientation_trials(trials)
    orients = list(m_orient_trials.keys())

    col_count = 4
    row_count = int(np.ceil(2 * len(m_orient_trials)/col_count))
    fig, ax = plt.subplots(row_count, col_count, figsize=(22, 25))

    i = 0
    for r in range(0, row_count, 2):
        for c in range(col_count):
            orient = orients[i]
            orient_trials = m_orient_trials[orient]

            ax[r, c] = plot_raster(orient_trials, ax=ax[r, c])
            ax[r, c].set_title(orient)
            ax[r, c].grid(False)

            ax[r+1, c] = plot_isi(orient_trials, ax=ax[r+1, c])

            i += 1
            
    plt.tight_layout()

    return fig

def plot_isi(trials, ax=None, height=0.18):
    """
    """

    from elephant.statistics import isi

    if ax is None:
        fig, ax = plt.subplots()

    trial_mean_isis = []
    trial_median_isis = []
    trial_std = []
    x_axis = []
    for i, sptr in enumerate(trials):
        if len(sptr) > 1:
            sptr_isi = isi(sptr)
            trial_median_isis.append(np.median(sptr_isi).magnitude)
            trial_mean_isis.append(np.mean(sptr_isi).magnitude)
            trial_std.append(np.std(sptr_isi).magnitude)
            x_axis.append(i+1)
        elif 0 <= len(sptr) < 2:
            pass
        else:
            msg = "Something went wrong len(<sptr>) is negative"
            raise(RuntimeError, msg)

    x_axis = np.array(x_axis)
    median = ax.barh(x_axis-height, trial_median_isis, height=height, color='b', align='center')
    mean = ax.barh(x_axis, trial_mean_isis, height=height, xerr=trial_std, color='r', align='center')
    ax.legend(('Median', 'Mean'))
    return ax
    

def plot_raster(trials, color="#3498db", lw=1, ax=None, marker='|', marker_size=45,
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

def plot_waveforms(sptr, color='r', fig=None, title='waveforms', lw=2, gs=None):
    """
    By @lepmik GitHub:
    Visualize waveforms on respective channels
    Parameters
    ----------
    sptr : neo.SpikeTrain
    color : color of waveforms
    title : figure title
    fig : matplotlib figure
    Returns
    -------
    out : fig
    """
    import matplotlib.gridspec as gridspec
    
    nrc = sptr.waveforms.shape[1]
    if fig is None:
        fig = plt.figure()
        fig.suptitle(title)
    axs = []
    for c in range(nrc):
        if gs is None:
            ax = fig.add_subplot(1, nrc, c+1, sharex=ax, sharey=ax)
        else:
            gs0 = gridspec.GridSpecFromSubplotSpec(1, nrc, subplot_spec=gs)
            ax = fig.add_subplot(gs0[:, c], sharex=ax, sharey=ax)
        axs.append(ax)
    for c in range(nrc):
        wf = sptr.waveforms[:, c, :]
        m = np.mean(wf, axis=0)
        stime = np.arange(m.size, dtype=np.float32)/sptr.sampling_rate
        stime.units = 'ms'
        sd = np.std(wf, axis=0)
        axs[c].plot(stime, m, color=color, lw=lw)
        axs[c].fill_between(stime, m-sd, m+sd, alpha=.1, color=color)
        if sptr.left_sweep is not None:
            sptr.left_sweep.units = 'ms'
            axs[c].axvspan(sptr.left_sweep, sptr.left_sweep, color='k',
                           ls='--')
        axs[c].set_xlabel(stime.dimensionality)
        axs[c].set_xlim([stime.min(), stime.max()])
        if c > 0:
            plt.setp(axs[c].get_yticklabels(), visible=False)
    axs[0].set_ylabel(r'amplitude $\pm$ std [%s]' % wf.dimensionality)

    return fig
