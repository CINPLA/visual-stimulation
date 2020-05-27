import numpy as np
import quantities as pq
import pytest
import pdb

from visualstimulation.analysis import (compute_osi,
                                        compute_dsi,
                                        compute_circular_variance,
                                        compute_spontan_rate,
                                        calculate_psth,
                                        calculate_psth_from_trials,
                                        )


def test_circular_variance():
    rates = 1.23*np.ones(8)*pq.Hz
    orients = np.linspace(0, 315, 8)*pq.deg
    assert abs(compute_circular_variance(rates, orients)-1) < 1e-12

    rates = np.zeros(8)*pq.Hz
    rates[4] = 12.4*pq.Hz
    orients = np.linspace(0, 315, 8)*pq.deg
    assert abs(compute_circular_variance(rates, orients)) < 1e-12

    rates = np.array([1, 2, 3])*pq.Hz
    orients = np.array([0, 45, 90])*pq.deg
    assert abs(compute_circular_variance(rates, orients)-0.52859547920896832) < 1e-12


def test_compute_osi():
    # normalise false
    orients = np.arange(0, 315, 45)*pq.deg
    rates = np.array([1, 1, 2, 1, 1, 2, 1, 1])*pq.Hz
    assert abs(compute_osi(rates, orients) - 1/3) < 1e-12

    orients = np.arange(0, 315, 45)*pq.deg
    rates = np.array([3, 1, 2, 1, 1, 2, 1, 1])*pq.Hz
    assert abs(compute_osi(rates, orients) - 1/5) < 1e-12

    orients = np.arange(0, 2*np.pi, np.pi/4)*pq.rad
    rates = np.array([3, 1, 2, 1, 1, 2, 1, 1])*pq.Hz
    assert abs(compute_osi(rates, orients) - 1/5) < 1e-12

    orients = np.arange(0, 315, 40)*pq.deg
    rates = np.array([3, 1, 2, 1, 1, 2, 1, 1])*pq.Hz
    with pytest.warns(UserWarning):
        assert abs(compute_osi(rates, orients) - 1/5) < 1e-12
    
    # normalise true
    orients = np.arange(0, 315, 45)*pq.deg
    rates = np.array([1, 1, 2, 1, 1, 2, 1, 1])*pq.Hz
    assert abs(compute_osi(rates, orients, normalise=True) - 1) < 1e-12

    orients = np.arange(0, 315, 45)*pq.deg
    rates = np.array([3, 1, 2, 1, 1, 2, 1, 1])*pq.Hz
    assert abs(compute_osi(rates, orients, normalise=True) - 1/3) < 1e-12

    orients = np.arange(0, 2*np.pi, np.pi/4)*pq.rad
    rates = np.array([3, 1, 2, 1, 1, 2, 1, 1])*pq.Hz
    assert abs(compute_osi(rates, orients, normalise=True) - 1/3) < 1e-12

    orients = np.arange(0, 315, 40)*pq.deg
    rates = np.array([3, 1, 2, 1, 1, 2, 1, 1])*pq.Hz
    with pytest.warns(UserWarning):
        assert abs(compute_osi(rates, orients, normalise=True) - 1/3) < 1e-12


def test_compute_dsi():
    orients = np.arange(0, 315, 45)*pq.deg
    rates = np.array([1, 1, 2, 1, 1, 3, 1, 1])*pq.Hz
    assert abs(compute_dsi(rates, orients) - 1/2) < 1e-12

    orients = np.arange(0, 315, 45)*pq.deg
    rates = np.array([3, 1, 4, 1, 1, 2, 3, 1])*pq.Hz
    assert abs(compute_dsi(rates, orients) - 1/7) < 1e-12

    orients = np.arange(0, 2*np.pi, np.pi/4)*pq.rad
    rates = np.array([3, 1, 4, 1, 1, 2, 1, 1])*pq.Hz
    assert abs(compute_dsi(rates, orients) - 3/5) < 1e-12

    orients = np.arange(0, 315, 40)*pq.deg
    rates = np.array([3, 1, 2, 1, 2.3, 2, 1, 1])*pq.Hz
    with pytest.warns(UserWarning):
        assert abs(compute_dsi(rates, orients) - 0.7/5.3) < 1e-12


def test_compute_orientation_tuning():
    from neo.core import SpikeTrain
    import quantities as pq
    from visualstimulation.analysis import compute_orientation_tuning
    from visualstimulation.data_processing import make_orientation_trials

    trials = [SpikeTrain(np.arange(0, 10, 2)*pq.s, t_stop=10*pq.s,
                         orient=315. * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.5)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad),
              SpikeTrain(np.arange(0, 10, 1)*pq.s, t_stop=10*pq.s,
                         orient=0 * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.3)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad)]
    sorted_orients = np.array([0, (np.pi/3 * pq.rad).rescale(pq.deg)/pq.deg, 315]) * pq.deg
    rates_e = np.array([1., 2.7, 0.5]) / pq.s

    trials = make_orientation_trials(trials)
    rates, orients = compute_orientation_tuning(trials)
    assert((rates == rates_e).all())
    assert(rates.units == rates_e.units)
    assert((orients == sorted_orients).all())
    assert(orients.units == sorted_orients.units)


def test_make_orientation_trials():
    from neo.core import SpikeTrain
    from visualstimulation.data_processing import make_orientation_trials
    from visualstimulation.helper import convert_string_to_quantity_scalar

    trials = [SpikeTrain(np.arange(0, 10, 2)*pq.s, t_stop=10*pq.s,
                         orient=315. * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.5)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad),
              SpikeTrain(np.arange(0, 10, 1)*pq.s, t_stop=10*pq.s,
                         orient=0 * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.3)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad)]

    sorted_trials = [[trials[2]], [trials[1], trials[3]], [trials[0]]]
    sorted_orients = [0 * pq.deg, (np.pi/3 * pq.rad).rescale(pq.deg), 315 * pq.deg]
    orient_trials = make_orientation_trials(trials, unit=pq.deg)

    for (key, value), trial, orient in zip(orient_trials.items(),
                                           sorted_trials,
                                           sorted_orients):
        key = convert_string_to_quantity_scalar(key)
        assert(key == orient.magnitude)
        for t, st in zip(value, trial):
            assert((t == st).all())
            assert(t.t_start == st.t_start)
            assert(t.t_stop == st.t_stop)
            assert(t.annotations["orient"] == orient)


def test_make_stimulus_off_epoch():
    from neo.core import Epoch
    from visualstimulation.data_processing import (make_stimulus_off_epoch)

    times = np.linspace(0, 10, 11) * pq.s
    durations = np.ones(len(times)) * pq.s
    labels = np.ones(len(times))

    stim_epoch = Epoch(labels=labels, durations=durations, times=times)
    stim_off_epoch = make_stimulus_off_epoch(stim_epoch)

    assert(stim_off_epoch.times == np.linspace(1, 10, 10)).all()
    assert(stim_off_epoch.durations == np.zeros(10)).all()
    assert(stim_off_epoch.labels == [None]*10).all()

    stim_off_epoch = make_stimulus_off_epoch(stim_epoch, include_boundary=True)
    assert(stim_off_epoch.times == np.linspace(0, 10, 11)).all()
    assert(stim_off_epoch.durations == np.zeros(11)).all()
    assert(stim_off_epoch.labels == [None]*11).all()

    times = np.arange(0.5, 11, 0.5)[::2] * pq.s
    durations = np.ones(len(times)) * 0.5 * pq.s
    labels = np.ones(len(times))

    stim_epoch = Epoch(labels=labels, durations=durations, times=times)
    stim_off_epoch = make_stimulus_off_epoch(stim_epoch)

    assert(stim_off_epoch.times == np.arange(1, 11, 1)).all()
    assert(stim_off_epoch.durations == np.ones(10) * 0.5).all()
    assert(stim_off_epoch.labels == [None]*10).all()

    stim_off_epoch = make_stimulus_off_epoch(stim_epoch, include_boundary=True)

    assert(stim_off_epoch.times == np.arange(0, 11, 1)).all()
    assert(stim_off_epoch.durations == np.ones(11) * 0.5).all()
    assert(stim_off_epoch.labels == [None]*11).all()


def test_calculate_psth():
    from neo.core import Epoch
    from neo.core import SpikeTrain

    # create epoch
    times = np.arange(0, 91, 10.) * pq.s
    durations = np.ones(len(times)) * pq.s
    labels = np.ones(len(times))

    epoch = Epoch(labels=labels, durations=durations, times=times)

    # create spiketrain, some regular spikes plus some manually inserted ones
    ts = list(np.arange(2, 93, 10.))
    ts.append(9)
    ts = np.array(ts) * pq.s
    ts = np.sort(ts)
    st = SpikeTrain(ts, t_stop=110*pq.s)

    # define psth
    lags = [-5., 5] * pq.s
    bin_size = 2.5 * pq.s

    # calculate psth
    psth = calculate_psth(st, epoch, lags, bin_size, unit='rate')
    psth_target = np.array([0., 0.04, 0.4, 0.])
    assert np.allclose(psth, psth_target)

def test_calculate_psth_from_trials():
    from neo.core import Epoch
    from neo.core import SpikeTrain
    from visualstimulation.data_processing import make_spiketrain_trials

    # create epoch
    times = np.arange(0, 91, 10.) * pq.s
    durations = np.ones(len(times)) * pq.s
    labels = np.ones(len(times))

    epoch = Epoch(labels=labels, durations=durations, times=times)

    # create spiketrain, some regular spikes plus some manually inserted ones
    ts = list(np.arange(2, 93, 10.))
    ts.append(9)
    ts = np.array(ts) * pq.s
    ts = np.sort(ts)
    st = SpikeTrain(ts, t_stop=110*pq.s)

    # define psth
    lags = [-5., 5] * pq.s
    bin_size = 2.5 * pq.s

    trials = make_spiketrain_trials(st, epoch, t_start=lags[0], t_stop=lags[1])
    # calculate psth
    psth = calculate_psth_from_trials(trials, bin_size, unit='rate')
    psth_target = np.array([0., 0.04, 0.4, 0.])
    assert np.allclose(psth, psth_target)
    
