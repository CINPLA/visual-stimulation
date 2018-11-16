import numpy as np
import quantities as pq
import pytest


from visualstimulation.tools import (compute_osi,
                                     compute_dsi,
                                     compute_circular_variance,
                                     compute_spontan_rate,
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


def test_wrap_angle_360():
    from visualstimulation.helper import wrap_angle
    angles = np.array([85.26, -437.34, 298.14, 57.47, -28.98, 681.25, -643.99,
                       43.71, -233.82, -549.63, 593.7, 164.48, 544.05, -52.66,
                       79.87, -21.11, 708.31, 29.45, 279.14, -586.88])

    angles_ex = np.array([85.26, 282.66, 298.14, 57.47, 331.02, 321.25, 76.01,
                          43.71, 126.18, 170.37, 233.7, 164.48, 184.05, 307.34,
                          79.87, 338.89, 348.31, 29.45, 279.14, 133.12])

    result = wrap_angle(angles, 360)
    np.testing.assert_almost_equal(result, angles_ex, decimal=13)


def test_wrap_angle_2pi():
    from visualstimulation.helper import wrap_angle
    angles = np.array([-7.15, -7.3, 7.74, 4.68, -9.33, 1.32, 4.18, 3.49,
                       8.21, 1.43, -0.96, 6.63, 1.32, 9.66, -10.57, -7.17,
                       1.84, -10.24, -7.31, -11.71, -1.82, 2.85, 1.99, -5.11,
                       -10.16, 3.6, 9.36, -3.13, -0.64, -1.77])

    angles_ex = np.array([5.41637061, 5.26637061, 1.45681469, 4.68, 3.23637061,
                          1.32, 4.18, 3.49, 1.92681469, 1.43,
                          5.32318531, 0.34681469, 1.32, 3.37681469, 1.99637061,
                          5.39637061, 1.84, 2.32637061, 5.25637061, 0.85637061,
                          4.46318531, 2.85, 1.99, 1.17318531, 2.40637061, 3.6, 3.07681469, 3.15318531, 5.64318531, 4.51318531])

    result = wrap_angle(angles, 2*np.pi)
    np.testing.assert_almost_equal(result, angles_ex, decimal=8)


def test_make_stimulus_off_epoch():
    from neo.core import Epoch
    from visualstimulation.tools import (make_stimulus_off_epoch)

    times = np.linspace(0, 10, 11) * pq.s
    durations = np.ones(len(times)) * pq.s
    labels = np.ones(len(times))

    stim_epoch = Epoch(labels=labels, durations=durations, times=times)
    stim_off_epoch = make_stimulus_off_epoch(stim_epoch)

    assert(stim_off_epoch.times == np.linspace(1, 10, 10)).all()
    assert(stim_off_epoch.durations == np.zeros(10)).all()
    assert(stim_off_epoch.labels == [None]*10)

    stim_off_epoch = make_stimulus_off_epoch(stim_epoch, include_boundary=True)
    assert(stim_off_epoch.times == np.linspace(0, 10, 11)).all()
    assert(stim_off_epoch.durations == np.zeros(11)).all()
    assert(stim_off_epoch.labels == [None]*11)

    times = np.arange(0.5, 11, 0.5)[::2] * pq.s
    durations = np.ones(len(times)) * 0.5 * pq.s
    labels = np.ones(len(times))

    stim_epoch = Epoch(labels=labels, durations=durations, times=times)
    stim_off_epoch = make_stimulus_off_epoch(stim_epoch)

    assert(stim_off_epoch.times == np.arange(1, 11, 1)).all()
    assert(stim_off_epoch.durations == np.ones(10) * 0.5).all()
    assert(stim_off_epoch.labels == [None]*10)

    stim_off_epoch = make_stimulus_off_epoch(stim_epoch, include_boundary=True)

    assert(stim_off_epoch.times == np.arange(0, 11, 1)).all()
    assert(stim_off_epoch.durations == np.ones(11) * 0.5).all()
    assert(stim_off_epoch.labels == [None]*11)


def test_compute_orientation_tuning():
    from neo.core import SpikeTrain
    import quantities as pq
    from visualstimulation.tools import (make_orientation_trials,
                                      compute_orientation_tuning)

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
    from visualstimulation.tools import (make_orientation_trials,
                                         convert_string_to_quantity_scalar)

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


def test_rescale_orients():
    from neo.core import SpikeTrain
    import quantities as pq
    from visualstimulation.helper import rescale_orients

    trials = [SpikeTrain(np.arange(0, 10, 1.)*pq.s, t_stop=10*pq.s,
                         orient=0 * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.5)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad),
              SpikeTrain(np.arange(0, 10, 2)*pq.s, t_stop=10*pq.s,
                         orient=315 * pq.deg)]
    scaled_trials = trials.copy()
    rescale_orients(scaled_trials[:])
    assert(scaled_trials is not trials)
    for t, st in zip(trials, scaled_trials):
        orient = list(t.annotations.values())[0].rescale(pq.deg)
        scaled_orient = list(st.annotations.values())[0]
        assert(scaled_orient.units == pq.deg)
        assert(scaled_orient == orient)
        assert((t == st).all())
        assert(t.t_start == st.t_start)
        assert(t.t_stop == st.t_stop)

    trials = [SpikeTrain(np.arange(0, 10, 1.)*pq.s, t_stop=10*pq.s,
                         orient=0 * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.5)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad),
              SpikeTrain(np.arange(0, 10, 2)*pq.s, t_stop=10*pq.s,
                         orient=315 * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.3)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad)]

    scaled_trials = trials.copy()
    rescale_orients(scaled_trials[:], unit=pq.rad)
    assert(scaled_trials is not trials)
    for t, st in zip(trials, scaled_trials):
        orient = list(t.annotations.values())[0].rescale(pq.rad)
        scaled_orient = list(st.annotations.values())[0]
        assert(scaled_orient.units == pq.rad)
        assert(scaled_orient == orient)
        assert((t == st).all())
        assert(t.t_start == st.t_start)
        assert(t.t_stop == st.t_stop)
