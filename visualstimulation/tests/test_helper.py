import quantities as pq
import numpy as np


def test_find_nearst():
    from visualstimulation.helper import find_nearest

    A = np.array([0, 1, 2, 3, 4, 5])
    v_A = 2.3
    assert find_nearest(A, v_A) == (2, 2)

    B = np.array([-2.3, 5.2, 8.2, 1.1, 2.67, -3.2])
    v_B = -2.4
    assert find_nearest(B, v_B) == (0, -2.3)

    C = np.array([9.66407446, 7.40204369, 8.47934683, 8.7268378, 9.537069,
                  8.94007828, 6.37876932, 7.84503963, 8.70901142])
    v_C = 7.5
    assert find_nearest(C, v_C) == (1, 7.40204369)

    D = np.array([4.20844744, 5.44088512, -1.44998235, 1.8764609,
                  -2.22633141, 0.33623971, 7.23507673])
    v_D = 0.0
    assert find_nearest(D, v_D) == (5, 0.33623971)


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


def test_gaussian():
    from visualstimulation.helper import gaussian
    assert gaussian(x=0, A=1, a=1, dx=0) == 1
    assert gaussian(x=0*pq.deg, A=1, a=2*pq.deg, dx=0*pq.deg) == 1
    assert abs(gaussian(x=0, A=1, a=1, dx=1) - 0.367879441171) < 1e-12
    assert abs(gaussian(x=1, A=1, a=1, dx=0) - 0.367879441171) < 1e-12
    assert abs(gaussian(x=9*pq.deg, A=-1, a=10.2*pq.deg, dx=6.9*pq.deg) + 0.958498249056) < 1e-12


def test_sum_of_gaussians():
    from visualstimulation.helper import sum_of_gaussians
    assert sum_of_gaussians(x=0, A=1, B=1, width=1, pref_orient=0, baseline=-2) == -1
    assert sum_of_gaussians(x=0, A=0, B=1, width=1, pref_orient=180, baseline=0) == 1
    assert abs(sum_of_gaussians(x=90, A=1.2, B=3.4, width=2.78,
                                pref_orient=90, baseline=2.4)-3.6) < 1e-12
    assert abs(sum_of_gaussians(x=270, A=1.2, B=3.4, width=2.78,
                                pref_orient=90, baseline=2.4)-5.8) < 1e-12
    assert abs(sum_of_gaussians(x=180, A=1.2, B=3.4, width=2.78, pref_orient=360, baseline=2.4)-5.8) < 1e-12


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
