import quantities as pq
import numpy as np


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
