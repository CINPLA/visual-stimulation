import numpy as np
import quantities as pq
import pytest

from visualstimulation.data_processing import (find_epoch_intersection,
                                               find_epoch_difference)

def test_find_epoch_intersection():
    from neo.core import Epoch

    # create epochs
    t = [np.array([10, 20])*pq.s, np.array([9, 21, 30])*pq.s]
    d = [np.array([4, 5])*pq.s, np.array([6, 5, 5])*pq.s]
    l = [np.array(['1', '2']), np.array(['6', '5', '5'])]
    
    ep0 = Epoch(durations=d[0], times=t[0], labels=l[0])
    ep1 = Epoch(durations=d[1], times=t[1], labels=l[1])
    ep0_int = find_epoch_intersection(ep0, ep1)
    t_int = ep0_int.times.rescale(pq.s).magnitude
    t_int_test = np.array([10])
    assert np.allclose(t_int, t_int_test)

    
def test_find_epoch_difference():
    from neo.core import Epoch

    # create epochs
    t = [np.array([0, 20])*pq.s, np.array([9, 21, 30])*pq.s]
    d = [np.array([4, 5])*pq.s, np.array([6, 5, 5])*pq.s]
    l = [np.array(['1', '2']), np.array(['6', '5', '5'])]
    
    ep0 = Epoch(durations=d[0], times=t[0], labels=l[0])
    ep1 = Epoch(durations=d[1], times=t[1], labels=l[1])
    ep0_int = find_epoch_difference(ep0, ep1)
    t_int = ep0_int.times.rescale(pq.s).magnitude
    t_int_test = np.array([0])
    assert np.allclose(t_int, t_int_test)
    
