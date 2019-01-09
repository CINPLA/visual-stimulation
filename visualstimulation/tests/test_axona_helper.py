import numpy as np
import quantities as pq
import os
import pytest


from visualstimulation.axona_helper import (get_synced_orientation_data,
                                            _find_identical_trialing_elements,
                                            convert_inp_values_to_keys)


def test_convert_inp_values_to_keys():
    keys = np.array(("space", "!", "+", "K", "L", "R", "Z", "s"))

    byte_7_values = np.array((32, 33, 43, 75, 76, 82, 90, 115), dtype=int)
    byte_6_values = np.zeros(len(byte_7_values), dtype=int)
    values = np.column_stack((byte_6_values, byte_7_values))

    assert((keys == convert_inp_values_to_keys(values))).all()

    values[:, 0] = np.ones(len(byte_7_values), dtype=int)
    with pytest.raises(ValueError):
        assert((keys == convert_inp_values_to_keys(values))).all()


def test_find_identical_trailing_elements():
    a = [0, 1, 1, 2, 3, 3, 4, 5, 5]
    ids = [1, 4, 7]
    assert(ids == _find_identical_trialing_elements(a))

    a = np.array([0, 0, 1, 1, 1, 10, 9, 9, 8])
    ids = [0, 2, 6]
    assert(ids == _find_identical_trialing_elements(a))

    a = np.array([0, 0, -2, -4, -3, -3])
    ids = [0, 4]
    assert(ids == _find_identical_trialing_elements(a))

    a = np.arange(100)
    ids = []
    assert(ids == _find_identical_trialing_elements(a))

    a = np.ones(10)
    ids = [0]
    assert(ids == _find_identical_trialing_elements(a))


def test_get_synced_orientation_data():
    current_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(current_path, "test_files")

    raw_inp_timestamps = os.path.join(data_path, "timestamps_1.npy")
    raw_inp_values = os.path.join(data_path, "values_1.npy")
    blank_file = os.path.join(data_path, "blank_1.csv")
    stim_file = os.path.join(data_path, "stim_1.csv")

    stim_data, orients_data = np.loadtxt(stim_file, delimiter=',', unpack=True)
    blank_data = np.loadtxt(blank_file, delimiter=',', unpack=True) * pq.s

    timestamps = np.load(raw_inp_timestamps) * pq.s
    values = np.load(raw_inp_values)
    orientations, t_stim, t_blank = get_synced_orientation_data(timestamps, values)

    assert((orients_data * pq.deg == orientations).all())
    assert((t_blank == blank_data).all())
    assert((t_stim == stim_data).all())

    raw_inp_timestamps = os.path.join(data_path, "timestamps_2.npy")
    raw_inp_values = os.path.join(data_path, "values_2.npy")
    blank_file = os.path.join(data_path, "blank_2.csv")
    stim_file = os.path.join(data_path, "stim_2.csv")

    stim_data, orients_data = np.loadtxt(stim_file, delimiter=',', unpack=True)
    blank_data = np.loadtxt(blank_file, delimiter=',', unpack=True) * pq.s

    timestamps = np.load(raw_inp_timestamps) * pq.s
    values = np.load(raw_inp_values)
    orientations, t_stim, t_blank = get_synced_orientation_data(timestamps, values)

    assert((t_blank == blank_data).all())
    assert((orients_data * pq.deg == orientations).all())
    assert((t_stim == stim_data).all())
