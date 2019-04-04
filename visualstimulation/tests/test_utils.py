from random import random
import numpy as np


def test_generate_gradiently_weighed_data():
    from visualstimulation.utils import generate_gradiently_weighed_data

    weight_start, weight_end = 1, 0.5     # 1.0 -> 0.5

    A = np.array([1, 2, 3, 4, 5, 6])
    expected_A = np.array([1. , 1.8, 2.4, 2.8, 3. , 3. ])
    
    weighed_A = generate_gradiently_weighed_data(data=A, weight_start=weight_start, weight_end=weight_end)

    assert all(np.isclose(expected_A, weighed_A))


    weight_start, weight_end = 1, random()     # 1 -> [0.0, 1.0)

    A = np.array([1, 2, 3, 4, 5, 6])
    expected_A = np.linspace(weight_start, weight_end, 6) * A
    
    weighed_A = generate_gradiently_weighed_data(data=A, weight_start=weight_start, weight_end=weight_end)

    assert all(np.isclose(expected_A, weighed_A))


if __name__ == "__main__":
    test_generate_gradiently_weighed_data()
