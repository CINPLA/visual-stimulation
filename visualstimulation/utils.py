import quantities as pq
import numpy as np

def generate_gradiently_weighed_data(data, weight_start=1, weight_end=0.5):
    """
    Creates weighed data using gradients from weight_start to weight_end.

    Example
    -------
    >>> A = np.array([2, 5, 7, 4, 8, 10, 3, 7, 3, 5])
    >>> generate_gradiently_weighed_data(data=A)
    array([2.        , 4.72222222, 6.22222222, 3.33333333, 6.22222222,
          7.22222222, 2.        , 4.27777778, 1.66666667, 2.5       ])

    Note
    -------
    In the example above the following weights are generated:
    array([1.        , 0.94444444, 0.88888889, 0.83333333, 0.77777778,
           0.72222222, 0.66666667, 0.61111111, 0.55555556, 0.5       ])

    Parameters
    ----------
    data : numpy.array
        0D numpy.array with data to be weighed
    weight_start : int, float
        Initial weight
    weight_end : int, float
        Last weight
    Returns
    ------
    out : numpy.array(Weighed data); data * weight
    """
    if not isinstance(data, type(np.empty(0))):
        msg = "data has to be numpy.array, and not {}".format(type(data))
        raise TypeError(msg)

    weights = np.linspace(weight_start, weight_end, len(data))
    weighed_data = data * weights
    return weighed_data


def minmax_scale(data, units=None):
    """
    Transforms features by scaling each feature to [0, 1]

    Parameters
    ----------
    data : numpy.array
        0D numpy.array with data to be weighed
    units : None; quantity.unitquantity; bool
        None or False, no action
        True, assign data.unit to every scaled element
        quantity.unitquantity, assign unit to every scaled element        
    Returns
    ------
    out : minmax scaled ([0, 1]) numpy.array
    """
    if not isinstance(data, type(np.empty(0))):
        msg = "data has to be numpy.array, and not {}".format(type(data))
        raise TypeError(msg)

    scaled_data = (data - data.min()) / (data.max() - data.min())
    if units is None or units is False:
        return scaled_data
    elif isinstance(units, pq.unitquantity.UnitQuantity):
        return scaled_data * units
    elif units is True:
        if not hasattr(spiketrain, 'units'):
            msg = "Data doesn't have a unit, but units is True"
            raise AttributeError(msg)
        return scaled_data * data.units