import numpy as np
import quantities as pq


def find_nearest(array, value):
    """
    Find the element in array which is closest to
    value.
    Parameters
    ----------
    array : array/quantity array
    value : float/quantity scalar
    Returns
    -------
        idx : index of closest element in array
        out : value of the array element
    """
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]


def wrap_angle(angle, wrap_range=360.):
    '''
    wraps angle in to the interval [0, wrap_range]

    Parameters
    ----------
    angle : numpy.array/float
        input array/float
    wrap_range : float
        wrap range (eg. 360 or 2pi)

    Returns
    -------
    out : numpy.array/float
        angle in interval [0, wrap_range]
    '''
    return angle - wrap_range * np.floor(angle/float(wrap_range))


def rescale_orients(trials, unit=pq.deg):
    """
    Rescales all orient annotations to the same unit

    Parameters
    ----------
    trials : neo.SpikeTrains
        list of spike trains where orientation is given as
        annotation 'orient' (quantity scalar) on each spike train.
    unit : Quantity, optional
        scaling unit. Default is degree.
    """
    if unit not in [pq.deg, pq.rad]:
        raise ValueError("unit can only be deg or rad, ", str(unit))

    for trial in trials:
        orient = trial.annotations["orient"]
        trial.annotations["orient"] = orient.rescale(unit)


def convert_quantity_scalar_to_string(value):
    """
    converts quantity scalar to string

    Parameters
    ----------
    value : quantity scalar

    Returns
    -------
    out : str
        magnitude and unit are separated with space.
    """
    return str(value.magnitude)+" "+value.dimensionality.string


def convert_string_to_quantity_scalar(value):
    """
    converts string to quantity scalar

    Parameters
    ----------
    value : str
        magnitude and unit are assumed to be separated with space.

    Returns
    -------
    out : quantity scalar
    """
    v = value.split(" ")
    return pq.Quantity(float(v[0]), v[1])


def gaussian(x, A, a, dx):
    """
    Evaluates the gauss function
    Parameters
    ----------
    x : array_like
    A : float
        Peak value
    a : float/quantity scalar
        Width
    dx: float/quantity scalar
        shift in center
    Returns
    -------
        out : ndarray
            Calculated values
    """
    return A * np.exp(-(x - dx)**2 / a**2)


def sum_of_gaussians(x, A, B, width, pref_orient, baseline):
    """
    Fit function: sum of Gaussians with
    equal width, but centered at: pref_orient + n
    for n=-360, -180, 0, 180, 360.
    Parameters
    ----------
    x : array_like
        Orientation array
    A : float
        Amplitude first gaussian
    B : float
        Amplitude second gaussian
    width : float/quantity scalar
        Width of Gaussians
    pref_orient: float/quantity scalar
        pref_orient Orientation
    baseline: boolean
        Constant baseline
    Returns
    -------
    func : callable
        Sum of Gaussians function
    Notes
    -----
    This function is used to fit the tuning curve according to
    Niell et al 2008 (sum of two Gaussians). However since
    the tuning curves are periodic in interval 0-360 deg
    we have added several more (shifted) Gaussians
    to incorporate the periodic nature of tuning curves.
    """

    func = baseline
    func += gaussian(x=x, A=A, a=width, dx=pref_orient-360)
    func += gaussian(x=x, A=B, a=width, dx=pref_orient-180)
    func += gaussian(x=x, A=A, a=width, dx=pref_orient)
    func += gaussian(x=x, A=B, a=width, dx=pref_orient+180)
    func += gaussian(x=x, A=A, a=width, dx=pref_orient+360)

    return func


def fit_arguments_sum_of_gaussians(rates, orients,
                                   initial_guess=None,
                                   min_params=None,
                                   max_params=None):
    """
    Sets the fit arguments for sum of two Gaussians.
    This function is custom made for this specific
    fit function.
    Parameters
    ----------
    rates : quantity array
        Array of firing rates
    orients : quantity array
        Array of orientations
    initial_guess : dict (optional)
        Initial guess for each parameter
    min_params : dict (optional)
        Lower bound for each parameter
    max_params : dict (optional)
        Upper bound for each parameter
    Returns
    -------
    guess : list
        Ordered list with initial guess
    bounds : tuple
        Ordered tuple with bounds
    """

    if initial_guess is None:
        initial_guess = {
            "A": rates.max().magnitude - np.maximum(rates.min().magnitude, 1e-6),
            "B": rates.max().magnitude - np.maximum(rates.min().magnitude, 1e-6),
            "width": 10,
            "pref_orient": orients[np.argmax(rates)].magnitude,
            "baseline": 0
        }

    if min_params is None:
        min_params = {
            "A": 0,
            "B": 0,
            "width": 0,
            "pref_orient": -0,
            "baseline": 0
        }

    if max_params is None:
        max_params = {
            "A": 1.1*rates.max().magnitude - np.maximum(rates.min().magnitude, 1e-6),
            "B": 1.1*rates.max().magnitude - np.maximum(rates.min().magnitude, 1e-6),
            "width": 100,
            "pref_orient": 360,
            "baseline": np.maximum(rates.min().magnitude, 1e-6)
        }

        guess = [initial_guess["A"], initial_guess["B"],
                 initial_guess["width"], initial_guess["pref_orient"],
                 initial_guess["baseline"]]

        bounds_lower = [min_params["A"], min_params["B"],
                        min_params["width"], min_params["pref_orient"],
                        min_params["baseline"]]

        bounds_upper = [max_params["A"], max_params["B"],
                        max_params["width"], max_params["pref_orient"],
                        max_params["baseline"]]

        return guess, (bounds_lower, bounds_upper)
