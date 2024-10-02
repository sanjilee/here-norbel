import numpy as np
import math

def xy(lnglat, truncate=False):
    """Convert longitude and latitude to web mercator x, y
    Parameters
    ----------
    lnglat : np.array
        Longitude and latitude array in decimal degrees, shape: (-1, 2)
    truncate : bool, optional
        Whether to truncate or clip inputs to web mercator limits.
    Returns
    -------
    np.array with x, y in webmercator
    >>> a = np.array([(0.0, 0.0), (-75.15963, -14.704620000000013)])
    >>> b = np.array(((0.0, 0.0), (-8366731.739810849, -1655181.9927159143)))
    >>> np.isclose(xy(a), b)
    array([[ True,  True],
           [ True,  True]], dtype=bool)
    """

    lng, lat = lnglat[:,0], lnglat[:, 1]
    if truncate:
        lng = numpy.clip(lng, -180.0, 180.0)
        lat = numpy.clip(lng, -90.0, 90.0)
    x = 6378137.0 * np.radians(lng)
    y = 6378137.0 * np.log(
        np.tan((math.pi * 0.25) + (0.5 * np.radians(lat))))
    return np.array((x, y)).T
