#!/usr/bin/env python
"""
utils.py -- utilities common to the strobesim module

"""
import numpy as np

def sample_sphere(N, d=3):
    """
    Sample *N* points from the surface of the unit sphere, returning
    the result as a Cartesian set of points.

    args
    ----
        N       :   int, the number of trajectories to simulate. Alternatively,
                    N may be a tuple of integers, in which case the return
                    array has shape (*N, d)
        d       :   int, number of dimensions for the (hyper)sphere

    returns
    -------
        2D ndarray of shape (N, d), the positions of the points 

    """
    if isinstance(N, int):
        N = [N]
    p = np.random.normal(size=(np.prod(N), d))
    p = (p.T / np.sqrt((p**2).sum(axis=1))).T 
    return p.reshape(tuple(list(N) + [d]))
