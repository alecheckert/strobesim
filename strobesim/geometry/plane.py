#!/usr/bin/env python
"""
plane.py -- simulate stroboscopic single particle trajectories
in a thin 3D slab of infinite XY extent

"""
# Numeric
import numpy as np 
import pandas as pd

# Convert tracks from a numpy.ndarray format to a pandas.DataFrame format
from .utils import tracks_to_dataframe_gapped

# Simulation utilities
from .utils import (
    impose_defoc,
    impose_bleach,
    impose_loc_error
)

def strobe_plane(tracks, dz=0.7, loc_error=0.035, n_gaps=0, radius=5.0, 
    bleach_prob=0.1, allow_start_outside=True, defoc=True, **kwargs):
    """
    Simulate 3D trajectories that are generated at any point inside a 
    3D slab with thickness *dz* and infinite XY extent.

    args
    ----
        tracks              :   3D ndarray of shape (n_tracks, track_length, 3),
                                the 3D coordinates of each trajectory at 
                                each frame.

                                ~~Each trajectory in *tracks*
                                should start from the origin.~~

        dz                  :   float, focal depth in microns 
        loc_error           :   float, standard deviation for normally
                                distributed localization error along 
                                each axis in microns
        n_gaps              :   int, the number of gaps frames to tolerate
                                during tracking
        radius              :   float, the radius of the sphere in microns
        bleach_prob         :   float, bleach probability per frame
        allow_start_outside :   bool, allow trajectories to start outside
                                the focal volume
        defoc               :   bool, simulate defocalization by killing 
                                detections that lie outside the focal volume
        kwargs              :   sieve; ignored

    returns
    -------
        pandas.DataFrame, the trajectories, with columns 
            ["trajectory", "frame", "z", "y", "x"]. See the docstring
            for *tracks_to_dataframe_gapped* for more information.

    """
    # Check that user has passed trajectories in the correct format
    try:
        n_tracks, track_len, n_dim = tracks.shape 
        assert n_dim == 3
    except (ValueError, AssertionError) as e:
        raise ValueError("tracks must be a numpy.ndarray of " \
            "shape (n_tracks, track_length, 3")

    # Half the observation slice width 
    hz = dz * 0.5


    ## STARTING POSITION FOR TRACKS
    # Add random starting positions along the axial direction that are 
    # sampled uniformly from the interval [-hz, hz]

    start_pos = np.random.uniform(-hz, hz, size=n_tracks)
    tracks[:,:,0] = (tracks[:,:,0].T + start_pos).T 


    ## DEFOCALIZATION
    # Exclude molecules outside of the focal volume from observation
    # by setting their positions to NaN. Some of these molecules may
    # subsequently reenter at later frames, if the number of gaps 
    # tolerated during tracking is greater than 0.
    if defoc: 
        tracks = impose_defoc(tracks, dz, allow_start_outside=allow_start_outside)


    ## BLEACHING
    # Stochastically and permanently bleach molecules. The bleaching
    # rate is assumed to be the same throughout the nucleus and is 
    # stationary in time.
    tracks = impose_bleach(tracks, bleach_prob=bleach_prob)


    ## LOCALIZATION ERROR
    # Add normally-distributed localization error to each point
    tracks = impose_loc_error(tracks, loc_error=loc_error)


    # Format output as pandas.DataFrame
    if len(tracks) == 0:
        return pd.DataFrame([])
    else:
        return tracks_to_dataframe_gapped(tracks, n_gaps=n_gaps)


