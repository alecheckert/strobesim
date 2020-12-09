#!/usr/bin/env python
"""
sphere.py -- simulate stroboscopic single particle trajectories only
observable in a thin slice of a sphere

"""
# Numeric
import numpy as np
import pandas as pd 

# Convert tracks from a numpy.ndarray to a pandas.DataFrame
from .utils import tracks_to_dataframe_gapped

# Simulation utilities
from .utils import (
    impose_defoc,
    impose_bleach,
    impose_loc_error
)

# Sample points uniformly from the surface of a sphere
from ..utils import sample_sphere 

def strobe_sphere(tracks, dz=0.7, loc_error=0.035, n_gaps=0,
    radius=5.0, bleach_prob=0.1, allow_start_outside=True, **kwargs):
    """
    Given a set of trajectories, place the trajectories at random points
    inside a 3D sphere and let them diffuse inside the sphere with 
    specular reflections at the boundaries. If *defoc* is *True*, then 
    these trajectories are only observed at points that coincide with 
    a thin plane that bisects the sphere. 

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

    # Nucleus diameter
    diameter = radius * 2


    ## STARTING POSITION FOR TRACKS
    # Add random starting positions that are sampled uniformly
    # from the volume inside the sphere

    # Angle relative to origin, expressed as a unit-length vector
    start_pos_ang = sample_sphere(n_tracks, d=3)

    # Radial distance from origin
    start_pos_rad = radius * np.cbrt(np.random.random(size=n_tracks))

    # Starting positions
    start_pos = (start_pos_ang.T * start_pos_rad).T 

    # Offset each trajectory by the starting position
    for d in range(3):
        tracks[:,:,d] = (tracks[:,:,d].T + start_pos[:,d]).T 

    # Destroy trajectories that start outside the focal volume
    # if desired
    if not allow_start_outside:
        tracks = tracks[np.abs(tracks[:,0,0] <= hz), :, :]


    ## BORDER REFLECTIONS
    # Deal with trajectories that cross over the sphere's border
    # by specular reflection
    for frame in range(tracks.shape[1]):

        # Determine the set of trajectories that are outside the 
        # sphere at this frame
        dist_from_origin = np.sqrt((tracks[:,frame,:]**2).sum(axis=1))
        outside = dist_from_origin > radius 

        # Get the closest point on the surface of the sphere for
        # each of these points
        reflect_points = radius * (tracks[outside,frame,:].T / dist_from_origin[outside]).T 

        # Reflect the trajectories back into the sphere for each 
        # subsequent frame
        for g in range(frame, tracks.shape[1]):
            tracks[outside, g, :] = 2 * reflect_points - tracks[outside, g, :]


    ## DEFOCALIZATION
    # Exclude molecules outside of the focal volume from observation
    # by setting their positions to NaN. Some of these molecules may
    # subsequently reenter at later frames, if the number of gaps 
    # tolerated during tracking is greater than 0.
    impose_defoc(tracks, dz, allow_start_outside=allow_start_outside)


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
