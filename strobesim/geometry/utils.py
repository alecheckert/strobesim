#!/usr/bin/env python
"""
utils.py -- utilities for the strobesim.geometry module

"""
import numpy as np 
import pandas as pd

###########################
## TRACK FORMAT HANDLING ##
###########################

def tracks_to_dataframe_gapped(positions, n_gaps=0):
    """
    Given a set of trajectories as a 3D ndarray indexed by (trajectory,
    frame, dimension), convert to a pandas.DataFrame format.

    This set of trajectories may be broken up by unobserved localizations,
    which are represented in *strobesim* as np.nan values. 

    If this is the case, then we only count each trajectory as contiguous
    if the number of unobserved points for in each gap is less than
    or equal to *n_gaps*.

    So, for instance, a *positions* argument that looks like this:

        [[[1.0, 2.0, 3.0],
          [1.5, 2.5, 3.5],
          [nan, nan, nan],
          [4.0, 4.5, 5.0]]]

    would be considered as two separate trajectories if n_gaps = 0, or 
    a single trajectory if n_gaps = 1.

    args
    ----
        positions       :   3D ndarray of shape (n_tracks, n_frames, 3),
                            the ZYX positions of each localization in each
                            trajectory
        n_gaps          :   int, the number of gap frames tolerated

    returns
    -------
        pandas.DataFrame, the trajectories in dataframe format. This includes
            the "trajectory", "frame", "z", "y", and "x"

    """
    # Work with a copy of the original set of positions
    positions = positions.copy()

    # Shape of the problem
    n_tracks, n_frames, n_dim = positions.shape 
    assert n_dim == 3

    # Finished tracks, in chunks
    all_tracks = []

    # Exclude trajectories that are never observed
    never_obs = np.isnan(positions[:,:,0]).all(axis=1)
    positions = positions[~never_obs, :, :]

    _round = 0

    while (~np.isnan(positions)).any():

        P = positions.copy()
        track_indices = np.arange(positions.shape[0])

        # Find the first observed localization in each trajectory
        first_obs_frame = np.argmin(np.isnan(positions[:,:,0]), axis=1)

        # Find the last observed localization in each trajectory, 
        # allowing for gaps
        last_obs_frame = first_obs_frame.copy()
        active = np.ones(positions.shape[0], dtype=bool)
        gap_count = np.zeros(positions.shape[0], dtype=np.int64)

        while active.any():

            # Remove points from future consideration
            P[track_indices, last_obs_frame, :] = np.nan 

            # Extend all active trajectories
            last_obs_frame[active] += 1

            # If we've reached the end of the available frames in the
            # simulation, terminate
            at_end = last_obs_frame >= positions.shape[1]
            last_obs_frame[at_end] = positions.shape[1] - 1
            active[at_end] = False

            # Drop trajectories if they haven't been observed for more 
            # than the maximum number of tolerated gap frames
            not_obs = np.isnan(positions[track_indices, last_obs_frame, 0])
            gap_count[not_obs] += 1
            gap_count[~not_obs] = 0
            dropped = gap_count > n_gaps 
            active = np.logical_and(active, ~dropped)

        # Aggregate finished trajectories
        for i, t in enumerate(range(positions.shape[0])):
            all_tracks.append(positions[t,first_obs_frame[i]:last_obs_frame[i]+1,:])

        # Remove trajectories that have been completely exhausted
        never_obs = np.isnan(P[:,:,0]).all(axis=1)
        positions = P[~never_obs, :, :].copy()

        _round += 1

    # Format the result as a dataframe
    n_points = sum([t.shape[0] for t in all_tracks])
    track_ids = np.zeros(n_points, dtype=np.int64)
    c = 0
    for i, t in enumerate(all_tracks):
        L = t.shape[0]
        track_ids[c:c+L] = i 
        c += L 
    if len(all_tracks) == 0:
        return pd.DataFrame(
            {
                "trajectory": np.zeros(0, dtype=np.int64),
                "frame": np.zeros(0, dtype=np.int64),
                "z": np.zeros(0, dtype=np.float64),
                "y": np.zeros(0, dtype=np.float64),
                "x": np.zeros(0, dtype=np.float64),
            }
        )
    tracks_concat = np.concatenate(all_tracks, axis=0)
    result = pd.DataFrame(tracks_concat, columns=["z", "y", "x"])
    result["trajectory"] = track_ids
    result["one"] = 1
    result["frame"] = result.groupby("trajectory")["one"].cumsum() - 1
    result = result.drop("one", axis=1)
    result = result[~pd.isnull(result["z"])]
    return result

##########################
## SIMULATION UTILITIES ##
##########################

def impose_defoc(tracks, dz, allow_start_outside=True):
    """
    Given a set of 3D trajectories, when the axial position of a 
    trajectories lies outside the interval [-dz/2, dz/2], remove 
    it from observation by setting its position to np.nan.

    args
    ----
        tracks              :   3D ndarray of shape (n_tracks, track_length, 3),
                                trajectory points in microns
        dz                  :   float, focal depth in microns
        allow_start_outside :   bool, allow trajectories to reenter after
                                a single defocalization event

    returns
    -------
        reference to *tracks*

    """
    if (not dz is None) and (not dz is np.inf):

        # Half the focal depth
        hz = dz * 0.5

        for frame in range(tracks.shape[1]):

            # Get the set of points that lie outside the focal plane
            # at this frame
            outside = np.abs(tracks[:,frame,0]) > hz 

            # Set the coordinates of these points to NaN 
            for d in range(3):
                tracks[:, frame, d][outside] = np.nan 

        # If not allowing trajectories to start outside the focal
        # volume, also set all the remaining points for the 
        # corresponding trajectories to np.nan
        if not allow_start_outside:
            lost = np.zeros(tracks.shape[0], dtype=bool)
            for frame in range(tracks.shape[1]):
                lost = np.logical_or(lost, np.isnan(tracks[:,frame,0]))
                tracks[lost,frame,:] = np.nan

    return tracks 

def impose_bleach(tracks, bleach_prob=0):
    """
    Given a set of 3D trajectories, bleach trajectories stochastically
    and permanently by setting their positions to np.nan.

    The bleaching rate is assumed to be the same for all trajectories
    and to be stationary in time.
    
    args
    ----
        tracks          :   3D ndarray of shape (n_tracks, track_length, 3),
                            trajectory points in microns
        bleach_prob     :   float, probability for any trajectory to bleach
                            in one frame

    returns
    -------
        reference to *tracks*

    """
    if bleach_prob > 0:
        bleached = np.zeros(tracks.shape[0], dtype=bool)
        for frame in range(tracks.shape[1]):
            bleached = np.logical_or(
                bleached,
                np.random.random(size=tracks.shape[0]) <= bleach_prob 
            )
            for d in range(3):
                tracks[:,frame,d][bleached] = np.nan 
    return tracks 

def impose_loc_error(tracks, loc_error=0.0):
    """
    Given a set of 3D trajectories, add normally-distributed localization
    error with standard deviation *loc_error* along each spatial 
    dimension.

    args
    ----
        tracks          :   3D ndarray of shape (n_tracks, track_length, 3),
                            trajectory points in microns
        loc_error       :   float, 1D localization error in microns, or 
                            1D ndarray of shape (n_tracks), the localization
                            error for each trajectory in microns

    returns
    -------
        reference to *tracks*

    """
    if isinstance(loc_error, float):
        if loc_error > 0.0:
            tracks = tracks + np.random.normal(scale=loc_error, size=tracks.shape)
    elif isinstance(loc_error, np.ndarray):
        errors = np.zeros(tracks.shape)
        for l in range(tracks.shape[1]):
            for d in range(tracks.shape[2]):
                errors[:,l,d] = np.random.normal(scale=loc_error, size=tracks.shape[0])
        tracks = tracks + errors 

    return tracks 
