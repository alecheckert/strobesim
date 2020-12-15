#!/usr/bin/env python
"""
multistate.py

"""
import numpy as np 
import pandas as pd 

# Types of motion supported
from .. import (
    FractionalBrownianMotion3D,
    LevyFlight3D
)

# Available motion types
MOTIONS = {
    "brownian": FractionalBrownianMotion3D,
    "fbm": FractionalBrownianMotion3D,
    "levy": LevyFlight3D
}

# Simulate tracking in a sphere or an infinite plane
from .. import (
    strobe_sphere,
    strobe_plane
)

# Available imaging geometries
GEOMETRIES = {
    "sphere": strobe_sphere,
    "plane": strobe_plane
}

# Custom utilities
from .utils import (
    concat_tracks
)

def strobe_multistate(n_tracks, diff_coefs, occupancies, motion="brownian", 
    geometry="sphere", motion_kwargs={}, track_len=100, dz=0.7,
    frame_interval=0.00548, loc_error=0.035, n_gaps=0, bleach_prob=0.1,
    n_rounds=1, allow_start_outside=True, **geometry_kwargs):
    """
    Simulate a stroboscopic SPT experiment with multiple kinds of 
    motion.

    args
    ----
        n_tracks            :   int, the number of trajectories to simulate
        diff_coefs          :   1D ndarray of shape (n_states), the diffusion
                                coefficients/dispersion parameters for each 
                                state
        occupancies         :   1D ndarray of shape (n_states), the fractional
                                occupancies of each state
        motion              :   str, the type of motion
        geometry            :   str, the type of observation geometry
        motion_kwargs       :   dict or list of dict. If dict, then these
                                are shared keyword arguments to the 
                                trajectory simulator (from *strobesim.motions*)
                                for all states. If a list of dict, then 
                                each dict is considered keyword arguments
                                for the corresponding state.
        track_len           :   int, the length of each trajectory in frames
        dz                  :   float, focal depth in um 
        frame_interval      :   float, frame interval in seconds
        loc_error           :   float, localization error in microns
        n_gaps              :   int, number of gaps to tolerate during tracking
        bleach_prob         :   float, probability to bleach in one frame
        n_rounds            :   int, the number of replicates of the entire
                                simulation to do
        allow_start_outside :   bool, allow trajectories to start outside
                                the focal volume
        geometry_kwargs     :   additional keyword arguments to the geometry
                                simulator in *strobesim.geometry*

    returns
    -------
        pandas.DataFrame, the trajectories, with columns ["trajectory",
            "frame", "y", "x", "z"]

    """
    # Single-state
    if isinstance(diff_coefs, float):
        diff_coefs = [diff_coefs]
        occupancies = [1.0]

    # Only consider states with nonzero occupancies
    occupancies = np.asarray(occupancies).copy()
    diff_coefs = np.asarray(diff_coefs).copy()
    nonzero = occupancies > 0
    occupancies = occupancies[nonzero]
    diff_coefs = diff_coefs[nonzero]

    # One of two modes for the motion keyword argument: either a 
    # single set of motion keywords for all states, or a separate
    # set for each state
    if isinstance(motion_kwargs, list):
        assert len(motion_kwargs) == len(occupancies), "number of elements in " \
            "motion_kwargs must match the number of states"
    else:
        if len(occupancies) > 0:
            motion_kwargs = [motion_kwargs for j in range(len(occupancies))]

    # Simulate all remaining states
    results = []
    for round_ in range(n_rounds):

        # Choose the number of trajectories in each state
        n_occ = np.random.multinomial(n_tracks, occupancies)

        # Generate the motions
        tracks = []
        for i, D in enumerate(diff_coefs):

            # Underlying trajectory simulator
            model_obj = MOTIONS[motion](D=D, frame_interval=frame_interval,
                track_len=track_len, **motion_kwargs[i])

            # Generate trajectories
            tracks_state = model_obj(n_occ[i])

            # Simulate tracking in the desired geometry
            tracks_state = GEOMETRIES[geometry](tracks_state, dz=dz, 
                loc_error=loc_error, n_gaps=n_gaps, bleach_prob=bleach_prob,
                allow_start_outside=allow_start_outside, **geometry_kwargs)

            tracks.append(tracks_state)

        # Concatenate trajectories from all states while incrementing
        # redundant trajectory indices
        results.append(concat_tracks(*tracks))

    # Concatenate across individual rounds
    if n_rounds > 1:
        return concat_tracks(*results)
    else:
        return results[0]

def strobe_generator(n_tracks, track_generator, geometry="sphere", motion_kwargs={},
    dz=0.7, frame_interval=0.00548, loc_error=0.035, n_gaps=0, bleach_prob=0.1,
    n_rounds=1, allow_start_outside=True, **geometry_kwargs):
    """
    Simulate a stroboscopic SPT experiment with an arbitrary generator for the 
    underlying motion. This is useful, for example, when simulating trajectories
    with state parameters that may not correspond to discrete states.

    args
    ----
        n_tracks            :   int, the number of trajectories to simulate
        track_generator     :   function with signature (n_tracks, **motion_kwargs),
                                the generator for the trajectories
        geometry            :   str, the type of observation geometry
        motion_kwargs       :   kwargs to *track_generator*, if relevant
        dz                  :   float, focal depth in um 
        frame_interval      :   float, frame interval in seconds
        loc_error           :   float, localization error in microns
        n_gaps              :   int, number of gaps to tolerate during tracking
        bleach_prob         :   float, probability to bleach in one frame
        n_rounds            :   int, the number of replicates of the entire
                                simulation to do
        allow_start_outside :   bool, allow trajectories to start outside
                                the focal volume
        geometry_kwargs     :   additional keyword arguments to the geometry
                                simulator in *strobesim.geometry*

    returns
    -------
        pandas.DataFrame, the trajectories, with columns ["trajectory",
            "frame", "y", "x", "z"]

    """
    # Multiple rounds of simulation to limit active memory, if desired
    results = []
    for round_ in range(n_rounds):

        # Generate trajectories
        tracks = track_generator(n_tracks, **motion_kwargs)

        # Simulate tracking in the desired geometry
        tracks = GEOMETRIES[geometry](tracks, dz=dz, loc_error=loc_error,
            n_gaps=n_gaps, bleach_prob=bleach_prob,
            allow_start_outside=allow_start_outside, **geometry_kwargs)

        results.append(tracks)

    # Concatenate across all individual rounds 
    if n_rounds > 1:
        return concat_tracks(*results)
    else:
        return results[0]
