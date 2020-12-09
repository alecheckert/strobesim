#!/usr/bin/env python
"""
utils.py -- utilities for the strobesim.multistate module

"""
import warnings
import numpy as np 
import pandas as pd 

#########################
## TRAJECTORY HANDLING ##
#########################

def concat_tracks(*tracks):
    """
    Join some trajectory dataframes together into a larger dataframe,
    while preserving uniqe trajectory indices.

    args
    ----
        tracks      :   pandas.DataFrame with the "trajectory" column

    returns
    -------
        pandas.DataFrame, the concatenated trajectories

    """
    # Only include trajectory dataframes that are (1) nonempty and 
    # (2) include the "trajectory" column
    tracks = [t for t in tracks if (not t.empty) and ("trajectory" in t.columns)]
    n = len(tracks)

    # If nothing remains, return an empty DataFrame
    if n == 0:
        warnings.warn("concat_tracks: no trajectories found in input; returning empty output")
        return pd.DataFrame([], columns=["trajectory", "frame", "y", "x", "dataframe_index"])

    # Sort the tracks dataframes by their size. The only important thing
    # here is that if at least one of the tracks dataframes is nonempty,
    # we need to put that one first.
    df_lens = [len(t) for t in tracks]
    try:
        tracks = [t for _, t in sorted(zip(df_lens, tracks))][::-1]
    except ValueError:
        pass

    # Iteratively concatenate each dataframe to the first while 
    # incrementing the trajectory index as necessary
    out = tracks[0].assign(dataframe_index=0)
    c_idx = out["trajectory"].max() + 1

    for t in range(1, n):

        # Get the next set of trajectories and keep track of the origin
        # dataframe
        new = tracks[t].assign(dataframe_index=t)

        # Ignore negative trajectory indices (facilitating a user filter)
        new.loc[new["trajectory"]>=0, "trajectory"] += c_idx 

        # Increment the total number of trajectories
        c_idx = new["trajectory"].max() + 1

        # Concatenate
        out = pd.concat([out, new], ignore_index=True, sort=False)

    return out 
