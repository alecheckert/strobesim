# strobesim
Simulate stroboscopic single particle tracking experiments with different kinds of motion

## Purpose

`strobesim` is a simple Python tool that makes short trajectories for different kinds
of motion, then simulates the act of observing them in a thin focal plane, as 
would be encountered in a typical stroboscopic single particle tracking (SPT)
experiment. 

`strobesim` supplements a manuscript in preparation that uses these simulations
to verify the accuracy of tools to retrieve diffusion coefficients and other
parameters governing motion.

## Details

Three types of motion are currently supported - regular (Gaussian) Brownian motion,
fractional Brownian motion, and Levy flights. More may be added in the future.

Two types of simulation "geometries" are available - a sphere or a "plane"
with some finite thickness and
infinite XY extent. A sphere is appropriate to model motion inside a confined
space such as a cell nucleus, while the plane is useful for assessing the degree
to which confinement affects the results.

## Dependencies

`numpy`, `pandas`, `hankel`. You can get `hankel` at [`conda-forge`](https://anaconda.org/conda-forge/hankel). 

## Install

Run
```
    python setup.py install
```

from the top-level `strobesim` directory.

## Usage 

The main tool in `strobesim` is the
`strobe_multistate` command. This takes a set of simulation 
parameters (see examples below) and outputs the observed trajectories
as a `pandas.DataFrame`. Each row is a separate point,
which belongs to one trajectory. This dataframe has the following columns:

 - `frame`: the index of the frame in which the point was found
 - `trajectory`: the index of the trajectory to which this point was assigned
 - `y`: the y-position of the point in microns
 - `x`: the x-position of the point in microns
 - `z`: the z-position of the point in microns

If a point belongs to a trajectory that has bleached, or is found 
outside the focal volume at any given frame, it is not included in 
the output `DataFrame`. 

### Examples

To simulate a single regular Brownian motion:
```
    from strobesim import strobe_multistate

    tracks = strobe_multistate(
        10000,   # Simulate 10000 trajectories. Note that some of 
                 # these may defocalize or bleach before entering
                 # the focal volume, and will not be observed
        3.0,     # diffusion coefficient, microns squared per sec
        1.0,     # state occupancy
        motion="brownian",
        geometry="sphere",
        radius=5.0,          # radius of sphere in microns
        dz=0.7,              # thickness of focal volume in microns
        frame_interval=0.01, # frame interval in seconds
        loc_error=0.035,     # 1-dimensional localization error in microns
        track_len=100,       # frames
        bleach_prob=0.1      # probability to bleach per frame
    )

```

To simulate three regular Brownian motion states (slow, medium, and 
fast), 
```
    tracks = strobe_multistate(
        10000,   # 10000 trajectories
        [0.1, 3.0, 8.0],     # diffusion coefficient, microns squared per sec
        [0.3, 0.5, 0.2],     # state occupancies
        motion="brownian",
        geometry="sphere",
        radius=5.0,
        dz=0.7,
        frame_interval=0.01,
        loc_error=0.035,
        track_len=100,
        bleach_prob=0.1
    )
   
```

To simulate two Levy flight states with stability parameters 2.0
and 1.5 and disperion parameters 0.1 and 3.0:
```
    tracks = strobe_multistate(
        10000,   # 10000 trajectories
        [0.1, 3.0],
        [0.3, 0.7],
        motion="levy",
        motion_kwargs=[{'alpha': 2.0}, {'alpha': 1.5}], # for each state
        geometry="sphere",
        radius=5.0,
        dz=0.7,
        frame_interval=0.01,
        loc_error=0.035,
        track_len=100,
        bleach_prob=0.1
    )

```

To simulate two fractional Brownian motion states in a planar 
geometry, killing trajectories that leave the focal volume without
allowing them to reenter:
```
    tracks = strobe_multistate(
        10000,   # 10000 trajectories
        [0.1, 3.0],
        [0.3, 0.7],
        motion="fbm",
        motion_kwargs=[{'hurst': 0.3}, {'hurst': 0.5}], # for each state
        geometry="plane",
        dz=0.7,
        frame_interval=0.01,
        loc_error=0.035,
        track_len=10,
        bleach_prob=0,
        allow_start_outside=False   # don't let trajectories reenter
                                    # after they leave the focal volume
    )

```

and so on. Consult the docstring for `strobesim.multistate.strobe_multistate`
for more information about the accepted arguments.
