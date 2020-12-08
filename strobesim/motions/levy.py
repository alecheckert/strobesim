#!/usr/bin/env python
"""
levy.py -- simulate 3D Levy flights

"""
# Numeric
import numpy as np 

# Cubic spline interpolation
from scipy.interpolate import InterpolatedUnivariateSpline as spline

# Custom utilities
from .utils import (
    pdf_from_cf_rad,  # get 2D/3D radial PDFs from characteristic functions
    radnorm           # marginalize hyperspherical densities on the angular parts
)

# Sample points uniformly from the surface of a 3D sphere
from ..utils import sample_sphere 

class LevyFlight3D(object):
    """
    Simulate a 3D Levy flight using an approximate inverse CDF approach.

    description of stochastic process
    ---------------------------------
        LevyFlight3D simulates a Markovian diffusion process whose 3D
        radial displacements are distributed according to a Levy stable
        random vector, and whose angular displacements are isotropically
        distributed on the surface of the unit sphere.

    description of simulation method
    --------------------------------
        LevyFlight3D uses one of two different methods to approximate the CDF
        for the radial displacements of a Levy flight, which is subsequently
        used for inverse CDF sampling. Both methods are based on the inverse
        Fourier transform of the Levy flight's characteristic function.

        The first method, "hankel", uses an explicit 3D inverse spherical Fourier
        transform of the CF to evaluate the PDF at a discrete set of bins.

        The second method, "radon", uses a convenient formula derived from the
        inverse Radon transform of the PDF. This corresponds to a modification of
        the CF and then an inverse 1D Fourier transform.

        "radon_alt" is a slight modification of "radon" that may be more 
        accurate for very long displacements.

        In both cases, once the PDF is evaluated at a finite set of bins, it
        is accumulated to approximate the CDF. A spline interpolation of the 
        CDF is then used to generate the inverse CDF function for sampling.

        Both methods improve in accuracy as the bin size is decreased. Ideally
        the bin size should be small enough that the CDF is approximately linear
        in the space between each bin, at which point the spline interpolation
        becomes very good.

        The direction of the displacements is chosen by sampling points 
        uniformly from the surface of the unit sphere.

    init
    ----
        R_max       :   float, the maximum radial displacement to consider in um
        bin_size    :   float, the size of the interpolation bins in um
        alpha       :   float between 0.0 and 2.0, Levy stability parameter
        D           :   float, the dispersion parameter
        frame_interval: float, frame interval in seconds
        method      :   str, either "radon", "radon_alt", or "hankel", the method
                        to use to simulate the radial displacements of the Levy flight.
                        These are mathematically identical ways to accomplish
                        the same thing. "hankel" is more naive, but is also slower.
        track_len   :   int, the number of "frames" (points) per trajectory

    example simulation
    ------------------
        # Generate the simulator object. Here, we use Brownian motion, considering
        # displacements up to 40.0 um
        L = LevyFlight3D(40.0, alpha=2.0, D=2.0, frame_interval=0.01, method="radon")

        # Generate 100000 trajectories of length 10
        tracks = L(100000, track_len=10)

    methods
    -------
        cf              :   evaluate the characteristic function
        pdf             :   evaluate the (approximate) probability mass function
        cdf             :   evaluate the (approximate) cumulative distribution function
        cdf_dev         :   evaluate the first derivative of the (approximate)
                            cumulative distribution function
        inverse_cdf     :   evaluate the (approximate) inverse cumulative distribution
                            function

    """
    def __init__(self, R_max=20.0, bin_size=0.001, alpha=2.0, D=1.0, frame_interval=0.01,
        method="radon", track_len=10):

        assert method in ["radon", "hankel", "radon_alt"]

        self.R_max = R_max 
        self.bin_size = bin_size 
        self.alpha = alpha 
        self.D = D 
        self.frame_interval = frame_interval
        self.method = method 
        self.track_len = track_len 

        # Generate real and frequency domain supports
        self._generate_support()

    def __call__(self, N, track_len=None, n_iter=20):
        """
        Simulate *N* instances of this Levy flight.

        args
        ----
            N           :   int, the number of Levy flights to simulate
            track_len   :   int, the length of each trajectory to simulate
            n_iter      :   int, the number of iterations of Newton's method
                            to use for inverse CDF sampling

        returns
        ------- 
            3D ndarray, shape (N, track_len, 3), the ZYX positions of each
                Levy flight at each frame

        """
        if track_len is None:
            track_len = self.track_len 

        # Generate the radial displacements
        p = np.random.random(size=(N * (track_len-1)))
        r = self.inverse_cdf(p, n_iter=n_iter).reshape((N, track_len-1))

        # Generate the angular displacements
        v = sample_sphere((N, track_len), d=3)

        # The first position for each trajectory is always zero
        v[:,0,:] = 0

        # Multiply radial and angular displacements
        for t in range(1, track_len):
            v[:,t,:] = (v[:,t,:].T * r[:,t-1]).T 

        # Accumulate the displacements to generate trajectories
        v = np.cumsum(v, axis=1)
        return v 

    def _generate_support(self):
        """
        Generate the real domain and frequency domain supports for this Levy flight.

        """
        # The edges of each interpolation bin. Note that for the Radon method, 
        # we need to evaluate twice as many points in the real domain as the user
        # specifies, to sufficiently separate the positive and negative parts of 
        # the Radon transform.
        self.r_full = np.arange(0.0, 2*self.R_max+self.bin_size, self.bin_size)

        # The centers of each interpolation bin 
        self.rc_full = self.r_full[:-1] + self.bin_size * 0.5

        # The last real domain point to use, for the "radon" method
        self.cutoff = self.r_full.shape[0] // 2

        # Truncated versions of the real domain supports
        self.r = self.r_full[:self.cutoff+1]
        self.rc = self.rc_full[:self.cutoff]

        # The frequency domain 
        self.k = 2.0 * np.pi * np.fft.rfftfreq(self.r_full.shape[0], d=self.bin_size)

    def cf(self, k=None):
        """
        Evaluate the characteristic function for the 3D radial displacements of 
        this Levy flight, expressed as a function of the radial frequency
        coordinate. That is, 

            k^2 = k_{x}^2 + k_{y}^2 + k_{z}^2 


        args
        ----
            k       :   1D ndarray. If *None*, default to this object's initial
                            support (self.k)

        returns
        -------
            1D ndarray, dtype float64, shape k.shape; the CF

        """
        if k is None:
            k = self.k
        return np.exp(-self.D * self.frame_interval * np.power(np.abs(k), self.alpha))

    def pdf(self, r=None, normalize=True):
        """
        Evaluate the probability density function for the 3D radial 
        displacements of this Levy flight. Note that this is really a 
        scaled approximation to the PMF rather than the PDF, since 
        we may evaluate it on a support with different bin size than the 
        support originally used to generate the PDF (self.rc).

        args
        ----
            r           :   1D ndarray, the set of points in the real
                            domain at which to evaluate the PDF
            normalize   :   bool, normalize this PDF to sum to 1

        returns
        -------
            1D ndarray, dtype float64, size r.shape; the PDF

        """
        # If not specified, default to the initial real domain support
        if r is None:
            r = self.rc[:self.cutoff]

        # Generate a spline interpolation to the PDF, if not done already
        if not hasattr(self, "_pdf_rad"):
            self._gen_pdf_rad()

        # Evaluate 
        result = self._pdf_rad(r)

        # Normalize
        if normalize:
            result /= result.sum()
        return result 

    def cdf(self, r=None):
        """
        Evaluate the cumulative distribution function for the 3D radial
        displacements of this Levy flight.

        args
        ----
            r           :   1D ndarray, the set of points at which to 
                            evaluate the CDF. If *None*, defaults to the
                            same set of bins that are used for spline 
                            interpolation

        returns
        -------
            1D ndarray, dtype float64, size r.shape, the CDF

        """
        # If not specified, default to the initial real domain support
        if r is None:
            r = self.r[:self.cutoff+1]

        # Generate a spline interpolation to the CDF, if not done already
        if not hasattr(self, "_cdf_rad"):
            self._gen_cdf_rad()

        # Evaluate
        return self._cdf_rad(r)

    def cdf_dev(self, r=None):
        """
        Evaluate the first derivative of the cumulative distribution function
        for the 3D radial displacements of this Levy flight.

        args
        ----
            r           :   1D ndarray, the set of points at which to evaluate
                            the CDF's derivative. If *None, defaults to the 
                            same set of bins that are used for spline interpolation
                            of the CDF

        returns
        -------
            1D ndarray, dtype float64, siez r.shape

        """
        if r is None:
            r = self.r[:self.cutoff+1]

        # Generate a spline interpolation to this function, if not already done
        if not hasattr(self, "_cdf_rad_dev"):
            self._gen_cdf_rad()

        # Evaluate
        return self._cdf_rad_dev(r)

    def _gen_pdf_rad(self):
        """
        Generate a spline interpolation to the PDF for the 3D radial 
        displacements of this Levy flight.

        """
        # Use a fast formula derived from the 3D Radon transform
        if self.method == "radon":
            arg = self.D * self.frame_interval * np.power(np.abs(self.k), self.alpha)
            cf_dev = (1 - self.alpha * arg) * np.exp(-arg)
            pdf = np.fft.irfft(cf_dev, n=self.r_full.shape[0])[:self.cutoff+1]
            pdf /= pdf.sum()

        # Similar to the "radon" method, but somewhat more stable/accurate for 
        # larger values of the dispersion parameter
        elif self.method == "radon_alt":
            k_cf = 1.0j * self.k * self.cf(self.k)
            pdf = -self.r * np.fft.irfft(k_cf, n=self.r_full.shape[0])[:self.cutoff+1]
            pdf /= pdf.sum()

        # Perform an explicit inverse spherical Fourier transform on 
        # the radially symmetric PDF
        elif self.method == "hankel":
            pdf = pdf_from_cf_rad(self.cf, self.r)
            pdf = radnorm(self.r, pdf, d=3)

        # Generate a spline interpolation
        self._pdf_rad = spline(self.r, pdf)

    def _gen_cdf_rad(self):
        """
        Generate a spline interpolation to the CDF for the 3D radial
        displacements of this Levy flight.

        """
        # Evaluate the PDF at the center of each spatial bin
        pdf = self.pdf(self.rc)

        # If using the "hankel" method, it's possible to erroneously get 
        # negative values in the PDF, especially near the origin (Gibbs
        # phenomenon and other effects) and when using insufficient iterations
        # of the numerical Hankel transform. Even when these are of inconsequential
        # magnitude, if they are allowed to exist they compromise the monotonicity
        # of the CDF, which makes inverse CDF sampling impossible. Here, kill
        # all such instances.
        # pdf[pdf<0] = 0

        # Accumulate to get the CDF, and include the 0 point
        cdf = np.concatenate(([0], np.cumsum(pdf)))
        r = np.concatenate(([0], self.rc+self.bin_size*0.5))

        # Generate a spline interpolation to the CDF
        self._cdf_rad = spline(r, cdf)

        # Generate a spline interpolation to the first derivative of the CDF, 
        # used in inverse CDF sampling
        self._cdf_rad_dev = self._cdf_rad.derivative(1)

    def inverse_cdf(self, p, n_iter=20):
        """
        Approximate the inverse cumulative distribution function for the 3D
        radial displacements of this Levy flight.

        args
        ----
            p       :   1D ndarray, floats between 0.0 and 1.0
            n_iter  :   int, the number of iterations of Newton's method to 
                        use to refine the estimate

        returns
        -------
            1D ndarray, dtype float64, shape p.shape; the inverse CDF

        """
        # Evaluate the CDF on the native support
        pdf = self.pdf(self.rc)
        pdf[pdf<0] = 0
        cdf = np.concatenate(([0], np.cumsum(pdf)))
        r = np.concatenate(([0], self.rc))

        # Take the closest bin to each point as the initial guess
        cdf = cdf / cdf[-1]
        r0 = r[np.digitize(p, cdf)-1]

        # Do a few rounds of Newton to refine the guess
        for i in range(n_iter):
            r0 += -0.5 * (self.cdf(r0) - p) / self.cdf_dev(r0)

        return r0 
