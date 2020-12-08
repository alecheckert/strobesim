#!/usr/bin/env python
"""
fbm.py -- simulate short fractional Brownian motion trajectories

"""
import numpy as np

class MultivariateNormalRandomVector(object):
    """
    A multivariate normal random vector X, with methods
    to get instances and to combine with other multivariate
    normal random vectors.

    init
    ----
        mean :  1D ndarray of shape (N,), the mean
                vector
        C    :  2D ndarray of shape (N, N), the 
                covariance matrix

    example
    -------
        To generate 10000 instances of a 3-step 
        Wiener process, do

            # Covariance matrix
            C = np.array([
                [1, 1, 1],
                [1, 2, 2],
                [1, 2, 3]
            ])

            # Simulator object
            MN = MultivariateNormalRandomVector(C)

            # Simulate 10000 instances
            random_vecs = MN(10000)

    methods
    -------
        cholesky        :   get the Cholesky decomposition
                            of the covariance matrix

        inv_covariance  :   get the inverse of the covariance
                            matrix

        det_covariance  :   get the determinant of the covariance
                            matrix

        pdf             :   return the probability density function
                            of the random vector at a point

        add             :   add another multivariate random
                            vector to the random vector X

        marginalize     :   marginalize the distribution on 
                            a subset of the elements of the 
                            random vector

        condition       :   condition the random vector on 
                            specific values for a subset of 
                            the elements of the random vector

    """

    def __init__(self, C, mean=None):
        # Check user inputs
        C = np.asarray(C)
        assert len(C.shape) == 2
        assert C.shape[0] == C.shape[1]
        if mean is None:
            mean = np.zeros(C.shape[0], dtype="float64")
        else:
            mean = np.asarray(mean)
            assert len(mean.shape) == 1
            assert mean.shape[0] == C.shape[0]

        self.mean = mean
        self.C = C
        self.N = C.shape[0]

    @property
    def cholesky(self):
        """Cholesky decomposition of covariance matrix"""
        if not hasattr(self, "E"):
            self.E = np.linalg.cholesky(self.C)
        return self.E

    @property
    def inv_covariance(self):
        """Inverse of covariance matrix"""
        if not hasattr(self, "C_inv"):
            self.C_inv = np.linalg.inv(self.C)
        return self.C_inv

    @property
    def det_covariance(self):
        """Determinant of the covariance matrix"""
        if not hasattr(self, "C_dev"):
            self.C_det = np.linalg.det(self.C)
        return self.C_det

    def __call__(self, n=1):
        """
        Simulate *n* instances of the random vector.

        args
        ----
            n :  int

        returns
        -------
            2D ndarray of shape (n, self.N), the
                simulated vectors

        """
        E = self.cholesky
        z = np.random.normal(size=(self.N, n))
        return (E @ z).T + self.mean

    def pdf(self, x):
        """
        Return the probability density function at 
        the random vector x.

        args
        ----
            x   :   1D ndarray of shape (self.N), or
                    2D ndarray of shape (m, self.N),
                    a set of vectors

        returns
        -------
            float or 1D ndarray of shape (m,),
                the PDF at the passed points

        """
        norm = np.power(2.0 * np.pi, self.N / 2.0) * np.power(self.det_covariance, 0.5)
        x_shift = x - self.mean
        print('x_shift:')
        print(x_shift)

        if len(x.shape) == 1:
            return np.exp(-0.5 * x_shift @ self.inv_covariance @ x_shift) / norm
        else:
            return np.exp(-0.5 * (x_shift * (self.inv_covariance @ x_shift.T).T).sum(1)) / norm

    def add(self, MN):
        """
        Add another multivariate random vector to
        this random vector, returning the sum as a
        new random vector.

        Parameters
        ----------
            MN  :   MultivariateNormalRandomVector,
                    the variable to be added

        Returns
        -------
            MultivariateNormalRandomVector corresponding
                to the sum

        """
        assert self.N == MN.N
        return MultivariateNormalRandomVector(self.C + MN.C, mean=(self.mean + MN.mean))

    def linear_transformation(self, A):
        """
        Return the random vector produced by the 
        matrix multiplication A @ X, where A is a
        matrix operator and X is the random vector
        described by this object.

        args
        ----
            A   :   2D ndarray of shape (m, self.N)

        returns
        -------
            MultivariateNormalRandomVector, the product 
                random vector

        """
        assert A.shape[0] == self.N
        return MultivariateNormalRandomVector(A.T @ self.C @ A, mean=(A.T @ self.mean))

    def marginalize(self, indices):
        """
        Marginalize the random vector on some 
        subset of its elements.

        args
        ----
            indices :  list of int, the indices
                of the elements of the random 
                vector on which to marginalize.

                For instance, if indices == [0],
                then we'll marginalize on the 
                first element.

        returns
        -------
            MultivariateNormalRandomVector, the 
                marginal random vector

        """
        if isinstance(indices, int):
            indices = [indices]
        out = [i for i in range(self.N) if i not in indices]
        m = len(out)
        P = np.zeros((self.N, m))
        P[tuple(out), tuple(range(m))] = 1
        return self.linear_transformation(P)

    def condition(self, indices, values):
        """
        Condition this random vector by constraining
        some subset of its elements to have specific
        values, returning a new random vector.

        args
        ----
            indices :   list of int, the indices of the 
                        elements of X on which to
                        condition
            values :    list of float, the corresponding
                        values of each element

        returns
        -------
            MultivariateNormalRandomVector, the conditional
                random vector

        """
        if isinstance(indices, int):
            indices = [indices]
        indices = tuple(indices)
        keep = tuple([i for i in range(self.N) if i not in indices])

        # Partition the covariance matrix according to
        # conditioned and unconditioned variables
        C11 = self.C[keep, :][:, keep]
        C22 = self.C[indices, :][:, indices]
        C12 = self.C[keep, :][:, indices]
        C21 = self.C[indices, :][:, keep]

        # Do the same for the offset vector
        mean1 = self.mean[list(keep)]
        mean2 = self.mean[list(indices)]

        # Compute the offset and covariance matrix
        # for the conditional distribution
        A = C12 @ np.linalg.inv(C22)
        sub_C = C11 - A @ C21
        sub_mean = mean1 + A @ (np.asarray(values) - mean2)

        return MultivariateNormalRandomVector(sub_C, mean=sub_mean)

class FractionalBrownianMotion(MultivariateNormalRandomVector):
    """
    A fractional Brownian motion under the Riemann-
    Liouville fractional integral (rather than 
    Mandelbrot's Weyl integral representation). 

    init
    ----
        track_len   :   int, the number of steps in
                        the FBM 
        hurst       :   float between 0.0 and 1.0,
                        the Hurst parameter
        D           :   float, diffusion coefficient
        frame_interval: float, the time interval for
                        each step
        D_type      :   int, the type of diffusion
                        coefficient (either 1, 2, or 3).

    methods
    -------
        all methods for MultivariateNormalRandomVector
        get_time    :   return the set of time indices
                        on which this FBM is defined

    """
    def __init__(self, track_len=10, hurst=0.5, D=1.0, frame_interval=0.01, D_type=4):
        self.track_len = track_len 
        self.N = track_len 
        self.hurst = hurst
        self.D = D
        self.frame_interval = frame_interval
        assert D_type in [1, 2, 3, 4]
        self.D_type = D_type

        # Build the FBM covariance matrix
        if D_type == 1:
            T, S = (np.indices((track_len, track_len)) + 1) * frame_interval
            self.C = D * (
                np.power(T, 2 * hurst)
                + np.power(S, 2 * hurst)
                - np.power(np.abs(T - S), 2 * hurst)
            )
        elif D_type == 2:
            T, S = (np.indices((track_len, track_len)) + 1) * frame_interval * D
            self.C = (
                np.power(T, 2 * hurst)
                + np.power(S, 2 * hurst)
                - np.power(np.abs(T - S), 2 * hurst)
            )
        elif D_type == 3:
            self.D_mod = D / (2 * hurst * np.power(frame_interval, 2 * hurst - 1))
            T, S = (np.indices((track_len, track_len)) + 1) * frame_interval
            self.C = self.D_mod * (
                np.power(T, 2 * hurst)
                + np.power(S, 2 * hurst)
                - np.power(np.abs(T - S), 2 * hurst)
            )
        elif D_type == 4:
            self.D_mod = D / np.power(frame_interval, 2 * hurst - 1)
            T, S = (np.indices((track_len, track_len)) + 1) * frame_interval
            self.C = self.D_mod * (
                np.power(T, 2 * hurst)
                + np.power(S, 2 * hurst)
                - np.power(np.abs(T - S), 2 * hurst)
            )           

        # Set zero mean
        self.mean = np.zeros(track_len, dtype="float64")

    def get_time(self):
        """
        Return the set of times on which the 
        Brownian motion is defined.

        """
        return np.arange(self.track_len) * self.frame_interval


class BrownianMotion(FractionalBrownianMotion):
    """
    A regular Brownian motion, corresponding to
    a fractional Brownian motion with the Hurst
    parameter set to 0.5.

    init
    ----
        track_len   :   int, the number of steps in
                        the FBM 
        D           :   float, diffusion coefficient
        frame_interval: float, the time interval for
                        each step

    example
    -------
        # Generate the Brownian motion object with 
        # 10 steps, diffusion coefficient 2.5, and 
        # time interval 0.01
        BM = BrownianMotion(N=10, D=2.5, frame_interval=0.01)

        # Simulate 100000 instances
        trajs = BM(10000)

    methods
    -------
        Inherits all methods from MultivariateNormalRandomVector,
        and the get_time method from FractionalBrownianMotion

    """

    def __init__(self, track_len=8, D=1.0, frame_interval=0.01):
        kwargs = {"track_len": track_len, "D": D, "frame_interval": frame_interval}
        super(BrownianMotion, self).__init__(**kwargs)

class FractionalBrownianMotion3D(object):
    """
    A wrapper on FractionalBrownianMotion that simulates independent
    FBMs in each of 3 spatial dimensions.

    """
    def __init__(self, **kwargs):
        self.fbm = FractionalBrownianMotion(**kwargs)
        self.track_len = self.fbm.track_len 

    def __call__(self, n=1):
        """
        Simulate *n* FBMs of length *track_len*.

        returns
        -------
            3D ndarray of shape (n, track_len, 3), the ZYX
                positions of each trajectory at each frame

        """
        result = np.zeros((n, self.fbm.track_len, 3), dtype=np.float64)
        for d in range(3):
            result[:,1:,d] = self.fbm(n)[:,:-1]
        return result