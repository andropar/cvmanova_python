"""
Core cross-validated MANOVA implementation.

This module contains the core computational engine for cross-validated
MANOVA as proposed by Allefeld and Haynes (2014).

Reference:
    Allefeld, C., & Haynes, J. D. (2014). Searchlight-based multi-voxel
    pattern analysis of fMRI by cross-validated MANOVA. NeuroImage, 89,
    345-357.
"""

import numpy as np
from numpy.linalg import pinv
import warnings
from typing import Optional

from .utils import sign_permutations, inestimability


class CvManovaCore:
    """
    Cross-validated MANOVA core computation engine.

    This class implements the core computational algorithm for cross-validated
    MANOVA as described in Allefeld & Haynes (2014). It provides a reusable
    computational kernel that can be applied to different voxel sets (e.g.,
    searchlight neighborhoods or ROIs).

    The algorithm uses a leave-one-session-out cross-validation scheme to compute
    an unbiased estimate of multivariate discriminability. For each cross-validation
    fold, it trains on all but one session and tests on the held-out session,
    computing the cross-validated Mahalanobis distance between conditions.

    Parameters
    ----------
    Ys : list of ndarray
        Per-session data matrices, each with shape (n_scans, n_voxels). Data should
        be preprocessed (whitened, filtered, mean-centered) before passing to this
        class. All sessions must have the same number of voxels.
    Xs : list of ndarray
        Per-session design matrices, each with shape (n_scans, n_regressors). The
        design matrices specify the GLM for each session. The number of regressors
        can vary across sessions, but contrasts must be defined on regressors
        common to all sessions.
    Cs : list of ndarray
        Contrast matrices or vectors to test. Each contrast has shape
        (n_rows, n_regressors) and specifies a linear combination of regressors
        to test. Contrasts must be estimable (span the column space) in all sessions.
    fE : array-like
        Per-session residual degrees of freedom (length m = n_sessions). This is
        the number of scans minus the rank of the design matrix, minus any additional
        loss from filtering. Used for bias correction of discriminability estimates.
    permute : bool, default=False
        Whether to compute permutation distribution for statistical testing. If True,
        generates sign permutations and computes discriminability for each. This
        provides a null distribution for inference.
    lambda_ : float, default=0.0
        Regularization parameter for shrinkage of error covariance matrices.
        Range: [0, 1], where 0 = no regularization, 1 = diagonal covariance.
        Shrinks off-diagonal elements toward zero, which can improve stability
        for small sample sizes or high-dimensional voxel sets.

    Attributes
    ----------
    m : int
        Number of sessions in the dataset.
    n : ndarray
        Number of scans (volumes) per session as a 1D array.
    n_contrasts : int
        Number of contrast matrices to test.
    n_perms : int
        Number of permutations to compute (1 if permute=False).
    Cs : list of ndarray
        Validated and trimmed contrast matrices.
    betas : list of ndarray
        Estimated GLM parameters for each session.
    xis : list of ndarray
        GLM residuals for each session.
    sp : ndarray
        Sign permutation matrix (shape: m × n_perms).

    Methods
    -------
    compute(vi)
        Compute discriminability for a specific set of voxels.
    get_output_size()
        Return the total number of output values per voxel set.

    See Also
    --------
    SearchlightCvManova : High-level searchlight estimator
    RegionCvManova : High-level region-based estimator
    AnalysisConfig : Configuration for regularization and permutation

    Notes
    -----
    Algorithm overview:
    1. Estimate GLM parameters (betas) and residuals for each session
    2. For each cross-validation fold l:
       a. Train on all sessions except l (estimate covariance from residuals)
       b. Compute cross-validated effect between test session l and training sessions
       c. Compute Mahalanobis distance using training covariance
    3. Average discriminability across folds
    4. Apply bias correction based on degrees of freedom

    Mathematical formulation:
    The discriminability D for a contrast C is computed as:

        D = (fE_train - p - 1) / n_train * trace(H @ inv(E_train))

    where:
    - H is the cross-validated effect sum of squares
    - E_train is the error covariance from training sessions
    - p is the number of voxels
    - fE_train is the residual degrees of freedom in training sessions
    - n_train is the number of scans in training sessions

    The bias correction term (fE - p - 1) / n ensures unbiased estimation.

    Reference:
        Allefeld, C., & Haynes, J. D. (2014). Searchlight-based multi-voxel
        pattern analysis of fMRI by cross-validated MANOVA. NeuroImage, 89,
        345-357. https://doi.org/10.1016/j.neuroimage.2013.11.043

    Important assumptions:
    - Data has been whitened to remove temporal autocorrelation
    - Residuals are approximately multivariate normal
    - Design matrices have full column rank
    - Contrasts are estimable in all sessions

    Examples
    --------
    Basic usage with single voxel set:

    >>> import numpy as np
    >>> from cvmanova.core import CvManovaCore
    >>> # Simulated data: 3 sessions, 100 scans each, 50 voxels
    >>> Ys = [np.random.randn(100, 50) for _ in range(3)]
    >>> # Design matrices: 100 scans, 4 regressors
    >>> Xs = [np.random.randn(100, 4) for _ in range(3)]
    >>> # Simple contrast: first vs second regressor
    >>> C = np.array([[1, -1, 0, 0]])
    >>> Cs = [C]
    >>> # Degrees of freedom: 100 scans - 4 regressors
    >>> fE = np.array([96, 96, 96])
    >>> # Initialize core
    >>> core = CvManovaCore(Ys, Xs, Cs, fE, permute=False)
    >>> # Compute for first 20 voxels
    >>> D = core.compute(np.arange(20))
    >>> print(f"Discriminability shape: {D.shape}")
    (1,)  # 1 contrast × 1 permutation

    With permutation testing:

    >>> core = CvManovaCore(Ys, Xs, Cs, fE, permute=True)
    >>> print(f"Computing {core.n_perms} permutations")
    >>> D = core.compute(np.arange(20))
    >>> print(f"Output shape: {D.shape}")
    # Shape depends on 2^(m-1) permutations

    With regularization for high-dimensional data:

    >>> # Many voxels relative to scans - use regularization
    >>> core = CvManovaCore(Ys, Xs, Cs, fE, lambda_=0.3)
    >>> D = core.compute(np.arange(100))

    Multiple contrasts:

    >>> # Test multiple effects
    >>> C1 = np.array([[1, -1, 0, 0]])  # Contrast 1
    >>> C2 = np.array([[0, 0, 1, -1]])  # Contrast 2
    >>> Cs = [C1, C2]
    >>> core = CvManovaCore(Ys, Xs, Cs, fE)
    >>> D = core.compute(np.arange(20))
    >>> D_reshaped = D.reshape(2, 1)  # 2 contrasts × 1 permutation
    """

    def __init__(
        self,
        Ys: list[np.ndarray],
        Xs: list[np.ndarray],
        Cs: list[np.ndarray],
        fE: np.ndarray,
        permute: bool = False,
        lambda_: float = 0.0,
    ):
        self.m = len(Ys)  # Number of sessions
        self.n = np.array([Y.shape[0] for Y in Ys])  # Scans per session
        self.n_contrasts = len(Cs)
        self.fE = np.asarray(fE)
        self.lambda_ = lambda_

        # Validate input
        n_scans_X = [X.shape[0] for X in Xs]
        assert np.array_equal(self.n, n_scans_X), (
            "Inconsistent number of scans between data and design!"
        )

        n_voxels = [Y.shape[1] for Y in Ys]
        assert len(set(n_voxels)) == 1, "Inconsistent number of voxels within data!"

        # Check contrasts
        q_min = min(X.shape[1] for X in Xs)
        Cs_trimmed = []
        for ci, C in enumerate(Cs):
            C = np.atleast_2d(C)
            if C.ndim == 1:
                C = C.reshape(-1, 1)
            # Trim trailing all-zero rows
            last_nonzero = np.where(~np.all(C == 0, axis=1))[0]
            if len(last_nonzero) > 0:
                q_C = last_nonzero[-1] + 1
                C = C[:q_C, :]
            else:
                C = C[:1, :]  # Keep at least one row

            assert C.shape[0] <= q_min, (
                f"Contrast {ci + 1} exceeds the {q_min} common regressors!"
            )

            for si in range(self.m):
                ie = inestimability(C, Xs[si])
                assert ie <= 1e-6, (
                    f"Contrast {ci + 1} is not estimable in session {si + 1}!"
                )

            Cs_trimmed.append(C)

        self.Cs = Cs_trimmed

        # Estimate GLM parameters and errors, prepare design inner products
        self.betas = []
        self.xis = []
        self.XXs = []
        for si in range(self.m):
            beta = pinv(Xs[si]) @ Ys[si]
            xi = Ys[si] - Xs[si] @ beta
            XX = Xs[si].T @ Xs[si]
            self.betas.append(beta)
            self.xis.append(xi)
            self.XXs.append(XX)

        # Prepare contrast projectors
        self.CCs = []
        for C in self.Cs:
            CC = pinv(C.T) @ C.T
            self.CCs.append(CC)

        # Generate sign permutations
        sp, n_perms = sign_permutations(self.m)
        n_perms = n_perms // 2  # The two halves are equivalent
        if not permute:
            n_perms = 1  # Neutral permutation only
        self.sp = sp[:, :n_perms]
        self.n_perms = n_perms

    def get_output_size(self) -> int:
        """Return the number of output values per voxel set."""
        return self.n_contrasts * self.n_perms

    def compute(self, vi: np.ndarray) -> np.ndarray:
        """
        Compute cross-validated MANOVA for specified voxels.

        Parameters
        ----------
        vi : ndarray
            Voxel indices (into columns of Ys).

        Returns
        -------
        D : ndarray
            Pattern distinctness, shape (n_contrasts * n_perms,).
            Values are arranged as contrasts × permutations in row-major order.
        """
        vi = np.asarray(vi).flatten()
        p = len(vi)
        m = self.m

        # Precompute per-session E (error covariance for selected voxels)
        Es = []
        for k in range(m):
            x = self.xis[k][:, vi]
            Es.append(x.T @ x)

        # Precompute inverse of per-fold summed E
        iEls = []
        for l in range(m):
            ks = [k for k in range(m) if k != l]
            El = sum(Es[k] for k in ks)

            # Shrinkage regularization towards diagonal
            if self.lambda_ > 0:
                El = (1 - self.lambda_) * El + self.lambda_ * np.diag(np.diag(El))

            # Compute inverse (eye / El is faster than inv)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*singular.*")
                iEl = np.eye(El.shape[0]) @ np.linalg.inv(El)
            iEls.append(iEl)

        D = np.zeros((self.n_contrasts, self.n_perms))

        # For each contrast
        for ci in range(self.n_contrasts):
            CC = self.CCs[ci]
            q_CC = CC.shape[0]

            # Precompute per-session betaDelta
            betaDelta = []
            for k in range(m):
                bd = CC @ self.betas[k][:q_CC, :][:, vi]
                betaDelta.append(bd)

            # Precompute per-session H
            Hs = [[None] * m for _ in range(m)]
            for k in range(m):
                for l in range(m):
                    if l == k:
                        continue
                    XX_sub = self.XXs[l][:q_CC, :q_CC]
                    Hs[k][l] = betaDelta[k].T @ XX_sub @ betaDelta[l]

            # For each permutation
            for pi in range(self.n_perms):
                # For each cross-validation fold
                for l in range(m):
                    ks = [k for k in range(m) if k != l]

                    # Sign-permuted, summed H
                    Hl = np.zeros((p, p))
                    for k in ks:
                        sign = self.sp[k, pi] * self.sp[l, pi]
                        Hl += sign * Hs[k][l]

                    # Fold-wise D
                    # trace(Hl @ iEls[l]) = sum(Hl.T * iEls[l])
                    Dl = np.sum(Hl.T * iEls[l])

                    # Bias correction (fold-specific)
                    fE_sum = sum(self.fE[k] for k in ks)
                    n_sum = sum(self.n[k] for k in ks)
                    Dl = (fE_sum - p - 1) / n_sum * Dl

                    # Sum across cross-validation folds
                    D[ci, pi] += Dl

        # Mean across folds
        D = D / m

        # Return as row vector
        return D.flatten()
