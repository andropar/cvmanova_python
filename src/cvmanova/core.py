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

    This class implements the core computational method for cross-validated
    MANOVA. It uses a two-step approach: first initialize with data and
    parameters, then call compute() for different voxel sets.

    Parameters
    ----------
    Ys : list of ndarray
        Per-session data matrices, each of shape (n_scans, n_voxels).
    Xs : list of ndarray
        Per-session design matrices, each of shape (n_scans, n_regressors).
    Cs : list of ndarray
        Contrast vectors or matrices.
    fE : array-like
        Per-session error degrees of freedom.
    permute : bool, optional
        Whether to compute permutation values (default: False).
    lambda_ : float, optional
        Regularization parameter, 0-1 (default: 0).

    Attributes
    ----------
    m : int
        Number of sessions.
    n : ndarray
        Number of scans per session.
    n_contrasts : int
        Number of contrasts.
    n_perms : int
        Number of permutations.

    Notes
    -----
    It is assumed that the data and design matrices have been whitened
    and possibly filtered. fE is the residual number of degrees of freedom,
    i.e., the number of scans per session minus the rank of the design
    matrix and minus further loss of dfs due to filtering.

    Examples
    --------
    >>> # Initialize with data
    >>> cmc = CvManovaCore(Ys, Xs, Cs, fE)
    >>> # Compute for specific voxels
    >>> D = cmc.compute(voxel_indices)
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
