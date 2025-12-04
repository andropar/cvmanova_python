"""
Utility functions for cvManova.

This module contains helper functions for sign permutations, contrast
estimability checking, searchlight size calculation, and checksums.
"""

import numpy as np
from numpy.linalg import svd, norm
from scipy.linalg import orth


def sign_permutations(n: int, max_perms: int = 5000) -> tuple[np.ndarray, int]:
    """
    Generate sign permutations.

    Parameters
    ----------
    n : int
        Number of data points (sessions/folds).
    max_perms : int, optional
        Maximum number of permutations (default: 5000).

    Returns
    -------
    perms : ndarray
        Permutations, n x n_perms array of +1/-1 values.
    n_perms : int
        Number of permutations.

    Notes
    -----
    Permutations are randomly selected if the full enumeration is larger
    than max_perms. The first permutation is always the neutral permutation
    (all ones).
    """
    if 2**n <= max_perms:
        # Full enumeration of permutations
        n_perms = 2**n
        # Generate all binary combinations
        indices = np.arange(n_perms)
        perms = np.zeros((n, n_perms), dtype=np.int8)
        for i in range(n):
            perms[i, :] = (indices >> i) & 1
    else:
        # Random (Monte Carlo) selection of permutations
        n_perms = max_perms
        perms = np.zeros((n, n_perms), dtype=np.int8)
        # First column is neutral permutation (all zeros -> all +1 after transform)
        perms[:, 1:] = (np.random.rand(n, n_perms - 1) > 0.5).astype(np.int8)

    # Convert 0/1 to +1/-1
    perms = (-1) ** perms

    return perms, n_perms


def null_space(A: np.ndarray, tol: float = None) -> np.ndarray:
    """
    Compute the null space of a matrix.

    Parameters
    ----------
    A : ndarray
        Input matrix.
    tol : float, optional
        Tolerance for singular values. Default is max(M, N) * eps * max(s).

    Returns
    -------
    ndarray
        Orthonormal basis for the null space of A.
    """
    # Ensure float type for SVD
    A = np.asarray(A, dtype=np.float64)
    u, s, vh = svd(A, full_matrices=True)
    M, N = A.shape
    if tol is None:
        tol = max(M, N) * np.finfo(np.float64).eps * s.max() if s.size > 0 else 0
    rank = np.sum(s > tol)
    return vh[rank:].T.conj()


def inestimability(C: np.ndarray, X: np.ndarray) -> float:
    """
    Compute the degree of inestimability of a contrast w.r.t. a design matrix.

    Parameters
    ----------
    C : ndarray
        Contrast matrix (column vectors).
    X : ndarray
        Design matrix.

    Returns
    -------
    float
        Degree of inestimability. For an estimable contrast, this should be 0
        (save for numerical error). A number smaller than 1 indicates that the
        contrast has an estimable part. For a completely inestimable contrast,
        ie = 1.

    Notes
    -----
    Maximum "0" observed so far: 4.79 * eps
    """
    # Extend C to match X columns if needed
    if C.shape[0] < X.shape[1]:
        C_extended = np.zeros((X.shape[1], C.shape[1] if C.ndim > 1 else 1))
        C_extended[: C.shape[0], :] = C.reshape(C.shape[0], -1)
        C = C_extended

    # Determine orthonormal basis of the range of the contrast matrix
    RC = orth(C)

    # Determine orthonormal basis of the null space of the design matrix
    NX = null_space(X)

    # NX.T @ RC is a matrix of the inner products ("correlations")
    # of all the C-range vectors with all the X-null vectors.
    # The worst-case proportion across the range of C is given by the matrix 2-norm
    if NX.size == 0 or RC.size == 0:
        return 0.0
    return norm(NX.T @ RC, ord=2)


def sl_size(sl_radius: float) -> int:
    """
    Calculate searchlight size for a given radius.

    Parameters
    ----------
    sl_radius : float
        Searchlight radius in voxels.

    Returns
    -------
    int
        Number of voxels in the searchlight sphere.

    Notes
    -----
    A voxel is included if its distance from the center is <= radius.
    """
    # Distances from center voxel on grid
    r = int(np.ceil(sl_radius))
    coords = np.arange(-r, r + 1)
    dxi, dyi, dzi = np.meshgrid(coords, coords, coords, indexing="ij")
    d = np.sqrt(dxi**2 + dyi**2 + dzi**2)

    return int(np.sum(d <= sl_radius))


def sl_size_table(max_radius: float = 5.0) -> None:
    """
    Print a table of searchlight radii and sizes.

    Parameters
    ----------
    max_radius : float, optional
        Maximum searchlight radius to tabulate (default: 5.0).
    """
    r = int(np.ceil(max_radius))
    coords = np.arange(-r, r + 1)
    dxi, dyi, dzi = np.meshgrid(coords, coords, coords, indexing="ij")
    d = np.sqrt(dxi**2 + dyi**2 + dzi**2)

    # Get unique distances up to max_radius
    unique_r = np.unique(d[d <= max_radius])

    print("slRadius  pMax")
    print("--------  ----")
    for radius in unique_r:
        p_max = int(np.sum(d <= radius))
        print(f"  {radius:<5g}   {p_max:4d}")


def fletcher16(data: bytes | np.ndarray | str) -> int:
    """
    Compute Fletcher-16 checksum.

    Parameters
    ----------
    data : bytes, ndarray, or str
        Data to be checksummed. If string, will be encoded as UTF-8.
        If ndarray, must contain integers in range 0-255.

    Returns
    -------
    int
        Fletcher-16 checksum (16-bit integer).

    References
    ----------
    https://en.wikipedia.org/wiki/Fletcher%27s_checksum
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    if isinstance(data, np.ndarray):
        data = data.flatten().astype(np.uint8)
    else:
        data = np.frombuffer(data, dtype=np.uint8)

    sum1 = int(np.sum(data) % 255)
    sum2 = int(np.sum(np.cumsum(data)) % 255)

    return sum2 * 256 + sum1
