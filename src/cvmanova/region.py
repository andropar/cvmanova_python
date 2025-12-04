"""
Region-of-interest (ROI) analysis for cvManova.

This module provides functions for computing cross-validated MANOVA
on predefined regions of interest.
"""

import numpy as np
import warnings
from typing import Optional, Union
from pathlib import Path

from .core import CvManovaCore
from .io import load_data_spm


def cv_manova_region(
    Ys: list[np.ndarray],
    Xs: list[np.ndarray],
    Cs: list[np.ndarray],
    fE: np.ndarray,
    region_indices: list[np.ndarray],
    permute: bool = False,
    lambda_: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-validated MANOVA on regions of interest.

    Parameters
    ----------
    Ys : list of ndarray
        Per-session data matrices, each of shape (n_scans, n_voxels).
    Xs : list of ndarray
        Per-session design matrices.
    Cs : list of ndarray
        Contrast vectors or matrices.
    fE : array-like
        Per-session error degrees of freedom.
    region_indices : list of ndarray
        List of arrays containing mask voxel indices for each region.
    permute : bool, optional
        Whether to compute permutation values (default: False).
    lambda_ : float, optional
        Regularization parameter, 0-1 (default: 0).

    Returns
    -------
    D : ndarray
        Pattern distinctness, shape (n_contrasts, n_perms, n_regions).
    p : ndarray
        Number of voxels in each region.

    Examples
    --------
    >>> D, p = cv_manova_region(Ys, Xs, Cs, fE, region_indices)
    >>> print(f"Region 1, Contrast 1: D = {D[0, 0, 0]:.6f}")
    """
    print("\ncvManovaRegion\n")

    fE = np.asarray(fE)
    n_regions = len(region_indices)
    fE_min = np.sum(fE) - np.max(fE)

    # Initialize CvManovaCore
    cmc = CvManovaCore(Ys, Xs, Cs, fE, permute=permute, lambda_=lambda_)

    # Compute for each region
    print("Computing cross-validated MANOVA on regions")
    p = np.array([len(ri) for ri in region_indices])
    D_list = []

    for i, rmvi in enumerate(region_indices):
        # Suppress near-singular matrix warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*singular.*")
            D_i = cmc.compute(rmvi)

        # Check if data is sufficient
        if p[i] > fE_min * 0.9:
            print(
                f"  Warning: data insufficient for the {p[i]} voxels of region {i + 1}!"
            )
            D_i = np.full_like(D_i, np.nan)

        D_list.append(D_i)

    # Stack and reshape results
    D = np.stack(D_list, axis=-1)
    D = D.reshape(cmc.n_contrasts, cmc.n_perms, n_regions)

    return D, p


def cv_manova_region_from_spm(
    dir_name: Union[str, Path],
    regions: list,
    Cs: list[np.ndarray],
    permute: bool = False,
    lambda_: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-validated MANOVA on regions, loading data from SPM.mat.

    Parameters
    ----------
    dir_name : str or Path
        Directory containing SPM.mat file.
    regions : list
        Region mask(s) as logical 3D volumes or filenames.
    Cs : list of ndarray
        Contrast vectors or matrices.
    permute : bool, optional
        Whether to compute permutation values (default: False).
    lambda_ : float, optional
        Regularization parameter, 0-1 (default: 0).

    Returns
    -------
    D : ndarray
        Pattern distinctness, shape (n_contrasts, n_perms, n_regions).
    p : ndarray
        Number of voxels in each region.
    """
    if not regions:
        raise ValueError("No region mask specified!")

    # Ensure dir_name ends with separator
    dir_name = Path(dir_name)

    # Load data
    Ys, Xs, mask, misc = load_data_spm(dir_name, regions)

    # Run analysis
    D, p = cv_manova_region(
        Ys, Xs, Cs, misc["fE"], misc["rmvi"], permute=permute, lambda_=lambda_
    )

    return D, p
