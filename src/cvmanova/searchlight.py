"""
Searchlight analysis implementation.

This module provides a general-purpose searchlight framework for
applying functions to data within sliding spherical windows over
a brain volume.
"""

import numpy as np
import os
import time
import pickle
from typing import Callable, Optional, Any
from pathlib import Path

from .core import CvManovaCore
from .utils import sl_size


def run_searchlight(
    mask: np.ndarray,
    sl_radius: float,
    fun: Callable[[np.ndarray], np.ndarray],
    checkpoint: Optional[str] = None,
    progress_interval: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    General purpose searchlight analysis.

    Apply a function to data contained in a sliding spherical window.

    Parameters
    ----------
    mask : ndarray
        3-dimensional logical array indicating which voxels to use.
    sl_radius : float
        Radius of searchlight in voxels.
    fun : callable
        Function to call with voxel indices within window.
        Must accept (mvi,) where mvi is a 1D array of mask voxel indices.
        Must return a 1D array of results.
    checkpoint : str, optional
        Name of checkpoint file (None to disable checkpointing).
    progress_interval : float, optional
        Interval in seconds for progress reports (default: 30).

    Returns
    -------
    res : ndarray
        Results, shape (n_volume_voxels, n_output_values).
    p : ndarray
        Number of voxels in each searchlight, shape (n_volume_voxels,).

    Notes
    -----
    The function is called with mvi as a column vector of linear indices
    into the mask voxels. Indices are sorted by increasing distance from
    the center, so the first index refers to the center voxel.

    A voxel is included in the searchlight if its distance from the center
    is smaller than or equal to the radius.

    Intermediate results are saved at regular intervals to the checkpoint
    file if given. On a subsequent run, if the checkpoint file exists, its
    contents are loaded and the computation continues from that point.
    """
    # Normalize checkpoint file name
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if checkpoint.suffix != ".pkl":
            checkpoint = checkpoint.with_suffix(".pkl")

    # Volume dimensions
    dim = mask.shape
    n_volume_voxels = np.prod(dim)
    n_mask_voxels = np.sum(mask)

    # Determine searchlight voxel offsets relative to center voxel
    r = int(np.ceil(sl_radius))
    coords = np.arange(-r, r + 1)
    dxi, dyi, dzi = np.meshgrid(coords, coords, coords, indexing="ij")

    # Prototype searchlight (sphere mask)
    PSL = dxi**2 + dyi**2 + dzi**2 <= sl_radius**2

    # Flatten offsets for voxels within sphere
    dxi_flat = dxi[PSL]
    dyi_flat = dyi[PSL]
    dzi_flat = dzi[PSL]

    # Compute linear index offsets
    # For a volume of shape (dim[0], dim[1], dim[2]) with 'F' order
    # linear index = x + y * dim[0] + z * dim[0] * dim[1]
    di = (
        dxi_flat.astype(np.int64)
        + dyi_flat.astype(np.int64) * dim[0]
        + dzi_flat.astype(np.int64) * dim[0] * dim[1]
    )

    # Sort offsets by increasing distance from center
    distances = dxi_flat**2 + dyi_flat**2 + dzi_flat**2
    sort_idx = np.argsort(distances)
    di = di[sort_idx]
    dxi_flat = dxi_flat[sort_idx]
    dyi_flat = dyi_flat[sort_idx]
    dzi_flat = dzi_flat[sort_idx]

    # Mapping from volume to mask voxel indices
    vvi2mvi = np.full(n_volume_voxels, -1, dtype=np.int64)
    mask_flat = mask.ravel(order="F")
    vvi2mvi[mask_flat] = np.arange(n_mask_voxels)

    # Get output size by calling function with empty input
    dummy_output = fun(np.array([], dtype=np.int64))
    n_outputs = len(dummy_output)

    # Initialize result arrays
    res = np.full((n_volume_voxels, n_outputs), np.nan)
    p = np.full(n_volume_voxels, np.nan)

    start_time = time.time()
    last_report_time = start_time
    cvvi = 0  # Searchlight center volume voxel index
    cmvi = 0  # Searchlight center mask voxel index

    # Load checkpoint if exists
    if checkpoint is not None and checkpoint.exists():
        with open(checkpoint, "rb") as f:
            ckpt = pickle.load(f)
        res = ckpt["res"]
        p = ckpt["p"]
        cvvi = ckpt["cvvi"]
        cmvi = ckpt["cmvi"]
        print(f"  *restart*  {cmvi:6d} voxels  {100 * cmvi / n_mask_voxels:5.1f} %")

    # Main loop
    while cvvi < n_volume_voxels:
        # Is center within mask?
        if mask_flat[cvvi]:
            # Searchlight center coordinates
            xi = cvvi % dim[0]
            yi = (cvvi // dim[0]) % dim[1]
            zi = cvvi // (dim[0] * dim[1])

            # Filter offsets to stay within volume boundaries
            valid = (
                (xi + dxi_flat >= 0)
                & (xi + dxi_flat < dim[0])
                & (yi + dyi_flat >= 0)
                & (yi + dyi_flat < dim[1])
                & (zi + dzi_flat >= 0)
                & (zi + dzi_flat < dim[2])
            )

            # Searchlight voxel volume indices
            vvi = cvvi + di[valid]

            # Discard out-of-mask voxels
            in_mask = mask_flat[vvi]
            vvi = vvi[in_mask]

            # Translate to mask voxel indices
            mvi = vvi2mvi[vvi]

            # Call function and store output
            res[cvvi, :] = fun(mvi)
            p[cvvi] = len(mvi)

            cmvi += 1

        # Progress report and checkpointing
        current_time = time.time()
        if (
            current_time - last_report_time > progress_interval
        ) or cvvi == n_volume_voxels - 1:
            elapsed = current_time - start_time
            print(
                f" {elapsed / 60:6.1f} min  {cmvi:6d} voxels  "
                f"{100 * cmvi / n_mask_voxels:5.1f} %"
            )
            last_report_time = current_time

            # Save checkpoint (not on final iteration)
            if checkpoint is not None and cvvi < n_volume_voxels - 1:
                with open(checkpoint, "wb") as f:
                    pickle.dump(
                        {"res": res, "p": p, "cvvi": cvvi + 1, "cmvi": cmvi}, f
                    )

        cvvi += 1

    # Delete checkpoint file after completion
    if checkpoint is not None and checkpoint.exists():
        checkpoint.unlink()

    return res, p


def cv_manova_searchlight(
    Ys: list[np.ndarray],
    Xs: list[np.ndarray],
    mask: np.ndarray,
    sl_radius: float,
    Cs: list[np.ndarray],
    fE: np.ndarray,
    permute: bool = False,
    lambda_: float = 0.0,
    checkpoint: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Cross-validated MANOVA on searchlight.

    Parameters
    ----------
    Ys : list of ndarray
        Per-session data matrices, each of shape (n_scans, n_voxels).
    Xs : list of ndarray
        Per-session design matrices.
    mask : ndarray
        3D logical mask indicating which voxels to use.
    sl_radius : float
        Radius of the searchlight sphere in voxels.
    Cs : list of ndarray
        Contrast vectors or matrices.
    fE : array-like
        Per-session error degrees of freedom.
    permute : bool, optional
        Whether to compute permutation values (default: False).
    lambda_ : float, optional
        Regularization parameter, 0-1 (default: 0).
    checkpoint : str, optional
        Base name for checkpoint file.

    Returns
    -------
    D : ndarray
        Pattern discriminability, shape (n_volume_voxels, n_contrasts, n_perms).
    p : ndarray
        Number of voxels per searchlight, shape (n_volume_voxels,).
    n_contrasts : int
        Number of contrasts.
    n_perms : int
        Number of permutations.

    Notes
    -----
    This function initializes the CvManovaCore and runs the searchlight
    analysis. For more control, use run_searchlight directly with a
    CvManovaCore instance.
    """
    print("\ncvManovaSearchlight\n")

    # Check searchlight size vs degrees of freedom
    fE = np.asarray(fE)
    fE_min = np.sum(fE) - np.max(fE)
    p_max = sl_size(sl_radius)

    if p_max > fE_min * 0.9:
        raise ValueError(f"Data insufficient for searchlight of size {p_max}!")

    print("Computing cross-validated MANOVA on searchlight")
    print(f"  Searchlight size: {p_max}")

    # Initialize CvManovaCore
    cmc = CvManovaCore(Ys, Xs, Cs, fE, permute=permute, lambda_=lambda_)

    # Run searchlight
    D, p = run_searchlight(
        mask=mask,
        sl_radius=sl_radius,
        fun=cmc.compute,
        checkpoint=checkpoint,
    )

    # Reshape to separate contrast and permutation dimensions
    n_contrasts = cmc.n_contrasts
    n_perms = cmc.n_perms
    D = D.reshape(-1, n_contrasts, n_perms)

    return D, p, n_contrasts, n_perms
