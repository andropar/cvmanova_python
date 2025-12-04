"""
High-level API for cvManova.

This module provides convenient wrapper functions that combine data loading
and analysis in single calls, similar to the original MATLAB API.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union

from .io import load_data_spm, write_image
from .searchlight import cv_manova_searchlight
from .region import cv_manova_region
from .utils import sl_size, fletcher16


def searchlight_analysis(
    dir_name: Union[str, Path],
    sl_radius: float,
    Cs: list[np.ndarray],
    permute: bool = False,
    lambda_: float = 0.0,
    output_dir: Optional[Union[str, Path]] = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Run cross-validated MANOVA searchlight analysis from SPM.mat.

    This is the main entry point for searchlight analysis, equivalent to
    cvManovaSearchlight in the MATLAB version.

    Parameters
    ----------
    dir_name : str or Path
        Directory containing SPM.mat file.
    sl_radius : float
        Radius of the searchlight sphere in voxels.
    Cs : list of ndarray
        Contrast vectors or matrices.
    permute : bool, optional
        Whether to compute permutation values (default: False).
    lambda_ : float, optional
        Regularization parameter, 0-1 (default: 0).
    output_dir : str or Path, optional
        Directory for output files. If None, uses dir_name.

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
    Output files written to output_dir:
    - spmD_C####_P####.nii: Pattern discriminability D maps
    - spmDs_C####_P####.nii: Standardized pattern discriminability D_s maps
    - VPSL.nii: Number of voxels per searchlight
    - cmsParameters.npz: Record of analysis parameters
    """
    print("\n\ncvManovaSearchlight\n")

    dir_name = Path(dir_name)
    if output_dir is None:
        output_dir = dir_name
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique ID for checkpointing
    import datetime
    spm_date = (dir_name / "SPM.mat").stat().st_mtime
    uid_str = f"SPM.mat of {datetime.datetime.fromtimestamp(spm_date)}"
    uid_str += f"\nslRadius={sl_radius}\nCs={Cs}\npermute={permute}\nlambda={lambda_}"
    uid = format(fletcher16(uid_str), "04X")

    checkpoint_name = output_dir / f"cmsCheckpoint{uid}.pkl"

    # Load data
    Ys, Xs, mask, misc = load_data_spm(dir_name)

    # Run searchlight
    print("\nComputing cross-validated MANOVA on searchlight")
    print(f"  Searchlight size: {sl_size(sl_radius)}")

    D, p, n_contrasts, n_perms = cv_manova_searchlight(
        Ys, Xs, mask, sl_radius, Cs, misc["fE"],
        permute=permute, lambda_=lambda_,
        checkpoint=str(checkpoint_name)
    )

    # Save results
    mask_shape = mask.shape
    for ci in range(n_contrasts):
        for pi in range(n_perms):
            # Pattern discriminability D
            D_volume = D[:, ci, pi].reshape(mask_shape, order="F")
            write_image(
                D_volume,
                output_dir / f"spmD_C{ci+1:04d}_P{pi+1:04d}.nii",
                misc["mat"],
                descrip="pattern discriminability"
            )

            # Standardized D (D / sqrt(p))
            with np.errstate(invalid="ignore"):
                Ds_volume = (D[:, ci, pi] / np.sqrt(p)).reshape(mask_shape, order="F")
            write_image(
                Ds_volume,
                output_dir / f"spmDs_C{ci+1:04d}_P{pi+1:04d}.nii",
                misc["mat"],
                descrip="standardized pattern discriminability"
            )

    # Save voxels per searchlight
    p_volume = p.reshape(mask_shape, order="F")
    write_image(
        p_volume,
        output_dir / "VPSL.nii",
        misc["mat"],
        descrip="voxels per searchlight"
    )

    # Save parameters
    np.savez(
        output_dir / "cmsParameters.npz",
        sl_radius=sl_radius,
        Cs=np.array(Cs, dtype=object),
        permute=permute,
        lambda_=lambda_,
        n_perms=n_perms,
        fE=misc["fE"],
        mat=misc["mat"]
    )

    return D, p, n_contrasts, n_perms


def region_analysis(
    dir_name: Union[str, Path],
    regions: list,
    Cs: list[np.ndarray],
    permute: bool = False,
    lambda_: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run cross-validated MANOVA region analysis from SPM.mat.

    This is the main entry point for ROI analysis, equivalent to
    cvManovaRegion in the MATLAB version.

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

    Examples
    --------
    >>> D, p = region_analysis(
    ...     "path/to/spm_dir",
    ...     ["roi1.nii", "roi2.nii"],
    ...     [contrast1, contrast2]
    ... )
    >>> # Print results
    >>> for ri in range(D.shape[2]):
    ...     for ci in range(D.shape[0]):
    ...         print(f"Region {ri+1}, Contrast {ci+1}: D = {D[ci, 0, ri]:.6f}")
    """
    from .region import cv_manova_region_from_spm

    return cv_manova_region_from_spm(
        dir_name, regions, Cs, permute=permute, lambda_=lambda_
    )
