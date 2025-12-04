"""
NIfTI I/O functions for cvManova.

This module provides functions for reading and writing NIfTI images,
including masked reading for efficiency and resampling to match
template voxel grids.
"""

import numpy as np
import nibabel as nib
from scipy.io import loadmat
from scipy.ndimage import affine_transform
from pathlib import Path
from typing import Optional, Union
import warnings


def read_vols_masked(
    volumes: Union[list, np.ndarray, str, Path],
    mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read in-mask MR image data from NIfTI files.

    Parameters
    ----------
    volumes : list, str, or Path
        List of volume file paths, or a single file path.
    mask : ndarray, optional
        Logical array with dimensions matching data volumes.
        If not specified, all voxels are read.

    Returns
    -------
    Y : ndarray
        Data of shape (n_volumes, n_mask_voxels).
    mask : ndarray
        The mask used (useful if mask was None).

    Notes
    -----
    This function reads only in-mask voxels to save memory.
    """
    # Handle single file input
    if isinstance(volumes, (str, Path)):
        volumes = [volumes]

    # Load first volume to get dimensions
    first_img = nib.load(volumes[0])
    dim = first_img.shape[:3]

    # Create or validate mask
    if mask is None:
        mask = np.ones(dim, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != dim:
            raise ValueError(f"Mask shape {mask.shape} doesn't match data {dim}")

    mask_indices = np.where(mask.ravel(order="F"))[0]
    n_mask_voxels = len(mask_indices)

    # Read data
    n_vols = len(volumes)
    Y = np.zeros((n_vols, n_mask_voxels))

    for vi, vol_path in enumerate(volumes):
        img = nib.load(vol_path)
        data = img.get_fdata()

        # Handle 4D images
        if data.ndim == 4:
            data = data[:, :, :, 0]

        # Extract masked voxels (Fortran order to match MATLAB)
        Y[vi, :] = data.ravel(order="F")[mask_indices]

        if (vi + 1) % 100 == 0 or vi == n_vols - 1:
            print(f"  {vi + 1} of {n_vols} volumes loaded")

    return Y, mask


def read_vol_matched(
    volume: Union[str, Path, nib.Nifti1Image],
    template: Union[str, Path, nib.Nifti1Image],
    order: int = 0,
) -> np.ndarray:
    """
    Read MR image data resampled to match a template voxel grid.

    Parameters
    ----------
    volume : str, Path, or Nifti1Image
        Volume to read from.
    template : str, Path, or Nifti1Image
        Template volume to match.
    order : int, optional
        Interpolation order:
        - 0: nearest neighbor (default)
        - 1: trilinear
        - 2+: higher-order spline

    Returns
    -------
    Y : ndarray
        Data (3D array) with same dimensions and physical locations as template.

    Notes
    -----
    The image data are interpolated to match the voxel grid of the template.
    This is useful for reading ROI masks that have been inverse-normalized.
    """
    # Load images
    if isinstance(volume, (str, Path)):
        vol_img = nib.load(volume)
    else:
        vol_img = volume

    if isinstance(template, (str, Path)):
        tpl_img = nib.load(template)
    else:
        tpl_img = template

    vol_data = vol_img.get_fdata()
    tpl_shape = tpl_img.shape[:3]

    # Compute affine transform from template voxels to volume voxels
    # template_voxel -> world -> volume_voxel
    # vol_inv @ tpl_affine
    tpl_affine = tpl_img.affine
    vol_affine = vol_img.affine
    vol_inv = np.linalg.inv(vol_affine)

    # Combined transform
    transform = vol_inv @ tpl_affine

    # Extract rotation/scaling and translation
    # For scipy.ndimage.affine_transform, we need the inverse mapping
    matrix = transform[:3, :3]
    offset = transform[:3, 3]

    # Apply affine transformation
    Y = affine_transform(
        vol_data,
        matrix,
        offset=offset,
        output_shape=tpl_shape,
        order=order,
        mode="constant",
        cval=0.0,
    )

    return Y


def write_image(
    data: np.ndarray,
    fname: Union[str, Path],
    affine: np.ndarray,
    descrip: str = "",
) -> nib.Nifti1Image:
    """
    Write a data array to a NIfTI image.

    Parameters
    ----------
    data : ndarray
        Data to be written (up to 7D array).
    fname : str or Path
        Name of the image file to write to.
    affine : ndarray
        4x4 transformation matrix from voxel indices to mm coordinates.
    descrip : str, optional
        Description of data.

    Returns
    -------
    img : Nifti1Image
        The saved NIfTI image object.
    """
    fname = Path(fname)

    # Handle logical data
    if data.dtype == bool:
        data = data.astype(np.uint8)

    # Create NIfTI image
    img = nib.Nifti1Image(data, affine)

    # Set description
    img.header["descrip"] = descrip.encode("utf-8")[:80]

    # Save
    nib.save(img, fname)

    return img


def load_data_spm(
    dir_name: Union[str, Path],
    regions: Optional[list] = None,
    whiten_filter: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, dict]:
    """
    Load fMRI data via SPM.mat file.

    Parameters
    ----------
    dir_name : str or Path
        Directory containing SPM.mat file.
    regions : list, optional
        Additional region mask(s) as logical 3D volumes or filenames.
    whiten_filter : bool, optional
        Whether to apply whitening and filtering (default: True).

    Returns
    -------
    Ys : list of ndarray
        MR data (within mask), one array per session of shape (scans, voxels).
    Xs : list of ndarray
        Design matrix for each session.
    mask : ndarray
        Analysis brain mask (logical 3D volume).
    misc : dict
        Additional data including:
        - 'mat': voxels to mm transformation matrix
        - 'fE': residual degrees of freedom for each session
        - 'rmvi': mask voxel indices for each region

    Notes
    -----
    Y & X are high-pass filtered and whitened if whiten_filter=True.
    Y includes only those voxels selected through the mask.

    This function loads SPM.mat files created by SPM8 or SPM12.
    """
    dir_name = Path(dir_name)
    spm_path = dir_name / "SPM.mat"

    print("Loading data")
    print(f"  via {spm_path}")

    # Load SPM.mat
    try:
        spm_data = loadmat(spm_path, struct_as_record=False, squeeze_me=True)
        SPM = spm_data["SPM"]
    except Exception as e:
        raise IOError(f"Could not load SPM.mat: {e}")

    # Extract volume information
    VY = SPM.xY.VY
    if not hasattr(VY, "__len__"):
        VY = [VY]

    # Get file paths
    vol_files = []
    for v in VY:
        fname = str(v.fname)
        if not Path(fname).is_absolute():
            fname = str(dir_name / fname)
        vol_files.append(fname)

    # Load first volume to get dimensions
    first_img = nib.load(vol_files[0])
    dim = first_img.shape[:3]
    affine = first_img.affine

    # Read analysis brain mask
    if hasattr(SPM, "VM"):
        VM = SPM.VM
        mask_fname = str(VM.fname)
        if not Path(mask_fname).is_absolute():
            mask_fname = str(dir_name / mask_fname)
        mask_img = nib.load(mask_fname)
        mask = mask_img.get_fdata() > 0
    else:
        raise ValueError("No analysis brain mask in SPM.VM!")

    print(f"  {np.sum(mask)} in-mask voxels")

    # Process region masks
    rmvi = []
    if regions is None or len(regions) == 0:
        print("  No region mask")
    else:
        if not isinstance(regions, list):
            regions = [regions]

        region_masks = []
        for i, region in enumerate(regions):
            if isinstance(region, (str, Path)):
                # Load from file
                region_data = read_vol_matched(region, first_img, order=0) > 0
            else:
                region_data = np.asarray(region) > 0

            if region_data.shape != dim:
                raise ValueError(f"Region mask {i + 1} shape doesn't match!")

            region_masks.append(region_data)

        # Restrict brain mask to union of regions
        combined_regions = np.any(np.stack(region_masks, axis=-1), axis=-1)
        mask = mask & combined_regions

        # Determine mask voxel indices for each region
        mask_flat = mask.ravel(order="F")
        for i, region_data in enumerate(region_masks):
            region_flat = region_data.ravel(order="F")
            region_in_mask = region_flat[mask_flat]
            rmvi.append(np.where(region_in_mask)[0])
            print(f"  {len(rmvi[-1])} in-mask voxels in region {i + 1}")

        print(f"  {np.sum(mask)} in-mask voxels in regions")

    # Read and mask data
    print("  Reading images")
    Y, mask = read_vols_masked(vol_files, mask)

    # Get design matrix
    X = np.asarray(SPM.xX.X)

    # Apply whitening and filtering if requested
    if whiten_filter:
        # Whitening
        if hasattr(SPM.xX, "W"):
            print("  Whitening")
            W = np.asarray(SPM.xX.W)
            if W.ndim == 1:
                W = np.diag(W)
            Y = W @ Y
            X = W @ X
        else:
            print("  * SPM.mat does not define whitening matrix!")

        # High-pass filtering
        print("  High-pass filtering")
        K = SPM.xX.K
        if not hasattr(K, "__len__"):
            K = [K]

        # Apply session-wise filtering
        row_start = 0
        for si, Ks in enumerate(K):
            n_scans = int(SPM.nscan[si]) if hasattr(SPM.nscan, "__len__") else int(SPM.nscan)
            row_end = row_start + n_scans

            if hasattr(Ks, "X0") and Ks.X0 is not None:
                X0 = np.asarray(Ks.X0)
                # Filter: remove low-frequency components
                # Y_filtered = Y - X0 @ (X0.T @ Y)
                Y[row_start:row_end, :] -= X0 @ (np.linalg.lstsq(X0, Y[row_start:row_end, :], rcond=None)[0])
                X[row_start:row_end, :] -= X0 @ (np.linalg.lstsq(X0, X[row_start:row_end, :], rcond=None)[0])

            row_start = row_end

    # Separate Y and X into session blocks
    nscan = SPM.nscan
    if not hasattr(nscan, "__len__"):
        nscan = [nscan]
    m = len(nscan)

    Ys = []
    Xs = []
    Bcovs = []

    # Get session info
    Sess = SPM.Sess
    if not hasattr(Sess, "__len__"):
        Sess = [Sess]

    row_start = 0
    for si in range(m):
        n_scans = int(nscan[si])
        row_end = row_start + n_scans
        rows = slice(row_start, row_end)

        Ys.append(Y[rows, :])

        # Get columns for this session (including constant)
        cols = list(Sess[si].col.flatten() - 1)  # MATLAB to Python indexing

        # Add constant regressor column
        iB = SPM.xX.iB
        if hasattr(iB, "__len__"):
            const_col = int(iB[si]) - 1
        else:
            const_col = int(iB) - 1
        if const_col not in cols:
            cols.append(const_col)

        Xs.append(X[rows, :][:, cols])

        # Parameter covariance (if available)
        if hasattr(SPM.xX, "Bcov"):
            Bcov = np.asarray(SPM.xX.Bcov)
            Bcovs.append(Bcov[np.ix_(cols, cols)])

        row_start = row_end

    # Compute degrees of freedom for each session
    fE = []
    for si in range(m):
        n_scans = int(nscan[si])
        n_regressors = np.linalg.matrix_rank(Xs[si])

        # Loss from filter
        K = SPM.xX.K
        if not hasattr(K, "__len__"):
            K = [K]
        if hasattr(K[si], "X0") and K[si].X0 is not None:
            n_filter = np.linalg.matrix_rank(np.asarray(K[si].X0))
        else:
            n_filter = 0

        fE.append(n_scans - n_filter - n_regressors)

    fE = np.array(fE)

    total_df = sum(int(n) for n in nscan)
    filter_df = sum(
        np.linalg.matrix_rank(np.asarray(K[si].X0))
        if hasattr(K[si], "X0") and K[si].X0 is not None
        else 0
        for si in range(m)
    )
    regressor_df = sum(np.linalg.matrix_rank(Xs[si]) for si in range(m))
    residual_df = sum(fE)

    print(
        f"  df: {total_df} - {filter_df} - {regressor_df} = {residual_df}"
    )
    if hasattr(SPM.xX, "trRV") and hasattr(SPM.xX, "erdf"):
        print(f"   [SPM: trRV = {SPM.xX.trRV}  erdf = {SPM.xX.erdf}]")

    # Miscellaneous output
    misc = {
        "mat": affine,
        "fE": fE,
        "rmvi": rmvi,
        "Bcovs": Bcovs,
    }

    return Ys, Xs, mask, misc
