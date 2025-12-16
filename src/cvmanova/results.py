"""
Result objects for cvManova analysis.

This module provides rich result containers that replace simple tuple returns
with structured objects offering visualization and export capabilities.
"""

from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path
import numpy as np
import warnings


@dataclass
class CvManovaResult:
    """
    Container for cross-validated MANOVA results.

    This rich result object replaces simple tuple returns with a structured container
    that provides convenient methods for accessing, visualizing, and exporting
    discriminability results. It supports both searchlight and region-based analyses
    with appropriate export formats for each.

    Parameters
    ----------
    discriminability : ndarray
        Pattern discriminability values with shape (n_locations, n_contrasts, n_perms).
        - For searchlight: n_locations is the number of voxels in the brain volume
          (flattened in Fortran order). Values outside the mask are NaN.
        - For region: n_locations is the number of regions (ROIs).
        Discriminability quantifies how well neural patterns distinguish experimental
        conditions, with higher values indicating greater discriminability.
    n_voxels : ndarray
        Number of voxels contributing to each discriminability value.
        - For searchlight: shape (n_volume_voxels,), counts voxels in each sphere.
        - For region: shape (n_regions,), total voxels per region.
    contrasts : list of ndarray
        Contrast matrices used in the analysis. Each contrast specifies a linear
        combination of design matrix regressors to test.
    contrast_names : list of str
        Descriptive names for each contrast (e.g., ['Face', 'House', 'Face×House']).
        Length must match the number of contrasts.
    mask : ndarray or None
        3D boolean mask defining the brain volume. Required for searchlight analysis,
        None for region analysis. Used for converting flat indices to 3D volumes.
    affine : ndarray
        4×4 affine transformation matrix mapping voxel to world coordinates.
        Extracted from NIfTI header and used when saving results to NIfTI format.
    analysis_type : {'searchlight', 'region'}, default='searchlight'
        Type of analysis. Determines which methods are available and how results
        are structured.
    region_names : list of str or None, default=None
        Names for each region (e.g., ['V1', 'V2', 'FFA']). Only used for region
        analysis. If None, regions are identified by index.

    Attributes
    ----------
    n_contrasts : int
        Number of contrasts tested.
    n_perms : int
        Number of permutations computed. If permutation testing was not performed,
        this is 1 (just the original unpermuted result).

    Methods
    -------
    get_contrast(contrast, perm=0)
        Extract discriminability values for a specific contrast and permutation.
    get_contrast_3d(contrast, perm=0)
        Get discriminability as a 3D brain volume (searchlight only).
    to_nifti(contrast, output_path, perm=0)
        Save contrast results as a NIfTI file (searchlight only).
    to_dataframe()
        Convert results to pandas DataFrame (region analysis only).
    get_peaks(contrast, n=10, perm=0)
        Find peak discriminability locations (searchlight only).
    save(output_dir)
        Save all results to a directory with appropriate format.

    See Also
    --------
    SearchlightCvManova : Searchlight estimator that produces these results
    RegionCvManova : Region-based estimator that produces these results
    SessionData : Input data structure
    DesignMatrix : Input design structure

    Notes
    -----
    The discriminability values represent the cross-validated multivariate effect
    size, quantifying how reliably neural patterns distinguish conditions. The
    method is described in:

        Allefeld, C., & Haynes, J. D. (2014). Searchlight-based multi-voxel
        pattern analysis of fMRI by cross-validated MANOVA. NeuroImage, 89,
        345-357.

    For searchlight results:
    - Discriminability is stored as a flat array with length equal to total brain
      volume voxels (in Fortran/column-major order).
    - Values outside the mask are NaN.
    - Use get_contrast_3d() to reshape to 3D for visualization.

    For region results:
    - Discriminability is stored with one value per region.
    - Use to_dataframe() for tabular output suitable for statistical analysis.

    Permutation testing (if enabled) provides a null distribution for statistical
    inference. The first permutation (index 0) is always the unpermuted result.

    Examples
    --------
    Access discriminability for a specific contrast:

    >>> from cvmanova import SearchlightCvManova
    >>> # ... run analysis ...
    >>> result = estimator.fit_score(data, design)
    >>> face_disc = result.get_contrast('Face')
    >>> print(f"Mean discriminability: {np.nanmean(face_disc):.3f}")

    Save searchlight results as NIfTI:

    >>> result.to_nifti('Face', 'face_discriminability.nii.gz')
    >>> result.to_nifti('House', 'house_discriminability.nii.gz')

    Get 3D volume for visualization:

    >>> face_volume = result.get_contrast_3d('Face')
    >>> # Visualize with nilearn, matplotlib, etc.

    Find peak discriminability locations:

    >>> peaks = result.get_peaks('Face', n=10)
    >>> for i, (x, y, z, disc) in enumerate(peaks):
    ...     print(f"Peak {i+1}: ({x}, {y}, {z}) D={disc:.3f}")

    Convert region results to DataFrame:

    >>> # For region analysis
    >>> df = result.to_dataframe()
    >>> print(df.groupby('region')['discriminability'].mean())

    Save all results to directory:

    >>> result.save('./results')
    >>> # Creates: results/discriminability_Face.nii.gz, etc.

    Access permutation distribution:

    >>> if result.n_perms > 1:
    ...     # Get permutation distribution
    ...     perm_discs = [result.get_contrast('Face', perm=i)
    ...                   for i in range(result.n_perms)]
    ...     # Compute empirical p-values, etc.
    """

    discriminability: np.ndarray
    n_voxels: np.ndarray
    contrasts: List[np.ndarray]
    contrast_names: List[str]
    mask: Optional[np.ndarray]
    affine: np.ndarray
    analysis_type: str = "searchlight"
    region_names: Optional[List[str]] = None

    def __post_init__(self):
        """Validate result data after initialization."""
        # Validate analysis type
        if self.analysis_type not in ["searchlight", "region"]:
            raise ValueError(
                f"analysis_type must be 'searchlight' or 'region', got {self.analysis_type}"
            )

        # Validate shapes
        if self.discriminability.ndim != 3:
            raise ValueError(
                f"discriminability must be 3D (locations × contrasts × perms), "
                f"got shape {self.discriminability.shape}"
            )

        n_locations = self.discriminability.shape[0]
        if len(self.n_voxels) != n_locations:
            raise ValueError(
                f"n_voxels length {len(self.n_voxels)} does not match "
                f"discriminability locations {n_locations}"
            )

        # Validate contrasts
        if len(self.contrasts) != self.n_contrasts:
            raise ValueError(
                f"Number of contrast matrices {len(self.contrasts)} does not match "
                f"discriminability contrasts {self.n_contrasts}"
            )

        if len(self.contrast_names) != self.n_contrasts:
            raise ValueError(
                f"Number of contrast names {len(self.contrast_names)} does not match "
                f"number of contrasts {self.n_contrasts}"
            )

        # Validate mask for searchlight analysis
        if self.analysis_type == "searchlight":
            if self.mask is None:
                raise ValueError("mask is required for searchlight analysis")
            if self.mask.ndim != 3:
                raise ValueError(f"mask must be 3D, got shape {self.mask.shape}")

        # Validate region names for region analysis
        if self.analysis_type == "region":
            if self.region_names is not None:
                if len(self.region_names) != n_locations:
                    raise ValueError(
                        f"Number of region names {len(self.region_names)} does not match "
                        f"number of regions {n_locations}"
                    )

        # Validate affine
        if self.affine.shape != (4, 4):
            raise ValueError(f"affine must be 4x4, got shape {self.affine.shape}")

    @property
    def n_contrasts(self) -> int:
        """Number of contrasts."""
        return self.discriminability.shape[1]

    @property
    def n_perms(self) -> int:
        """Number of permutations."""
        return self.discriminability.shape[2]

    def get_contrast(
        self, contrast: Union[str, int], perm: int = 0
    ) -> np.ndarray:
        """
        Get discriminability values for a specific contrast.

        Parameters
        ----------
        contrast : str or int
            Contrast name or index.
        perm : int, optional
            Permutation index (default: 0 = non-permuted).

        Returns
        -------
        ndarray
            Discriminability values for the contrast. Shape (n_volume_voxels,) for
            searchlight or (n_regions,) for region analysis.

        Raises
        ------
        ValueError
            If contrast name not found or index out of range.
        """
        if isinstance(contrast, str):
            try:
                contrast_idx = self.contrast_names.index(contrast)
            except ValueError:
                raise ValueError(
                    f"Contrast '{contrast}' not found. "
                    f"Available: {self.contrast_names}"
                )
        else:
            contrast_idx = contrast
            if not 0 <= contrast_idx < self.n_contrasts:
                raise ValueError(
                    f"Contrast index {contrast_idx} out of range [0, {self.n_contrasts})"
                )

        if not 0 <= perm < self.n_perms:
            raise ValueError(
                f"Permutation index {perm} out of range [0, {self.n_perms})"
            )

        return self.discriminability[:, contrast_idx, perm]

    def get_contrast_3d(
        self, contrast: Union[str, int], perm: int = 0
    ) -> np.ndarray:
        """
        Get discriminability as 3D volume (searchlight only).

        Parameters
        ----------
        contrast : str or int
            Contrast name or index.
        perm : int, optional
            Permutation index (default: 0 = non-permuted).

        Returns
        -------
        ndarray
            3D volume of discriminability values.

        Raises
        ------
        ValueError
            If called on region analysis results.
        """
        if self.analysis_type != "searchlight":
            raise ValueError("get_contrast_3d only available for searchlight analysis")

        disc_1d = self.get_contrast(contrast, perm)

        # Reshape to 3D volume (discriminability is already volume-indexed with NaN outside mask)
        volume = disc_1d.reshape(self.mask.shape, order="F")

        return volume

    def to_nifti(
        self, contrast: Union[str, int], output_path: Union[str, Path],
        perm: int = 0
    ):
        """
        Save contrast results as NIfTI file (searchlight only).

        Parameters
        ----------
        contrast : str or int
            Contrast name or index.
        output_path : str or Path
            Output file path (will add .nii.gz if not present).
        perm : int, optional
            Permutation index (default: 0 = non-permuted).

        Raises
        ------
        ValueError
            If called on region analysis results.
        """
        if self.analysis_type != "searchlight":
            raise ValueError("to_nifti only available for searchlight analysis")

        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required for to_nifti. Install with: pip install nibabel"
            )

        output_path = Path(output_path)
        if output_path.suffix not in [".nii", ".gz"]:
            output_path = output_path.with_suffix(".nii.gz")

        volume = self.get_contrast_3d(contrast, perm)
        img = nib.Nifti1Image(volume, self.affine)
        nib.save(img, str(output_path))

    def to_dataframe(self):
        """
        Convert results to pandas DataFrame (region analysis only).

        Returns
        -------
        DataFrame
            Results with columns: region, contrast, permutation, discriminability, n_voxels.

        Raises
        ------
        ValueError
            If called on searchlight analysis results (too large for DataFrame).
        """
        if self.analysis_type != "searchlight":
            try:
                import pandas as pd
            except ImportError:
                raise ImportError(
                    "pandas is required for to_dataframe. Install with: pip install pandas"
                )

            records = []
            n_locations = self.discriminability.shape[0]

            for loc in range(n_locations):
                region_name = (
                    self.region_names[loc] if self.region_names else f"region_{loc}"
                )
                for ci, contrast_name in enumerate(self.contrast_names):
                    for pi in range(self.n_perms):
                        records.append({
                            "region": region_name,
                            "contrast": contrast_name,
                            "permutation": pi,
                            "discriminability": self.discriminability[loc, ci, pi],
                            "n_voxels": self.n_voxels[loc],
                        })

            return pd.DataFrame(records)
        else:
            raise ValueError(
                "to_dataframe only available for region analysis "
                "(searchlight results are too large)"
            )

    def get_peaks(
        self, contrast: Union[str, int], n: int = 10, perm: int = 0
    ) -> np.ndarray:
        """
        Get peak discriminability locations (searchlight only).

        Parameters
        ----------
        contrast : str or int
            Contrast name or index.
        n : int, optional
            Number of peaks to return (default: 10).
        perm : int, optional
            Permutation index (default: 0 = non-permuted).

        Returns
        -------
        ndarray
            Array of shape (n, 4) with columns [x, y, z, discriminability].

        Raises
        ------
        ValueError
            If called on region analysis results.
        """
        if self.analysis_type != "searchlight":
            raise ValueError("get_peaks only available for searchlight analysis")

        disc_3d = self.get_contrast_3d(contrast, perm)

        # Find peak coordinates
        valid_mask = ~np.isnan(disc_3d)
        disc_values = disc_3d[valid_mask]
        coords = np.argwhere(valid_mask)

        # Sort by discriminability (descending)
        sorted_indices = np.argsort(disc_values)[::-1]
        top_n = min(n, len(sorted_indices))

        peaks = np.zeros((top_n, 4))
        for i in range(top_n):
            idx = sorted_indices[i]
            peaks[i, :3] = coords[idx]
            peaks[i, 3] = disc_values[idx]

        return peaks

    def save(self, output_dir: Union[str, Path]):
        """
        Save all results to directory.

        For searchlight: Saves one NIfTI file per contrast.
        For region: Saves results as CSV file.

        Parameters
        ----------
        output_dir : str or Path
            Output directory (will be created if it doesn't exist).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.analysis_type == "searchlight":
            # Save each contrast as NIfTI
            for i, name in enumerate(self.contrast_names):
                output_path = output_dir / f"discriminability_{name}.nii.gz"
                self.to_nifti(i, output_path, perm=0)

            # Save n_voxels map
            try:
                import nibabel as nib
                volume = np.full(self.mask.shape, np.nan)
                volume[self.mask] = self.n_voxels[self.mask.ravel(order="F")]
                img = nib.Nifti1Image(volume, self.affine)
                nib.save(img, str(output_dir / "n_voxels.nii.gz"))
            except ImportError:
                warnings.warn("nibabel not available, skipping n_voxels map")

        else:
            # Save as CSV
            try:
                df = self.to_dataframe()
                df.to_csv(output_dir / "results.csv", index=False)
            except ImportError:
                warnings.warn("pandas not available, skipping CSV export")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CvManovaResult(analysis_type='{self.analysis_type}', "
            f"n_contrasts={self.n_contrasts}, n_perms={self.n_perms}, "
            f"contrasts={self.contrast_names})"
        )
