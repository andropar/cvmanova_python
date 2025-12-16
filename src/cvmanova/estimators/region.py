"""
Region (ROI) estimator for cvManova analysis.
"""

from typing import List, Optional, Union
from pathlib import Path
import numpy as np

from .base import BaseCvManova
from ..config import RegionConfig, AnalysisConfig, ContrastSpec
from ..data import SessionData, DesignMatrix
from ..results import CvManovaResult
from ..region import cv_manova_region
from ..utils import sign_permutations


class RegionCvManova(BaseCvManova):
    """
    Region-based (ROI) cross-validated MANOVA estimator.

    This estimator performs cross-validated MANOVA analysis on predefined regions
    of interest (ROIs). Unlike searchlight analysis which scans the entire brain,
    this approach focuses on specific anatomical or functional regions, computing
    discriminability separately for each ROI.

    Region-based analysis is ideal when you have strong anatomical hypotheses or
    want to compare discriminability across specific brain areas (e.g., visual
    cortex regions, language areas, etc.).

    Parameters
    ----------
    region_config : RegionConfig
        Configuration specifying ROI masks and related parameters. This includes
        paths to mask files or numpy arrays, region names, and validation settings.
    contrasts : list of ndarray or ContrastSpec or None, default=None
        Specification of contrasts to test. Can be:
        - List of contrast matrices (shape (n_contrasts_i, n_regressors))
        - ContrastSpec object for automatic factorial design contrasts
        - None (must be set before calling fit())
    analysis_config : AnalysisConfig or None, default=None
        Configuration for analysis parameters (regularization, permutation testing).
        If None, uses default AnalysisConfig().
    verbose : int or None, default=None
        Verbosity level (0=silent, 1=progress, 2=detailed). If provided, overrides
        the verbose setting in analysis_config.

    Attributes
    ----------
    data_ : SessionData or None
        Fitted brain data (set by fit()).
    design_ : DesignMatrix or None
        Fitted design matrices (set by fit()).
    contrast_matrices_ : list of ndarray or None
        Processed contrast matrices (set by fit()).
    contrast_names_ : list of str or None
        Names for each contrast (set by fit()).
    region_masks_ : list of ndarray or None
        3D boolean masks for each region (loaded during fit()).
    result_ : CvManovaResult or None
        Results from most recent score() call. Access via get_result().
    is_fitted_ : bool
        Whether the estimator has been fitted to data.

    Methods
    -------
    fit(data, design)
        Fit estimator to data, design, and load/validate region masks.
    score(data, design)
        Compute discriminability for each region.
    fit_score(data, design)
        Convenience method: fit().score() in one call.
    get_result()
        Retrieve results from last score() call.

    See Also
    --------
    SearchlightCvManova : Searchlight-based estimator
    RegionConfig : Configuration for region parameters
    AnalysisConfig : Configuration for analysis parameters
    CvManovaResult : Result container with export methods

    Notes
    -----
    Region-based analysis offers several advantages over searchlight:
    - Faster computation (only analyzes specified regions)
    - Anatomically interpretable results (tied to known brain areas)
    - Better statistical power (more voxels per test)
    - Easier to export and analyze statistically (fewer comparisons)

    The analysis computes discriminability independently for each region using
    all voxels within that region. This differs from searchlight which uses
    small local neighborhoods.

    Regions can be defined from:
    - Anatomical atlases (e.g., Harvard-Oxford, AAL)
    - Functional localizers (subject-specific ROIs)
    - Meta-analysis coordinates (spheres around peaks)
    - Custom masks from other analyses

    Results can be easily exported to DataFrame for statistical analysis, plotting,
    and comparison across regions, participants, or conditions.

    Examples
    --------
    Basic region analysis with anatomical ROIs:

    >>> from cvmanova import RegionCvManova
    >>> from cvmanova.config import RegionConfig, ContrastSpec
    >>> from cvmanova.loaders import SPMLoader
    >>> from pathlib import Path
    >>>
    >>> # Load data
    >>> loader = SPMLoader('/path/to/spm')
    >>> data, design = loader.load()
    >>>
    >>> # Configure regions
    >>> region_config = RegionConfig(
    ...     regions=[
    ...         Path('V1_left.nii.gz'),
    ...         Path('V1_right.nii.gz'),
    ...         Path('FFA_left.nii.gz'),
    ...         Path('FFA_right.nii.gz')
    ...     ],
    ...     region_names=['V1_L', 'V1_R', 'FFA_L', 'FFA_R']
    ... )
    >>> contrasts = ContrastSpec(['Face', 'House'], [2, 2])
    >>>
    >>> # Run analysis
    >>> estimator = RegionCvManova(
    ...     region_config=region_config,
    ...     contrasts=contrasts
    ... )
    >>> result = estimator.fit_score(data, design)
    >>>
    >>> # Export to DataFrame for analysis
    >>> df = result.to_dataframe()
    >>> print(df.groupby('region')['discriminability'].mean())

    Region analysis with permutation testing:

    >>> from cvmanova.config import AnalysisConfig
    >>> analysis_config = AnalysisConfig(
    ...     permute=True,
    ...     max_permutations=5000,
    ...     random_state=42
    ... )
    >>> estimator = RegionCvManova(
    ...     region_config=region_config,
    ...     contrasts=contrasts,
    ...     analysis_config=analysis_config
    ... )
    >>> result = estimator.fit_score(data, design)
    >>> df = result.to_dataframe()
    >>> # Analyze permutation distribution
    >>> perm_df = df[df['permutation'] > 0]

    Using numpy arrays as region masks:

    >>> import numpy as np
    >>> # Create custom regions
    >>> mask1 = np.zeros((64, 64, 30), dtype=bool)
    >>> mask1[20:30, 20:30, 10:15] = True
    >>> mask2 = np.zeros((64, 64, 30), dtype=bool)
    >>> mask2[35:45, 35:45, 10:15] = True
    >>>
    >>> region_config = RegionConfig(
    ...     regions=[mask1, mask2],
    ...     region_names=['Custom_Region_1', 'Custom_Region_2']
    ... )
    >>> estimator = RegionCvManova(
    ...     region_config=region_config,
    ...     contrasts=contrasts
    ... )
    >>> result = estimator.fit_score(data, design)

    Comparing discriminability across regions:

    >>> result = estimator.fit_score(data, design)
    >>> df = result.to_dataframe()
    >>> # Get unpermuted results only
    >>> df_real = df[df['permutation'] == 0]
    >>> # Compare regions for Face contrast
    >>> face_by_region = df_real[df_real['contrast'] == 'Face'].groupby('region')
    >>> print(face_by_region['discriminability'].describe())
    """

    def __init__(
        self,
        region_config: RegionConfig,
        contrasts: Optional[Union[List[np.ndarray], ContrastSpec]] = None,
        analysis_config: Optional[AnalysisConfig] = None,
        verbose: Optional[int] = None,
    ):
        super().__init__(
            contrasts=contrasts,
            analysis_config=analysis_config,
            verbose=verbose,
        )

        if region_config is None:
            raise ValueError("region_config is required for RegionCvManova")

        self.region_config = region_config

        # Additional state
        self.region_masks_: Optional[List[np.ndarray]] = None
        self.result_: Optional[CvManovaResult] = None

    def _load_region_masks(self, data: SessionData) -> List[np.ndarray]:
        """
        Load region masks from paths or use provided arrays.

        Parameters
        ----------
        data : SessionData
            Brain data (for shape validation).

        Returns
        -------
        list of ndarray
            3D boolean masks for each region.

        Raises
        ------
        ValueError
            If masks have wrong shape or too few voxels.
        """
        masks = []

        for i, region in enumerate(self.region_config.regions):
            if isinstance(region, Path):
                # Load from NIfTI file
                try:
                    import nibabel as nib
                except ImportError:
                    raise ImportError(
                        "nibabel is required to load region masks from files. "
                        "Install with: pip install nibabel"
                    )

                img = nib.load(region)
                mask = np.asarray(img.dataobj) > 0

                # Validate shape
                if mask.shape != data.mask.shape:
                    raise ValueError(
                        f"Region {i} ({self.region_config.region_names[i]}): "
                        f"mask shape {mask.shape} does not match data shape "
                        f"{data.mask.shape}"
                    )
            else:
                # Use provided array
                mask = region.astype(bool)

                if mask.shape != data.mask.shape:
                    raise ValueError(
                        f"Region {i} ({self.region_config.region_names[i]}): "
                        f"mask shape {mask.shape} does not match data shape "
                        f"{data.mask.shape}"
                    )

            # Check minimum voxels
            n_voxels = np.sum(mask)
            if n_voxels < self.region_config.min_voxels:
                raise ValueError(
                    f"Region {i} ({self.region_config.region_names[i]}): "
                    f"only {n_voxels} voxels, minimum is "
                    f"{self.region_config.min_voxels}"
                )

            masks.append(mask)

        # Check for overlaps if not allowed
        if not self.region_config.allow_overlap:
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    overlap = np.sum(masks[i] & masks[j])
                    if overlap > 0:
                        raise ValueError(
                            f"Regions {i} and {j} overlap in {overlap} voxels "
                            f"but allow_overlap=False"
                        )

        return masks

    def fit(
        self,
        data: SessionData,
        design: DesignMatrix,
    ) -> "RegionCvManova":
        """
        Fit the estimator (store and validate data).

        Parameters
        ----------
        data : SessionData
            Brain data with mask and affine.
        design : DesignMatrix
            Design matrices (one per session).

        Returns
        -------
        self : RegionCvManova
            Fitted estimator.

        Raises
        ------
        ValueError
            If data and design are incompatible or regions are invalid.
        """
        # Validate inputs
        self._validate_data(data, design)
        self._validate_and_prepare_contrasts(design)

        # Load and validate region masks
        self.region_masks_ = self._load_region_masks(data)

        self.data_ = data
        self.design_ = design
        self.is_fitted_ = True

        if self.analysis_config.verbose >= 1:
            print(f"Fitted RegionCvManova with {data.n_sessions} sessions")
            print(f"  Data shape: {data.sessions[0].shape}")
            print(f"  Regions: {self.region_config.n_regions}")
            for i, name in enumerate(self.region_config.region_names):
                n_voxels = np.sum(self.region_masks_[i])
                print(f"    {name}: {n_voxels} voxels")
            print(f"  Contrasts: {len(self.contrast_matrices_)}")

        return self

    def score(
        self,
        data: Optional[SessionData] = None,
        design: Optional[DesignMatrix] = None,
    ) -> CvManovaResult:
        """
        Compute discriminability scores for each region.

        Parameters
        ----------
        data : SessionData, optional
            Brain data. If None, uses data from fit().
        design : DesignMatrix, optional
            Design matrices. If None, uses design from fit().

        Returns
        -------
        CvManovaResult
            Region analysis results.

        Raises
        ------
        ValueError
            If estimator not fitted and no data provided.
        """
        # Use provided data or fitted data
        if data is None and design is None:
            self._check_is_fitted()
            data = self.data_
            design = self.design_
        elif data is not None and design is not None:
            self._validate_data(data, design)
            self._validate_and_prepare_contrasts(design)
            # Reload masks if new data
            self.region_masks_ = self._load_region_masks(data)
        else:
            raise ValueError("Must provide both data and design, or neither")

        # Generate sign permutations if needed
        if self.analysis_config.permute:
            sp = sign_permutations(
                data.n_sessions,
                self.analysis_config.max_permutations,
                seed=self.analysis_config.random_state,
            )
        else:
            sp = np.ones((data.n_sessions, 1))

        if self.analysis_config.verbose >= 1:
            print("Running region analysis...")
            print(f"  Permutations: {sp.shape[1]}")
            print(f"  Regularization: {self.analysis_config.regularization}")

        # Convert 3D boolean masks to flat indices for each region
        # region_indices should be indices into the masked data (n_voxels dimension)
        region_indices = []
        mask_flat = data.mask.ravel(order="F")
        mask_indices = np.where(mask_flat)[0]  # Indices of True values in flattened mask

        for region_mask in self.region_masks_:
            region_flat = region_mask.ravel(order="F")
            # Find which mask voxels are in this region
            region_in_mask = region_flat[mask_indices]
            # Get indices relative to masked data
            region_idx = np.where(region_in_mask)[0]
            region_indices.append(region_idx)

        # Run region analysis
        D, p = cv_manova_region(
            Ys=data.sessions,
            Xs=design.matrices,
            Cs=self.contrast_matrices_,
            fE=data.degrees_of_freedom,
            region_indices=region_indices,
            permute=self.analysis_config.permute,
            lambda_=self.analysis_config.regularization,
        )

        # Transpose D from (n_contrasts, n_perms, n_regions) to (n_regions, n_contrasts, n_perms)
        D = np.transpose(D, (2, 0, 1))

        # Create result object
        self.result_ = CvManovaResult(
            discriminability=D,
            n_voxels=p,
            contrasts=self.contrast_matrices_,
            contrast_names=self.contrast_names_,
            mask=None,  # No single mask for region analysis
            affine=data.affine,
            analysis_type="region",
            region_names=self.region_config.region_names,
        )

        if self.analysis_config.verbose >= 1:
            print("Region analysis complete")
            print(f"  Result shape: {D.shape}")

        return self.result_

    def get_result(self) -> CvManovaResult:
        """
        Get results from last score() call.

        Returns
        -------
        CvManovaResult
            Analysis results.

        Raises
        ------
        ValueError
            If score() has not been called yet.
        """
        if self.result_ is None:
            raise ValueError("No results available. Call score() first.")
        return self.result_

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"RegionCvManova(n_regions={self.region_config.n_regions}, "
            f"{fitted_str})"
        )
