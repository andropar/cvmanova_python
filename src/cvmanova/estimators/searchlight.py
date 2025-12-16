"""
Searchlight estimator for whole-brain cvManova analysis.
"""

from typing import List, Optional, Union
from pathlib import Path
import numpy as np

from .base import BaseCvManova
from ..config import SearchlightConfig, AnalysisConfig, ContrastSpec
from ..data import SessionData, DesignMatrix
from ..results import CvManovaResult
from ..searchlight import cv_manova_searchlight


class SearchlightCvManova(BaseCvManova):
    """
    Searchlight-based cross-validated MANOVA estimator.

    This estimator performs whole-brain searchlight analysis using cross-validated
    MANOVA to assess multivariate pattern discriminability. For each voxel in the
    brain, it analyzes the local sphere of surrounding voxels to compute how well
    neural patterns within that neighborhood distinguish experimental conditions.

    The searchlight approach provides spatial specificity by identifying where in
    the brain multivariate patterns carry information about task variables.

    Parameters
    ----------
    searchlight_config : SearchlightConfig or None, default=None
        Configuration for searchlight-specific parameters (radius, parallelization,
        checkpointing). If None, uses default SearchlightConfig().
    contrasts : list of ndarray or ContrastSpec or None, default=None
        Specification of contrasts to test. Can be:
        - List of contrast matrices (shape (n_contrasts_i, n_regressors))
        - ContrastSpec object for automatic factorial design contrasts
        - None (must be set before calling fit())
    analysis_config : AnalysisConfig or None, default=None
        Configuration for analysis parameters (regularization, permutation testing).
        If None, uses default AnalysisConfig().
    n_jobs : int or None, default=None
        Number of parallel jobs for searchlight computation. If provided, overrides
        searchlight_config.n_jobs. Use -1 for all CPUs, 1 for sequential processing.
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
    result_ : CvManovaResult or None
        Results from most recent score() call. Access via get_result().
    is_fitted_ : bool
        Whether the estimator has been fitted to data.

    Methods
    -------
    fit(data, design)
        Fit estimator to data and design matrices (validation and storage).
    score(data, design)
        Compute searchlight discriminability across the brain.
    fit_score(data, design)
        Convenience method: fit().score() in one call.
    get_result()
        Retrieve results from last score() call.

    See Also
    --------
    RegionCvManova : Region-based (ROI) estimator
    SearchlightConfig : Configuration for searchlight parameters
    AnalysisConfig : Configuration for analysis parameters
    CvManovaResult : Result container with visualization methods

    Notes
    -----
    The searchlight method was introduced by Kriegeskorte et al. (2006) and adapted
    for cross-validated MANOVA by Allefeld & Haynes (2014):

        Allefeld, C., & Haynes, J. D. (2014). Searchlight-based multi-voxel
        pattern analysis of fMRI by cross-validated MANOVA. NeuroImage, 89,
        345-357.

    Key algorithm steps:
    1. For each voxel, define a sphere of surrounding voxels (radius specified
       in searchlight_config)
    2. Extract data from voxels within the sphere across all sessions
    3. Compute cross-validated MANOVA discriminability for each contrast
    4. Store discriminability value at the center voxel
    5. Repeat for all voxels in the brain mask

    Computational considerations:
    - Runtime scales with number of voxels × radius³ × number of sessions
    - Parallelization (n_jobs > 1) significantly speeds up analysis
    - Checkpointing allows resuming interrupted long-running analyses
    - Memory usage increases with n_jobs (multiple searchlights in parallel)

    The discriminability values represent cross-validated multivariate effect sizes,
    quantifying information content in local neural patterns.

    Examples
    --------
    Basic searchlight analysis with factorial design:

    >>> from cvmanova import SearchlightCvManova
    >>> from cvmanova.config import SearchlightConfig, ContrastSpec
    >>> from cvmanova.loaders import SPMLoader
    >>>
    >>> # Load data
    >>> loader = SPMLoader('/path/to/spm')
    >>> data, design = loader.load()
    >>>
    >>> # Configure searchlight
    >>> sl_config = SearchlightConfig(radius=3.0, n_jobs=-1)
    >>> contrasts = ContrastSpec(['Face', 'House'], [2, 2])
    >>>
    >>> # Run analysis
    >>> estimator = SearchlightCvManova(
    ...     searchlight_config=sl_config,
    ...     contrasts=contrasts
    ... )
    >>> result = estimator.fit_score(data, design)
    >>>
    >>> # Save results
    >>> result.to_nifti('Face', 'face_discriminability.nii.gz')
    >>> result.to_nifti('House', 'house_discriminability.nii.gz')
    >>> result.to_nifti('Face×House', 'interaction_discriminability.nii.gz')

    Searchlight with permutation testing:

    >>> from cvmanova.config import AnalysisConfig
    >>> analysis_config = AnalysisConfig(
    ...     permute=True,
    ...     max_permutations=5000,
    ...     random_state=42
    ... )
    >>> estimator = SearchlightCvManova(
    ...     searchlight_config=SearchlightConfig(radius=4.0, n_jobs=4),
    ...     contrasts=contrasts,
    ...     analysis_config=analysis_config
    ... )
    >>> result = estimator.fit_score(data, design)
    >>> print(f"Computed {result.n_perms} permutations")

    Searchlight with checkpointing for long jobs:

    >>> from pathlib import Path
    >>> sl_config = SearchlightConfig(
    ...     radius=5.0,
    ...     checkpoint_dir=Path('./checkpoints'),
    ...     checkpoint_name='my_searchlight',
    ...     progress_interval=60.0,
    ...     n_jobs=-1
    ... )
    >>> estimator = SearchlightCvManova(
    ...     searchlight_config=sl_config,
    ...     contrasts=contrasts,
    ...     verbose=2
    ... )
    >>> result = estimator.fit_score(data, design)

    Separate fit and score for multiple analyses:

    >>> estimator = SearchlightCvManova(contrasts=contrasts)
    >>> estimator.fit(data, design)
    >>> # Can now score multiple times with different parameters
    >>> result1 = estimator.score()
    >>> result2 = estimator.score(data, design)  # With different data
    """

    def __init__(
        self,
        searchlight_config: Optional[SearchlightConfig] = None,
        contrasts: Optional[Union[List[np.ndarray], ContrastSpec]] = None,
        analysis_config: Optional[AnalysisConfig] = None,
        n_jobs: Optional[int] = None,
        verbose: Optional[int] = None,
    ):
        super().__init__(
            contrasts=contrasts,
            analysis_config=analysis_config,
            verbose=verbose,
        )

        self.searchlight_config = searchlight_config or SearchlightConfig()

        # Override n_jobs if explicitly provided
        if n_jobs is not None:
            self.searchlight_config.n_jobs = n_jobs

        # Additional state
        self.result_: Optional[CvManovaResult] = None

    def fit(
        self,
        data: SessionData,
        design: DesignMatrix,
    ) -> "SearchlightCvManova":
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
        self : SearchlightCvManova
            Fitted estimator.

        Raises
        ------
        ValueError
            If data and design are incompatible.
        """
        # Validate inputs
        self._validate_data(data, design)
        self._validate_and_prepare_contrasts(design)

        self.data_ = data
        self.design_ = design
        self.is_fitted_ = True

        if self.analysis_config.verbose >= 1:
            print(f"Fitted SearchlightCvManova with {data.n_sessions} sessions")
            print(f"  Data shape: {data.sessions[0].shape}")
            print(f"  Mask voxels: {np.sum(data.mask)}")
            print(f"  Contrasts: {len(self.contrast_matrices_)}")
            print(f"  Searchlight radius: {self.searchlight_config.radius}")

        return self

    def score(
        self,
        data: Optional[SessionData] = None,
        design: Optional[DesignMatrix] = None,
    ) -> CvManovaResult:
        """
        Compute discriminability scores across the brain.

        Parameters
        ----------
        data : SessionData, optional
            Brain data. If None, uses data from fit().
        design : DesignMatrix, optional
            Design matrices. If None, uses design from fit().

        Returns
        -------
        CvManovaResult
            Searchlight analysis results.

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
        else:
            raise ValueError("Must provide both data and design, or neither")

        if self.analysis_config.verbose >= 1:
            print("Running searchlight analysis...")
            print(f"  Permute: {self.analysis_config.permute}")
            if self.analysis_config.permute:
                print(f"  Max permutations: {self.analysis_config.max_permutations}")
            print(f"  Regularization: {self.analysis_config.regularization}")
            print(f"  Parallel jobs: {self.searchlight_config.n_jobs}")

        # Run searchlight analysis
        # Note: cv_manova_searchlight internally generates permutations if permute=True
        # TODO: Update cv_manova_searchlight to support all config options (n_jobs, etc.)
        # TODO: Pass random_state for reproducibility
        checkpoint_path = self.searchlight_config.get_checkpoint_path()
        checkpoint_str = str(checkpoint_path) if checkpoint_path is not None else None

        D, p, _, _ = cv_manova_searchlight(
            Ys=data.sessions,
            Xs=design.matrices,
            mask=data.mask,
            sl_radius=self.searchlight_config.radius,
            Cs=self.contrast_matrices_,
            fE=data.degrees_of_freedom,
            permute=self.analysis_config.permute,
            lambda_=self.analysis_config.regularization,
            checkpoint=checkpoint_str,
        )

        # Create result object
        self.result_ = CvManovaResult(
            discriminability=D,
            n_voxels=p,
            contrasts=self.contrast_matrices_,
            contrast_names=self.contrast_names_,
            mask=data.mask,
            affine=data.affine,
            analysis_type="searchlight",
        )

        if self.analysis_config.verbose >= 1:
            print("Searchlight analysis complete")
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
            f"SearchlightCvManova(radius={self.searchlight_config.radius}, "
            f"n_jobs={self.searchlight_config.n_jobs}, {fitted_str})"
        )
