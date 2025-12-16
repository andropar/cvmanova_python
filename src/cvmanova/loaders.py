"""
Data loaders for various input formats.

This module provides flexible data loading from different sources:
- SPM.mat files
- NIfTI files
- Nilearn maskers (optional)
- BIDS datasets (future)
"""

from typing import List, Optional, Union, Tuple
from pathlib import Path
import numpy as np

from .data import SessionData, DesignMatrix
from .io import load_data_spm


class SPMLoader:
    """
    Load data from SPM.mat file.

    This loader wraps the legacy load_data_spm function with modern structured
    data containers. It reads fMRI data, design matrices, and metadata from an
    SPM analysis directory, applying optional preprocessing (whitening and filtering).

    SPM.mat files contain the complete GLM specification from an SPM analysis,
    including scans, design matrices, filtering, and whitening parameters. This
    loader extracts this information and packages it into SessionData and
    DesignMatrix objects suitable for cvManova analysis.

    Parameters
    ----------
    spm_dir : Path or str
        Directory containing SPM.mat file. This is typically the output directory
        of an SPM first-level analysis.
    whiten : bool, default=True
        Whether to apply whitening (pre-whitening) to the data. Whitening
        decorrelates temporal autocorrelation using SPM's covariance model.
        Recommended to keep True for most analyses.
    high_pass_filter : bool, default=True
        Whether to apply high-pass filtering to remove low-frequency drifts.
        Uses the filter specification from SPM.mat. Recommended to keep True.
    tr : float or None, default=None
        Repetition time in seconds. If None, attempts to extract from SPM.mat.
        Currently not used in cvManova but may be useful for future features.

    Attributes
    ----------
    spm_dir : Path
        Path to SPM directory.
    whiten : bool
        Whitening setting.
    high_pass_filter : bool
        Filtering setting.
    tr : float or None
        Repetition time.

    Methods
    -------
    load()
        Load data and design matrices from SPM.mat.

    See Also
    --------
    NiftiLoader : Load from NIfTI files directly
    NilearnMaskerLoader : Load using nilearn preprocessing
    SessionData : Output data container
    DesignMatrix : Output design container

    Notes
    -----
    The SPM.mat file must contain:
    - VY: Cell array of memory-mapped nifti objects for each scan
    - xX: Design specification including design matrix and filtering
    - xM: Mask specification
    - Sess: Session information
    - xVi: Covariance structure (for whitening)

    Preprocessing steps applied (if enabled):
    1. High-pass filtering using SPM's specified cutoff
    2. Whitening using SPM's estimated covariance structure
    3. Data is extracted from masked voxels only

    The degrees of freedom returned account for:
    - Number of scans per session
    - Number of regressors in design matrix
    - Loss due to filtering

    Examples
    --------
    Basic loading from SPM directory:

    >>> from cvmanova.loaders import SPMLoader
    >>> loader = SPMLoader('/path/to/spm')
    >>> data, design = loader.load()
    >>> print(f"Loaded {data.n_sessions} sessions")
    >>> print(f"  Voxels: {data.n_voxels}")
    >>> print(f"  Scans per session: {data.n_scans}")

    Load without preprocessing:

    >>> loader = SPMLoader(
    ...     '/path/to/spm',
    ...     whiten=False,
    ...     high_pass_filter=False
    ... )
    >>> data, design = loader.load()

    Use with estimator:

    >>> from cvmanova import SearchlightCvManova
    >>> from cvmanova.config import ContrastSpec
    >>> loader = SPMLoader('/path/to/spm')
    >>> data, design = loader.load()
    >>> estimator = SearchlightCvManova(
    ...     contrasts=ContrastSpec(['Face', 'House'], [2, 2])
    ... )
    >>> result = estimator.fit_score(data, design)
    """

    def __init__(
        self,
        spm_dir: Union[str, Path],
        whiten: bool = True,
        high_pass_filter: bool = True,
        tr: Optional[float] = None,
    ):
        self.spm_dir = Path(spm_dir)
        self.whiten = whiten
        self.high_pass_filter = high_pass_filter
        self.tr = tr

        # Validate SPM.mat exists
        spm_file = self.spm_dir / "SPM.mat"
        if not spm_file.exists():
            raise FileNotFoundError(f"SPM.mat not found in {self.spm_dir}")

    def load(self) -> Tuple[SessionData, DesignMatrix]:
        """
        Load data and design matrices from SPM.mat.

        Returns
        -------
        data : SessionData
            Brain data with mask and metadata.
        design : DesignMatrix
            Design matrices for each session.

        Raises
        ------
        FileNotFoundError
            If SPM.mat or required files not found.
        ValueError
            If data is invalid or inconsistent.
        """
        # Use existing load_data_spm function
        Ys, Xs, mask, affine, fE = load_data_spm(
            str(self.spm_dir), whiten=self.whiten, highpass=self.high_pass_filter
        )

        # Create SessionData
        data = SessionData(
            sessions=Ys,
            mask=mask,
            affine=affine,
            degrees_of_freedom=fE,
        )

        # Create DesignMatrix
        design = DesignMatrix(matrices=Xs)

        return data, design

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SPMLoader(spm_dir={self.spm_dir}, whiten={self.whiten}, "
            f"high_pass_filter={self.high_pass_filter})"
        )


class NiftiLoader:
    """
    Load data directly from NIfTI files.

    This loader provides a flexible way to load fMRI data from NIfTI files without
    requiring SPM. It handles data extraction, masking, and basic preprocessing,
    making it suitable for custom analysis pipelines or when working with data
    from non-SPM sources.

    Parameters
    ----------
    bold_files : list of Path or str
        Paths to NIfTI files, one per session. Each file should be a 4D image
        (x × y × z × time) containing the BOLD time series. Files can be
        compressed (.nii.gz) or uncompressed (.nii).
    mask_file : Path or str
        Path to 3D binary mask file defining which voxels to include. The mask
        should have the same spatial dimensions as the BOLD files. Values > 0
        are treated as inside the mask.
    design_matrices : list of ndarray
        Design matrices, one per session. Each should have shape
        (n_scans, n_regressors) matching the number of volumes in the
        corresponding BOLD file. These should be constructed from your
        experimental design.
    degrees_of_freedom : array-like or None, default=None
        Per-session residual degrees of freedom (length n_sessions). If None,
        automatically computed as n_scans - n_regressors for each session.
        Provide custom values if you have additional constraints (e.g., from
        filtering).
    tr : float or None, default=None
        Repetition time in seconds. Currently not used in cvManova but may be
        useful for documentation or future features.
    preprocess : bool, default=True
        Whether to apply basic preprocessing (mean-centering each voxel's time
        series). Set to False if your data is already preprocessed.

    Attributes
    ----------
    bold_files : list of Path
        Paths to BOLD NIfTI files.
    mask_file : Path
        Path to mask file.
    design_matrices : list of ndarray
        Design matrices for each session.
    degrees_of_freedom : ndarray or None
        Degrees of freedom setting.
    tr : float or None
        Repetition time.
    preprocess : bool
        Preprocessing setting.

    Methods
    -------
    load()
        Load and prepare data for analysis.

    See Also
    --------
    SPMLoader : Load from SPM.mat files
    NilearnMaskerLoader : Load with advanced nilearn preprocessing
    SessionData : Output data container
    DesignMatrix : Output design container

    Notes
    -----
    This loader performs minimal preprocessing, making it suitable when you want
    full control over preprocessing steps. For more advanced preprocessing
    (smoothing, filtering, standardization), consider using NilearnMaskerLoader.

    Loading and preprocessing steps:
    1. Load mask and extract voxel coordinates
    2. For each session:
       - Load 4D BOLD data
       - Extract masked voxels (shape: n_scans × n_voxels)
       - Apply mean-centering if preprocess=True
    3. Compute or validate degrees of freedom
    4. Package into SessionData and DesignMatrix objects

    The data is stored in Fortran (column-major) order to match MATLAB/SPM
    conventions, which is important for compatibility with existing analyses.

    Examples
    --------
    Load data from NIfTI files with custom design:

    >>> import numpy as np
    >>> from cvmanova.loaders import NiftiLoader
    >>> # Create design matrices (example: simple 2-condition design)
    >>> n_scans = 100
    >>> X1 = np.column_stack([
    ...     np.sin(np.linspace(0, 4*np.pi, n_scans)),  # Condition 1
    ...     np.cos(np.linspace(0, 4*np.pi, n_scans)),  # Condition 2
    ... ])
    >>> X2 = X1.copy()  # Same design for session 2
    >>>
    >>> loader = NiftiLoader(
    ...     bold_files=['sub-01_run-1_bold.nii.gz', 'sub-01_run-2_bold.nii.gz'],
    ...     mask_file='sub-01_mask.nii.gz',
    ...     design_matrices=[X1, X2],
    ...     tr=2.0,
    ...     preprocess=True
    ... )
    >>> data, design = loader.load()

    Load without preprocessing (if data already preprocessed):

    >>> loader = NiftiLoader(
    ...     bold_files=['preprocessed_run1.nii.gz', 'preprocessed_run2.nii.gz'],
    ...     mask_file='mask.nii.gz',
    ...     design_matrices=[X1, X2],
    ...     preprocess=False
    ... )
    >>> data, design = loader.load()

    Specify custom degrees of freedom:

    >>> # Account for additional filtering that removed 10 df per session
    >>> custom_dof = np.array([90, 90])  # 100 scans - 2 regressors - 10 filter
    >>> loader = NiftiLoader(
    ...     bold_files=['run1.nii.gz', 'run2.nii.gz'],
    ...     mask_file='mask.nii.gz',
    ...     design_matrices=[X1, X2],
    ...     degrees_of_freedom=custom_dof
    ... )
    >>> data, design = loader.load()

    Complete analysis pipeline:

    >>> from cvmanova import SearchlightCvManova
    >>> from cvmanova.config import ContrastSpec
    >>> loader = NiftiLoader(
    ...     bold_files=['run1.nii.gz', 'run2.nii.gz', 'run3.nii.gz'],
    ...     mask_file='mask.nii.gz',
    ...     design_matrices=[X1, X2, X3]
    ... )
    >>> data, design = loader.load()
    >>> estimator = SearchlightCvManova(
    ...     contrasts=ContrastSpec(['A', 'B'], [2, 2])
    ... )
    >>> result = estimator.fit_score(data, design)
    """

    def __init__(
        self,
        bold_files: List[Union[str, Path]],
        mask_file: Union[str, Path],
        design_matrices: List[np.ndarray],
        degrees_of_freedom: Optional[np.ndarray] = None,
        tr: Optional[float] = None,
        preprocess: bool = True,
    ):
        self.bold_files = [Path(f) for f in bold_files]
        self.mask_file = Path(mask_file)
        self.design_matrices = design_matrices
        self.degrees_of_freedom = degrees_of_freedom
        self.tr = tr
        self.preprocess = preprocess

        # Validate inputs
        if len(self.bold_files) != len(self.design_matrices):
            raise ValueError(
                f"Number of BOLD files ({len(self.bold_files)}) does not match "
                f"number of design matrices ({len(self.design_matrices)})"
            )

        # Check files exist
        if not self.mask_file.exists():
            raise FileNotFoundError(f"Mask file not found: {self.mask_file}")

        for bf in self.bold_files:
            if not bf.exists():
                raise FileNotFoundError(f"BOLD file not found: {bf}")

    def load(self) -> Tuple[SessionData, DesignMatrix]:
        """
        Load data from NIfTI files.

        Returns
        -------
        data : SessionData
            Brain data with mask and metadata.
        design : DesignMatrix
            Design matrices for each session.

        Raises
        ------
        ImportError
            If nibabel is not installed.
        ValueError
            If data dimensions are inconsistent.
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required for NiftiLoader. Install with: pip install nibabel"
            )

        # Load mask
        mask_img = nib.load(self.mask_file)
        mask = np.asarray(mask_img.dataobj) > 0
        affine = mask_img.affine

        # Load BOLD data for each session
        Ys = []
        for i, bold_file in enumerate(self.bold_files):
            bold_img = nib.load(bold_file)
            bold_data = np.asarray(bold_img.dataobj)

            # Validate shape matches mask
            if bold_data.shape[:3] != mask.shape:
                raise ValueError(
                    f"BOLD file {i} shape {bold_data.shape[:3]} does not match "
                    f"mask shape {mask.shape}"
                )

            # Extract masked voxels (n_scans × n_voxels)
            n_scans = bold_data.shape[3]
            mask_flat = mask.ravel(order="F")
            n_voxels = np.sum(mask_flat)

            Y = np.zeros((n_scans, n_voxels))
            for t in range(n_scans):
                volume = bold_data[:, :, :, t]
                Y[t, :] = volume.ravel(order="F")[mask_flat]

            # Validate dimensions
            if Y.shape[0] != self.design_matrices[i].shape[0]:
                raise ValueError(
                    f"Session {i}: BOLD has {Y.shape[0]} scans but design matrix "
                    f"has {self.design_matrices[i].shape[0]} rows"
                )

            # Basic preprocessing
            if self.preprocess:
                # Mean-center each voxel
                Y = Y - Y.mean(axis=0, keepdims=True)

            Ys.append(Y)

        # Compute degrees of freedom if not provided
        if self.degrees_of_freedom is None:
            fE = np.array(
                [
                    Y.shape[0] - X.shape[1]
                    for Y, X in zip(Ys, self.design_matrices)
                ]
            )
        else:
            fE = np.asarray(self.degrees_of_freedom)

        # Create data structures
        data = SessionData(
            sessions=Ys, mask=mask, affine=affine, degrees_of_freedom=fE
        )

        design = DesignMatrix(matrices=self.design_matrices)

        return data, design

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NiftiLoader(n_sessions={len(self.bold_files)}, "
            f"preprocess={self.preprocess})"
        )


class NilearnMaskerLoader:
    """
    Load data using nilearn maskers.

    This loader integrates with nilearn's powerful NiftiMasker for advanced
    preprocessing. It allows you to apply sophisticated preprocessing pipelines
    (smoothing, filtering, standardization, detrending, confound removal) before
    cvManova analysis.

    Nilearn maskers provide a high-level interface for common fMRI preprocessing
    operations, making it easy to apply best practices for data preparation.

    Parameters
    ----------
    bold_files : list of Path or str
        Paths to NIfTI files, one per session. Each file should be a 4D image
        (x × y × z × time) containing the BOLD time series.
    masker : nilearn masker object
        Configured nilearn masker instance (e.g., NiftiMasker, MultiNiftiMasker).
        The masker should be configured with desired preprocessing parameters but
        not yet fitted. Common parameters include:
        - mask_img: Brain mask
        - smoothing_fwhm: Spatial smoothing kernel size
        - standardize: Z-score normalization
        - detrend: Remove linear trends
        - high_pass/low_pass: Temporal filtering
        - t_r: Repetition time
    design_matrices : list of ndarray
        Design matrices, one per session. Each should have shape
        (n_scans, n_regressors) matching the number of volumes in the
        corresponding BOLD file.
    degrees_of_freedom : array-like or None, default=None
        Per-session residual degrees of freedom. If None, computed as
        n_scans - n_regressors for each session. Provide custom values if
        preprocessing steps (e.g., filtering) reduce degrees of freedom.

    Attributes
    ----------
    bold_files : list of Path
        Paths to BOLD NIfTI files.
    masker : nilearn masker object
        Configured masker instance.
    design_matrices : list of ndarray
        Design matrices.
    degrees_of_freedom : ndarray or None
        Degrees of freedom setting.

    Methods
    -------
    load()
        Load and preprocess data using the masker.

    See Also
    --------
    SPMLoader : Load from SPM.mat files
    NiftiLoader : Load from NIfTI with minimal preprocessing
    SessionData : Output data container
    DesignMatrix : Output design container

    Notes
    -----
    This loader provides the most flexible preprocessing pipeline by leveraging
    nilearn's maskers. It's particularly useful when:
    - You need advanced preprocessing (smoothing, filtering, etc.)
    - You want to use nilearn's preprocessing conventions
    - You're working with BIDS datasets
    - You need confound regression

    The masker.fit_transform() is called for each session, which:
    1. Fits preprocessing parameters to the data
    2. Applies all configured preprocessing steps
    3. Extracts masked voxel time series
    4. Returns preprocessed data as (n_scans, n_voxels)

    Important: The masker should not be pre-fitted. This loader will call
    fit_transform() for each session independently.

    For details on masker parameters and preprocessing options, see:
    https://nilearn.github.io/stable/modules/maskers.html

    Examples
    --------
    Advanced preprocessing with smoothing and filtering:

    >>> from nilearn.maskers import NiftiMasker
    >>> from cvmanova.loaders import NilearnMaskerLoader
    >>> masker = NiftiMasker(
    ...     mask_img='mask.nii.gz',
    ...     smoothing_fwhm=6.0,
    ...     standardize=True,
    ...     detrend=True,
    ...     high_pass=0.01,
    ...     low_pass=0.1,
    ...     t_r=2.0
    ... )
    >>> loader = NilearnMaskerLoader(
    ...     bold_files=['run1.nii.gz', 'run2.nii.gz'],
    ...     masker=masker,
    ...     design_matrices=[X1, X2]
    ... )
    >>> data, design = loader.load()

    Using with confound regression:

    >>> masker = NiftiMasker(
    ...     mask_img='mask.nii.gz',
    ...     smoothing_fwhm=5.0,
    ...     standardize='zscore_sample',
    ...     detrend=False,  # Will use GLM to remove confounds
    ...     t_r=2.0
    ... )
    >>> # Note: For confound regression, typically handled in design matrices
    >>> # or by using nilearn's clean() function before this loader
    >>> loader = NilearnMaskerLoader(
    ...     bold_files=['run1.nii.gz', 'run2.nii.gz'],
    ...     masker=masker,
    ...     design_matrices=[X1, X2]
    ... )
    >>> data, design = loader.load()

    Minimal preprocessing (just masking):

    >>> masker = NiftiMasker(
    ...     mask_img='mask.nii.gz',
    ...     smoothing_fwhm=None,
    ...     standardize=False,
    ...     detrend=False
    ... )
    >>> loader = NilearnMaskerLoader(
    ...     bold_files=['run1.nii.gz', 'run2.nii.gz'],
    ...     masker=masker,
    ...     design_matrices=[X1, X2]
    ... )
    >>> data, design = loader.load()

    Complete workflow:

    >>> from cvmanova import SearchlightCvManova
    >>> from cvmanova.config import SearchlightConfig, ContrastSpec
    >>> # Configure preprocessing
    >>> masker = NiftiMasker(
    ...     mask_img='mask.nii.gz',
    ...     smoothing_fwhm=6.0,
    ...     standardize=True,
    ...     high_pass=0.01,
    ...     t_r=2.0
    ... )
    >>> # Load data
    >>> loader = NilearnMaskerLoader(
    ...     bold_files=['run1.nii.gz', 'run2.nii.gz', 'run3.nii.gz'],
    ...     masker=masker,
    ...     design_matrices=[X1, X2, X3]
    ... )
    >>> data, design = loader.load()
    >>> # Run analysis
    >>> estimator = SearchlightCvManova(
    ...     searchlight_config=SearchlightConfig(radius=3.0, n_jobs=-1),
    ...     contrasts=ContrastSpec(['Face', 'House'], [2, 2])
    ... )
    >>> result = estimator.fit_score(data, design)
    """

    def __init__(
        self,
        bold_files: List[Union[str, Path]],
        masker,  # nilearn masker object
        design_matrices: List[np.ndarray],
        degrees_of_freedom: Optional[np.ndarray] = None,
    ):
        self.bold_files = [Path(f) for f in bold_files]
        self.masker = masker
        self.design_matrices = design_matrices
        self.degrees_of_freedom = degrees_of_freedom

        # Validate
        if len(self.bold_files) != len(self.design_matrices):
            raise ValueError(
                f"Number of BOLD files ({len(self.bold_files)}) does not match "
                f"number of design matrices ({len(self.design_matrices)})"
            )

    def load(self) -> Tuple[SessionData, DesignMatrix]:
        """
        Load and preprocess data using nilearn masker.

        Returns
        -------
        data : SessionData
            Preprocessed brain data.
        design : DesignMatrix
            Design matrices.

        Raises
        ------
        ImportError
            If nilearn is not installed.
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required. Install with: pip install nibabel nilearn"
            )

        # Load and transform data for each session
        Ys = []
        for bold_file in self.bold_files:
            # nilearn's transform returns (n_scans, n_voxels)
            Y = self.masker.fit_transform(str(bold_file))
            Ys.append(Y)

        # Extract mask and affine from masker
        mask_img = self.masker.mask_img_
        mask = np.asarray(mask_img.dataobj) > 0
        affine = mask_img.affine

        # Compute degrees of freedom
        if self.degrees_of_freedom is None:
            fE = np.array(
                [
                    Y.shape[0] - X.shape[1]
                    for Y, X in zip(Ys, self.design_matrices)
                ]
            )
        else:
            fE = np.asarray(self.degrees_of_freedom)

        # Create data structures
        data = SessionData(
            sessions=Ys, mask=mask, affine=affine, degrees_of_freedom=fE
        )

        design = DesignMatrix(matrices=self.design_matrices)

        return data, design

    def __repr__(self) -> str:
        """String representation."""
        return f"NilearnMaskerLoader(n_sessions={len(self.bold_files)})"
