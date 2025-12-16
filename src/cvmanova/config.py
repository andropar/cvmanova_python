"""
Configuration dataclasses for cvManova analysis.

This module provides validated configuration objects that replace long
parameter lists with structured, type-safe configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Literal
from pathlib import Path
import numpy as np


@dataclass
class SearchlightConfig:
    """
    Configuration for searchlight analysis.

    This configuration controls the behavior of searchlight-based cvManova analysis,
    including sphere size, parallel processing, checkpointing, and progress reporting.

    Parameters
    ----------
    radius : float, default=3.0
        Radius of searchlight sphere in voxels. Larger radii capture more spatial
        context but increase computation time. Typical values range from 2.0 to 5.0
        voxels, corresponding to approximately 4-10mm radius depending on voxel size.
    checkpoint_dir : Path or None, default=None
        Directory for saving checkpoint files during analysis. If None, checkpointing
        is disabled. Checkpointing allows resuming interrupted analyses and is
        recommended for long-running searchlight jobs.
    checkpoint_name : str, default='searchlight_checkpoint'
        Base filename for checkpoint file (without extension). The checkpoint will
        be saved as {checkpoint_name}.pkl in checkpoint_dir.
    progress_interval : float, default=30.0
        Interval in seconds between checkpoint saves. Shorter intervals provide more
        frequent backups but may slow down analysis slightly.
    min_voxels : int, default=10
        Minimum number of voxels required in a searchlight sphere. Searchlights with
        fewer voxels (e.g., at brain edges) are skipped. Typical values: 10-20.
    n_jobs : int, default=1
        Number of parallel jobs for computation. Use -1 to use all available CPUs,
        1 for sequential processing. Parallel processing significantly speeds up
        analysis but uses more memory.
    show_progress : bool, default=True
        Whether to display a progress bar during analysis. Useful for monitoring
        long-running jobs.
    chunk_size : int or None, default=None
        Number of voxels processed per parallel chunk. If None, automatically
        calculated based on n_jobs. Larger chunks reduce overhead but use more memory.
    backend : {'loky', 'threading', 'multiprocessing'}, default='loky'
        Joblib parallel backend. 'loky' is recommended for most cases (robust process
        isolation). 'threading' may be faster but has GIL limitations. 'multiprocessing'
        is an alternative process-based backend.

    See Also
    --------
    SearchlightCvManova : Searchlight estimator that uses this configuration
    AnalysisConfig : Analysis-specific parameters (regularization, permutations)
    RegionConfig : Configuration for region-based analysis

    Notes
    -----
    Memory usage scales with n_jobs and chunk_size. For large datasets or limited
    memory, reduce n_jobs or use smaller chunk_size.

    The recommended backend 'loky' provides better error handling and process
    isolation than 'multiprocessing', especially on Windows.

    Examples
    --------
    Basic searchlight with default settings:

    >>> from cvmanova.config import SearchlightConfig
    >>> config = SearchlightConfig(radius=3.0)

    Parallel searchlight with all CPUs:

    >>> config = SearchlightConfig(
    ...     radius=4.0,
    ...     n_jobs=-1,
    ...     show_progress=True
    ... )

    Searchlight with checkpointing for long jobs:

    >>> from pathlib import Path
    >>> config = SearchlightConfig(
    ...     radius=5.0,
    ...     checkpoint_dir=Path('./checkpoints'),
    ...     checkpoint_name='my_analysis',
    ...     progress_interval=60.0
    ... )

    Custom chunk size for memory optimization:

    >>> config = SearchlightConfig(
    ...     radius=3.0,
    ...     n_jobs=4,
    ...     chunk_size=100
    ... )
    """

    radius: float = 3.0
    checkpoint_dir: Optional[Path] = None
    checkpoint_name: str = "searchlight_checkpoint"
    progress_interval: float = 30.0
    min_voxels: int = 10
    n_jobs: int = 1
    show_progress: bool = True
    chunk_size: Optional[int] = None
    backend: str = "loky"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")

        if self.progress_interval <= 0:
            raise ValueError(
                f"progress_interval must be positive, got {self.progress_interval}"
            )

        if self.min_voxels < 1:
            raise ValueError(f"min_voxels must be >= 1, got {self.min_voxels}")

        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ValueError(
                f"n_jobs must be -1 (all CPUs) or positive, got {self.n_jobs}"
            )

        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        if self.backend not in ["loky", "threading", "multiprocessing"]:
            raise ValueError(
                f"backend must be 'loky', 'threading', or 'multiprocessing', "
                f"got '{self.backend}'"
            )

        # Convert checkpoint_dir to Path if string
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.checkpoint_dir)

    def get_checkpoint_path(self) -> Optional[Path]:
        """
        Get full checkpoint file path.

        Returns
        -------
        Path or None
            Full path to checkpoint file, or None if checkpointing disabled.
        """
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / f"{self.checkpoint_name}.pkl"


@dataclass
class RegionConfig:
    """
    Configuration for region-based (ROI) analysis.

    This configuration specifies regions of interest for ROI-based cvManova analysis.
    Regions can be provided as NIfTI mask files or as numpy boolean arrays.

    Parameters
    ----------
    regions : list of Path or list of ndarray
        Region definitions. Can be either:
        - Paths to NIfTI mask files (binary or integer-valued)
        - 3D boolean numpy arrays indicating region voxels
        All regions must have the same spatial dimensions as the data.
    region_names : list of str or None, default=None
        Descriptive names for each region (e.g., ['V1', 'V2', 'FFA']). If None,
        regions are automatically named as 'region_1', 'region_2', etc. Names are
        used in result outputs and visualizations.
    min_voxels : int, default=10
        Minimum number of voxels required per region. Regions with fewer voxels
        are rejected with an error. This prevents analysis on regions too small
        for reliable multivariate statistics.
    allow_overlap : bool, default=True
        Whether to allow regions to overlap. If False, raises an error if any
        voxels are shared between regions. Set to False for parcellation-based
        analyses where regions should be mutually exclusive.

    See Also
    --------
    RegionCvManova : Region-based estimator that uses this configuration
    SearchlightConfig : Configuration for searchlight analysis
    AnalysisConfig : Analysis-specific parameters

    Notes
    -----
    When loading from NIfTI files, any non-zero values are treated as part of
    the region (i.e., the mask is binarized).

    Region overlap is allowed by default because many anatomical atlases have
    overlapping definitions (e.g., probabilistic atlases thresholded at different
    levels).

    The min_voxels threshold ensures sufficient data for multivariate pattern
    analysis. Values below 10 may result in unstable covariance estimates.

    Examples
    --------
    Load regions from NIfTI files:

    >>> from pathlib import Path
    >>> from cvmanova.config import RegionConfig
    >>> config = RegionConfig(
    ...     regions=[Path('V1.nii.gz'), Path('V2.nii.gz'), Path('V3.nii.gz')],
    ...     region_names=['V1', 'V2', 'V3']
    ... )

    Use numpy arrays as region masks:

    >>> import numpy as np
    >>> mask1 = np.zeros((64, 64, 30), dtype=bool)
    >>> mask1[20:30, 20:30, 10:15] = True
    >>> mask2 = np.zeros((64, 64, 30), dtype=bool)
    >>> mask2[35:45, 35:45, 10:15] = True
    >>> config = RegionConfig(
    ...     regions=[mask1, mask2],
    ...     region_names=['Region A', 'Region B'],
    ...     allow_overlap=False
    ... )

    Require non-overlapping regions with higher voxel threshold:

    >>> config = RegionConfig(
    ...     regions=[Path('parcel1.nii'), Path('parcel2.nii')],
    ...     region_names=['Parcel 1', 'Parcel 2'],
    ...     min_voxels=50,
    ...     allow_overlap=False
    ... )
    """

    regions: Union[List[Path], List[np.ndarray]]
    region_names: Optional[List[str]] = None
    min_voxels: int = 10
    allow_overlap: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.regions:
            raise ValueError("regions cannot be empty")

        if self.min_voxels < 1:
            raise ValueError(f"min_voxels must be >= 1, got {self.min_voxels}")

        # Convert string paths to Path objects
        if self.regions and isinstance(self.regions[0], (str, Path)):
            self.regions = [Path(r) for r in self.regions]

        # Validate region names if provided
        if self.region_names is not None:
            if len(self.region_names) != len(self.regions):
                raise ValueError(
                    f"region_names length {len(self.region_names)} does not match "
                    f"number of regions {len(self.regions)}"
                )
        else:
            # Auto-generate region names
            self.region_names = [f"region_{i+1}" for i in range(len(self.regions))]

    @property
    def n_regions(self) -> int:
        """Number of regions."""
        return len(self.regions)


@dataclass
class AnalysisConfig:
    """
    Configuration for MANOVA analysis parameters.

    This configuration controls the core computational aspects of cvManova analysis,
    including regularization, permutation testing, and verbosity.

    Parameters
    ----------
    regularization : float, default=0.0
        Regularization parameter (lambda) for shrinkage of error covariance matrices.
        Range: [0, 1], where 0 = no regularization (MLE), 1 = diagonal covariance.
        Regularization shrinks the covariance estimate toward the diagonal, which
        can improve stability when the number of voxels is large relative to the
        number of scans. Typical values: 0.0-0.3.
    permute : bool, default=False
        Whether to compute permutation distribution for statistical testing. If True,
        computes sign permutations up to max_permutations. Permutation testing provides
        exact p-values under the null hypothesis of no discriminability.
    max_permutations : int, default=5000
        Maximum number of sign permutations to compute (only used if permute=True).
        Actual number may be less if 2^(n_sessions-1) < max_permutations. More
        permutations provide finer p-value resolution but increase computation time.
    n_jobs : int, default=1
        Number of parallel jobs for computation. Use -1 for all CPUs, 1 for sequential.
        Note: For searchlight analysis, parallelization is controlled by
        SearchlightConfig.n_jobs instead.
    verbose : int, default=1
        Verbosity level controlling printed output:
        - 0: Silent (no output)
        - 1: Progress information (default)
        - 2: Detailed diagnostic information
    memory : Path or None, default=None
        Directory for joblib memory caching of intermediate computations. If None,
        caching is disabled. Caching can speed up repeated analyses with the same
        data but uses disk space.
    random_state : int or None, default=None
        Random seed for reproducible permutation generation. If None, permutations
        are generated non-deterministically. Set to an integer for reproducibility.

    See Also
    --------
    SearchlightConfig : Configuration for searchlight-specific parameters
    RegionConfig : Configuration for region-based analysis
    ContrastSpec : Specification for generating contrast matrices

    Notes
    -----
    Regularization is applied as:
        E_regularized = (1 - lambda) * E + lambda * diag(E)

    where E is the error covariance matrix. This shrinks off-diagonal elements
    toward zero, reducing the influence of noise in covariance estimation.

    The number of possible unique sign permutations is 2^(m-1) where m is the
    number of sessions. For example:
    - 3 sessions: 4 permutations
    - 5 sessions: 16 permutations
    - 10 sessions: 512 permutations
    - 15 sessions: 16,384 permutations

    Examples
    --------
    Standard analysis without permutation testing:

    >>> from cvmanova.config import AnalysisConfig
    >>> config = AnalysisConfig(regularization=0.1, verbose=1)

    Analysis with permutation testing for p-values:

    >>> config = AnalysisConfig(
    ...     permute=True,
    ...     max_permutations=5000,
    ...     random_state=42
    ... )

    High-regularization for small sample sizes:

    >>> config = AnalysisConfig(
    ...     regularization=0.5,
    ...     verbose=2
    ... )

    Parallel analysis with caching:

    >>> from pathlib import Path
    >>> config = AnalysisConfig(
    ...     n_jobs=-1,
    ...     memory=Path('./cache'),
    ...     verbose=1
    ... )
    """

    regularization: float = 0.0
    permute: bool = False
    max_permutations: int = 5000
    n_jobs: int = 1
    verbose: int = 1
    memory: Optional[Path] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 <= self.regularization <= 1:
            raise ValueError(
                f"regularization must be in [0, 1], got {self.regularization}"
            )

        if self.max_permutations < 1:
            raise ValueError(
                f"max_permutations must be positive, got {self.max_permutations}"
            )

        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ValueError(
                f"n_jobs must be -1 (all CPUs) or positive, got {self.n_jobs}"
            )

        if self.verbose not in [0, 1, 2]:
            raise ValueError(f"verbose must be 0, 1, or 2, got {self.verbose}")

        # Convert memory to Path if string
        if self.memory is not None:
            self.memory = Path(self.memory)


@dataclass
class ContrastSpec:
    """
    Specification for auto-generating contrasts from factorial design.

    This class provides automatic generation of contrast matrices for testing
    main effects and interactions in factorial experimental designs. It eliminates
    the need to manually construct contrast matrices, ensuring correct orthogonal
    contrasts for standard factorial designs.

    Parameters
    ----------
    factors : list of str
        Names of experimental factors (e.g., ['Face', 'Object', 'Animate']).
        Factor names are used to generate descriptive labels for contrasts.
    levels : list of int
        Number of levels for each factor (e.g., [2, 3, 2] for a 2×3×2 design).
        Must have the same length as factors. Each value must be at least 2.
    effects : {'all', 'main', 'interaction'}, default='all'
        Which effects to generate contrasts for:
        - 'all': Generate contrasts for main effects and all interactions
        - 'main': Generate contrasts for main effects only
        - 'interaction': Generate contrasts for interactions only (no main effects)
    include_intercept : bool, default=False
        Whether to include an intercept term in the design matrix. Typically False
        for fMRI analysis where data is already mean-centered.

    Attributes
    ----------
    n_conditions : int
        Total number of conditions in the factorial design (product of levels).
    n_contrasts : int
        Number of contrast matrices that will be generated.

    See Also
    --------
    AnalysisConfig : Analysis configuration including regularization
    contrasts.contrasts : Underlying function for contrast generation

    Notes
    -----
    The generated contrasts are orthogonal and follow standard factorial design
    conventions. For a 2×2 design, this generates:
    1. Main effect of Factor 1
    2. Main effect of Factor 2
    3. Interaction between Factor 1 and Factor 2

    For designs with more than 2 factors, all possible interactions (2-way, 3-way,
    etc.) are generated when effects='all'.

    The contrast matrices are suitable for testing multivariate discriminability
    of neural patterns associated with each effect.

    Examples
    --------
    Simple 2×2 factorial design:

    >>> from cvmanova.config import ContrastSpec
    >>> spec = ContrastSpec(
    ...     factors=['Face', 'Object'],
    ...     levels=[2, 2]
    ... )
    >>> contrasts, names = spec.to_matrices()
    >>> print(names)
    ['Face', 'Object', 'Face×Object']
    >>> print(f"Total conditions: {spec.n_conditions}")
    Total conditions: 4

    2×3 design with only main effects:

    >>> spec = ContrastSpec(
    ...     factors=['Stimulus', 'Task'],
    ...     levels=[2, 3],
    ...     effects='main'
    ... )
    >>> contrasts, names = spec.to_matrices()
    >>> print(names)
    ['Stimulus', 'Task']

    Complex 2×2×2 design with all effects:

    >>> spec = ContrastSpec(
    ...     factors=['Category', 'Viewpoint', 'Exemplar'],
    ...     levels=[2, 2, 2],
    ...     effects='all'
    ... )
    >>> contrasts, names = spec.to_matrices()
    >>> len(names)  # 3 main + 3 two-way + 1 three-way = 7 total
    7
    >>> print(spec.n_conditions)
    8

    Only interactions (no main effects):

    >>> spec = ContrastSpec(
    ...     factors=['Face', 'House'],
    ...     levels=[2, 2],
    ...     effects='interaction'
    ... )
    >>> contrasts, names = spec.to_matrices()
    >>> print(names)
    ['Face×House']
    """

    factors: List[str]
    levels: List[int]
    effects: Literal["main", "interaction", "all"] = "all"
    include_intercept: bool = False

    def __post_init__(self):
        """Validate contrast specification."""
        if not self.factors:
            raise ValueError("factors cannot be empty")

        if len(self.factors) != len(self.levels):
            raise ValueError(
                f"factors length {len(self.factors)} does not match "
                f"levels length {len(self.levels)}"
            )

        for i, level in enumerate(self.levels):
            if level < 2:
                raise ValueError(
                    f"Factor '{self.factors[i]}' must have at least 2 levels, "
                    f"got {level}"
                )

        if self.effects not in ["main", "interaction", "all"]:
            raise ValueError(
                f"effects must be 'main', 'interaction', or 'all', got '{self.effects}'"
            )

    def to_matrices(self) -> tuple[List[np.ndarray], List[str]]:
        """
        Generate contrast matrices and names from specification.

        Returns
        -------
        contrasts : list of ndarray
            Contrast matrices.
        names : list of str
            Names for each contrast.

        Examples
        --------
        >>> spec = ContrastSpec(['Face', 'House'], [2, 2])
        >>> Cs, names = spec.to_matrices()
        >>> len(Cs), len(names)
        (3, 3)
        >>> names
        ['Face', 'House', 'Face×House']
        """
        from .contrasts import contrasts as generate_contrasts

        # Generate contrasts using existing function (returns Cs and auto-generated names)
        Cs, auto_names = generate_contrasts(self.levels, self.factors)

        # Filter based on effects setting
        n_factors = len(self.factors)

        if self.effects == "main":
            # Only main effects (first n_factors contrasts)
            Cs = Cs[:n_factors]
            names = auto_names[:n_factors]
        elif self.effects == "interaction":
            # Only interactions (everything after main effects)
            Cs = Cs[n_factors:]
            names = auto_names[n_factors:]
        else:  # "all"
            names = auto_names

        return Cs, names

    @property
    def n_conditions(self) -> int:
        """Total number of conditions in the design."""
        return int(np.prod(self.levels))

    @property
    def n_contrasts(self) -> int:
        """Number of contrasts that will be generated."""
        n_factors = len(self.factors)

        if self.effects == "main":
            return n_factors

        # Total number of non-trivial subsets (2^n - 1)
        n_all = 2**n_factors - 1

        if self.effects == "all":
            return n_all
        else:  # interaction
            return n_all - n_factors
