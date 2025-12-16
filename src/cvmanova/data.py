"""
Modern data structures for cvManova.

This module provides structured containers to replace MATLAB-style
list[np.ndarray] patterns with type-safe, validated Python dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class SessionData:
    """
    Container for multi-session fMRI data.

    This structured container replaces the legacy list-of-arrays pattern with a
    type-safe, validated data structure that bundles fMRI data with its spatial
    metadata. It ensures consistency across sessions and provides convenient
    access methods.

    Parameters
    ----------
    sessions : list of ndarray
        Per-session data matrices. Each array has shape (n_scans, n_voxels) where
        n_scans is the number of volumes in that session and n_voxels is the number
        of brain voxels (must be consistent across sessions). Data should be
        preprocessed (whitened, filtered) before creating SessionData.
    mask : ndarray
        3D boolean array defining which voxels are included in the analysis.
        Shape should match the original 3D brain volume. The number of True values
        must equal n_voxels in the session data.
    affine : ndarray
        4×4 affine transformation matrix mapping voxel coordinates to world (scanner)
        coordinates. This is typically extracted from the NIfTI header and is used
        for saving results back to NIfTI format.
    degrees_of_freedom : ndarray
        Per-session residual degrees of freedom (length n_sessions). This is the
        number of scans minus the rank of the design matrix, minus any additional
        loss due to filtering. Used for bias correction in discriminability estimation.
    session_ids : list of str or None, default=None
        Optional descriptive identifiers for each session (e.g., ['run1', 'run2']).
        If None, sessions are identified by index.

    Attributes
    ----------
    n_sessions : int
        Number of sessions in the dataset.
    n_voxels : int
        Number of voxels per session (extracted from data).
    n_scans : ndarray
        Number of scans (volumes) per session as a 1D array.
    total_scans : int
        Total number of scans across all sessions.

    Methods
    -------
    get_session(idx)
        Retrieve data for a specific session by index.
    to_list()
        Convert to legacy list format for backward compatibility.

    See Also
    --------
    DesignMatrix : Container for design matrices
    loaders.SPMLoader : Load SessionData from SPM.mat files
    loaders.NiftiLoader : Load SessionData from NIfTI files

    Notes
    -----
    All validation is performed in __post_init__, ensuring that:
    - All sessions have the same number of voxels
    - Mask is 3D boolean with correct number of True values
    - Affine is 4×4
    - Degrees of freedom has correct length and positive values

    The data matrices should be in (n_scans, n_voxels) format, which differs
    from some neuroimaging conventions that use (n_voxels, n_scans). This format
    is more natural for GLM-style analyses.

    Examples
    --------
    Create SessionData from preprocessed arrays:

    >>> import numpy as np
    >>> from cvmanova.data import SessionData
    >>> # Three sessions with varying number of scans
    >>> Y1 = np.random.randn(100, 5000)  # 100 scans, 5000 voxels
    >>> Y2 = np.random.randn(100, 5000)
    >>> Y3 = np.random.randn(100, 5000)
    >>> mask = np.zeros((64, 64, 30), dtype=bool)
    >>> mask[10:50, 10:50, 5:25] = True  # 5000 voxels
    >>> affine = np.eye(4)
    >>> affine[:3, :3] *= 3.0  # 3mm isotropic
    >>> data = SessionData(
    ...     sessions=[Y1, Y2, Y3],
    ...     mask=mask,
    ...     affine=affine,
    ...     degrees_of_freedom=np.array([90, 90, 90]),
    ...     session_ids=['run1', 'run2', 'run3']
    ... )
    >>> print(f"Total data: {data.n_sessions} sessions, "
    ...       f"{data.n_voxels} voxels, {data.total_scans} scans")

    Access individual sessions:

    >>> session1_data = data.get_session(0)
    >>> print(session1_data.shape)
    (100, 5000)

    Use with loaders:

    >>> from cvmanova.loaders import SPMLoader
    >>> loader = SPMLoader('/path/to/spm')
    >>> data, design = loader.load()
    >>> print(data)
    SessionData(n_sessions=3, n_voxels=5000, ...)
    """

    sessions: List[np.ndarray]
    mask: np.ndarray
    affine: np.ndarray
    degrees_of_freedom: np.ndarray
    session_ids: Optional[List[str]] = None

    def __post_init__(self):
        """Validate data consistency after initialization."""
        # Validate sessions
        if not self.sessions:
            raise ValueError("sessions cannot be empty")

        # Check all sessions have same number of voxels
        n_voxels_per_session = [Y.shape[1] for Y in self.sessions]
        if len(set(n_voxels_per_session)) != 1:
            raise ValueError(
                f"Inconsistent number of voxels across sessions: {n_voxels_per_session}"
            )

        # Validate mask
        if self.mask.ndim != 3:
            raise ValueError(f"mask must be 3D, got shape {self.mask.shape}")

        if not np.issubdtype(self.mask.dtype, np.bool_):
            raise ValueError(f"mask must be boolean, got dtype {self.mask.dtype}")

        # Check mask voxels match data voxels
        n_mask_voxels = np.sum(self.mask)
        if n_mask_voxels != self.n_voxels:
            raise ValueError(
                f"mask has {n_mask_voxels} voxels but data has {self.n_voxels} voxels"
            )

        # Validate affine
        if self.affine.shape != (4, 4):
            raise ValueError(f"affine must be 4x4, got shape {self.affine.shape}")

        # Validate degrees of freedom
        self.degrees_of_freedom = np.asarray(self.degrees_of_freedom)
        if len(self.degrees_of_freedom) != self.n_sessions:
            raise ValueError(
                f"degrees_of_freedom has {len(self.degrees_of_freedom)} entries "
                f"but there are {self.n_sessions} sessions"
            )

        if np.any(self.degrees_of_freedom <= 0):
            raise ValueError("degrees_of_freedom must be positive")

        # Validate session IDs if provided
        if self.session_ids is not None:
            if len(self.session_ids) != self.n_sessions:
                raise ValueError(
                    f"session_ids has {len(self.session_ids)} entries "
                    f"but there are {self.n_sessions} sessions"
                )

    @property
    def n_sessions(self) -> int:
        """Number of sessions."""
        return len(self.sessions)

    @property
    def n_voxels(self) -> int:
        """Number of voxels per session."""
        return self.sessions[0].shape[1] if self.sessions else 0

    @property
    def n_scans(self) -> np.ndarray:
        """Number of scans per session."""
        return np.array([Y.shape[0] for Y in self.sessions])

    @property
    def total_scans(self) -> int:
        """Total number of scans across all sessions."""
        return sum(Y.shape[0] for Y in self.sessions)

    def get_session(self, idx: int) -> np.ndarray:
        """
        Get data for a specific session.

        Parameters
        ----------
        idx : int
            Session index (0-based).

        Returns
        -------
        ndarray
            Data matrix of shape (n_scans, n_voxels).
        """
        return self.sessions[idx]

    def to_list(self) -> List[np.ndarray]:
        """
        Convert to legacy list format for backward compatibility.

        Returns
        -------
        list of ndarray
            Per-session data matrices.
        """
        return self.sessions

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SessionData(n_sessions={self.n_sessions}, "
            f"n_voxels={self.n_voxels}, "
            f"n_scans={self.n_scans.tolist()}, "
            f"mask_shape={self.mask.shape})"
        )


@dataclass
class DesignMatrix:
    """
    Container for multi-session design matrices.

    This structured container replaces the legacy list-of-arrays pattern for design
    matrices with a validated, type-safe structure. It bundles design matrices with
    their metadata and provides convenient access methods for GLM-style analyses.

    Parameters
    ----------
    matrices : list of ndarray
        Per-session design matrices. Each array has shape (n_scans, n_regressors)
        where n_scans is the number of volumes in that session and n_regressors is
        the number of predictors. The number of regressors can vary across sessions
        (e.g., session-specific motion parameters), but all contrasts must be defined
        on regressors common to all sessions.
    regressor_names : list of str or None, default=None
        Optional names for each regressor/column (e.g., ['Face', 'House', 'Motion_X']).
        Length should match the maximum number of regressors across sessions. Names
        are used for documentation and contrast specification.
    session_ids : list of str or None, default=None
        Optional descriptive identifiers for each session (e.g., ['run1', 'run2']).
        If None, sessions are identified by index.

    Attributes
    ----------
    n_sessions : int
        Number of sessions in the dataset.
    n_regressors : ndarray
        Number of regressors per session as a 1D array (can vary across sessions).
    n_scans : ndarray
        Number of scans (volumes) per session as a 1D array.
    min_regressors : int
        Minimum number of regressors across all sessions.
    max_regressors : int
        Maximum number of regressors across all sessions.

    Methods
    -------
    get_session(idx)
        Retrieve design matrix for a specific session by index.
    to_list()
        Convert to legacy list format for backward compatibility.

    See Also
    --------
    SessionData : Container for fMRI data
    ContrastSpec : Automatic contrast generation from factorial designs
    loaders.SPMLoader : Load DesignMatrix from SPM.mat files

    Notes
    -----
    All validation is performed in __post_init__, ensuring that:
    - All matrices are 2D
    - Regressor names (if provided) match the number of columns
    - Session IDs (if provided) match the number of sessions

    Design matrices should contain the GLM regressors for each session. For fMRI
    analysis, these typically include:
    - Task-related regressors (conditions of interest)
    - Motion parameters
    - Other nuisance regressors (drift, etc.)

    Contrasts are defined on the common subset of regressors across all sessions.
    The number of regressors can vary (e.g., different motion parameters per session),
    but contrasts should only test regressors present in all sessions.

    Examples
    --------
    Create DesignMatrix from GLM design matrices:

    >>> import numpy as np
    >>> from cvmanova.data import DesignMatrix
    >>> # Three sessions with 100 scans each, 5 regressors
    >>> X1 = np.random.randn(100, 5)
    >>> X2 = np.random.randn(100, 5)
    >>> X3 = np.random.randn(100, 5)
    >>> design = DesignMatrix(
    ...     matrices=[X1, X2, X3],
    ...     regressor_names=['Face', 'House', 'Motion_X', 'Motion_Y', 'Motion_Z'],
    ...     session_ids=['run1', 'run2', 'run3']
    ... )
    >>> print(f"Design: {design.n_sessions} sessions, "
    ...       f"{design.min_regressors} regressors")

    Handle varying numbers of regressors per session:

    >>> # Sessions have different numbers of regressors
    >>> X1 = np.random.randn(100, 5)  # 5 regressors
    >>> X2 = np.random.randn(100, 7)  # 7 regressors (extra motion params)
    >>> X3 = np.random.randn(100, 6)  # 6 regressors
    >>> design = DesignMatrix(matrices=[X1, X2, X3])
    >>> print(f"Regressors per session: {design.n_regressors}")
    [5 7 6]
    >>> print(f"Min: {design.min_regressors}, Max: {design.max_regressors}")
    Min: 5, Max: 7

    Access individual session designs:

    >>> session1_design = design.get_session(0)
    >>> print(session1_design.shape)
    (100, 5)

    Use with SessionData for complete analysis:

    >>> from cvmanova.data import SessionData
    >>> # Ensure design and data have matching scans
    >>> assert all(X.shape[0] == Y.shape[0]
    ...           for X, Y in zip(design.matrices, data.sessions))
    """

    matrices: List[np.ndarray]
    regressor_names: Optional[List[str]] = None
    session_ids: Optional[List[str]] = None

    def __post_init__(self):
        """Validate design matrices after initialization."""
        # Validate matrices
        if not self.matrices:
            raise ValueError("matrices cannot be empty")

        # Check all matrices are 2D
        for i, X in enumerate(self.matrices):
            if X.ndim != 2:
                raise ValueError(
                    f"Design matrix {i} must be 2D, got shape {X.shape}"
                )

        # Validate regressor names if provided
        if self.regressor_names is not None:
            max_regressors = max(X.shape[1] for X in self.matrices)
            if len(self.regressor_names) > max_regressors:
                raise ValueError(
                    f"regressor_names has {len(self.regressor_names)} entries "
                    f"but max regressors is {max_regressors}"
                )

        # Validate session IDs if provided
        if self.session_ids is not None:
            if len(self.session_ids) != self.n_sessions:
                raise ValueError(
                    f"session_ids has {len(self.session_ids)} entries "
                    f"but there are {self.n_sessions} sessions"
                )

    @property
    def n_sessions(self) -> int:
        """Number of sessions."""
        return len(self.matrices)

    @property
    def n_regressors(self) -> np.ndarray:
        """Number of regressors per session."""
        return np.array([X.shape[1] for X in self.matrices])

    @property
    def n_scans(self) -> np.ndarray:
        """Number of scans per session."""
        return np.array([X.shape[0] for X in self.matrices])

    @property
    def min_regressors(self) -> int:
        """Minimum number of regressors across sessions."""
        return min(X.shape[1] for X in self.matrices) if self.matrices else 0

    @property
    def max_regressors(self) -> int:
        """Maximum number of regressors across sessions."""
        return max(X.shape[1] for X in self.matrices) if self.matrices else 0

    def get_session(self, idx: int) -> np.ndarray:
        """
        Get design matrix for a specific session.

        Parameters
        ----------
        idx : int
            Session index (0-based).

        Returns
        -------
        ndarray
            Design matrix of shape (n_scans, n_regressors).
        """
        return self.matrices[idx]

    def to_list(self) -> List[np.ndarray]:
        """
        Convert to legacy list format for backward compatibility.

        Returns
        -------
        list of ndarray
            Per-session design matrices.
        """
        return self.matrices

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DesignMatrix(n_sessions={self.n_sessions}, "
            f"n_regressors={self.n_regressors.tolist()}, "
            f"n_scans={self.n_scans.tolist()})"
        )
