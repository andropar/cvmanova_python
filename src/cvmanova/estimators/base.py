"""
Base estimator class for cvManova analysis.

Provides common functionality for all cvManova estimators.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np

from ..config import AnalysisConfig, ContrastSpec
from ..data import SessionData, DesignMatrix
from ..results import CvManovaResult


class BaseCvManova(ABC):
    """
    Base class for cvManova estimators.

    This abstract base class provides the common interface and functionality for all
    cvManova estimators, following scikit-learn's estimator conventions. It implements
    shared validation, contrast handling, and the fit/score/fit_score API pattern.

    Subclasses must implement the abstract fit() and score() methods to define
    specific analysis strategies (searchlight, region-based, etc.).

    Parameters
    ----------
    contrasts : list of ndarray or ContrastSpec or None
        Specification of contrasts to test. Can be either:
        - List of contrast matrices (each of shape (n_contrasts_i, n_regressors))
        - ContrastSpec object for automatic generation from factorial design
        - None (must be set before calling fit)
    analysis_config : AnalysisConfig or None, default=None
        Configuration for analysis parameters (regularization, permutation testing,
        verbosity). If None, uses default AnalysisConfig().
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
    is_fitted_ : bool
        Whether the estimator has been fitted.

    Methods
    -------
    fit(data, design)
        Fit the estimator to data and design matrices.
    score(data, design)
        Compute discriminability scores.
    fit_score(data, design)
        Convenience method combining fit() and score().

    See Also
    --------
    SearchlightCvManova : Searchlight-based estimator
    RegionCvManova : Region-based estimator
    AnalysisConfig : Configuration for analysis parameters
    ContrastSpec : Automatic contrast generation

    Notes
    -----
    This class follows scikit-learn conventions:
    - fit() stores data and validates inputs
    - score() performs computation and returns results
    - fit_score() is a convenience method for fit().score()
    - Fitted attributes end with underscore (_)
    - is_fitted_ tracks whether fit() has been called

    Contrasts must be estimable in all sessions (i.e., they must span the column
    space of the design matrices). This is validated in _validate_and_prepare_contrasts().

    Examples
    --------
    Subclasses implement specific estimator types:

    >>> from cvmanova import SearchlightCvManova
    >>> from cvmanova.config import ContrastSpec, SearchlightConfig
    >>> # Automatic contrast generation
    >>> contrasts = ContrastSpec(['Face', 'House'], [2, 2])
    >>> estimator = SearchlightCvManova(
    ...     contrasts=contrasts,
    ...     searchlight_config=SearchlightConfig(radius=3.0)
    ... )
    >>> result = estimator.fit_score(data, design)

    Manual contrast specification:

    >>> import numpy as np
    >>> # Define contrasts manually
    >>> C_face = np.array([[1, -1, 0, 0]])  # Face > House
    >>> C_house = np.array([[0, 0, 1, -1]])  # House > Object
    >>> estimator = SearchlightCvManova(contrasts=[C_face, C_house])
    >>> result = estimator.fit_score(data, design)
    """

    def __init__(
        self,
        contrasts: Optional[Union[List[np.ndarray], ContrastSpec]] = None,
        analysis_config: Optional[AnalysisConfig] = None,
        verbose: Optional[int] = None,
    ):
        self.contrasts = contrasts
        self.analysis_config = analysis_config or AnalysisConfig()

        # Override verbose if explicitly provided
        if verbose is not None:
            self.analysis_config.verbose = verbose

        # State variables
        self.data_: Optional[SessionData] = None
        self.design_: Optional[DesignMatrix] = None
        self.contrast_matrices_: Optional[List[np.ndarray]] = None
        self.contrast_names_: Optional[List[str]] = None
        self.is_fitted_ = False

    def _validate_and_prepare_contrasts(self, design: DesignMatrix):
        """
        Validate and prepare contrast matrices.

        Parameters
        ----------
        design : DesignMatrix
            Design matrix to validate contrasts against.

        Raises
        ------
        ValueError
            If contrasts are invalid or incompatible with design.
        """
        if self.contrasts is None:
            raise ValueError(
                "contrasts must be specified. Provide either a list of contrast "
                "matrices or a ContrastSpec for auto-generation."
            )

        if isinstance(self.contrasts, ContrastSpec):
            # Auto-generate from specification
            self.contrast_matrices_, self.contrast_names_ = (
                self.contrasts.to_matrices()
            )
        else:
            # Use provided matrices
            self.contrast_matrices_ = self.contrasts

            # Auto-generate names if not provided
            self.contrast_names_ = [
                f"contrast_{i+1}" for i in range(len(self.contrast_matrices_))
            ]

        # Validate contrast dimensions
        n_regressors = design.matrices[0].shape[1]
        for i, C in enumerate(self.contrast_matrices_):
            if C.ndim != 2:
                raise ValueError(
                    f"Contrast {i} must be 2D, got shape {C.shape}"
                )
            if C.shape[1] > n_regressors:
                raise ValueError(
                    f"Contrast {i} has {C.shape[1]} columns but design has "
                    f"only {n_regressors} regressors"
                )

    def _validate_data(self, data: SessionData, design: DesignMatrix):
        """
        Validate that data and design are compatible.

        Parameters
        ----------
        data : SessionData
            Brain data.
        design : DesignMatrix
            Design matrices.

        Raises
        ------
        ValueError
            If data and design are incompatible.
        """
        if data.n_sessions != len(design.matrices):
            raise ValueError(
                f"Number of data sessions ({data.n_sessions}) does not match "
                f"number of design matrices ({len(design.matrices)})"
            )

        for i, (Y, X) in enumerate(zip(data.sessions, design.matrices)):
            if Y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Session {i}: number of scans in data ({Y.shape[0]}) "
                    f"does not match design matrix ({X.shape[0]})"
                )

    @abstractmethod
    def fit(
        self, data: SessionData, design: DesignMatrix
    ) -> "BaseCvManova":
        """
        Fit the estimator (store data and validate).

        Parameters
        ----------
        data : SessionData
            Brain data.
        design : DesignMatrix
            Design matrices.

        Returns
        -------
        self : BaseCvManova
            Fitted estimator.
        """
        pass

    @abstractmethod
    def score(
        self,
        data: Optional[SessionData] = None,
        design: Optional[DesignMatrix] = None,
    ) -> CvManovaResult:
        """
        Compute discriminability scores.

        Parameters
        ----------
        data : SessionData, optional
            Brain data. Uses fitted data if None.
        design : DesignMatrix, optional
            Design matrices. Uses fitted design if None.

        Returns
        -------
        CvManovaResult
            Analysis results.
        """
        pass

    def fit_score(
        self, data: SessionData, design: DesignMatrix
    ) -> CvManovaResult:
        """
        Fit and score in one step (convenience method).

        Parameters
        ----------
        data : SessionData
            Brain data.
        design : DesignMatrix
            Design matrices.

        Returns
        -------
        CvManovaResult
            Analysis results.
        """
        return self.fit(data, design).score()

    def _check_is_fitted(self):
        """Check if estimator has been fitted."""
        if not self.is_fitted_:
            raise ValueError(
                "This estimator is not fitted yet. Call fit() before score()."
            )
