"""
cvManova - Cross-validated MANOVA for fMRI data analysis

This package implements multivariate pattern analysis (MVPA) using
cross-validated MANOVA for fMRI data analysis, as introduced by
Allefeld & Haynes (2014).

Reference:
    Allefeld, C., & Haynes, J. D. (2014). Searchlight-based multi-voxel
    pattern analysis of fMRI by cross-validated MANOVA. NeuroImage, 89,
    345-357.
"""

from .core import CvManovaCore
from .searchlight import cv_manova_searchlight, run_searchlight
from .region import cv_manova_region
from .contrasts import contrasts
from .utils import (
    sign_permutations,
    inestimability,
    sl_size,
    fletcher16,
)
from .io import (
    load_data_spm,
    read_vols_masked,
    read_vol_matched,
    write_image,
)
from .api import searchlight_analysis, region_analysis
# Modern data structures (Phase 4)
from .data import SessionData, DesignMatrix
from .results import CvManovaResult
# Configuration system (Phase 5)
from .config import SearchlightConfig, RegionConfig, AnalysisConfig, ContrastSpec
# Scikit-learn style estimators (Phase 6)
from .estimators import SearchlightCvManova, RegionCvManova
# Flexible data loaders (Phase 7)
from .loaders import SPMLoader, NiftiLoader, NilearnMaskerLoader

__version__ = "4.0.0"
__author__ = "Carsten Allefeld"

__all__ = [
    # Core
    "CvManovaCore",
    # Main API
    "cv_manova_searchlight",
    "cv_manova_region",
    "run_searchlight",
    # Modern data structures (Phase 4)
    "SessionData",
    "DesignMatrix",
    "CvManovaResult",
    # Configuration system (Phase 5)
    "SearchlightConfig",
    "RegionConfig",
    "AnalysisConfig",
    "ContrastSpec",
    # Scikit-learn style estimators (Phase 6)
    "SearchlightCvManova",
    "RegionCvManova",
    # Flexible data loaders (Phase 7)
    "SPMLoader",
    "NiftiLoader",
    "NilearnMaskerLoader",
    # Utilities
    "contrasts",
    "sign_permutations",
    "inestimability",
    "sl_size",
    "fletcher16",
    # I/O
    "load_data_spm",
    "read_vols_masked",
    "read_vol_matched",
    "write_image",
    # High-level API
    "searchlight_analysis",
    "region_analysis",
]
