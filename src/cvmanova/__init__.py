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

__version__ = "3.0.0"
__author__ = "Carsten Allefeld"

__all__ = [
    # Core
    "CvManovaCore",
    # Main API
    "cv_manova_searchlight",
    "cv_manova_region",
    "run_searchlight",
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
