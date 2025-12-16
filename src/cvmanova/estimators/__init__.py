"""
Scikit-learn style estimators for cvManova analysis.

This module provides the main user-facing API following scikit-learn conventions.
"""

from .searchlight import SearchlightCvManova
from .region import RegionCvManova

__all__ = [
    "SearchlightCvManova",
    "RegionCvManova",
]
