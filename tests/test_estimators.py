"""
Tests for scikit-learn style estimators.
"""

import numpy as np
import pytest
from pathlib import Path

from cvmanova.estimators import SearchlightCvManova, RegionCvManova
from cvmanova.config import (
    SearchlightConfig,
    RegionConfig,
    AnalysisConfig,
    ContrastSpec,
)
from cvmanova.data import SessionData, DesignMatrix
from cvmanova.results import CvManovaResult


class TestSearchlightCvManova:
    """Tests for SearchlightCvManova estimator."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        # Small 3D volume with 2 sessions
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True  # 3x3x3 = 27 voxels

        # Session data: random with some signal
        Y1 = np.random.randn(20, 27) + 0.5
        Y2 = np.random.randn(20, 27) + 0.5

        data = SessionData(
            sessions=[Y1, Y2],
            mask=mask,
            affine=np.eye(4),
            degrees_of_freedom=np.array([15, 15]),
        )

        # Design matrices: 2 conditions
        X1 = np.hstack([
            np.ones((20, 1)),  # intercept
            np.tile([1, -1], 10).reshape(-1, 1),  # condition effect
        ])
        X2 = np.hstack([
            np.ones((20, 1)),
            np.tile([1, -1], 10).reshape(-1, 1),
        ])

        design = DesignMatrix(matrices=[X1, X2])

        # Simple contrast
        contrasts = [np.array([[0, 1]])]  # Test condition effect

        return data, design, contrasts

    def test_initialization_defaults(self):
        """Test initialization with default config."""
        estimator = SearchlightCvManova(contrasts=[np.array([[1, -1]])])

        assert estimator.searchlight_config is not None
        assert estimator.analysis_config is not None
        assert not estimator.is_fitted_

    def test_initialization_with_configs(self):
        """Test initialization with custom configs."""
        sl_config = SearchlightConfig(radius=5.0, n_jobs=2)
        an_config = AnalysisConfig(regularization=0.1, verbose=0)
        contrasts = [np.array([[1, -1]])]

        estimator = SearchlightCvManova(
            searchlight_config=sl_config,
            analysis_config=an_config,
            contrasts=contrasts,
        )

        assert estimator.searchlight_config.radius == 5.0
        assert estimator.searchlight_config.n_jobs == 2
        assert estimator.analysis_config.regularization == 0.1

    def test_initialization_with_contrast_spec(self):
        """Test initialization with ContrastSpec."""
        contrast_spec = ContrastSpec(["A", "B"], [2, 2])

        estimator = SearchlightCvManova(contrasts=contrast_spec)
        assert estimator.contrasts == contrast_spec

    def test_n_jobs_override(self):
        """Test n_jobs parameter overrides config."""
        sl_config = SearchlightConfig(n_jobs=1)
        estimator = SearchlightCvManova(
            searchlight_config=sl_config,
            contrasts=[np.array([[1, -1]])],
            n_jobs=4,
        )

        assert estimator.searchlight_config.n_jobs == 4

    def test_verbose_override(self):
        """Test verbose parameter overrides config."""
        an_config = AnalysisConfig(verbose=1)
        estimator = SearchlightCvManova(
            analysis_config=an_config,
            contrasts=[np.array([[1, -1]])],
            verbose=0,
        )

        assert estimator.analysis_config.verbose == 0

    def test_fit_basic(self, simple_data):
        """Test basic fit functionality."""
        data, design, contrasts = simple_data

        estimator = SearchlightCvManova(contrasts=contrasts, verbose=0)
        result = estimator.fit(data, design)

        assert result is estimator  # Returns self
        assert estimator.is_fitted_
        assert estimator.data_ is data
        assert estimator.design_ is design
        assert len(estimator.contrast_matrices_) == 1

    def test_fit_validates_contrast_dimensions(self, simple_data):
        """Test that fit validates contrast dimensions."""
        data, design, contrasts = simple_data

        # Create contrast with wrong dimensions (too many columns)
        bad_contrast = [np.array([[1, -1, 0, 0, 0]])]  # 5 columns, but design has 2

        estimator = SearchlightCvManova(contrasts=bad_contrast, verbose=0)

        with pytest.raises(ValueError, match="columns but design has"):
            estimator.fit(data, design)

    def test_fit_validates_data_design_compatibility(self, simple_data):
        """Test that fit validates data/design compatibility."""
        data, design, contrasts = simple_data

        # Create incompatible design (wrong number of sessions)
        bad_design = DesignMatrix(matrices=[design.matrices[0]])

        estimator = SearchlightCvManova(contrasts=contrasts, verbose=0)

        with pytest.raises(ValueError, match="Number of data sessions"):
            estimator.fit(data, bad_design)

    def test_fit_validates_scan_counts(self, simple_data):
        """Test that fit validates scan counts match."""
        data, design, contrasts = simple_data

        # Create design with wrong number of scans
        bad_X = np.random.randn(10, 2)  # Only 10 scans instead of 20
        bad_design = DesignMatrix(matrices=[bad_X, design.matrices[1]])

        estimator = SearchlightCvManova(contrasts=contrasts, verbose=0)

        with pytest.raises(ValueError, match="number of scans"):
            estimator.fit(data, bad_design)

    def test_fit_with_contrast_spec(self, simple_data):
        """Test fit with ContrastSpec."""
        data, design, contrasts = simple_data

        # Use ContrastSpec instead of matrices
        # Need to adjust design to have 4 conditions for 2x2 design
        X1 = np.random.randn(20, 5)  # 5 regressors (1 intercept + 4 conditions)
        X2 = np.random.randn(20, 5)
        design = DesignMatrix(matrices=[X1, X2])

        contrast_spec = ContrastSpec(["A", "B"], [2, 2])

        estimator = SearchlightCvManova(contrasts=contrast_spec, verbose=0)
        estimator.fit(data, design)

        assert estimator.is_fitted_
        assert len(estimator.contrast_matrices_) == 3  # 2 main + 1 interaction
        assert len(estimator.contrast_names_) == 3

    def test_score_without_fit_raises(self):
        """Test that score without fit raises error."""
        estimator = SearchlightCvManova(contrasts=[np.array([[1, -1]])])

        with pytest.raises(ValueError, match="not fitted yet"):
            estimator.score()

    def test_score_requires_both_or_neither(self, simple_data):
        """Test that score requires both data and design, or neither."""
        data, design, contrasts = simple_data

        estimator = SearchlightCvManova(contrasts=contrasts, verbose=0)
        estimator.fit(data, design)

        # Providing only data should raise
        with pytest.raises(ValueError, match="Must provide both"):
            estimator.score(data=data)

        # Providing only design should raise
        with pytest.raises(ValueError, match="Must provide both"):
            estimator.score(design=design)

    def test_score_returns_result(self, simple_data):
        """Test that score returns CvManovaResult."""
        data, design, contrasts = simple_data

        estimator = SearchlightCvManova(
            contrasts=contrasts,
            verbose=0,
            searchlight_config=SearchlightConfig(radius=1.0, show_progress=False),
        )
        estimator.fit(data, design)
        result = estimator.score()

        assert isinstance(result, CvManovaResult)
        assert result.analysis_type == "searchlight"
        assert result.n_contrasts == 1
        assert result.discriminability.shape[0] == np.prod(data.mask.shape)

    def test_fit_score(self, simple_data):
        """Test convenience fit_score method."""
        data, design, contrasts = simple_data

        estimator = SearchlightCvManova(
            contrasts=contrasts,
            verbose=0,
            searchlight_config=SearchlightConfig(radius=1.0, show_progress=False),
        )
        result = estimator.fit_score(data, design)

        assert isinstance(result, CvManovaResult)
        assert estimator.is_fitted_

    def test_get_result(self, simple_data):
        """Test get_result method."""
        data, design, contrasts = simple_data

        estimator = SearchlightCvManova(
            contrasts=contrasts,
            verbose=0,
            searchlight_config=SearchlightConfig(radius=1.0, show_progress=False),
        )

        # Before scoring, should raise
        with pytest.raises(ValueError, match="No results available"):
            estimator.get_result()

        # After scoring, should return result
        estimator.fit_score(data, design)
        result = estimator.get_result()

        assert isinstance(result, CvManovaResult)

    def test_permutation_analysis(self, simple_data):
        """Test with permutation testing."""
        data, design, contrasts = simple_data

        estimator = SearchlightCvManova(
            contrasts=contrasts,
            searchlight_config=SearchlightConfig(radius=1.0, show_progress=False),
            analysis_config=AnalysisConfig(
                permute=True, max_permutations=10, random_state=42
            ),
            verbose=0,
        )

        result = estimator.fit_score(data, design)

        assert result.n_perms > 1
        assert result.discriminability.shape[2] == result.n_perms

    def test_repr(self):
        """Test string representation."""
        estimator = SearchlightCvManova(
            contrasts=[np.array([[1, -1]])],
            searchlight_config=SearchlightConfig(radius=5.0),
        )

        repr_str = repr(estimator)
        assert "SearchlightCvManova" in repr_str
        assert "radius=5.0" in repr_str
        assert "not fitted" in repr_str


class TestRegionCvManova:
    """Tests for RegionCvManova estimator."""

    @pytest.fixture
    def region_data(self):
        """Create test data with regions."""
        # Larger 3D volume
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:8, 2:8, 2:8] = True  # 6x6x6 = 216 voxels

        # Create 3 non-overlapping regions with enough voxels
        region1 = np.zeros((10, 10, 10), dtype=bool)
        region1[2:4, 2:5, 2:5] = True  # 2x3x3 = 18 voxels

        region2 = np.zeros((10, 10, 10), dtype=bool)
        region2[4:6, 2:5, 2:5] = True  # 2x3x3 = 18 voxels

        region3 = np.zeros((10, 10, 10), dtype=bool)
        region3[6:8, 2:5, 2:5] = True  # 2x3x3 = 18 voxels

        # Session data
        n_voxels = np.sum(mask)
        Y1 = np.random.randn(20, n_voxels) + 0.5
        Y2 = np.random.randn(20, n_voxels) + 0.5

        data = SessionData(
            sessions=[Y1, Y2],
            mask=mask,
            affine=np.eye(4),
            degrees_of_freedom=np.array([15, 15]),
        )

        # Design matrices
        X1 = np.hstack([
            np.ones((20, 1)),
            np.tile([1, -1], 10).reshape(-1, 1),
        ])
        X2 = np.hstack([
            np.ones((20, 1)),
            np.tile([1, -1], 10).reshape(-1, 1),
        ])

        design = DesignMatrix(matrices=[X1, X2])

        # Contrasts
        contrasts = [np.array([[0, 1]])]

        return data, design, contrasts, [region1, region2, region3]

    def test_initialization_requires_region_config(self):
        """Test that initialization requires RegionConfig."""
        with pytest.raises(ValueError, match="region_config is required"):
            RegionCvManova(region_config=None, contrasts=[np.array([[1, -1]])])

    def test_initialization_with_config(self, region_data):
        """Test initialization with RegionConfig."""
        _, _, contrasts, regions = region_data

        region_config = RegionConfig(
            regions=regions, region_names=["R1", "R2", "R3"]
        )

        estimator = RegionCvManova(
            region_config=region_config, contrasts=contrasts
        )

        assert estimator.region_config is not None
        assert not estimator.is_fitted_

    def test_fit_basic(self, region_data):
        """Test basic fit functionality."""
        data, design, contrasts, regions = region_data

        region_config = RegionConfig(regions=regions)

        estimator = RegionCvManova(
            region_config=region_config, contrasts=contrasts, verbose=0
        )
        result = estimator.fit(data, design)

        assert result is estimator
        assert estimator.is_fitted_
        assert len(estimator.region_masks_) == 3

    def test_fit_validates_region_shapes(self, region_data):
        """Test that fit validates region shapes."""
        data, design, contrasts, regions = region_data

        # Create region with wrong shape
        bad_region = np.ones((5, 5, 5), dtype=bool)

        region_config = RegionConfig(regions=[bad_region])

        estimator = RegionCvManova(
            region_config=region_config, contrasts=contrasts, verbose=0
        )

        with pytest.raises(ValueError, match="mask shape .* does not match"):
            estimator.fit(data, design)

    def test_fit_validates_minimum_voxels(self, region_data):
        """Test that fit validates minimum voxels."""
        data, design, contrasts, _ = region_data

        # Create region with only 1 voxel
        tiny_region = np.zeros((10, 10, 10), dtype=bool)
        tiny_region[2, 2, 2] = True

        region_config = RegionConfig(regions=[tiny_region], min_voxels=10)

        estimator = RegionCvManova(
            region_config=region_config, contrasts=contrasts, verbose=0
        )

        with pytest.raises(ValueError, match="only .* voxels, minimum is"):
            estimator.fit(data, design)

    def test_fit_detects_overlap(self, region_data):
        """Test that fit detects overlapping regions."""
        data, design, contrasts, regions = region_data

        # Create overlapping regions
        overlap_region = regions[0].copy()  # Same as region 1

        region_config = RegionConfig(
            regions=[regions[0], overlap_region], allow_overlap=False
        )

        estimator = RegionCvManova(
            region_config=region_config, contrasts=contrasts, verbose=0
        )

        with pytest.raises(ValueError, match="overlap"):
            estimator.fit(data, design)

    def test_fit_allows_overlap(self, region_data):
        """Test that fit allows overlapping regions when configured."""
        data, design, contrasts, regions = region_data

        # Create overlapping regions
        overlap_region = regions[0].copy()

        region_config = RegionConfig(
            regions=[regions[0], overlap_region], allow_overlap=True
        )

        estimator = RegionCvManova(
            region_config=region_config, contrasts=contrasts, verbose=0
        )

        # Should not raise
        estimator.fit(data, design)
        assert estimator.is_fitted_

    def test_score_returns_result(self, region_data):
        """Test that score returns CvManovaResult."""
        data, design, contrasts, regions = region_data

        region_config = RegionConfig(regions=regions)

        estimator = RegionCvManova(
            region_config=region_config, contrasts=contrasts, verbose=0
        )
        estimator.fit(data, design)
        result = estimator.score()

        assert isinstance(result, CvManovaResult)
        assert result.analysis_type == "region"
        assert result.n_contrasts == 1
        assert result.discriminability.shape[0] == 3  # 3 regions
        assert result.region_names == ["region_1", "region_2", "region_3"]

    def test_fit_score(self, region_data):
        """Test convenience fit_score method."""
        data, design, contrasts, regions = region_data

        region_config = RegionConfig(
            regions=regions, region_names=["V1", "V2", "V3"]
        )

        estimator = RegionCvManova(
            region_config=region_config, contrasts=contrasts, verbose=0
        )
        result = estimator.fit_score(data, design)

        assert isinstance(result, CvManovaResult)
        assert estimator.is_fitted_
        assert result.region_names == ["V1", "V2", "V3"]

    def test_get_result(self, region_data):
        """Test get_result method."""
        data, design, contrasts, regions = region_data

        region_config = RegionConfig(regions=regions)

        estimator = RegionCvManova(
            region_config=region_config, contrasts=contrasts, verbose=0
        )

        # Before scoring, should raise
        with pytest.raises(ValueError, match="No results available"):
            estimator.get_result()

        # After scoring, should return result
        estimator.fit_score(data, design)
        result = estimator.get_result()

        assert isinstance(result, CvManovaResult)

    def test_repr(self, region_data):
        """Test string representation."""
        _, _, contrasts, regions = region_data

        region_config = RegionConfig(regions=regions)

        estimator = RegionCvManova(
            region_config=region_config, contrasts=contrasts
        )

        repr_str = repr(estimator)
        assert "RegionCvManova" in repr_str
        assert "n_regions=3" in repr_str
        assert "not fitted" in repr_str


class TestEstimatorIntegration:
    """Integration tests using both estimators."""

    def test_searchlight_and_region_same_data(self):
        """Test that both estimators can work on same dataset."""
        # Create simple dataset with more voxels
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[3:7, 3:7, 3:7] = True  # 4x4x4 = 64 voxels

        Y1 = np.random.randn(20, 64)
        Y2 = np.random.randn(20, 64)

        data = SessionData(
            sessions=[Y1, Y2],
            mask=mask,
            affine=np.eye(4),
            degrees_of_freedom=np.array([15, 15]),
        )

        X1 = np.random.randn(20, 3)
        X2 = np.random.randn(20, 3)
        design = DesignMatrix(matrices=[X1, X2])

        contrasts = [np.array([[0, 1, 0]])]

        # Run searchlight with small radius
        sl_estimator = SearchlightCvManova(
            contrasts=contrasts,
            searchlight_config=SearchlightConfig(radius=1.0, show_progress=False),
            verbose=0,
        )
        sl_result = sl_estimator.fit_score(data, design)

        # Run region analysis on same mask
        region_config = RegionConfig(regions=[mask])
        roi_estimator = RegionCvManova(
            region_config=region_config, contrasts=contrasts, verbose=0
        )
        roi_result = roi_estimator.fit_score(data, design)

        # Both should return valid results
        assert isinstance(sl_result, CvManovaResult)
        assert isinstance(roi_result, CvManovaResult)
        assert sl_result.n_contrasts == roi_result.n_contrasts

    def test_contrast_spec_with_both_estimators(self):
        """Test ContrastSpec works with both estimators."""
        # Setup data with more voxels
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[4:6, 4:6, 4:6] = True  # 2x2x2 = 8 voxels, but we need at least 10 for region
        mask[4:7, 4:7, 4:5] = True  # Extend to have ~12 voxels

        n_voxels = np.sum(mask)
        Y1 = np.random.randn(20, n_voxels)
        Y2 = np.random.randn(20, n_voxels)

        data = SessionData(
            sessions=[Y1, Y2],
            mask=mask,
            affine=np.eye(4),
            degrees_of_freedom=np.array([15, 15]),
        )

        # Design with 4 conditions (2x2)
        X1 = np.random.randn(20, 5)
        X2 = np.random.randn(20, 5)
        design = DesignMatrix(matrices=[X1, X2])

        contrast_spec = ContrastSpec(["A", "B"], [2, 2])

        # Test with searchlight
        sl_estimator = SearchlightCvManova(
            contrasts=contrast_spec,
            searchlight_config=SearchlightConfig(radius=1.0, show_progress=False),
            verbose=0,
        )
        sl_result = sl_estimator.fit_score(data, design)

        # Test with region
        region_config = RegionConfig(regions=[mask])
        roi_estimator = RegionCvManova(
            region_config=region_config, contrasts=contrast_spec, verbose=0
        )
        roi_result = roi_estimator.fit_score(data, design)

        # Both should have same contrasts
        assert sl_result.contrast_names == roi_result.contrast_names
        assert len(sl_result.contrast_names) == 3  # 2 main + 1 interaction
