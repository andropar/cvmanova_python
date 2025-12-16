"""
Tests for configuration dataclasses.
"""

import numpy as np
import pytest
from pathlib import Path
from cvmanova.config import (
    SearchlightConfig,
    RegionConfig,
    AnalysisConfig,
    ContrastSpec,
)


class TestSearchlightConfig:
    """Tests for SearchlightConfig."""

    def test_default_creation(self):
        """Test creation with default values."""
        config = SearchlightConfig()

        assert config.radius == 3.0
        assert config.checkpoint_dir is None
        assert config.progress_interval == 30.0
        assert config.min_voxels == 10
        assert config.n_jobs == 1
        assert config.show_progress is True
        assert config.chunk_size is None
        assert config.backend == "loky"

    def test_custom_values(self):
        """Test creation with custom values."""
        config = SearchlightConfig(
            radius=5.0,
            checkpoint_dir=Path("/tmp/checkpoints"),
            n_jobs=-1,
            backend="threading",
        )

        assert config.radius == 5.0
        assert config.checkpoint_dir == Path("/tmp/checkpoints")
        assert config.n_jobs == -1
        assert config.backend == "threading"

    def test_negative_radius(self):
        """Test error on negative radius."""
        with pytest.raises(ValueError, match="radius must be positive"):
            SearchlightConfig(radius=-1.0)

    def test_zero_radius(self):
        """Test error on zero radius."""
        with pytest.raises(ValueError, match="radius must be positive"):
            SearchlightConfig(radius=0.0)

    def test_negative_progress_interval(self):
        """Test error on negative progress interval."""
        with pytest.raises(ValueError, match="progress_interval must be positive"):
            SearchlightConfig(progress_interval=-10.0)

    def test_zero_min_voxels(self):
        """Test error on zero min_voxels."""
        with pytest.raises(ValueError, match="min_voxels must be >= 1"):
            SearchlightConfig(min_voxels=0)

    def test_invalid_n_jobs_zero(self):
        """Test error on n_jobs=0."""
        with pytest.raises(ValueError, match="n_jobs must be -1"):
            SearchlightConfig(n_jobs=0)

    def test_invalid_n_jobs_less_than_minus_one(self):
        """Test error on n_jobs < -1."""
        with pytest.raises(ValueError, match="n_jobs must be -1"):
            SearchlightConfig(n_jobs=-2)

    def test_negative_chunk_size(self):
        """Test error on negative chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SearchlightConfig(chunk_size=-100)

    def test_invalid_backend(self):
        """Test error on invalid backend."""
        with pytest.raises(ValueError, match="backend must be"):
            SearchlightConfig(backend="invalid")

    def test_checkpoint_path_conversion(self):
        """Test conversion of string to Path."""
        config = SearchlightConfig(checkpoint_dir="/tmp/test")
        assert isinstance(config.checkpoint_dir, Path)
        assert config.checkpoint_dir == Path("/tmp/test")

    def test_get_checkpoint_path(self):
        """Test getting full checkpoint path."""
        config = SearchlightConfig(checkpoint_dir=Path("/tmp"))
        path = config.get_checkpoint_path()
        assert path == Path("/tmp/searchlight_checkpoint.pkl")

    def test_get_checkpoint_path_custom_name(self):
        """Test getting checkpoint path with custom name."""
        config = SearchlightConfig(
            checkpoint_dir=Path("/tmp"), checkpoint_name="my_analysis"
        )
        path = config.get_checkpoint_path()
        assert path == Path("/tmp/my_analysis.pkl")

    def test_get_checkpoint_path_none(self):
        """Test getting checkpoint path when disabled."""
        config = SearchlightConfig(checkpoint_dir=None)
        path = config.get_checkpoint_path()
        assert path is None


class TestRegionConfig:
    """Tests for RegionConfig."""

    def test_creation_with_paths(self):
        """Test creation with file paths."""
        paths = [Path("/data/V1.nii"), Path("/data/V2.nii")]
        config = RegionConfig(regions=paths, region_names=["V1", "V2"])

        assert config.n_regions == 2
        assert config.region_names == ["V1", "V2"]
        assert config.min_voxels == 10
        assert config.allow_overlap is True

    def test_creation_with_arrays(self):
        """Test creation with numpy arrays."""
        masks = [np.random.rand(10, 10, 10) > 0.5, np.random.rand(10, 10, 10) > 0.5]
        config = RegionConfig(regions=masks)

        assert config.n_regions == 2
        assert config.region_names == ["region_1", "region_2"]

    def test_auto_generate_names(self):
        """Test auto-generation of region names."""
        paths = [Path("r1.nii"), Path("r2.nii"), Path("r3.nii")]
        config = RegionConfig(regions=paths)

        assert config.region_names == ["region_1", "region_2", "region_3"]

    def test_empty_regions(self):
        """Test error on empty regions list."""
        with pytest.raises(ValueError, match="regions cannot be empty"):
            RegionConfig(regions=[])

    def test_zero_min_voxels(self):
        """Test error on zero min_voxels."""
        with pytest.raises(ValueError, match="min_voxels must be >= 1"):
            RegionConfig(regions=[Path("test.nii")], min_voxels=0)

    def test_region_names_length_mismatch(self):
        """Test error when region names don't match regions."""
        paths = [Path("r1.nii"), Path("r2.nii")]
        with pytest.raises(ValueError, match="region_names length"):
            RegionConfig(regions=paths, region_names=["R1"])

    def test_string_paths_converted(self):
        """Test conversion of string paths to Path objects."""
        config = RegionConfig(regions=["/data/r1.nii", "/data/r2.nii"])

        assert all(isinstance(r, Path) for r in config.regions)
        assert config.regions[0] == Path("/data/r1.nii")


class TestAnalysisConfig:
    """Tests for AnalysisConfig."""

    def test_default_creation(self):
        """Test creation with default values."""
        config = AnalysisConfig()

        assert config.regularization == 0.0
        assert config.permute is False
        assert config.max_permutations == 5000
        assert config.n_jobs == 1
        assert config.verbose == 1
        assert config.memory is None
        assert config.random_state is None

    def test_custom_values(self):
        """Test creation with custom values."""
        config = AnalysisConfig(
            regularization=0.1,
            permute=True,
            max_permutations=1000,
            n_jobs=-1,
            verbose=2,
            random_state=42,
        )

        assert config.regularization == 0.1
        assert config.permute is True
        assert config.max_permutations == 1000
        assert config.n_jobs == -1
        assert config.verbose == 2
        assert config.random_state == 42

    def test_regularization_out_of_range_negative(self):
        """Test error on negative regularization."""
        with pytest.raises(ValueError, match="regularization must be in"):
            AnalysisConfig(regularization=-0.1)

    def test_regularization_out_of_range_above_one(self):
        """Test error on regularization > 1."""
        with pytest.raises(ValueError, match="regularization must be in"):
            AnalysisConfig(regularization=1.5)

    def test_negative_max_permutations(self):
        """Test error on negative max_permutations."""
        with pytest.raises(ValueError, match="max_permutations must be positive"):
            AnalysisConfig(max_permutations=0)

    def test_invalid_n_jobs(self):
        """Test error on invalid n_jobs."""
        with pytest.raises(ValueError, match="n_jobs must be -1"):
            AnalysisConfig(n_jobs=0)

    def test_invalid_verbose(self):
        """Test error on invalid verbose level."""
        with pytest.raises(ValueError, match="verbose must be 0, 1, or 2"):
            AnalysisConfig(verbose=3)

    def test_memory_path_conversion(self):
        """Test conversion of string to Path."""
        config = AnalysisConfig(memory="/tmp/cache")
        assert isinstance(config.memory, Path)
        assert config.memory == Path("/tmp/cache")


class TestContrastSpec:
    """Tests for ContrastSpec."""

    def test_two_factor_all_effects(self):
        """Test 2x2 factorial design with all effects."""
        spec = ContrastSpec(factors=["Face", "House"], levels=[2, 2])

        assert spec.n_conditions == 4
        assert spec.n_contrasts == 3  # 2 main + 1 interaction

        Cs, names = spec.to_matrices()
        assert len(Cs) == 3
        assert len(names) == 3
        assert names == ["Face", "House", "Face×House"]

    def test_two_factor_main_only(self):
        """Test 2x2 factorial with main effects only."""
        spec = ContrastSpec(factors=["A", "B"], levels=[2, 2], effects="main")

        assert spec.n_contrasts == 2

        Cs, names = spec.to_matrices()
        assert len(Cs) == 2
        assert names == ["A", "B"]

    def test_two_factor_interaction_only(self):
        """Test 2x2 factorial with interaction only."""
        spec = ContrastSpec(factors=["A", "B"], levels=[2, 2], effects="interaction")

        assert spec.n_contrasts == 1

        Cs, names = spec.to_matrices()
        assert len(Cs) == 1
        assert names == ["A×B"]

    def test_three_factor(self):
        """Test 2x2x2 factorial design."""
        spec = ContrastSpec(factors=["A", "B", "C"], levels=[2, 2, 2])

        assert spec.n_conditions == 8
        assert spec.n_contrasts == 7  # 3 main + 3 two-way + 1 three-way

        Cs, names = spec.to_matrices()
        assert len(Cs) == 7
        assert len(names) == 7
        # Main effects
        assert "A" in names
        assert "B" in names
        assert "C" in names
        # 2-way interactions
        assert "A×B" in names
        assert "A×C" in names
        assert "B×C" in names
        # 3-way interaction
        assert "A×B×C" in names

    def test_unbalanced_design(self):
        """Test unbalanced factorial design (2x3)."""
        spec = ContrastSpec(factors=["Stim", "Task"], levels=[2, 3])

        assert spec.n_conditions == 6
        assert spec.n_contrasts == 3

        Cs, names = spec.to_matrices()
        assert names == ["Stim", "Task", "Stim×Task"]

    def test_empty_factors(self):
        """Test error on empty factors."""
        with pytest.raises(ValueError, match="factors cannot be empty"):
            ContrastSpec(factors=[], levels=[])

    def test_factors_levels_mismatch(self):
        """Test error when factors and levels don't match."""
        with pytest.raises(ValueError, match="factors length .* does not match"):
            ContrastSpec(factors=["A", "B"], levels=[2])

    def test_single_level_factor(self):
        """Test error on factor with only 1 level."""
        with pytest.raises(ValueError, match="must have at least 2 levels"):
            ContrastSpec(factors=["A"], levels=[1])

    def test_invalid_effects(self):
        """Test error on invalid effects specification."""
        with pytest.raises(ValueError, match="effects must be"):
            ContrastSpec(factors=["A"], levels=[2], effects="invalid")

    def test_contrast_matrices_shape(self):
        """Test that generated contrast matrices have correct shapes."""
        spec = ContrastSpec(factors=["A", "B"], levels=[2, 3])
        Cs, names = spec.to_matrices()

        # Each contrast should be a 2D array
        for C in Cs:
            assert C.ndim == 2
            # Number of columns should match number of conditions minus intercept
            assert C.shape[1] <= 6

    def test_four_factor_design(self):
        """Test 2x2x2x2 factorial design."""
        spec = ContrastSpec(factors=["A", "B", "C", "D"], levels=[2, 2, 2, 2])

        assert spec.n_conditions == 16
        # 4 main + 6 two-way + 4 three-way + 1 four-way = 15
        assert spec.n_contrasts == 15

        Cs, names = spec.to_matrices()
        assert len(Cs) == 15
        # Check that 4-way interaction exists
        assert "A×B×C×D" in names


class TestConfigIntegration:
    """Integration tests using multiple config objects together."""

    def test_searchlight_with_analysis_config(self):
        """Test combining searchlight and analysis configs."""
        sl_config = SearchlightConfig(radius=4.0, n_jobs=-1)
        an_config = AnalysisConfig(regularization=0.1, permute=True)

        assert sl_config.radius == 4.0
        assert an_config.regularization == 0.1

    def test_region_with_contrast_spec(self):
        """Test combining region config with contrast spec."""
        region_config = RegionConfig(
            regions=[Path("V1.nii"), Path("V2.nii")], region_names=["V1", "V2"]
        )
        contrast_spec = ContrastSpec(factors=["Face", "House"], levels=[2, 2])

        Cs, names = contrast_spec.to_matrices()
        assert region_config.n_regions == 2
        assert len(Cs) == 3

    def test_full_analysis_configuration(self):
        """Test creating a complete analysis configuration."""
        searchlight = SearchlightConfig(
            radius=3.0, checkpoint_dir=Path("/tmp"), n_jobs=-1
        )

        analysis = AnalysisConfig(regularization=0.05, permute=True, verbose=2)

        contrasts = ContrastSpec(
            factors=["Stimulus", "Task", "Attention"], levels=[2, 2, 2]
        )

        # Verify all configs are valid
        assert searchlight.get_checkpoint_path() is not None
        assert analysis.regularization == 0.05
        Cs, names = contrasts.to_matrices()
        assert len(Cs) == 7  # All effects for 2x2x2
