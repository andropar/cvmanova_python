"""
Tests for result objects (CvManovaResult).
"""

import numpy as np
import pytest
from pathlib import Path
from cvmanova.results import CvManovaResult


class TestCvManovaResult:
    """Tests for CvManovaResult class."""

    @pytest.fixture
    def searchlight_result(self):
        """Create a basic searchlight result for testing."""
        n_voxels_total = 1000
        n_contrasts = 2
        n_perms = 1

        discriminability = np.random.randn(n_voxels_total, n_contrasts, n_perms)
        n_voxels = np.random.randint(50, 150, size=n_voxels_total)
        contrasts = [np.array([[1, -1, 0]]), np.array([[0, 1, -1]])]
        contrast_names = ["Face-House", "House-Cat"]
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask_flat = mask.ravel(order="F")
        mask_flat[:n_voxels_total] = True
        mask = mask_flat.reshape((10, 10, 10), order="F")
        affine = np.eye(4)

        return CvManovaResult(
            discriminability=discriminability,
            n_voxels=n_voxels,
            contrasts=contrasts,
            contrast_names=contrast_names,
            mask=mask,
            affine=affine,
            analysis_type="searchlight"
        )

    @pytest.fixture
    def region_result(self):
        """Create a basic region result for testing."""
        n_regions = 5
        n_contrasts = 2
        n_perms = 1

        discriminability = np.random.randn(n_regions, n_contrasts, n_perms)
        n_voxels = np.random.randint(100, 500, size=n_regions)
        contrasts = [np.array([[1, -1, 0]]), np.array([[0, 1, -1]])]
        contrast_names = ["Face-House", "House-Cat"]
        affine = np.eye(4)
        region_names = ["V1", "V2", "FFA", "PPA", "EBA"]

        return CvManovaResult(
            discriminability=discriminability,
            n_voxels=n_voxels,
            contrasts=contrasts,
            contrast_names=contrast_names,
            mask=None,
            affine=affine,
            analysis_type="region",
            region_names=region_names
        )

    def test_searchlight_creation(self, searchlight_result):
        """Test basic searchlight result creation."""
        result = searchlight_result

        assert result.analysis_type == "searchlight"
        assert result.n_contrasts == 2
        assert result.n_perms == 1
        assert len(result.contrast_names) == 2

    def test_region_creation(self, region_result):
        """Test basic region result creation."""
        result = region_result

        assert result.analysis_type == "region"
        assert result.n_contrasts == 2
        assert result.n_perms == 1
        assert result.region_names == ["V1", "V2", "FFA", "PPA", "EBA"]

    def test_invalid_analysis_type(self):
        """Test error on invalid analysis type."""
        with pytest.raises(ValueError, match="analysis_type must be"):
            CvManovaResult(
                discriminability=np.random.randn(100, 2, 1),
                n_voxels=np.ones(100),
                contrasts=[np.array([[1, -1]])],
                contrast_names=["test"],
                mask=None,
                affine=np.eye(4),
                analysis_type="invalid"
            )

    def test_discriminability_not_3d(self):
        """Test error when discriminability is not 3D."""
        with pytest.raises(ValueError, match="discriminability must be 3D"):
            CvManovaResult(
                discriminability=np.random.randn(100, 2),  # Only 2D
                n_voxels=np.ones(100),
                contrasts=[np.array([[1, -1]])],
                contrast_names=["test"],
                mask=None,
                affine=np.eye(4),
                analysis_type="region"
            )

    def test_n_voxels_length_mismatch(self):
        """Test error when n_voxels length doesn't match discriminability."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask.ravel(order="F")[:100] = True

        with pytest.raises(ValueError, match="n_voxels length .* does not match"):
            CvManovaResult(
                discriminability=np.random.randn(100, 2, 1),
                n_voxels=np.ones(50),  # Wrong length
                contrasts=[np.array([[1, -1]]), np.array([[1, 0]])],
                contrast_names=["c1", "c2"],
                mask=mask,
                affine=np.eye(4),
                analysis_type="searchlight"
            )

    def test_contrast_count_mismatch(self):
        """Test error when contrast count doesn't match."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask.ravel(order="F")[:100] = True

        with pytest.raises(ValueError, match="Number of contrast matrices"):
            CvManovaResult(
                discriminability=np.random.randn(100, 2, 1),
                n_voxels=np.ones(100),
                contrasts=[np.array([[1, -1]])],  # Only 1 contrast
                contrast_names=["c1", "c2"],  # But 2 names
                mask=mask,
                affine=np.eye(4),
                analysis_type="searchlight"
            )

    def test_contrast_names_mismatch(self):
        """Test error when contrast names count doesn't match."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask.ravel(order="F")[:100] = True

        with pytest.raises(ValueError, match="Number of contrast names"):
            CvManovaResult(
                discriminability=np.random.randn(100, 2, 1),
                n_voxels=np.ones(100),
                contrasts=[np.array([[1, -1]]), np.array([[1, 0]])],
                contrast_names=["c1"],  # Only 1 name
                mask=mask,
                affine=np.eye(4),
                analysis_type="searchlight"
            )

    def test_searchlight_requires_mask(self):
        """Test error when searchlight analysis has no mask."""
        with pytest.raises(ValueError, match="mask is required for searchlight"):
            CvManovaResult(
                discriminability=np.random.randn(100, 2, 1),
                n_voxels=np.ones(100),
                contrasts=[np.array([[1, -1]]), np.array([[1, 0]])],
                contrast_names=["c1", "c2"],
                mask=None,  # No mask
                affine=np.eye(4),
                analysis_type="searchlight"
            )

    def test_mask_not_3d(self):
        """Test error when mask is not 3D."""
        with pytest.raises(ValueError, match="mask must be 3D"):
            CvManovaResult(
                discriminability=np.random.randn(100, 2, 1),
                n_voxels=np.ones(100),
                contrasts=[np.array([[1, -1]]), np.array([[1, 0]])],
                contrast_names=["c1", "c2"],
                mask=np.zeros((10, 10), dtype=bool),  # 2D
                affine=np.eye(4),
                analysis_type="searchlight"
            )

    def test_region_names_length_mismatch(self):
        """Test error when region names length doesn't match."""
        with pytest.raises(ValueError, match="Number of region names"):
            CvManovaResult(
                discriminability=np.random.randn(5, 2, 1),
                n_voxels=np.ones(5),
                contrasts=[np.array([[1, -1]]), np.array([[1, 0]])],
                contrast_names=["c1", "c2"],
                mask=None,
                affine=np.eye(4),
                analysis_type="region",
                region_names=["r1", "r2"]  # Only 2 names for 5 regions
            )

    def test_invalid_affine(self):
        """Test error on invalid affine matrix."""
        with pytest.raises(ValueError, match="affine must be 4x4"):
            CvManovaResult(
                discriminability=np.random.randn(5, 2, 1),
                n_voxels=np.ones(5),
                contrasts=[np.array([[1, -1]]), np.array([[1, 0]])],
                contrast_names=["c1", "c2"],
                mask=None,
                affine=np.eye(3),  # Wrong shape
                analysis_type="region"
            )

    def test_get_contrast_by_name(self, searchlight_result):
        """Test getting contrast by name."""
        disc = searchlight_result.get_contrast("Face-House")
        assert disc.shape == (1000,)

    def test_get_contrast_by_index(self, searchlight_result):
        """Test getting contrast by index."""
        disc = searchlight_result.get_contrast(0)
        assert disc.shape == (1000,)

    def test_get_contrast_invalid_name(self, searchlight_result):
        """Test error on invalid contrast name."""
        with pytest.raises(ValueError, match="Contrast .* not found"):
            searchlight_result.get_contrast("Invalid")

    def test_get_contrast_invalid_index(self, searchlight_result):
        """Test error on invalid contrast index."""
        with pytest.raises(ValueError, match="Contrast index .* out of range"):
            searchlight_result.get_contrast(10)

    def test_get_contrast_invalid_perm(self, searchlight_result):
        """Test error on invalid permutation index."""
        with pytest.raises(ValueError, match="Permutation index .* out of range"):
            searchlight_result.get_contrast("Face-House", perm=10)

    def test_get_contrast_3d(self, searchlight_result):
        """Test getting 3D contrast volume."""
        volume = searchlight_result.get_contrast_3d("Face-House")
        assert volume.shape == (10, 10, 10)

    def test_get_contrast_3d_region_error(self, region_result):
        """Test error when getting 3D volume from region result."""
        with pytest.raises(ValueError, match="only available for searchlight"):
            region_result.get_contrast_3d("Face-House")

    def test_get_peaks(self, searchlight_result):
        """Test getting peak locations."""
        peaks = searchlight_result.get_peaks("Face-House", n=5)
        assert peaks.shape == (5, 4)  # 5 peaks × (x, y, z, discriminability)

    def test_get_peaks_region_error(self, region_result):
        """Test error when getting peaks from region result."""
        with pytest.raises(ValueError, match="only available for searchlight"):
            region_result.get_peaks("Face-House")

    def test_repr(self, searchlight_result):
        """Test string representation."""
        repr_str = repr(searchlight_result)
        assert "CvManovaResult" in repr_str
        assert "searchlight" in repr_str
        assert "n_contrasts=2" in repr_str

    # Tests requiring optional dependencies are marked as skip if not available

    def test_to_nifti(self, searchlight_result, tmp_path):
        """Test saving to NIfTI file."""
        pytest.importorskip("nibabel")

        output_path = tmp_path / "test_result.nii.gz"
        searchlight_result.to_nifti("Face-House", output_path)

        assert output_path.exists()

    def test_to_nifti_region_error(self, region_result, tmp_path):
        """Test error when saving region result to NIfTI."""
        pytest.importorskip("nibabel")

        with pytest.raises(ValueError, match="only available for searchlight"):
            region_result.to_nifti("Face-House", tmp_path / "test.nii.gz")

    def test_to_dataframe(self, region_result):
        """Test conversion to DataFrame."""
        pd = pytest.importorskip("pandas")

        df = region_result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "region" in df.columns
        assert "contrast" in df.columns
        assert "discriminability" in df.columns
        assert len(df) == 5 * 2 * 1  # 5 regions × 2 contrasts × 1 perm

    def test_to_dataframe_searchlight_error(self, searchlight_result):
        """Test error when converting searchlight to DataFrame."""
        pytest.importorskip("pandas")

        with pytest.raises(ValueError, match="only available for region"):
            searchlight_result.to_dataframe()

    def test_save_searchlight(self, searchlight_result, tmp_path):
        """Test saving searchlight results."""
        pytest.importorskip("nibabel")

        output_dir = tmp_path / "results"
        searchlight_result.save(output_dir)

        assert output_dir.exists()
        assert (output_dir / "discriminability_Face-House.nii.gz").exists()
        assert (output_dir / "discriminability_House-Cat.nii.gz").exists()
        assert (output_dir / "n_voxels.nii.gz").exists()

    def test_save_region(self, region_result, tmp_path):
        """Test saving region results."""
        pytest.importorskip("pandas")

        output_dir = tmp_path / "results"
        region_result.save(output_dir)

        assert output_dir.exists()
        assert (output_dir / "results.csv").exists()
