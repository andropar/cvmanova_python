"""
Tests for data loaders.
"""

import numpy as np
import pytest
from pathlib import Path
from cvmanova.loaders import SPMLoader, NiftiLoader, NilearnMaskerLoader
from cvmanova.data import SessionData, DesignMatrix


class TestSPMLoader:
    """Tests for SPMLoader."""

    def test_initialization_with_valid_path(self, tmp_path):
        """Test initialization with path containing SPM.mat."""
        # Create a dummy SPM.mat file
        spm_file = tmp_path / "SPM.mat"
        spm_file.touch()

        loader = SPMLoader(tmp_path)

        assert loader.spm_dir == tmp_path
        assert loader.whiten is True
        assert loader.high_pass_filter is True

    def test_initialization_with_string_path(self, tmp_path):
        """Test initialization with string path."""
        spm_file = tmp_path / "SPM.mat"
        spm_file.touch()

        loader = SPMLoader(str(tmp_path))

        assert isinstance(loader.spm_dir, Path)
        assert loader.spm_dir == tmp_path

    def test_initialization_missing_spm_file(self, tmp_path):
        """Test error when SPM.mat not found."""
        with pytest.raises(FileNotFoundError, match="SPM.mat not found"):
            SPMLoader(tmp_path)

    def test_custom_parameters(self, tmp_path):
        """Test initialization with custom parameters."""
        spm_file = tmp_path / "SPM.mat"
        spm_file.touch()

        loader = SPMLoader(
            tmp_path, whiten=False, high_pass_filter=False, tr=2.5
        )

        assert loader.whiten is False
        assert loader.high_pass_filter is False
        assert loader.tr == 2.5

    def test_repr(self, tmp_path):
        """Test string representation."""
        spm_file = tmp_path / "SPM.mat"
        spm_file.touch()

        loader = SPMLoader(tmp_path)
        repr_str = repr(loader)

        assert "SPMLoader" in repr_str
        assert str(tmp_path) in repr_str
        assert "whiten=True" in repr_str


class TestNiftiLoader:
    """Tests for NiftiLoader."""

    @pytest.fixture
    def mock_nifti_data(self, tmp_path, monkeypatch):
        """Create mock NIfTI data for testing."""
        import sys
        from unittest.mock import MagicMock

        # Create mock mask
        mask_data = np.zeros((5, 5, 5), dtype=bool)
        mask_data[1:4, 1:4, 1:4] = True  # 3x3x3 = 27 voxels

        mask_img = MagicMock()
        mask_img.dataobj = mask_data
        mask_img.affine = np.eye(4)

        # Create mock BOLD data (5x5x5x20 = 20 timepoints)
        bold_data = np.random.randn(5, 5, 5, 20)

        bold_img = MagicMock()
        bold_img.dataobj = bold_data
        bold_img.affine = np.eye(4)
        bold_img.shape = bold_data.shape

        def mock_load(filename):
            if "mask" in str(filename):
                return mask_img
            return bold_img

        # Mock nibabel module
        nib = MagicMock()
        nib.load = mock_load

        monkeypatch.setitem(sys.modules, "nibabel", nib)

        # Create dummy files
        mask_file = tmp_path / "mask.nii.gz"
        bold1_file = tmp_path / "run1.nii.gz"
        bold2_file = tmp_path / "run2.nii.gz"

        mask_file.touch()
        bold1_file.touch()
        bold2_file.touch()

        # Design matrices
        X1 = np.random.randn(20, 3)
        X2 = np.random.randn(20, 3)

        return {
            "mask_file": mask_file,
            "bold_files": [bold1_file, bold2_file],
            "design_matrices": [X1, X2],
            "bold_data": bold_data,
            "mask_data": mask_data,
        }

    def test_initialization(self, mock_nifti_data):
        """Test initialization with valid inputs."""
        loader = NiftiLoader(
            bold_files=mock_nifti_data["bold_files"],
            mask_file=mock_nifti_data["mask_file"],
            design_matrices=mock_nifti_data["design_matrices"],
        )

        assert len(loader.bold_files) == 2
        assert loader.preprocess is True

    def test_initialization_file_count_mismatch(self, mock_nifti_data):
        """Test error when file count doesn't match design matrices."""
        with pytest.raises(ValueError, match="Number of BOLD files"):
            NiftiLoader(
                bold_files=[mock_nifti_data["bold_files"][0]],
                mask_file=mock_nifti_data["mask_file"],
                design_matrices=mock_nifti_data["design_matrices"],  # 2 matrices
            )

    def test_initialization_missing_mask(self, tmp_path, mock_nifti_data):
        """Test error when mask file doesn't exist."""
        fake_mask = tmp_path / "nonexistent_mask.nii.gz"

        with pytest.raises(FileNotFoundError, match="Mask file not found"):
            NiftiLoader(
                bold_files=mock_nifti_data["bold_files"],
                mask_file=fake_mask,
                design_matrices=mock_nifti_data["design_matrices"],
            )

    def test_initialization_missing_bold(self, tmp_path, mock_nifti_data):
        """Test error when BOLD file doesn't exist."""
        fake_bold = tmp_path / "nonexistent.nii.gz"

        with pytest.raises(FileNotFoundError, match="BOLD file not found"):
            NiftiLoader(
                bold_files=[fake_bold],
                mask_file=mock_nifti_data["mask_file"],
                design_matrices=[mock_nifti_data["design_matrices"][0]],
            )

    def test_load_returns_correct_types(self, mock_nifti_data):
        """Test that load returns SessionData and DesignMatrix."""
        loader = NiftiLoader(
            bold_files=mock_nifti_data["bold_files"],
            mask_file=mock_nifti_data["mask_file"],
            design_matrices=mock_nifti_data["design_matrices"],
        )

        data, design = loader.load()

        assert isinstance(data, SessionData)
        assert isinstance(design, DesignMatrix)

    def test_load_correct_dimensions(self, mock_nifti_data):
        """Test that loaded data has correct dimensions."""
        loader = NiftiLoader(
            bold_files=mock_nifti_data["bold_files"],
            mask_file=mock_nifti_data["mask_file"],
            design_matrices=mock_nifti_data["design_matrices"],
        )

        data, design = loader.load()

        assert data.n_sessions == 2
        assert data.n_voxels == 27  # 3x3x3
        assert len(data.sessions[0]) == 20  # 20 timepoints

    def test_load_with_no_preprocessing(self, mock_nifti_data):
        """Test loading without preprocessing."""
        loader = NiftiLoader(
            bold_files=mock_nifti_data["bold_files"],
            mask_file=mock_nifti_data["mask_file"],
            design_matrices=mock_nifti_data["design_matrices"],
            preprocess=False,
        )

        data, design = loader.load()

        assert isinstance(data, SessionData)

    def test_repr(self, mock_nifti_data):
        """Test string representation."""
        loader = NiftiLoader(
            bold_files=mock_nifti_data["bold_files"],
            mask_file=mock_nifti_data["mask_file"],
            design_matrices=mock_nifti_data["design_matrices"],
        )

        repr_str = repr(loader)

        assert "NiftiLoader" in repr_str
        assert "n_sessions=2" in repr_str


class TestNilearnMaskerLoader:
    """Tests for NilearnMaskerLoader."""

    @pytest.fixture
    def mock_masker(self, monkeypatch):
        """Create a mock nilearn masker."""
        from unittest.mock import MagicMock

        masker = MagicMock()

        # Mock mask
        mask_data = np.zeros((5, 5, 5), dtype=bool)
        mask_data[1:4, 1:4, 1:4] = True

        mask_img = MagicMock()
        mask_img.dataobj = mask_data
        mask_img.affine = np.eye(4)

        masker.mask_img_ = mask_img

        # Mock transform to return (n_scans, n_voxels)
        def mock_transform(filename):
            return np.random.randn(20, 27)  # 20 timepoints, 27 voxels

        masker.fit_transform = mock_transform

        return masker

    def test_initialization(self, tmp_path, mock_masker):
        """Test initialization with masker."""
        bold1 = tmp_path / "run1.nii.gz"
        bold2 = tmp_path / "run2.nii.gz"
        bold1.touch()
        bold2.touch()

        X1 = np.random.randn(20, 3)
        X2 = np.random.randn(20, 3)

        loader = NilearnMaskerLoader(
            bold_files=[bold1, bold2], masker=mock_masker, design_matrices=[X1, X2]
        )

        assert len(loader.bold_files) == 2

    def test_initialization_file_count_mismatch(self, tmp_path, mock_masker):
        """Test error on file/design count mismatch."""
        bold1 = tmp_path / "run1.nii.gz"
        bold1.touch()

        X1 = np.random.randn(20, 3)
        X2 = np.random.randn(20, 3)

        with pytest.raises(ValueError, match="Number of BOLD files"):
            NilearnMaskerLoader(
                bold_files=[bold1], masker=mock_masker, design_matrices=[X1, X2]
            )

    def test_load_returns_correct_types(self, tmp_path, mock_masker, monkeypatch):
        """Test that load returns correct types."""
        import sys
        from unittest.mock import MagicMock

        # Mock nibabel
        nib = MagicMock()
        monkeypatch.setitem(sys.modules, "nibabel", nib)

        bold1 = tmp_path / "run1.nii.gz"
        bold2 = tmp_path / "run2.nii.gz"
        bold1.touch()
        bold2.touch()

        X1 = np.random.randn(20, 3)
        X2 = np.random.randn(20, 3)

        loader = NilearnMaskerLoader(
            bold_files=[bold1, bold2], masker=mock_masker, design_matrices=[X1, X2]
        )

        data, design = loader.load()

        assert isinstance(data, SessionData)
        assert isinstance(design, DesignMatrix)

    def test_repr(self, tmp_path, mock_masker):
        """Test string representation."""
        bold1 = tmp_path / "run1.nii.gz"
        bold1.touch()

        X1 = np.random.randn(20, 3)

        loader = NilearnMaskerLoader(
            bold_files=[bold1], masker=mock_masker, design_matrices=[X1]
        )

        repr_str = repr(loader)

        assert "NilearnMaskerLoader" in repr_str
        assert "n_sessions=1" in repr_str


class TestLoaderIntegration:
    """Integration tests across loaders."""

    def test_loaders_produce_compatible_output(self, tmp_path, monkeypatch):
        """Test that all loaders produce compatible SessionData/DesignMatrix."""
        import sys
        from unittest.mock import MagicMock

        mask_data = np.zeros((5, 5, 5), dtype=bool)
        mask_data[2, 2, 2] = True

        mask_img = MagicMock()
        mask_img.dataobj = mask_data
        mask_img.affine = np.eye(4)

        bold_data = np.random.randn(5, 5, 5, 10)

        bold_img = MagicMock()
        bold_img.dataobj = bold_data
        bold_img.affine = np.eye(4)
        bold_img.shape = bold_data.shape

        def mock_load(filename):
            if "mask" in str(filename):
                return mask_img
            return bold_img

        # Mock nibabel
        nib = MagicMock()
        nib.load = mock_load
        monkeypatch.setitem(sys.modules, "nibabel", nib)

        # Create files
        mask_file = tmp_path / "mask.nii.gz"
        bold_file = tmp_path / "run1.nii.gz"
        mask_file.touch()
        bold_file.touch()

        X = np.random.randn(10, 2)

        # Test NiftiLoader
        nifti_loader = NiftiLoader(
            bold_files=[bold_file], mask_file=mask_file, design_matrices=[X]
        )

        data_nifti, design_nifti = nifti_loader.load()

        assert isinstance(data_nifti, SessionData)
        assert isinstance(design_nifti, DesignMatrix)
        assert data_nifti.n_sessions == 1
