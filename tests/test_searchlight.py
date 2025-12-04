"""Tests for searchlight analysis."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from numpy.testing import assert_allclose

from cvmanova.searchlight import run_searchlight, cv_manova_searchlight
from cvmanova.core import CvManovaCore


class TestRunSearchlight:
    """Tests for run_searchlight function."""

    def test_basic_searchlight(self):
        """Test basic searchlight operation."""
        # Create a simple 3D mask
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[3:7, 3:7, 3:7] = True  # 4x4x4 cube

        # Simple function that returns voxel count
        def count_voxels(mvi):
            if len(mvi) == 0:
                return np.array([0.0])
            return np.array([float(len(mvi))])

        res, p = run_searchlight(mask, sl_radius=1.0, fun=count_voxels)

        # Check output shapes
        assert res.shape[0] == np.prod(mask.shape)
        assert p.shape[0] == np.prod(mask.shape)

        # Check that only masked voxels have results
        mask_flat = mask.ravel(order="F")
        assert np.all(np.isnan(res[~mask_flat]))
        assert np.all(~np.isnan(res[mask_flat]))

    def test_searchlight_voxel_count(self):
        """Test that searchlight includes correct number of voxels."""
        mask = np.ones((11, 11, 11), dtype=bool)

        def count_voxels(mvi):
            if len(mvi) == 0:
                return np.array([0.0])
            return np.array([float(len(mvi))])

        res, p = run_searchlight(mask, sl_radius=1.0, fun=count_voxels)

        # Center voxel at (5,5,5) should have full searchlight of 7 voxels
        # (center + 6 face neighbors)
        center_idx = 5 + 5 * 11 + 5 * 11 * 11
        assert p[center_idx] == 7

    def test_searchlight_boundary_handling(self):
        """Test that boundary voxels have smaller searchlights."""
        mask = np.ones((5, 5, 5), dtype=bool)

        def count_voxels(mvi):
            if len(mvi) == 0:
                return np.array([0.0])
            return np.array([float(len(mvi))])

        res, p = run_searchlight(mask, sl_radius=1.0, fun=count_voxels)

        # Corner voxel should have smaller searchlight
        corner_idx = 0
        assert p[corner_idx] < 7  # Less than full searchlight

        # Center voxel should have full searchlight
        center_idx = 2 + 2 * 5 + 2 * 5 * 5
        assert p[center_idx] == 7

    def test_searchlight_with_mask_holes(self):
        """Test searchlight with non-contiguous mask."""
        mask = np.ones((7, 7, 7), dtype=bool)
        mask[3, 3, 3] = False  # Hole at center

        def count_voxels(mvi):
            if len(mvi) == 0:
                return np.array([0.0])
            return np.array([float(len(mvi))])

        res, p = run_searchlight(mask, sl_radius=1.0, fun=count_voxels)

        # Voxels adjacent to hole should have reduced count
        neighbor_idx = 2 + 3 * 7 + 3 * 7 * 7  # (2, 3, 3)
        assert p[neighbor_idx] == 6  # Missing one neighbor

    def test_checkpointing(self):
        """Test that checkpointing works."""
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True

        call_count = [0]

        def counting_fun(mvi):
            if len(mvi) == 0:
                return np.array([0.0])
            call_count[0] += 1
            return np.array([float(call_count[0])])

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "test_checkpoint"

            res, p = run_searchlight(
                mask, sl_radius=1.0, fun=counting_fun, checkpoint=str(checkpoint)
            )

            # Checkpoint file should be deleted after completion
            assert not checkpoint.with_suffix(".pkl").exists()

    def test_multi_output_function(self):
        """Test function returning multiple values."""
        mask = np.ones((5, 5, 5), dtype=bool)

        def multi_output(mvi):
            if len(mvi) == 0:
                return np.array([0.0, 0.0, 0.0])
            n = float(len(mvi))
            return np.array([n, n * 2, n * 3])

        res, p = run_searchlight(mask, sl_radius=1.0, fun=multi_output)

        assert res.shape == (125, 3)  # 5^3 voxels, 3 outputs


class TestCvManovaSearchlight:
    """Tests for cv_manova_searchlight function."""

    def test_basic_integration(self):
        """Test basic integration of searchlight with CvManovaCore."""
        np.random.seed(42)

        # Create small test case
        n_sessions = 4
        n_scans = 20
        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[2:6, 2:6, 2:6] = True
        n_voxels = np.sum(mask)

        # Create data
        Ys = []
        Xs = []
        for _ in range(n_sessions):
            X = np.zeros((n_scans, 3))
            X[:10, 0] = 1
            X[10:, 1] = 1
            X[:, 2] = 1
            Xs.append(X)

            Y = np.random.randn(n_scans, n_voxels)
            Ys.append(Y)

        fE = np.array([n_scans - 3] * n_sessions)
        Cs = [np.array([[1, -1, 0]]).T]

        D, p, n_contrasts, n_perms = cv_manova_searchlight(
            Ys, Xs, mask, sl_radius=1.0, Cs=Cs, fE=fE
        )

        assert D.shape == (512, 1, 1)  # 8^3 voxels, 1 contrast, 1 perm
        assert n_contrasts == 1
        assert n_perms == 1

    def test_insufficient_data_raises(self):
        """Test that insufficient data raises error."""
        n_sessions = 4
        n_scans = 10  # Too few
        mask = np.ones((5, 5, 5), dtype=bool)
        n_voxels = 125

        Ys = [np.random.randn(n_scans, n_voxels) for _ in range(n_sessions)]
        Xs = [np.ones((n_scans, 2)) for _ in range(n_sessions)]
        fE = np.array([n_scans - 2] * n_sessions)
        Cs = [np.array([[1, -1]]).T]

        # Searchlight of radius 2 has 33 voxels, but we only have
        # about 30 degrees of freedom
        with pytest.raises(ValueError, match="insufficient"):
            cv_manova_searchlight(
                Ys, Xs, mask, sl_radius=2.0, Cs=Cs, fE=fE
            )
