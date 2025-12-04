"""Tests for core cross-validated MANOVA computation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cvmanova.core import CvManovaCore


def create_test_data(
    n_sessions=4,
    n_scans_per_session=20,
    n_voxels=100,
    n_conditions=3,
    effect_size=1.0,
    noise_level=1.0,
    seed=42,
):
    """Create synthetic test data with known structure."""
    np.random.seed(seed)

    Ys = []
    Xs = []
    fE = []

    for si in range(n_sessions):
        # Design matrix: conditions + constant
        X = np.zeros((n_scans_per_session, n_conditions + 1))
        scans_per_cond = n_scans_per_session // n_conditions
        for ci in range(n_conditions):
            start = ci * scans_per_cond
            end = start + scans_per_cond
            X[start:end, ci] = 1
        X[:, -1] = 1  # Constant

        # True betas: different patterns for each condition
        true_betas = np.random.randn(n_conditions + 1, n_voxels) * effect_size

        # Data = design @ betas + noise
        Y = X @ true_betas + np.random.randn(n_scans_per_session, n_voxels) * noise_level

        Ys.append(Y)
        Xs.append(X)
        fE.append(n_scans_per_session - np.linalg.matrix_rank(X))

    return Ys, Xs, np.array(fE)


class TestCvManovaCore:
    """Tests for CvManovaCore class."""

    def test_initialization(self):
        """Test basic initialization."""
        Ys, Xs, fE = create_test_data()
        Cs = [np.array([[1, -1, 0]]).T]  # Simple contrast

        cmc = CvManovaCore(Ys, Xs, Cs, fE)

        assert cmc.m == 4
        assert cmc.n_contrasts == 1
        assert len(cmc.betas) == 4
        assert len(cmc.xis) == 4

    def test_compute_returns_correct_shape(self):
        """Test that compute returns correct shape."""
        Ys, Xs, fE = create_test_data(n_voxels=50)
        Cs = [
            np.array([[1, -1, 0]]).T,
            np.array([[0, 1, -1]]).T,
        ]

        cmc = CvManovaCore(Ys, Xs, Cs, fE, permute=False)
        vi = np.arange(10)  # First 10 voxels

        D = cmc.compute(vi)

        assert D.shape == (2,)  # 2 contrasts, 1 permutation

    def test_compute_with_permutations(self):
        """Test compute with permutations enabled."""
        Ys, Xs, fE = create_test_data(n_voxels=50)
        Cs = [np.array([[1, -1, 0]]).T]

        cmc = CvManovaCore(Ys, Xs, Cs, fE, permute=True)
        vi = np.arange(10)

        D = cmc.compute(vi)

        # Should have multiple permutation values
        assert D.shape[0] == cmc.n_perms

    def test_effect_detected(self):
        """Test that real effects are detected."""
        # Create data with strong effect
        Ys, Xs, fE = create_test_data(
            n_voxels=30, effect_size=3.0, noise_level=0.5
        )
        Cs = [np.array([[1, -1, 0]]).T]

        cmc = CvManovaCore(Ys, Xs, Cs, fE)
        vi = np.arange(20)

        D = cmc.compute(vi)

        # D should be positive for real effect
        assert D[0] > 0

    def test_no_effect_near_zero(self):
        """Test that null data gives D near zero."""
        np.random.seed(123)
        n_sessions = 4
        n_scans = 20
        n_voxels = 30

        # Pure noise data
        Ys = [np.random.randn(n_scans, n_voxels) for _ in range(n_sessions)]
        Xs = [np.ones((n_scans, 2)) for _ in range(n_sessions)]
        for X in Xs:
            X[:10, 0] = 0
            X[10:, 1] = 0
        fE = np.array([n_scans - 2] * n_sessions)

        Cs = [np.array([[1, -1]]).T]

        cmc = CvManovaCore(Ys, Xs, Cs, fE)
        vi = np.arange(20)

        D = cmc.compute(vi)

        # D should be close to zero for null data
        # (could be slightly negative due to bias correction)
        assert abs(D[0]) < 1.0

    def test_regularization(self):
        """Test that regularization works."""
        Ys, Xs, fE = create_test_data(n_voxels=50)
        Cs = [np.array([[1, -1, 0]]).T]

        # Without regularization
        cmc0 = CvManovaCore(Ys, Xs, Cs, fE, lambda_=0.0)
        D0 = cmc0.compute(np.arange(10))

        # With regularization
        cmc1 = CvManovaCore(Ys, Xs, Cs, fE, lambda_=0.5)
        D1 = cmc1.compute(np.arange(10))

        # Results should be different
        assert not np.allclose(D0, D1)

    def test_inconsistent_scans_raises(self):
        """Test that inconsistent scan counts raise error."""
        Ys = [np.random.randn(20, 50), np.random.randn(25, 50)]  # Different n_scans
        Xs = [np.random.randn(20, 3), np.random.randn(20, 3)]  # Inconsistent

        with pytest.raises(AssertionError):
            CvManovaCore(Ys, Xs, [np.array([[1, -1, 0]]).T], np.array([15, 15]))

    def test_inconsistent_voxels_raises(self):
        """Test that inconsistent voxel counts raise error."""
        Ys = [np.random.randn(20, 50), np.random.randn(20, 60)]  # Different n_voxels
        Xs = [np.random.randn(20, 3), np.random.randn(20, 3)]

        with pytest.raises(AssertionError):
            CvManovaCore(Ys, Xs, [np.array([[1, -1, 0]]).T], np.array([15, 15]))

    def test_inestimable_contrast_raises(self):
        """Test that inestimable contrasts raise error."""
        Ys = [np.random.randn(20, 50) for _ in range(4)]
        # Rank-deficient design
        Xs = [np.ones((20, 3)) for _ in range(4)]
        fE = np.array([17] * 4)

        # Contrast that can't be estimated
        Cs = [np.array([[1, -1, 0]]).T]

        with pytest.raises(AssertionError, match="not estimable"):
            CvManovaCore(Ys, Xs, Cs, fE)

    def test_contrast_trimming(self):
        """Test that trailing zeros in contrasts are trimmed."""
        Ys, Xs, fE = create_test_data()
        # Contrast with trailing zeros
        C = np.array([[1, -1, 0, 0, 0]]).T

        cmc = CvManovaCore(Ys, Xs, [C], fE)

        # Should be trimmed to first 2 rows
        assert cmc.Cs[0].shape[0] == 2
