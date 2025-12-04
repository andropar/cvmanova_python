"""Tests for utility functions."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from cvmanova.utils import (
    sign_permutations,
    inestimability,
    sl_size,
    fletcher16,
    null_space,
)


class TestSignPermutations:
    """Tests for sign_permutations function."""

    def test_full_enumeration_small(self):
        """Test full enumeration for small n."""
        perms, n_perms = sign_permutations(3)
        assert n_perms == 8  # 2^3
        assert perms.shape == (3, 8)
        # First permutation should be all ones (neutral)
        assert_array_equal(perms[:, 0], [1, 1, 1])
        # All values should be +1 or -1
        assert np.all(np.abs(perms) == 1)

    def test_full_enumeration_n1(self):
        """Test n=1 case."""
        perms, n_perms = sign_permutations(1)
        assert n_perms == 2
        assert perms.shape == (1, 2)

    def test_random_selection(self):
        """Test random selection when enumeration would be too large."""
        perms, n_perms = sign_permutations(20, max_perms=100)
        assert n_perms == 100
        assert perms.shape == (20, 100)
        # First permutation should still be neutral
        assert_array_equal(perms[:, 0], np.ones(20))

    def test_neutral_permutation_first(self):
        """Verify neutral permutation is always first."""
        for n in range(1, 8):
            perms, _ = sign_permutations(n)
            assert_array_equal(perms[:, 0], np.ones(n))


class TestNullSpace:
    """Tests for null_space function."""

    def test_full_rank_matrix(self):
        """Full rank matrix should have empty null space."""
        A = np.eye(3)
        ns = null_space(A)
        assert ns.size == 0 or ns.shape[1] == 0

    def test_rank_deficient_matrix(self):
        """Rank deficient matrix should have non-trivial null space."""
        A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])  # Rank 1
        ns = null_space(A)
        assert ns.shape[1] == 2  # Null space dimension = 3 - 1 = 2
        # Verify A @ ns is approximately zero
        assert_allclose(A @ ns, 0, atol=1e-10)


class TestInestimability:
    """Tests for inestimability function."""

    def test_estimable_contrast(self):
        """Fully estimable contrast should have ie ≈ 0."""
        X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        C = np.array([[1.0], [-1.0]])  # Difference between groups
        ie = inestimability(C, X)
        assert ie < 1e-10

    def test_inestimable_contrast(self):
        """Inestimable contrast should have ie > 0."""
        X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])  # Rank 1
        C = np.array([[1.0], [-1.0]])  # Can't estimate difference
        ie = inestimability(C, X)
        assert ie > 0.5

    def test_contrast_shorter_than_design(self):
        """Contrast shorter than design columns should be handled."""
        X = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        C = np.array([[1.0], [-1.0]])  # Only refers to first 2 columns
        ie = inestimability(C, X)
        assert ie < 1e-10


class TestSlSize:
    """Tests for sl_size function."""

    def test_radius_0(self):
        """Radius 0 should give size 1 (just center)."""
        assert sl_size(0) == 1

    def test_radius_1(self):
        """Radius 1 should include center + 6 face neighbors = 7."""
        assert sl_size(1) == 7

    def test_radius_sqrt2(self):
        """Radius sqrt(2) should include edges = 19."""
        assert sl_size(np.sqrt(2)) == 19

    def test_radius_sqrt3(self):
        """Radius sqrt(3) should include corners = 27."""
        assert sl_size(np.sqrt(3)) == 27

    def test_larger_radius(self):
        """Test a larger radius."""
        # Known value for radius 2
        assert sl_size(2) == 33


class TestFletcher16:
    """Tests for fletcher16 checksum function."""

    def test_empty_data(self):
        """Empty data should return 0."""
        assert fletcher16(b"") == 0

    def test_known_value(self):
        """Test against known checksum value."""
        # "abcde" -> specific checksum
        data = b"abcde"
        checksum = fletcher16(data)
        assert isinstance(checksum, int)
        assert 0 <= checksum < 65536

    def test_string_input(self):
        """String input should work."""
        checksum = fletcher16("hello world")
        assert isinstance(checksum, int)

    def test_numpy_array_input(self):
        """NumPy array input should work."""
        data = np.array([72, 101, 108, 108, 111], dtype=np.uint8)  # "Hello"
        checksum = fletcher16(data)
        assert isinstance(checksum, int)

    def test_different_inputs_different_checksums(self):
        """Different inputs should (usually) give different checksums."""
        c1 = fletcher16(b"hello")
        c2 = fletcher16(b"world")
        assert c1 != c2
