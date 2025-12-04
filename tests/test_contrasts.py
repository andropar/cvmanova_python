"""Tests for contrast generation functions."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from cvmanova.contrasts import contrasts


class TestContrasts:
    """Tests for contrasts function."""

    def test_two_factor_2x2(self):
        """Test 2x2 factorial design."""
        c_matrix, c_name = contrasts([2, 2])

        # Should have 3 contrasts: A, B, A×B
        assert len(c_matrix) == 3
        assert len(c_name) == 3

        # Check names
        assert c_name[0] == "A"
        assert c_name[1] == "B"
        assert c_name[2] == "A×B"

        # Main effect A: compares levels of first factor
        assert c_matrix[0].shape == (4, 1)

        # Main effect B: compares levels of second factor
        assert c_matrix[1].shape == (4, 1)

        # Interaction: 1 column (2-1) × (2-1)
        assert c_matrix[2].shape == (4, 1)

    def test_two_factor_2x3(self):
        """Test 2x3 factorial design."""
        c_matrix, c_name = contrasts([2, 3])

        assert len(c_matrix) == 3

        # Main effect A: 1 column (2-1)
        assert c_matrix[0].shape == (6, 1)

        # Main effect B: 2 columns (3-1)
        assert c_matrix[1].shape == (6, 2)

        # Interaction: 2 columns (2-1) × (3-1)
        assert c_matrix[2].shape == (6, 2)

    def test_three_factor(self):
        """Test 3-factor design."""
        c_matrix, c_name = contrasts([2, 2, 2])

        # Should have 7 contrasts: A, B, C, A×B, A×C, B×C, A×B×C
        assert len(c_matrix) == 7
        assert len(c_name) == 7

        # Check that names include all expected effects
        expected_names = {"A", "B", "C", "A×B", "A×C", "B×C", "A×B×C"}
        assert set(c_name) == expected_names

    def test_custom_names(self):
        """Test with custom factor names."""
        c_matrix, c_name = contrasts([2, 3], ["Face", "House"])

        assert c_name[0] == "Face"
        assert c_name[1] == "House"
        assert c_name[2] == "Face×House"

    def test_single_factor(self):
        """Test single factor design."""
        c_matrix, c_name = contrasts([3])

        assert len(c_matrix) == 1
        assert c_name[0] == "A"
        assert c_matrix[0].shape == (3, 2)  # 3-1 = 2 columns

    def test_contrast_sum_property(self):
        """Contrasts should sum to zero across conditions."""
        c_matrix, _ = contrasts([2, 2])

        for c in c_matrix:
            # Each column should sum to zero
            for col in range(c.shape[1]):
                assert_allclose(np.sum(c[:, col]), 0, atol=1e-10)

    def test_kronecker_structure(self):
        """Verify Kronecker product structure for interactions."""
        c_matrix, _ = contrasts([2, 2])

        # For 2x2: difference coding using -diff(eye(n)).T
        # Main effect A should be kron([-1;1], [1;1]) = [[1],[1],[-1],[-1]]
        expected_A = np.kron(np.array([[1], [-1]]), np.array([[1], [1]]))
        assert_allclose(c_matrix[0], expected_A)
