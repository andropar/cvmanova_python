"""
Tests for data structures (SessionData and DesignMatrix).
"""

import numpy as np
import pytest
from cvmanova.data import SessionData, DesignMatrix


def create_mask(shape, n_voxels):
    """Helper to create a mask with exactly n_voxels True values."""
    mask = np.zeros(shape, dtype=bool)
    flat = mask.ravel(order="F")
    flat[:n_voxels] = True
    return flat.reshape(shape, order="F")


class TestSessionData:
    """Tests for SessionData class."""

    def test_basic_creation(self):
        """Test basic creation with valid inputs."""
        sessions = [
            np.random.randn(100, 1000),
            np.random.randn(100, 1000),
            np.random.randn(100, 1000),
        ]
        mask = create_mask((10, 10, 10), 1000)

        affine = np.eye(4)
        dof = np.array([90, 90, 90])

        data = SessionData(
            sessions=sessions,
            mask=mask,
            affine=affine,
            degrees_of_freedom=dof
        )

        assert data.n_sessions == 3
        assert data.n_voxels == 1000
        assert np.array_equal(data.n_scans, [100, 100, 100])
        assert data.total_scans == 300

    def test_session_ids(self):
        """Test with session IDs."""
        sessions = [np.random.randn(100, 500), np.random.randn(100, 500)]
        mask = create_mask((10, 10, 5), 500)
        affine = np.eye(4)
        dof = np.array([90, 90])

        data = SessionData(
            sessions=sessions,
            mask=mask,
            affine=affine,
            degrees_of_freedom=dof,
            session_ids=["run1", "run2"]
        )

        assert data.session_ids == ["run1", "run2"]

    def test_inconsistent_voxels(self):
        """Test error on inconsistent voxel counts."""
        sessions = [
            np.random.randn(100, 1000),
            np.random.randn(100, 900),  # Different voxel count
        ]
        mask = create_mask((10, 10, 10), 1000)
        affine = np.eye(4)
        dof = np.array([90, 90])

        with pytest.raises(ValueError, match="Inconsistent number of voxels"):
            SessionData(
                sessions=sessions,
                mask=mask,
                affine=affine,
                degrees_of_freedom=dof
            )

    def test_mask_not_3d(self):
        """Test error on non-3D mask."""
        sessions = [np.random.randn(100, 1000)]
        mask = np.zeros((10, 100), dtype=bool)  # 2D mask
        affine = np.eye(4)
        dof = np.array([90])

        with pytest.raises(ValueError, match="mask must be 3D"):
            SessionData(
                sessions=sessions,
                mask=mask,
                affine=affine,
                degrees_of_freedom=dof
            )

    def test_mask_not_boolean(self):
        """Test error on non-boolean mask."""
        sessions = [np.random.randn(100, 1000)]
        mask = np.ones((10, 10, 10), dtype=int)  # Not boolean
        affine = np.eye(4)
        dof = np.array([90])

        with pytest.raises(ValueError, match="mask must be boolean"):
            SessionData(
                sessions=sessions,
                mask=mask,
                affine=affine,
                degrees_of_freedom=dof
            )

    def test_mask_voxel_mismatch(self):
        """Test error when mask voxels don't match data voxels."""
        sessions = [np.random.randn(100, 1000)]
        mask = create_mask((10, 10, 10), 500)  # Only 500 voxels
        affine = np.eye(4)
        dof = np.array([90])

        with pytest.raises(ValueError, match="mask has .* voxels but data has"):
            SessionData(
                sessions=sessions,
                mask=mask,
                affine=affine,
                degrees_of_freedom=dof
            )

    def test_invalid_affine(self):
        """Test error on invalid affine matrix."""
        sessions = [np.random.randn(100, 1000)]
        mask = create_mask((10, 10, 10), 1000)
        affine = np.eye(3)  # Wrong shape
        dof = np.array([90])

        with pytest.raises(ValueError, match="affine must be 4x4"):
            SessionData(
                sessions=sessions,
                mask=mask,
                affine=affine,
                degrees_of_freedom=dof
            )

    def test_dof_length_mismatch(self):
        """Test error when DOF length doesn't match sessions."""
        sessions = [np.random.randn(100, 1000), np.random.randn(100, 1000)]
        mask = create_mask((10, 10, 10), 1000)
        affine = np.eye(4)
        dof = np.array([90])  # Only 1 DOF for 2 sessions

        with pytest.raises(ValueError, match="degrees_of_freedom has .* entries"):
            SessionData(
                sessions=sessions,
                mask=mask,
                affine=affine,
                degrees_of_freedom=dof
            )

    def test_negative_dof(self):
        """Test error on negative degrees of freedom."""
        sessions = [np.random.randn(100, 1000)]
        mask = create_mask((10, 10, 10), 1000)
        affine = np.eye(4)
        dof = np.array([-90])  # Negative

        with pytest.raises(ValueError, match="degrees_of_freedom must be positive"):
            SessionData(
                sessions=sessions,
                mask=mask,
                affine=affine,
                degrees_of_freedom=dof
            )

    def test_session_ids_length_mismatch(self):
        """Test error when session IDs length doesn't match sessions."""
        sessions = [np.random.randn(100, 1000), np.random.randn(100, 1000)]
        mask = create_mask((10, 10, 10), 1000)
        affine = np.eye(4)
        dof = np.array([90, 90])

        with pytest.raises(ValueError, match="session_ids has .* entries"):
            SessionData(
                sessions=sessions,
                mask=mask,
                affine=affine,
                degrees_of_freedom=dof,
                session_ids=["run1"]  # Only 1 ID for 2 sessions
            )

    def test_get_session(self):
        """Test getting individual session data."""
        Y1 = np.random.randn(100, 1000)
        Y2 = np.random.randn(100, 1000)
        sessions = [Y1, Y2]
        mask = create_mask((10, 10, 10), 1000)
        affine = np.eye(4)
        dof = np.array([90, 90])

        data = SessionData(
            sessions=sessions,
            mask=mask,
            affine=affine,
            degrees_of_freedom=dof
        )

        assert np.array_equal(data.get_session(0), Y1)
        assert np.array_equal(data.get_session(1), Y2)

    def test_to_list(self):
        """Test conversion to legacy list format."""
        sessions = [np.random.randn(100, 1000), np.random.randn(100, 1000)]
        mask = create_mask((10, 10, 10), 1000)
        affine = np.eye(4)
        dof = np.array([90, 90])

        data = SessionData(
            sessions=sessions,
            mask=mask,
            affine=affine,
            degrees_of_freedom=dof
        )

        assert data.to_list() == sessions

    def test_repr(self):
        """Test string representation."""
        sessions = [np.random.randn(100, 1000)]
        mask = create_mask((10, 10, 10), 1000)
        affine = np.eye(4)
        dof = np.array([90])

        data = SessionData(
            sessions=sessions,
            mask=mask,
            affine=affine,
            degrees_of_freedom=dof
        )

        repr_str = repr(data)
        assert "SessionData" in repr_str
        assert "n_sessions=1" in repr_str
        assert "n_voxels=1000" in repr_str


class TestDesignMatrix:
    """Tests for DesignMatrix class."""

    def test_basic_creation(self):
        """Test basic creation with valid inputs."""
        matrices = [
            np.random.randn(100, 10),
            np.random.randn(100, 10),
            np.random.randn(100, 10),
        ]

        design = DesignMatrix(matrices=matrices)

        assert design.n_sessions == 3
        assert np.array_equal(design.n_regressors, [10, 10, 10])
        assert np.array_equal(design.n_scans, [100, 100, 100])
        assert design.min_regressors == 10
        assert design.max_regressors == 10

    def test_with_regressor_names(self):
        """Test with regressor names."""
        matrices = [np.random.randn(100, 5), np.random.randn(100, 5)]
        regressor_names = ["Face", "House", "Cat", "Shoe", "Bottle"]

        design = DesignMatrix(
            matrices=matrices,
            regressor_names=regressor_names
        )

        assert design.regressor_names == regressor_names

    def test_with_session_ids(self):
        """Test with session IDs."""
        matrices = [np.random.randn(100, 5), np.random.randn(100, 5)]

        design = DesignMatrix(
            matrices=matrices,
            session_ids=["run1", "run2"]
        )

        assert design.session_ids == ["run1", "run2"]

    def test_non_2d_matrix(self):
        """Test error on non-2D design matrix."""
        matrices = [np.random.randn(100, 10, 5)]  # 3D

        with pytest.raises(ValueError, match="Design matrix .* must be 2D"):
            DesignMatrix(matrices=matrices)

    def test_regressor_names_too_many(self):
        """Test error when too many regressor names."""
        matrices = [np.random.randn(100, 5)]
        regressor_names = ["a", "b", "c", "d", "e", "f", "g"]  # 7 names for 5 regressors

        with pytest.raises(ValueError, match="regressor_names has .* entries"):
            DesignMatrix(
                matrices=matrices,
                regressor_names=regressor_names
            )

    def test_session_ids_length_mismatch(self):
        """Test error when session IDs length doesn't match sessions."""
        matrices = [np.random.randn(100, 5), np.random.randn(100, 5)]

        with pytest.raises(ValueError, match="session_ids has .* entries"):
            DesignMatrix(
                matrices=matrices,
                session_ids=["run1"]  # Only 1 ID for 2 sessions
            )

    def test_variable_regressors(self):
        """Test with different number of regressors per session."""
        matrices = [
            np.random.randn(100, 10),
            np.random.randn(100, 8),
            np.random.randn(100, 12),
        ]

        design = DesignMatrix(matrices=matrices)

        assert design.min_regressors == 8
        assert design.max_regressors == 12
        assert np.array_equal(design.n_regressors, [10, 8, 12])

    def test_get_session(self):
        """Test getting individual session design matrix."""
        X1 = np.random.randn(100, 10)
        X2 = np.random.randn(100, 10)
        matrices = [X1, X2]

        design = DesignMatrix(matrices=matrices)

        assert np.array_equal(design.get_session(0), X1)
        assert np.array_equal(design.get_session(1), X2)

    def test_to_list(self):
        """Test conversion to legacy list format."""
        matrices = [np.random.randn(100, 10), np.random.randn(100, 10)]

        design = DesignMatrix(matrices=matrices)

        assert design.to_list() == matrices

    def test_repr(self):
        """Test string representation."""
        matrices = [np.random.randn(100, 10)]

        design = DesignMatrix(matrices=matrices)

        repr_str = repr(design)
        assert "DesignMatrix" in repr_str
        assert "n_sessions=1" in repr_str
        assert "n_regressors=[10]" in repr_str
