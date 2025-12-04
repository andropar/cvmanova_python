"""
Setup script for Haxby dataset integration test.

This script prepares the Haxby et al. (2001) dataset for cvManova testing,
replicating the MATLAB cvManovaTest pipeline using Python tools.

The MATLAB version uses SPM for:
1. Motion correction (realignment)
2. GLM model estimation

This Python version uses:
1. Simple detrending (no motion correction for simplicity)
2. Direct GLM estimation with numpy

Note: Results may differ slightly from MATLAB/SPM due to:
- Different preprocessing (no motion correction here)
- Different GLM estimation approach
- Floating point differences
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import signal
import sys


def parse_labels(labels_file: Path) -> list:
    """Parse the labels.txt file to extract stimulus timing."""
    with open(labels_file) as f:
        lines = f.readlines()

    labels = [line.strip().split()[0] for line in lines if line.strip()]
    return labels


def create_design_matrix(
    labels: list,
    n_vols_per_run: int = 121,
    n_runs: int = 12,
    tr: float = 2.5,
    conditions: list = None,
) -> tuple:
    """
    Create design matrix from volume labels.

    Returns per-session design matrices like SPM does.
    """
    if conditions is None:
        conditions = [
            "face", "house", "cat", "bottle",
            "scissors", "shoe", "chair", "scrambledpix"
        ]

    n_conditions = len(conditions)
    total_vols = n_vols_per_run * n_runs

    # Reshape labels into runs
    labels_array = np.array(labels[:total_vols]).reshape(n_runs, n_vols_per_run)

    # Create HRF (simplified canonical HRF)
    def hrf(t):
        """Simplified canonical HRF."""
        from scipy.stats import gamma
        # Parameters for canonical HRF
        a1, b1 = 6, 1  # response
        a2, b2 = 16, 1  # undershoot
        c = 1/6  # ratio of response to undershoot

        h = gamma.pdf(t, a1, scale=b1) - c * gamma.pdf(t, a2, scale=b2)
        return h / h.max() if h.max() > 0 else h

    # Time points for HRF (up to 32 seconds)
    t_hrf = np.arange(0, 32, tr)
    hrf_kernel = hrf(t_hrf)

    Xs = []
    for run in range(n_runs):
        run_labels = labels_array[run]

        # Create stimulus regressors
        X_run = np.zeros((n_vols_per_run, n_conditions + 1))

        for ci, cond in enumerate(conditions):
            # Binary indicator for this condition
            stim = (run_labels == cond).astype(float)
            # Convolve with HRF
            convolved = np.convolve(stim, hrf_kernel)[:n_vols_per_run]
            X_run[:, ci] = convolved

        # Add constant (intercept)
        X_run[:, -1] = 1

        Xs.append(X_run)

    return Xs, conditions


def load_and_preprocess_data(
    bold_file: Path,
    mask_file: Path = None,
    n_vols_per_run: int = 121,
    n_runs: int = 12,
) -> tuple:
    """
    Load BOLD data and apply basic preprocessing.

    Returns:
        Ys: list of session data matrices
        mask: 3D boolean mask
        affine: affine matrix
    """
    print(f"Loading BOLD data from {bold_file}")
    bold_img = nib.load(bold_file)
    bold_data = bold_img.get_fdata()
    affine = bold_img.affine

    print(f"  Shape: {bold_data.shape}")

    # Create or load mask
    if mask_file is not None and mask_file.exists():
        mask_img = nib.load(mask_file)
        mask = mask_img.get_fdata() > 0
    else:
        # Create mask from data (non-zero voxels)
        mask = np.std(bold_data, axis=3) > 0

    n_mask_voxels = np.sum(mask)
    print(f"  Mask voxels: {n_mask_voxels}")

    # Extract masked data
    mask_flat = mask.ravel(order='F')
    total_vols = n_vols_per_run * n_runs

    # Reshape to 2D (voxels x time)
    data_2d = bold_data.reshape(-1, bold_data.shape[3], order='F')
    data_masked = data_2d[mask_flat, :total_vols].T  # time x voxels

    # Split into sessions/runs
    Ys = []
    for run in range(n_runs):
        start = run * n_vols_per_run
        end = start + n_vols_per_run
        Y_run = data_masked[start:end, :].copy()

        # High-pass filter (detrending)
        # Remove linear trend and mean
        Y_run = signal.detrend(Y_run, axis=0, type='linear')

        Ys.append(Y_run)

    return Ys, mask, affine


def estimate_glm_and_prepare_data(
    Ys: list,
    Xs: list,
) -> tuple:
    """
    Estimate GLM and prepare whitened/filtered data.

    This mimics what SPM does: estimates GLM, computes residuals,
    and returns whitened data suitable for cvManova.

    In the original SPM workflow:
    1. Design matrix is filtered and whitened
    2. Data is filtered and whitened
    3. Beta estimates and residuals are computed

    For simplicity, we:
    1. Apply whitening based on AR(1) model of residuals
    2. Return the preprocessed Ys and Xs
    """
    n_sessions = len(Ys)

    # First pass: estimate betas and residuals to get AR coefficient
    residuals_all = []
    for si in range(n_sessions):
        Y = Ys[si]
        X = Xs[si]
        # OLS estimate
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        residuals = Y - X @ beta
        residuals_all.append(residuals)

    # Estimate AR(1) coefficient from residuals
    # (simplified: use median across voxels and sessions)
    ar_coeffs = []
    for res in residuals_all:
        for v in range(min(100, res.shape[1])):  # Sample voxels
            r = res[:, v]
            if np.std(r) > 0:
                ar = np.corrcoef(r[:-1], r[1:])[0, 1]
                if not np.isnan(ar):
                    ar_coeffs.append(ar)

    rho = np.median(ar_coeffs) if ar_coeffs else 0.0
    print(f"  Estimated AR(1) coefficient: {rho:.3f}")

    # Create whitening matrix based on AR(1)
    # W is approximately (I - rho * L) where L is lag operator
    Ys_whitened = []
    Xs_whitened = []

    for si in range(n_sessions):
        Y = Ys[si]
        X = Xs[si]

        # Simple AR(1) whitening: y_w[t] = y[t] - rho * y[t-1]
        Y_w = Y.copy()
        X_w = X.copy()

        Y_w[1:] = Y[1:] - rho * Y[:-1]
        X_w[1:] = X[1:] - rho * X[:-1]

        # Scale first observation
        Y_w[0] = Y[0] * np.sqrt(1 - rho**2)
        X_w[0] = X[0] * np.sqrt(1 - rho**2)

        Ys_whitened.append(Y_w)
        Xs_whitened.append(X_w)

    return Ys_whitened, Xs_whitened


def compute_degrees_of_freedom(Xs: list, n_filter_params: int = 0) -> np.ndarray:
    """Compute residual degrees of freedom for each session."""
    fE = []
    for X in Xs:
        n_scans = X.shape[0]
        n_params = np.linalg.matrix_rank(X)
        fE.append(n_scans - n_params - n_filter_params)
    return np.array(fE)


def setup_haxby_data(data_dir: Path) -> dict:
    """
    Full setup of Haxby data for cvManova testing.

    Returns dict with all data needed for analysis.
    """
    data_dir = Path(data_dir)

    # File paths
    bold_file = data_dir / "bold.nii"
    labels_file = data_dir / "labels.txt"
    mask_files = [
        data_dir / "mask4_vt.nii",
        data_dir / "mask8_face_vt.nii",
        data_dir / "mask8_house_vt.nii",
    ]

    # Parameters (matching MATLAB test)
    TR = 2.5
    n_runs = 12
    n_vols_per_run = 121
    conditions = [
        "face", "house", "cat", "bottle",
        "scissors", "shoe", "chair", "scrambledpix"
    ]

    print("Setting up Haxby dataset for cvManova")
    print("=" * 50)

    # Parse labels
    print("\nParsing labels...")
    labels = parse_labels(labels_file)
    print(f"  Total volumes: {len(labels)}")

    # Create design matrices
    print("\nCreating design matrices...")
    Xs, _ = create_design_matrix(
        labels, n_vols_per_run, n_runs, TR, conditions
    )
    print(f"  Sessions: {len(Xs)}")
    print(f"  Design shape per session: {Xs[0].shape}")

    # Load full brain mask for analysis
    print("\nLoading and preprocessing data...")
    # Use the VT mask as the analysis mask (intersection of brain + regions)
    analysis_mask_file = mask_files[0]  # mask4_vt.nii
    Ys, mask, affine = load_and_preprocess_data(
        bold_file, analysis_mask_file, n_vols_per_run, n_runs
    )

    # Apply whitening
    print("\nEstimating GLM and whitening...")
    Ys, Xs = estimate_glm_and_prepare_data(Ys, Xs)

    # Compute degrees of freedom
    fE = compute_degrees_of_freedom(Xs)
    print(f"  Degrees of freedom per session: {fE}")
    print(f"  Total residual df: {np.sum(fE)}")

    # Load region masks and compute indices
    print("\nLoading region masks...")
    mask_flat = mask.ravel(order='F')
    region_indices = []

    for i, mask_file in enumerate(mask_files):
        if mask_file.exists():
            region_img = nib.load(mask_file)
            region_mask = region_img.get_fdata() > 0
            region_flat = region_mask.ravel(order='F')

            # Indices within the analysis mask
            region_in_mask = region_flat[mask_flat]
            indices = np.where(region_in_mask)[0]
            region_indices.append(indices)
            print(f"  Region {i+1} ({mask_file.name}): {len(indices)} voxels")

    # Set up contrasts (matching MATLAB test)
    print("\nSetting up contrasts...")
    Cs = []

    # Contrast 1: Main effect of stimulus (all 8 conditions)
    C1 = np.array([
        [1, -1, 0, 0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1, -1],
    ]).T
    Cs.append(C1)
    print(f"  Contrast 1: Main effect ({C1.shape})")

    # Contrast 2: Category within object
    C2 = np.array([
        [0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 0],
    ]).T
    Cs.append(C2)
    print(f"  Contrast 2: Category within object ({C2.shape})")

    print("\n" + "=" * 50)
    print("Setup complete!")

    return {
        'Ys': Ys,
        'Xs': Xs,
        'mask': mask,
        'affine': affine,
        'fE': fE,
        'Cs': Cs,
        'region_indices': region_indices,
        'conditions': conditions,
    }


if __name__ == "__main__":
    # Default data directory
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path("/tmp/cvmanova_test/subj1")

    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        print("Please download the Haxby dataset first")
        sys.exit(1)

    data = setup_haxby_data(data_dir)

    # Run cvManova region analysis
    print("\n" + "=" * 50)
    print("Running cvManova region analysis...")
    print("=" * 50)

    from cvmanova import cv_manova_region

    D, p = cv_manova_region(
        data['Ys'],
        data['Xs'],
        data['Cs'],
        data['fE'],
        data['region_indices'],
        permute=False,
        lambda_=0.0,
    )

    print("\nResults:")
    print("-" * 40)
    for ri in range(D.shape[2]):
        for ci in range(D.shape[0]):
            print(f"  Region {ri+1}, Contrast {ci+1}: D = {D[ci, 0, ri]:.6f}")

    print("\nExpected values (from MATLAB with SPM12):")
    print("-" * 40)
    expected = [
        (1, 1, 5.443427),
        (1, 2, 1.021870),
        (2, 1, 0.314915),
        (2, 2, 0.021717),
        (3, 1, 1.711423),
        (3, 2, 0.241187),
    ]
    for ri, ci, val in expected:
        print(f"  Region {ri}, Contrast {ci}: D = {val:.6f}")

    print("\nNote: Values may differ due to preprocessing differences")
    print("(Python uses simple detrending, MATLAB uses SPM motion correction)")
