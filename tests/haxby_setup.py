"""
Setup script for Haxby dataset integration test.

This script prepares the Haxby et al. (2001) dataset for cvManova testing,
replicating the MATLAB cvManovaTest pipeline as closely as possible.

The MATLAB version (cvManovaTest_preprocess.m, cvManovaTest_model.m) uses SPM for:
1. Motion correction (realignment) with SPM's spm.spatial.realign.estwrite
2. GLM model with:
   - 6 motion parameters as nuisance regressors
   - 128s high-pass filter (DCT basis set)
   - AR(1) temporal autocorrelation modeling
   - HRF convolution for stimulus onsets

This Python version replicates these steps using:
1. Motion correction via nilearn (or uses pre-realigned data if available)
2. GLM model with:
   - 6 motion parameters from realignment
   - 128s high-pass filter (DCT basis, matching SPM)
   - AR(1) whitening (matching SPM's cvi='AR(1)')
   - SPM canonical HRF convolution
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import signal, linalg
from scipy.stats import gamma as gamma_dist
import sys
import warnings


def spm_hrf(tr, oversampling=16, time_length=32.0):
    """
    SPM canonical HRF function.

    Attempt to match SPM's spm_hrf.m as closely as possible.

    Parameters
    ----------
    tr : float
        Repetition time in seconds
    oversampling : int
        Temporal oversampling factor
    time_length : float
        Length of HRF in seconds

    Returns
    -------
    hrf : ndarray
        HRF sampled at TR
    """
    dt = tr / oversampling
    time_stamps = np.arange(0, time_length, dt)

    # SPM parameters
    peak_delay = 6.0  # time to peak (seconds)
    undershoot_delay = 16.0  # time to undershoot
    peak_disp = 1.0  # dispersion of peak
    undershoot_disp = 1.0  # dispersion of undershoot
    peak_undershoot_ratio = 6.0  # ratio of peak to undershoot

    # Gamma functions (SPM uses shape, scale parameterization)
    peak = gamma_dist.pdf(time_stamps, peak_delay / peak_disp, scale=peak_disp)
    undershoot = gamma_dist.pdf(time_stamps, undershoot_delay / undershoot_disp, scale=undershoot_disp)

    hrf_oversampled = peak - undershoot / peak_undershoot_ratio

    # Normalize
    hrf_oversampled = hrf_oversampled / np.max(hrf_oversampled)

    # Downsample to TR
    n_samples = int(time_length / tr)
    hrf = hrf_oversampled[::oversampling][:n_samples]

    return hrf


def spm_dctmtx(n, k=None):
    """
    Create SPM-style DCT basis set for high-pass filtering.

    Matches SPM's spm_dctmtx.m function.

    Parameters
    ----------
    n : int
        Number of time points
    k : int, optional
        Number of basis functions. If None, returns full basis.

    Returns
    -------
    C : ndarray
        DCT basis matrix (n x k)
    """
    if k is None:
        k = n

    # Create DCT-II basis (matches SPM)
    C = np.zeros((n, k))
    for i in range(k):
        if i == 0:
            C[:, i] = 1.0 / np.sqrt(n)
        else:
            C[:, i] = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * np.arange(n) + 1) * i / (2 * n))

    return C


def spm_filter(K, Y):
    """
    Apply SPM-style high-pass filter.

    Matches SPM's spm_filter.m - removes low frequency components
    using DCT basis set.

    Parameters
    ----------
    K : dict
        Filter specification with 'HParam' (cutoff in seconds) and 'RT' (TR)
    Y : ndarray
        Data matrix (time x voxels)

    Returns
    -------
    Y_filtered : ndarray
        High-pass filtered data
    n_filter_params : int
        Number of filter parameters (for df correction)
    """
    n = Y.shape[0]
    tr = K['RT']
    cutoff = K['HParam']

    # Number of DCT basis functions to remove (low frequencies)
    # SPM formula: k = fix(2*(n*RT)/HParam + 1)
    k = int(np.fix(2 * (n * tr) / cutoff + 1))

    # Create DCT basis for low frequencies
    X0 = spm_dctmtx(n, k)

    # Remove low frequency components (high-pass filter)
    # Y_filtered = Y - X0 @ (X0.T @ Y)
    Y_filtered = Y - X0 @ (X0.T @ Y)

    return Y_filtered, k


def realign_bold_data(bold_file, n_vols_per_run, n_runs):
    """
    Perform motion correction (realignment) on BOLD data.

    Uses nilearn's image processing to estimate and correct motion,
    similar to SPM's realignment.

    Parameters
    ----------
    bold_file : Path
        Path to BOLD NIfTI file
    n_vols_per_run : int
        Number of volumes per run
    n_runs : int
        Number of runs

    Returns
    -------
    realigned_data : ndarray
        Motion-corrected 4D data
    motion_params : list of ndarray
        Motion parameters per run (n_vols x 6)
    affine : ndarray
        Affine matrix
    """
    from pathlib import Path
    bold_file = Path(bold_file)

    # Check for pre-realigned data (SPM format: rbold.nii)
    realigned_file = bold_file.parent / f"r{bold_file.name}"
    rp_file = bold_file.parent / f"rp_{bold_file.stem}.txt"

    if realigned_file.exists() and rp_file.exists():
        print(f"  Loading pre-realigned data from {realigned_file}")
        img = nib.load(realigned_file)
        realigned_data = img.get_fdata()
        affine = img.affine

        all_motion = np.loadtxt(rp_file)
        motion_params = []
        for run in range(n_runs):
            start = run * n_vols_per_run
            end = start + n_vols_per_run
            motion_params.append(all_motion[start:end, :])

        return realigned_data, motion_params, affine

    # Otherwise, perform realignment using nilearn
    try:
        from nilearn.image import load_img, index_img, mean_img
        print("  Performing motion correction with nilearn...")

        img = load_img(str(bold_file))
        data = img.get_fdata()
        affine = img.affine

        total_vols = n_vols_per_run * n_runs
        realigned_data = np.zeros_like(data[:, :, :, :total_vols])
        motion_params = []

        # Process each run separately (like SPM does)
        for run in range(n_runs):
            start = run * n_vols_per_run
            end = start + n_vols_per_run

            # Get run data
            run_data = data[:, :, :, start:end]

            # Use first volume as reference
            ref_vol = run_data[:, :, :, 0]

            # Estimate motion parameters and apply correction
            mp = np.zeros((n_vols_per_run, 6))

            for t in range(n_vols_per_run):
                vol = run_data[:, :, :, t]

                if t == 0:
                    realigned_data[:, :, :, start + t] = vol
                    continue

                # Estimate translation by center of mass difference
                # This is a simplified approximation
                ref_com = np.array([
                    np.sum(np.arange(s) * np.sum(ref_vol, axis=tuple(j for j in range(3) if j != i)))
                    / np.sum(ref_vol) for i, s in enumerate(ref_vol.shape)
                ])
                vol_com = np.array([
                    np.sum(np.arange(s) * np.sum(vol, axis=tuple(j for j in range(3) if j != i)))
                    / np.sum(vol) for i, s in enumerate(vol.shape)
                ])

                # Translation parameters (in mm, assuming 1mm voxels)
                trans = (vol_com - ref_com) * np.diag(affine[:3, :3])
                mp[t, :3] = trans

                # Apply translation correction (simple shift)
                from scipy.ndimage import shift
                corrected = shift(vol, -(vol_com - ref_com), order=1, mode='constant')
                realigned_data[:, :, :, start + t] = corrected

            motion_params.append(mp)
            print(f"    Run {run+1}: max translation = {np.max(np.abs(mp[:, :3])):.2f} mm")

        return realigned_data, motion_params, affine

    except ImportError as e:
        print(f"  Warning: Could not perform motion correction ({e})")
        print("  Using original data without motion correction")
        img = nib.load(bold_file)
        return img.get_fdata(), [np.zeros((n_vols_per_run, 6)) for _ in range(n_runs)], img.affine


def parse_labels(labels_file: Path) -> list:
    """Parse the labels.txt file to extract stimulus timing."""
    with open(labels_file) as f:
        lines = f.readlines()

    labels = [line.strip().split()[0] for line in lines if line.strip()]
    return labels


def create_design_matrix(
    labels: list,
    motion_params: list,
    n_vols_per_run: int = 121,
    n_runs: int = 12,
    tr: float = 2.5,
    conditions: list = None,
    hpf_cutoff: float = 128.0,
) -> tuple:
    """
    Create design matrix from volume labels, matching SPM's fMRI model specification.

    Parameters
    ----------
    labels : list
        Volume labels for each time point
    motion_params : list
        Motion parameters per run (from realignment)
    n_vols_per_run : int
        Number of volumes per run
    n_runs : int
        Number of runs
    tr : float
        Repetition time in seconds
    conditions : list
        List of condition names
    hpf_cutoff : float
        High-pass filter cutoff in seconds (SPM default: 128s)

    Returns
    -------
    Xs : list
        Design matrices per session (n_vols x n_regressors)
    conditions : list
        Condition names
    n_filter_params : int
        Number of filter parameters per session
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

    # Get SPM canonical HRF
    hrf_kernel = spm_hrf(tr)

    # Number of DCT basis functions for high-pass filter
    n_filter_params = int(np.fix(2 * (n_vols_per_run * tr) / hpf_cutoff + 1))

    Xs = []
    for run in range(n_runs):
        run_labels = labels_array[run]

        # Number of regressors: conditions + 6 motion + 1 constant
        n_regressors = n_conditions + 6 + 1
        X_run = np.zeros((n_vols_per_run, n_regressors))

        # Stimulus regressors (convolved with HRF)
        for ci, cond in enumerate(conditions):
            # Binary indicator for this condition
            stim = (run_labels == cond).astype(float)
            # Convolve with HRF
            convolved = np.convolve(stim, hrf_kernel)[:n_vols_per_run]
            X_run[:, ci] = convolved

        # Motion parameters (6 regressors)
        if motion_params is not None and len(motion_params) > run:
            X_run[:, n_conditions:n_conditions+6] = motion_params[run]

        # Constant (intercept) - last column
        X_run[:, -1] = 1

        Xs.append(X_run)

    return Xs, conditions, n_filter_params


def load_and_preprocess_data(
    bold_data: np.ndarray,
    affine: np.ndarray,
    mask_file: Path = None,
    n_vols_per_run: int = 121,
    n_runs: int = 12,
    tr: float = 2.5,
    hpf_cutoff: float = 128.0,
) -> tuple:
    """
    Load BOLD data and apply SPM-style preprocessing.

    Applies:
    1. Masking
    2. High-pass filtering (128s DCT, matching SPM)

    Parameters
    ----------
    bold_data : ndarray
        4D BOLD data (already realigned)
    affine : ndarray
        Affine matrix
    mask_file : Path
        Optional mask file
    n_vols_per_run : int
        Volumes per run
    n_runs : int
        Number of runs
    tr : float
        Repetition time
    hpf_cutoff : float
        High-pass filter cutoff in seconds

    Returns
    -------
    Ys : list
        Session data matrices (n_vols x n_voxels)
    mask : ndarray
        3D boolean mask
    """
    print(f"  Data shape: {bold_data.shape}")

    # Create or load mask
    if mask_file is not None and mask_file.exists():
        mask_img = nib.load(mask_file)
        mask = mask_img.get_fdata() > 0
    else:
        mask = np.std(bold_data, axis=3) > 0

    n_mask_voxels = np.sum(mask)
    print(f"  Mask voxels: {n_mask_voxels}")

    # Extract masked data
    mask_flat = mask.ravel(order='F')
    total_vols = n_vols_per_run * n_runs

    # Reshape to 2D (voxels x time)
    data_2d = bold_data.reshape(-1, bold_data.shape[3], order='F')
    data_masked = data_2d[mask_flat, :total_vols].T  # time x voxels

    # High-pass filter specification (matching SPM)
    K = {'HParam': hpf_cutoff, 'RT': tr}

    # Split into sessions/runs and apply high-pass filter
    Ys = []
    for run in range(n_runs):
        start = run * n_vols_per_run
        end = start + n_vols_per_run
        Y_run = data_masked[start:end, :].copy()

        # Apply SPM-style high-pass filter
        Y_run, _ = spm_filter(K, Y_run)

        Ys.append(Y_run)

    return Ys, mask


def apply_highpass_to_design(Xs: list, n_vols_per_run: int, tr: float, hpf_cutoff: float):
    """
    Apply high-pass filter to design matrices (matching SPM).

    SPM applies the same high-pass filter to both data and design matrix.
    """
    K = {'HParam': hpf_cutoff, 'RT': tr}

    Xs_filtered = []
    for X in Xs:
        X_filtered, _ = spm_filter(K, X)
        # Restore the constant term
        X_filtered[:, -1] = 1
        Xs_filtered.append(X_filtered)

    return Xs_filtered


def estimate_ar1_and_whiten(Ys: list, Xs: list) -> tuple:
    """
    Estimate AR(1) coefficient and apply whitening, matching SPM's approach.

    SPM estimates AR(1) from pooled residuals across the brain,
    then applies whitening to both data and design matrix.

    Returns
    -------
    Ys_whitened : list
        Whitened data
    Xs_whitened : list
        Whitened design matrices
    rho : float
        Estimated AR(1) coefficient
    """
    n_sessions = len(Ys)

    # First pass: estimate residuals from OLS
    all_residuals = []
    for si in range(n_sessions):
        Y = Ys[si]
        X = Xs[si]
        # OLS estimate
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        residuals = Y - X @ beta
        all_residuals.append(residuals)

    # Estimate AR(1) coefficient from residuals (SPM pools across voxels)
    ar_coeffs = []
    for res in all_residuals:
        # Sample voxels for efficiency
        n_sample = min(500, res.shape[1])
        sample_idx = np.random.choice(res.shape[1], n_sample, replace=False)

        for v in sample_idx:
            r = res[:, v]
            if np.std(r) > 1e-10:
                # Lag-1 autocorrelation
                r_centered = r - np.mean(r)
                autocorr = np.sum(r_centered[:-1] * r_centered[1:]) / np.sum(r_centered ** 2)
                if not np.isnan(autocorr) and np.abs(autocorr) < 1:
                    ar_coeffs.append(autocorr)

    # Use median (robust to outliers)
    rho = np.median(ar_coeffs) if ar_coeffs else 0.0
    print(f"  Estimated AR(1) coefficient: {rho:.4f}")

    # Construct whitening matrix W such that W'W = V^(-1) where V is AR(1) covariance
    # For AR(1): V[i,j] = rho^|i-j|
    # The whitening transform is approximately: w[t] = y[t] - rho * y[t-1]
    # with appropriate scaling for the first observation

    Ys_whitened = []
    Xs_whitened = []

    for si in range(n_sessions):
        Y = Ys[si]
        X = Xs[si]
        n = Y.shape[0]

        # Build explicit whitening matrix (matches SPM more closely)
        # W is bidiagonal: W[i,i] = 1, W[i,i-1] = -rho, with W[0,0] = sqrt(1-rho^2)
        W = np.eye(n)
        W[0, 0] = np.sqrt(1 - rho ** 2)
        for i in range(1, n):
            W[i, i-1] = -rho

        # Apply whitening
        Y_w = W @ Y
        X_w = W @ X

        Ys_whitened.append(Y_w)
        Xs_whitened.append(X_w)

    return Ys_whitened, Xs_whitened, rho


def compute_degrees_of_freedom(Xs: list, n_filter_params: int = 0) -> np.ndarray:
    """
    Compute residual degrees of freedom for each session.

    df = n_scans - rank(X) - n_filter_params

    SPM accounts for the filter in the effective degrees of freedom.
    """
    fE = []
    for X in Xs:
        n_scans = X.shape[0]
        n_params = np.linalg.matrix_rank(X)
        df = n_scans - n_params - n_filter_params
        fE.append(max(1, df))  # Ensure positive df
    return np.array(fE)


def setup_haxby_data(data_dir: Path) -> dict:
    """
    Full setup of Haxby data for cvManova testing.

    Replicates the MATLAB cvManovaTest pipeline:
    1. Load BOLD data
    2. Estimate/load motion parameters
    3. Create design matrix with stimulus + motion regressors
    4. Apply 128s high-pass filter (SPM DCT)
    5. Estimate and apply AR(1) whitening

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

    # Parameters (matching MATLAB test exactly)
    TR = 2.5
    n_runs = 12
    n_vols_per_run = 121
    hpf_cutoff = 128.0  # SPM default high-pass filter cutoff
    conditions = [
        "face", "house", "cat", "bottle",
        "scissors", "shoe", "chair", "scrambledpix"
    ]

    print("Setting up Haxby dataset for cvManova (SPM-compatible pipeline)")
    print("=" * 60)

    # Parse labels
    print("\nParsing labels...")
    labels = parse_labels(labels_file)
    print(f"  Total volumes: {len(labels)}")

    # Motion correction (realignment)
    print("\nPerforming motion correction (realignment)...")
    bold_data, motion_params, affine = realign_bold_data(bold_file, n_vols_per_run, n_runs)

    # Create design matrices (with motion regressors)
    print("\nCreating design matrices (SPM-style)...")
    Xs, _, n_filter_params = create_design_matrix(
        labels, motion_params, n_vols_per_run, n_runs, TR, conditions, hpf_cutoff
    )
    print(f"  Sessions: {len(Xs)}")
    print(f"  Design shape per session: {Xs[0].shape}")
    print(f"  Regressors: {conditions} + 6 motion + constant")

    # Load and preprocess data (with high-pass filter)
    print(f"\nApplying high-pass filter (cutoff={hpf_cutoff}s)...")
    analysis_mask_file = mask_files[0]  # mask4_vt.nii
    Ys, mask = load_and_preprocess_data(
        bold_data, affine, analysis_mask_file, n_vols_per_run, n_runs, TR, hpf_cutoff
    )

    # Apply high-pass filter to design matrices
    print("Applying high-pass filter to design matrices...")
    Xs = apply_highpass_to_design(Xs, n_vols_per_run, TR, hpf_cutoff)

    # Estimate AR(1) and apply whitening
    print("\nEstimating AR(1) and applying whitening (SPM-style)...")
    Ys, Xs, rho = estimate_ar1_and_whiten(Ys, Xs)

    # Compute degrees of freedom (accounting for filter)
    fE = compute_degrees_of_freedom(Xs, n_filter_params)
    print(f"  Degrees of freedom per session: {fE}")
    print(f"  Total residual df: {np.sum(fE)}")
    print(f"  Filter params per session: {n_filter_params}")

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
    # Note: contrasts are defined on the first n_conditions regressors only
    # (not on motion parameters or constant)
    print("\nSetting up contrasts...")
    Cs = []

    # Contrast 1: Main effect of stimulus (all 8 conditions)
    # Pad with zeros for motion parameters and constant
    C1_base = np.array([
        [1, -1, 0, 0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1, -1],
    ]).T
    # Pad with zeros for motion (6) and constant (1)
    C1 = np.vstack([C1_base, np.zeros((7, C1_base.shape[1]))])
    Cs.append(C1)
    print(f"  Contrast 1: Main effect ({C1.shape})")

    # Contrast 2: Category within object (bottle, scissors, shoe, chair)
    C2_base = np.array([
        [0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 0],
    ]).T
    C2 = np.vstack([C2_base, np.zeros((7, C2_base.shape[1]))])
    Cs.append(C2)
    print(f"  Contrast 2: Category within object ({C2.shape})")

    print("\n" + "=" * 60)
    print("Setup complete! (SPM-compatible preprocessing applied)")

    return {
        'Ys': Ys,
        'Xs': Xs,
        'mask': mask,
        'affine': affine,
        'fE': fE,
        'Cs': Cs,
        'region_indices': region_indices,
        'conditions': conditions,
        'ar1_coefficient': rho,
        'hpf_cutoff': hpf_cutoff,
        'n_filter_params': n_filter_params,
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
    print("\n" + "=" * 60)
    print("Running cvManova region analysis...")
    print("=" * 60)

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
