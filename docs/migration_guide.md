# Migration Guide: cvManova 3.x → 4.x

This guide helps you migrate from the procedural API (3.x) to the modern estimator-based API (4.x).

## Overview of Changes

cvManova 4.0 introduces a modern, scikit-learn-style API while maintaining backward compatibility with the 3.x procedural API. The new API offers:

- **Estimator classes** (`SearchlightCvManova`, `RegionCvManova`) following scikit-learn conventions
- **Data structures** (`SessionData`, `DesignMatrix`, `CvManovaResult`) for type safety
- **Configuration objects** (`SearchlightConfig`, `AnalysisConfig`, etc.) for clear parameter organization
- **Flexible data loaders** (`SPMLoader`, `NiftiLoader`, `NilearnMaskerLoader`) for various input formats
- **Rich result objects** with visualization and export methods

**Good news:** Your existing code will continue to work! The old API is still available.

---

## Quick Migration Examples

### Example 1: Basic Searchlight Analysis

#### Old API (3.x)
```python
from cvmanova import cv_manova_searchlight, load_data_spm, contrasts

# Load data
Ys, Xs, mask, affine, fE = load_data_spm('/path/to/spm', whiten=True, highpass=True)

# Generate contrasts
Cs, names = contrasts([2, 2], ['Face', 'House'])

# Run analysis
D, p, _, _ = cv_manova_searchlight(
    Ys=Ys,
    Xs=Xs,
    mask=mask,
    sl_radius=3.0,
    Cs=Cs,
    fE=fE,
    permute=True,
    lambda_=0.0,
    checkpoint=None
)

# Save results manually
from cvmanova import write_image
write_image(D[:, :, :, 0, 0], 'face_disc.nii.gz', affine)
```

#### New API (4.x)
```python
from cvmanova import SearchlightCvManova, SPMLoader, SearchlightConfig, ContrastSpec

# Load data with type-safe loader
loader = SPMLoader('/path/to/spm', whiten=True, high_pass_filter=True)
data, design = loader.load()

# Configure searchlight and contrasts
sl_config = SearchlightConfig(radius=3.0)
contrasts = ContrastSpec(factors=['Face', 'House'], levels=[2, 2])

# Run analysis with estimator
estimator = SearchlightCvManova(
    searchlight_config=sl_config,
    contrasts=contrasts,
    analysis_config=AnalysisConfig(permute=True, regularization=0.0)
)

result = estimator.fit_score(data, design)

# Save with one line
result.to_nifti('Face', 'face_disc.nii.gz')
```

**Benefits:**
- Clearer code organization
- Type-safe data structures
- Automatic result management
- Built-in visualization methods

---

### Example 2: Region-Based Analysis

#### Old API (3.x)
```python
from cvmanova import cv_manova_region, load_data_spm
import numpy as np
import nibabel as nib

# Load data
Ys, Xs, mask, affine, fE = load_data_spm('/path/to/spm')

# Load region masks manually
v1_img = nib.load('V1.nii.gz')
v1_mask = v1_img.get_fdata() > 0

# Convert 3D mask to indices manually
mask_flat = mask.ravel(order='F')
mask_indices = np.where(mask_flat)[0]
region_flat = v1_mask.ravel(order='F')
region_in_mask = region_flat[mask_indices]
region_idx = np.where(region_in_mask)[0]

# Contrasts
Cs = [np.array([[0, 1, 0]])]

# Run analysis
D, p = cv_manova_region(
    Ys=Ys,
    Xs=Xs,
    Cs=Cs,
    fE=fE,
    region_indices=[region_idx],
    permute=True,
    lambda_=0.0
)

# Process results manually
print(f"V1 discriminability: {D[0, 0, 0]:.4f}")
```

#### New API (4.x)
```python
from cvmanova import RegionCvManova, SPMLoader, RegionConfig, ContrastSpec
from pathlib import Path

# Load data
loader = SPMLoader('/path/to/spm')
data, design = loader.load()

# Configure regions (handles mask conversion automatically)
region_config = RegionConfig(
    regions=[Path('V1.nii.gz'), Path('V2.nii.gz')],
    region_names=['V1', 'V2'],
    min_voxels=10
)

# Contrasts (or manually: contrasts = [np.array([[0, 1, 0]])])
contrasts = ContrastSpec(factors=['Condition'], levels=[2], effects='main')

# Run analysis
estimator = RegionCvManova(
    region_config=region_config,
    contrasts=contrasts,
    analysis_config=AnalysisConfig(permute=True)
)

result = estimator.fit_score(data, design)

# Export to pandas DataFrame
df = result.to_dataframe()
print(df)

# Access specific region/contrast
D_v1 = result.get_region_contrast('V1', 'Condition')
print(f"V1 discriminability: {D_v1[0]:.4f}")
```

**Benefits:**
- Automatic mask-to-index conversion
- Named regions for clarity
- Easy DataFrame export
- Structured result access

---

### Example 3: Custom Contrasts

#### Old API (3.x)
```python
from cvmanova import cv_manova_searchlight, load_data_spm
import numpy as np

Ys, Xs, mask, affine, fE = load_data_spm('/path/to/spm')

# Define contrasts manually
C_faces = np.array([[0, 1, 0, 0]])
C_houses = np.array([[0, 0, 1, 0]])
C_interaction = np.array([[0, 0, 0, 1]])
Cs = [C_faces, C_houses, C_interaction]

D, p, _, _ = cv_manova_searchlight(
    Ys=Ys, Xs=Xs, mask=mask, sl_radius=3.0, Cs=Cs, fE=fE
)

# Results for each contrast at D[:,:,:,i,0]
```

#### New API (4.x)
```python
from cvmanova import SearchlightCvManova, SPMLoader
import numpy as np

loader = SPMLoader('/path/to/spm')
data, design = loader.load()

# Option A: Auto-generate from factorial design
contrasts = ContrastSpec(
    factors=['Face', 'House'],
    levels=[2, 2],
    effects='all'  # main effects + interactions
)

# Option B: Specify manually with names
contrasts = [
    np.array([[0, 1, 0, 0]]),
    np.array([[0, 0, 1, 0]]),
    np.array([[0, 0, 0, 1]]),
]
# Will be auto-named as contrast_1, contrast_2, contrast_3

estimator = SearchlightCvManova(contrasts=contrasts)
result = estimator.fit_score(data, design)

# Access by name or index
D_faces = result.get_contrast('Face')  # or result.get_contrast(0)
```

**Benefits:**
- Named contrasts
- Auto-generation from factorial designs
- Clear access methods

---

### Example 4: NIfTI Files Instead of SPM

#### Old API (3.x)
```python
from cvmanova import cv_manova_searchlight, read_vols_masked
import numpy as np
import nibabel as nib

# Load mask
mask_img = nib.load('mask.nii.gz')
mask = mask_img.get_fdata() > 0
affine = mask_img.affine

# Load BOLD data manually
bold_files = ['run1.nii.gz', 'run2.nii.gz']
Ys = []
for bf in bold_files:
    # ... manual voxel extraction ...
    pass

# Create design matrices manually
Xs = [np.random.randn(200, 3), np.random.randn(200, 3)]

# Compute DOF manually
fE = np.array([Y.shape[0] - X.shape[1] for Y, X in zip(Ys, Xs)])

Cs = [np.array([[0, 1, 0]])]

D, p, _, _ = cv_manova_searchlight(
    Ys=Ys, Xs=Xs, mask=mask, sl_radius=3.0, Cs=Cs, fE=fE
)
```

#### New API (4.x)
```python
from cvmanova import SearchlightCvManova, NiftiLoader
import numpy as np

# Design matrices
X1 = np.random.randn(200, 3)
X2 = np.random.randn(200, 3)

# Load with NiftiLoader (handles everything automatically)
loader = NiftiLoader(
    bold_files=['run1.nii.gz', 'run2.nii.gz'],
    mask_file='mask.nii.gz',
    design_matrices=[X1, X2],
    tr=2.0,
    preprocess=True  # Applies mean-centering
)

data, design = loader.load()

# Run analysis
estimator = SearchlightCvManova(contrasts=[np.array([[0, 1, 0]])])
result = estimator.fit_score(data, design)
```

**Benefits:**
- Automatic voxel extraction
- Automatic DOF computation
- Optional preprocessing
- Clean, declarative syntax

---

### Example 5: Parallelization

#### Old API (3.x)
```python
from cvmanova import run_searchlight, load_data_spm

Ys, Xs, mask, affine, fE = load_data_spm('/path/to/spm')

# Parallelization not built-in
# Would need to manually parallelize with joblib or multiprocessing
```

#### New API (4.x)
```python
from cvmanova import SearchlightCvManova, SPMLoader, SearchlightConfig

loader = SPMLoader('/path/to/spm')
data, design = loader.load()

# Built-in parallelization
sl_config = SearchlightConfig(
    radius=3.0,
    n_jobs=-1,  # Use all CPU cores
    show_progress=True,  # Progress bar
    checkpoint_dir='./checkpoints'  # Auto-save progress
)

estimator = SearchlightCvManova(searchlight_config=sl_config)
result = estimator.fit_score(data, design)
```

**Benefits:**
- Built-in parallelization with `n_jobs`
- Progress bars with `tqdm`
- Automatic checkpointing

---

## API Correspondence Table

| Old (3.x) | New (4.x) | Notes |
|-----------|-----------|-------|
| `load_data_spm()` | `SPMLoader` | Returns `SessionData` and `DesignMatrix` |
| `cv_manova_searchlight()` | `SearchlightCvManova` | Estimator with `fit()` and `score()` |
| `cv_manova_region()` | `RegionCvManova` | Estimator with automatic mask handling |
| `run_searchlight()` | `SearchlightCvManova` | Estimator is more flexible |
| `contrasts()` | `ContrastSpec` | Auto-generation from factorial design |
| `write_image()` | `result.to_nifti()` | Method on result object |
| Manual DataFrame creation | `result.to_dataframe()` | Built-in pandas export |
| Manual peak finding | `result.get_peaks()` | Built-in peak detection |

---

## Configuration Parameters

### SearchlightConfig

| Old Parameter (3.x) | New Parameter (4.x) |
|---------------------|---------------------|
| `sl_radius` | `SearchlightConfig(radius=...)` |
| `checkpoint` | `SearchlightConfig(checkpoint_dir=..., checkpoint_name=...)` |
| No equivalent | `SearchlightConfig(n_jobs=-1)` for parallelization |
| No equivalent | `SearchlightConfig(show_progress=True)` for progress bars |

### AnalysisConfig

| Old Parameter (3.x) | New Parameter (4.x) |
|---------------------|---------------------|
| `lambda_` | `AnalysisConfig(regularization=...)` |
| `permute` | `AnalysisConfig(permute=...)` |
| No equivalent | `AnalysisConfig(max_permutations=...)` |
| No equivalent | `AnalysisConfig(verbose=...)` |

---

## Data Structures

### SessionData

Replaces: `list[np.ndarray]` for Ys

```python
# Old
Ys = [Y1, Y2, Y3]  # List of arrays

# New
from cvmanova import SessionData
data = SessionData(
    sessions=[Y1, Y2, Y3],
    mask=mask,
    affine=affine,
    degrees_of_freedom=fE
)

# Properties
data.n_sessions  # Number of sessions
data.n_voxels    # Number of voxels
data.n_scans     # Scans per session
```

### DesignMatrix

Replaces: `list[np.ndarray]` for Xs

```python
# Old
Xs = [X1, X2, X3]

# New
from cvmanova import DesignMatrix
design = DesignMatrix(
    matrices=[X1, X2, X3],
    regressor_names=['intercept', 'task_A', 'task_B']
)

# Access
design.n_sessions
design.n_regressors
design.to_dataframe()  # Convert to pandas
```

### CvManovaResult

Replaces: tuple returns `(D, p, ...)`

```python
# Old
D, p, _, _ = cv_manova_searchlight(...)
# D[:,:,:,contrast_idx,perm_idx]

# New
result = estimator.fit_score(data, design)

# Rich methods
result.get_contrast('Face')  # Get specific contrast
result.to_nifti('Face', 'output.nii.gz')  # Save to file
result.get_peaks('Face', n=10)  # Find peaks
result.to_dataframe()  # Export to pandas
result.plot_glass_brain('Face')  # Visualize (requires nilearn)
```

---

## Backward Compatibility

The old API remains available:

```python
# This still works in 4.x
from cvmanova import cv_manova_searchlight, load_data_spm

Ys, Xs, mask, affine, fE = load_data_spm('/path/to/spm')
Cs = [np.array([[0, 1, 0]])]

D, p, _, _ = cv_manova_searchlight(
    Ys=Ys, Xs=Xs, mask=mask, sl_radius=3.0, Cs=Cs, fE=fE
)
```

However, we recommend migrating to the new API for:
- Better performance (parallelization, vectorization)
- Type safety and validation
- Richer result objects
- Easier debugging
- Future features and improvements

---

## Step-by-Step Migration Strategy

1. **Keep your old code working**: No immediate changes required

2. **Try the new API on a new analysis**: Start fresh with an estimator

3. **Gradually migrate existing code**:
   - Replace `load_data_spm()` with `SPMLoader`
   - Replace `cv_manova_searchlight()` with `SearchlightCvManova`
   - Replace manual result handling with `CvManovaResult` methods

4. **Leverage new features**:
   - Add parallelization with `n_jobs=-1`
   - Use `ContrastSpec` for factorial designs
   - Export results with `to_dataframe()` and `to_nifti()`

---

## Common Migration Patterns

### Pattern 1: Result Indexing

```python
# Old: Manual indexing
D, p, _, _ = cv_manova_searchlight(...)
face_map = D[:, :, :, 0, 0]  # First contrast, observed data

# New: Named access
result = estimator.fit_score(data, design)
face_map = result.get_contrast('Face')
```

### Pattern 2: Saving Results

```python
# Old: Manual writes
from cvmanova import write_image
write_image(D[:, :, :, 0, 0], 'face.nii.gz', affine)
write_image(D[:, :, :, 1, 0], 'house.nii.gz', affine)

# New: One-liners
result.to_nifti('Face', 'face.nii.gz')
result.to_nifti('House', 'house.nii.gz')
```

### Pattern 3: Contrast Generation

```python
# Old: Manual matrix creation
Cs = [
    np.array([[0, 1, 0, 0]]),
    np.array([[0, 0, 1, 0]]),
    np.array([[0, 0, 0, 1]])
]

# New: Auto-generation
contrasts = ContrastSpec(
    factors=['Face', 'House'],
    levels=[2, 2],
    effects='all'
)
```

---

## Getting Help

- **Documentation**: See `/docs` folder for detailed API reference
- **Examples**: See `/examples` folder for complete workflows
- **Issues**: Report bugs at https://github.com/your-repo/cvmanova/issues

---

## Summary

✅ **Old API still works** - No breaking changes
✅ **New API is recommended** - Better performance, cleaner code
✅ **Migration is gradual** - Adopt features incrementally
✅ **Rich ecosystem** - Integration with nilearn, pandas, scikit-learn

Happy analyzing! 🧠
