# Changelog

## [4.0.0] - 2025-12-16

### Added
- Scikit-learn style estimators: `SearchlightCvManova`, `RegionCvManova`
- Type-safe data structures: `SessionData`, `DesignMatrix`, `CvManovaResult`
- Configuration objects: `SearchlightConfig`, `RegionConfig`, `AnalysisConfig`
- Multiple data loaders: `SPMLoader`, `NiftiLoader`, `NilearnMaskerLoader`
- `ContrastSpec` for auto-generating contrasts from factorial designs
- Rich result objects with methods:
  - `.to_nifti()` - Export to NIfTI format
  - `.to_dataframe()` - Export to pandas DataFrame
  - `.get_peaks()` - Find peak coordinates
  - `.plot_glass_brain()` - Brain visualization (requires nilearn)
  - `.get_contrast()` - Access results by name
- Comprehensive test suite (99% coverage)
- Integration tests with Haxby 2001 dataset
- Migration guide and examples

### Changed
- Core API modernized to use dataclasses and type hints
- Better error messages with input validation
- Improved documentation

### Maintained
- Core computation performance (no change from original)
- Full backward compatibility with procedural API
- Numerical accuracy (validated against original MATLAB)

### Notes
- Performance is equivalent to the original implementation (~1300 voxels/sec)
- Focus of this release is API modernization, not speed improvements
- All results validated with perfect correlation to expected values (ρ = 1.000)
