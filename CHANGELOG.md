# Changelog

All notable changes to EEG-Dash will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CNAME configuration for custom domain support
- Documentation improvements for API reference

### Fixed
- Sphinx documentation build warnings
- Import path resolution in documentation generation

### Changed
- Updated repository organization and structure

## [0.4.1] - 2025-10-21

### Added
- Treemap visualization for dataset statistics
- Sankey, bubble, and ridgeline plots for data exploration
- Time estimation feature for tutorials
- User guide documentation for EEGDash
- Warning system for nonexistent query conditions
- Comprehensive API documentation

### Fixed
- Python 3.10 compatibility issues with type annotations
- S3 download timeout handling for large files
- Cache directory inconsistencies in CI/CD
- MongoDB connection warnings and error messages
- Import errors in feature modules
- Documentation generation warnings
- Orphaned documentation files

### Changed
- Optimized offline mode performance (2x speedup)
- Replaced isort with ruff for import sorting
- Improved BIDS metadata caching efficiency
- Enhanced dataset loading speed
- Cleaned up legacy code and removed dead code

### Performance
- Significant speedup in offline dataset loading
- Optimized feature extraction with better vectorization
- Improved S3 download retry logic with exponential backoff

### Documentation
- Added comprehensive user guide
- Improved API documentation structure
- Added tutorial time estimates
- Updated logo and visual assets
- Enhanced developer notes

## [0.4.0] - 2025-10-11

### Added
- Dataset registry system for dynamic OpenNeuro dataset registration
- Support for multiple data releases
- Mini-release functionality for testing and development
- Enhanced BIDS dataset integration
- Feature extraction preprocessing as standalone functions
- PyArrow support for saving/loading feature dataframes
- Visualization tools for dataset distribution (bubble plots, Sankey diagrams)

### Fixed
- GitHub worker configuration to reduce costs (Linux-only by default)
- Cache directory path resolution across different platforms
- Download functionality for BIDS dependencies
- Pre-commit hook configuration issues

### Changed
- Refactored `load_eeg_attrs_from_bids_file` for better modularity
- Moved feature preprocessing logic to separate functions
- Improved CI/CD caching strategy for datasets
- Updated GitHub Actions workflow for better efficiency

### Security
- Improved MongoDB connection string handling
- Enhanced environment variable management

## [0.3.x] - 2025-09-xx

### Added
- Sphinx documentation system with GitHub Pages deployment
- Custom API documentation pages
- Sex classification tutorial
- P3 oddball and audio task tutorials
- Field consistency validation tests
- Support for EEGLAB (.set) file format
- Rich console output for better user experience

### Fixed
- Downloader module isolation and error handling
- Cache directory default behavior
- Import paths for various modules
- Logo and branding assets

### Changed
- Improved documentation structure and organization
- Enhanced tutorial examples
- Better error messages and logging

### Documentation
- Initial documentation deployment to GitHub Pages
- Added comprehensive examples
- Created tutorial notebooks for common tasks

## [0.2.0] - 2025-xx-xx

### Added
- NeurIPS 2025 challenge support
- Custom S3 bucket specification capability
- Support for braindecode 1.0+
- Type hints for top-level functionality
- Field consistency testing

### Changed
- Upgraded to latest braindecode version
- Improved API consistency
- Enhanced database query capabilities

### Fixed
- Pre-commit hook configurations
- API compatibility with newer dependencies
- Various import and dependency issues

## [0.1.x] - Initial Releases

### Added
- Initial EEGDash API for MongoDB queries
- EEGDashDataset for PyTorch-compatible data loading
- EEGDashBaseDataset for single recording access
- BIDS format support
- S3 data downloading functionality
- MongoDB connection management
- Feature extraction framework with 60+ features:
  - Complexity features (entropy, Lempel-Ziv complexity)
  - Connectivity features (coherence, imaginary coherence)
  - CSP (Common Spatial Pattern) features
  - Dimensionality features (fractal dimensions, Hurst exponent)
  - Signal features (statistical measures)
  - Spectral features (power, entropy, bands)
- OpenNeuro dataset integration
- HBN (Healthy Brain Network) specific preprocessing
- Braindecode integration for preprocessing
- pytest-based testing infrastructure

### Infrastructure
- GitHub Actions CI/CD pipeline
- Pre-commit hooks with ruff linting
- Sphinx documentation setup
- PyPI package publishing automation

---

## Release Schedule

- **Patch releases** (0.x.Y): Bug fixes, documentation updates (as needed)
- **Minor releases** (0.X.0): New features, non-breaking changes (monthly to quarterly)
- **Major releases** (X.0.0): Breaking changes, major refactors (when necessary)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Links

- **Homepage**: https://eegdash.org/
- **Repository**: https://github.com/sccn/EEG-Dash-Data
- **Documentation**: https://sccn.github.io/eegdash
- **PyPI**: https://pypi.org/project/eegdash/
- **Issues**: https://github.com/sccn/EEG-Dash-Data/issues

## Authors and Contributors

### Core Team
- **Young Truong** (dt.young112@gmail.com)
- **Arnaud Delorme** (adelorme@gmail.com) - Swartz Center for Computational Neuroscience, UCSD
- **Aviv Dotan** (avivd220@gmail.com) - Ben-Gurion University
- **Oren Shriki** (oren70@gmail.com) - Ben-Gurion University
- **Bruno Aristimunha** (b.aristimunha@gmail.com)

### Contributors
- Dung Truong
- Pierre Guetschel
- Vivian Chen
- Christian Kothe
- And many others who have contributed through pull requests and issue reports

## Acknowledgments

EEG-DaSh is a collaborative initiative between the United States and Israel, supported by the **National Science Foundation (NSF)**. The partnership brings together experts from:
- **Swartz Center for Computational Neuroscience (SCCN)** at UC San Diego
- **Ben-Gurion University (BGU)** in Israel

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## Version History Summary

| Version | Release Date | Key Features |
|---------|-------------|--------------|
| 0.4.1 | 2025-10-21 | Performance optimizations, visualization tools, comprehensive documentation |
| 0.4.0 | 2025-10-11 | Dataset registry, feature preprocessing refactor, multi-release support |
| 0.3.x | 2025-09-xx | Documentation system, tutorials, GitHub Pages deployment |
| 0.2.0 | 2025-xx-xx | NeurIPS challenge support, braindecode 1.0+ compatibility |
| 0.1.x | 2024-2025 | Initial release with core functionality |

---

*This changelog is automatically maintained based on commit history. For more detailed information about specific changes, please refer to the [commit history](https://github.com/sccn/EEG-Dash-Data/commits/main).*
