# Changelog

All notable changes to EEG-Dash will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Tutorial gallery refactor: 22 sphinx-gallery tutorials covering Categories A through H (Start Here, Core Workflow, Event-Related, Resting State, Features, Evaluation, Applied, Transfer/Foundation), each with a published spec under `docs/tutorials/_spec/` declaring learning goal, audience, runtime budget, network budget, expected outputs and GPU/network requirements
- Five new HPC and scaling how-tos under `examples/applied/`: `how_to_download_a_dataset`, `how_to_work_offline`, `how_to_use_hpc_cache`, `how_to_run_preprocessing_on_slurm`, `how_to_parallelize_feature_extraction`
- Six concept pages under `docs/source/concepts/` covering EEGDash objects, lazy loading and cache, leakage-safe splits, windowing semantics, evaluation protocols and the BDF transfer-foundation track
- Public APIs for the rewritten tutorials: `eegdash.splits` (subject-aware split utilities), `eegdash.tasks` (task helpers), `EEGDashDataset.summary` / `EEGDashDataset.preview` / `EEGDashDataset.filter` / `EEGDashDataset.search_datasets` for in-memory query and inspection, and `EEGTask.make_windows` for declarative window generation
- Tutorial audit pipeline (`scripts/tutorial_audit/`) with 49-rule rubric, spec coherence checks, runtime tracker, and reviewer-only score artifact (`docs/evidence/tutorials/<id>/reviewer_score.json`)
- Three-stage tutorial CI matrix (`.github/workflows/tutorial-audit.yml`): static rubric on every PR, anchor-tutorial nbclient execution on PRs touching tutorial sources, full gallery sweep on nightly cron and `workflow_dispatch`
- `make -f tutorials.mk tutorial-runtime` target and `scripts.tutorial_audit.runtime_tracker` CLI: aggregates declared `budgets.max_runtime_seconds` and `budgets.max_network_mb` across all 27 specs and emits `docs/evidence/runtime_tracker_<date>.md`, with optional `--measured` JSON for delta against measured timings

### Changed
- Reviewer rubric appendix in `CONTRIBUTING.md` now documents the eight reviewer-only rules and the per-tutorial scoring grid; CI parses each `reviewer_score.json` and gates merge on every reviewer-only rule scoring at least 3
- `eegdash.features.fit_feature_extractor` retained as a backward-compat alias for the new feature-extraction entrypoint used by the rewritten feature tutorials

### Removed
- Legacy tutorial entrypoints superseded by the new gallery: `examples/tutorial_minimal.py`, `examples/tutorial_transfer_learning.py` and several `examples/noplot_tutorial_*.py` scripts (their pedagogical content has been rewritten under `examples/tutorials/` and their applied counterparts under `examples/applied/`).

## [0.7.2] - 2026-04-29

### Added
- NEMAR digest-time sidecar inlining: `storage.sidecar_inline` carries small text BIDS sidecars (TSV/JSON/README) so the runtime never round-trips to GitHub for an enriched record (#334)

### Changed
- Quieted canonical-alias collision logs in the dataset registry; refreshed `uv.lock` (#333)

## [0.7.1] - 2026-04-29

### Fixed
- NEMAR anonymous S3 + GitHub access end-to-end: stop probing with `ListBucket`, use `s3fs.get_file()` for direct anonymous `GetObject`, and route via GitHub-pointer-first with the largefiles annex rule (#332)

## [0.7.0] - 2026-04-28

### Added
- Pick-channels feature for feature extractors (#302)
- Canonical-name aliases for dataset class registry (#306)
- Feature-as-preprocessor wrapper, including `preprocessor_as_feature` (#308)
- Montage registry pipeline with S3 FIF header fallback (#325)
- Embedded electrode-explorer iframe on each dataset documentation page (#326)
- Feature dataset and extractor improvements (#323)
- Feature extractor addons (#331)

### Fixed
- Reconcile `source` / `storage.base` against `dataset_id` pattern in ingest (#329)
- Feature column-naming consistency in extractor output (#330)
- `signal_decorrelation_time` logic correction (#328)
- NEMAR source detection and removal of three redirecting OpenNeuro IDs (#327)
- Dataset hero badge row alignment by pinning each `:width:` to SVG natural size (#324)
- `requestIdleCallback` signature plus shield-badge aspect ratio (#322)
- Restore `_rewrite_sitemap_index` dropped during #318 rebase (#321)
- Rewrite homepage entry to canonical bare-host URL in sitemap (#319)
- SEO batch-2: short/missing metas, duplicate tags, plus local validator and audit log (#318)
- Cap long meta descriptions broken by an apostrophe regex (#317)
- P0+P1 figure fixes — Sankey SVG export, clinical split, growth log, bubble labels

### Changed
- Refactor bivariate iterator to enforce directed/undirected logic on metrics (#309, #310)
- Convert feature class decorators to functions (#305)
- Minimize runtime dependencies; replace parquet with safetensors for feature serialization

### Performance
- Drop 2.2 MiB per-page payload, recover 1.1K+ 404s, add GSC/Bing/IndexNow hooks (#315)
- Vendor Fuse.js plus post-build asset fingerprinting (#320)

### Documentation
- Polish catalog charts, table, intro and extend author-year alias resolution (#307)
- Structured data, Open Graph cards, meta descriptions, JS gating (#311)
- Agent-readiness, library-first hero, paper-accurate citation block (#312)
- `llms.txt` coverage, `dataset_summary.md` shrink, directive hints (#314)
- Remove AI-isms across user-facing documentation (#316)
- Redesigned social card with brand identity and code snippet

## [0.6.0] - 2026-04-06

### Added
- Publication-quality chart redesign with PDF export (#299)
- EEGDash API tutorial (#283)
- Intel Mac system support with conditional `numba` and `torch` pinning (#294)
- Preprocessor output type system with documentation and inspection (#278, #289, #290, #291, #292)
- Feature extractor configuration files (#286)
- Optional recording-info parameter for feature extractors (#284, #285)
- `on_error` parameter to skip bad records during data loading (#276)
- CTF direct reader for `.ds` directories (#279)
- BrainVision handling with auto-repair and metadata generation (#236)
- Enhanced documentation search with autocomplete and live search (#238)
- fNIRS metadata extraction support (#238)
- EDF/BDF metadata extraction via MNE
- Auto-sync dataset CSV with git-annex size correction (#298)
- Duration computation for recordings
- `datasets_dict` dictionary for programmatic dataset access (#209)
- HTTP API client (#214)
- `acq-` BIDS entity support in record pipeline (#247)
- Pathology and modality labels for dataset summary (#212)
- Number of sessions column in dataset summary table
- Unified color palette across plots and CSS tags

### Fixed
- Reliable data loading across 522 EEG datasets (#282)
- Git-annex key path resolution to BIDS names for S3 downloads (#280)
- BIDS TSV whitespace-padded `n/a` values (#278)
- BIDS coordinate system validation and extended non-numeric run fallback (#277)
- EEGLAB format: truncated `.fdt` handling (#266), epoch/trial mismatch (#275), latin-1 encoding (#269), error handling (#272)
- CTF "Illegal date" handling for numeric dash dates (#262)
- Corrupt MAT/EEGLAB file detection (#265)
- Participants.tsv subject/session ID repair (#264)
- Non-numeric run entities via `check=False` in BIDSPath (#263)
- Empty or malformed `channels.tsv` handling (#270)
- Auto-discovery and download of companion files `.fdt`, `.eeg`, `.vmrk` (#268)
- Split FIF file handling and continuation file downloads (#273)
- TSV encoding fallback order (#240, #241)
- Case-insensitive BIDS sidecar matching for task entity mismatches (#242, #243)
- Duplicate `participant_id` in `participants.tsv` (#244)
- European comma decimal separators in BIDS TSV files (#257)
- iEEG `coordsystem.json` key handling (#236)
- Git-annex key pointer rewriting in VHDR files (#259, #261)
- Corrupt data, bad timestamps, and invalid BIDS entity handling (#260)
- Directory-based format recursive download support (#253)
- Dynamic `cumulative_sizes` and accurate `nchans` computation (#243)
- BIDSPath entity extraction for git-annex datasets (#245)
- Subject/participant TSV fallback handling (#248)
- Signal filter preprocessor bug (#288)
- Anonymous hyperlinks in docstrings to avoid duplicate target warnings
- MNE/Dataset compatibility with retry handlers for multiple failing datasets (#271)

### Changed
- Switched to released PyPI versions for MNE and MNE-BIDS dependencies
- Required `pandas>=2.0` (#210)
- Consolidated `_load_raw` error handling into single retry loop
- Expanded default BIDS modalities to include SNIRF
- Bumped `braindecode` dependency to `>=1.4.0`

### Documentation
- Full docstrings for Features module (#281, #287)
- Dataset documentation with tags, feedback section, and visualization updates (#235)
- P3 oddball tutorial updates (#192)
- Improved dataset page layout and README parser

## [0.5.0] - 2026-01-07

### Added
- New EEGDash logo and branding assets (#200)
- CNAME configuration for custom domain support (#187)
- Favicon configuration for documentation
- Preprocessors as standalone functions (#194)
- Data digestion pipeline v2 (#193, #215)
- Mass ingestion fixes and optimization (#227)
- HPC, Docker, and Singularity usage examples
- Age regression tutorial
- iEEG documentation

### Fixed
- Warning spam during EEGChallengeDataset download (#226)
- Sphinx documentation build warnings
- Import path resolution in documentation generation
- Completed missing entries in the dataset summary table

### Changed
- Updated repository organization to eegdash/EEGDash
- Updated license from GPL to BSD-3-Clause (#199)
- Updated color theme to match new logo (primary blue: #003D82, accent orange: #FF8000)
- Updated GitHub references to new organization
- Cleaned up legacy code and removed dead code (#182, #184)
- Use API dataset summary for CI docs (#224)

### Documentation
- Parallelized documentation build for faster generation (#232)
- Improved quickstart guide aesthetics (#232)
- Regenerated dataset summary and documentation (#220)
- Enhanced footer with logo branding
- Updated tutorial documentation (#202)
- Improved documentation styling and visual hierarchy

## [0.4.1] - 2025-10-21

### Added
- Treemap visualization for dataset statistics
- Sankey, bubble, and ridgeline plots for data exploration
- Time estimation feature for tutorials
- User guide documentation for EEGDash
- Warning system for nonexistent query conditions
- Full API documentation

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
- 2x speedup in offline dataset loading
- Vectorized feature extraction
- S3 download retry logic with exponential backoff

### Documentation
- Added user guide
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
- Added end-to-end examples
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
- **Bruno Aristimunha** (b.aristimunha@gmail.com)
- **Arnaud Delorme** (adelorme@gmail.com) - Swartz Center for Computational Neuroscience, UCSD
- **Young Truong** (dt.young112@gmail.com)
- **Aviv Dotan** (avivd220@gmail.com) - Ben-Gurion University
- **Oren Shriki** (oren70@gmail.com) - Ben-Gurion University

### Contributors
- Pierre Guetschel
- Vivian Chen
- Christian Kothe
- And many others who have contributed through pull requests and issue reports

## Acknowledgments

EEG-DaSh is a collaborative initiative between the United States and Israel, supported by the **National Science Foundation (NSF)**. The partnership brings together experts from:
- **Swartz Center for Computational Neuroscience (SCCN)** at UC San Diego
- **Ben-Gurion University (BGU)** in Israel

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

---

## Version History Summary

| Version | Release Date | Key Features |
|---------|-------------|--------------|
| 0.7.2 | 2026-04-29 | NEMAR digest-time sidecar inlining; runtime never round-trips to GitHub for enriched records |
| 0.7.1 | 2026-04-29 | NEMAR anonymous S3 access fix (GitHub-pointer-first, anonymous GetObject) |
| 0.7.0 | 2026-04-28 | Montage registry, electrode-explorer integration, feature-as-preprocessor, parquet→safetensors, SEO/perf overhaul |
| 0.6.0 | 2026-04-06 | Reliable loading across 522 datasets, publication charts, API tutorial, fNIRS support |
| 0.5.0 | 2026-01-07 | New branding, digestion pipeline v2, preprocessors as functions, BSD-3 license |
| 0.4.1 | 2025-10-21 | Performance optimizations, visualization tools, comprehensive documentation |
| 0.4.0 | 2025-10-11 | Dataset registry, feature preprocessing refactor, multi-release support |
| 0.3.x | 2025-09-xx | Documentation system, tutorials, GitHub Pages deployment |
| 0.2.0 | 2025-xx-xx | NeurIPS challenge support, braindecode 1.0+ compatibility |
| 0.1.x | 2024-2025 | Initial release with core functionality |

