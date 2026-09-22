# Changelog

This file records notable changes to the project.

We use [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and follow
[Semantic Versioning](https://semver.org/) where practical.

## [Unreleased]

### Added

- Mitra-v2 fine-tuning and inference through AutoGluon with a pinned checkpoint.
- Local TabFM classification with pinned source and model revisions, bounded context, and CUDA out-of-memory fallback settings.
- Optional `mitra` and `tabfm` uv dependency groups.
- Pipeline regression tests for label semantics, feature selection, metrics, and raw-feature validation.
- Training metrics, timings, source revisions, licenses, and foundation-model artifact locations in the schema-v2 model bundle.
- Technical guides for architecture, development, training, bundle compatibility, deployment, and troubleshooting.

### Changed

- Migrated project dependency management and Docker installation from pip requirements to uv.
- Split preprocessing by model family: classical models use one-hot encoding, SMOTE, and scaling, while foundation models use raw mixed-type features.
- Updated the Streamlit dashboard to serve available foundation models sequentially so they do not compete for GPU memory.
- Updated Docker deployment to include Mitra-v2 while keeping TabFM local-only.

### Fixed

- Corrected target encoding so `Attrited Customer` is the positive churn class.
- Released Mitra CUDA memory before loading TabFM in combined runs.

## [2026-06-13]

### Added

- OSS companion documentation initialized (license, contributing, security, conduct, changelog).
