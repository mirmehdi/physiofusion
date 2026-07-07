# physiofusion

A reusable, uncertainty-aware ML stack for multimodal physiological signals,
demonstrated across three glucose tracks (CGM forecasting, non-invasive
estimation, optical-spectral modeling). See PROJECT_CHARTER_v1.2.md.

## Layout
- src/physiofusion/ — the reusable library (Stages 1-8 of the charter)
  - splits/ — subject-grouped, leakage-proof splitting (the P0 keystone)
- tracks/ — per-track code that imports the library
- tests/ — including the CI leakage guard
- configs/ — Hydra configs (added later in P0)
- Data/ — raw datasets (git-ignored; versioned with DVC)

## Setup
    conda env create -f environment.yml
    conda activate physiofusion
    pytest

## Data discipline
A subject_id is in exactly one of {train, val, test}. The CI build fails if any
subject crosses splits. See tests/test_grouped_splits.py.
