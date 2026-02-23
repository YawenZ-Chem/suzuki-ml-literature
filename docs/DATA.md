## Data Policy (Phase 1)

This project uses literature-derived reaction data that are manually curated and structured for modeling purposes.

### Directory Structure

- `data/raw/`  
  Raw extracted tables or notes derived from published papers.  
  Large files should not be committed. Extraction scripts and documentation should be retained for reproducibility.

- `data/interim/`  
  Cleaned and standardized reaction tables prior to final feature generation.

- `data/processed/`  
  Machine-learning-ready datasets (CSV/JSON) used for training and validation.

### Reproducibility Principles

- Prefer committing:
  - Small derived datasets
  - All extraction and preprocessing scripts

- Avoid committing:
  - Large raw supplementary files
  - Full journal PDFs

All data originate from publicly available peer-reviewed publications. This repository contains structured reaction data only and does not redistribute original published materials.
