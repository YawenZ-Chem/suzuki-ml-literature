# Data policy (Phase 1)

- `data/raw/`: raw extracted tables/files from papers (do not commit if large; keep scripts + notes)
- `data/interim/`: cleaned but not finalized
- `data/processed/`: ML-ready datasets (CSV/JSON) used for training/validation
- Prefer committing small derived datasets + full extraction scripts so results are reproducible.
