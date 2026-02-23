# Dataset Schema (phase 1)

Each row represents one Suzuki cross-coupling reaction outcome.

---

## Required Columns (for current 3-bin workflow)

- **paper_id**: Short identifier for source paper (e.g., `"paper_A"`)
- **reaction_id**: Unique identifier within paper (e.g., `"A_001"`)
- **smiles_halide**: Electrophile SMILES
- **smiles_boronic**: Boronic acid/ester SMILES
- **smiles_product**: Product SMILES  
- **catalyst**: Normalized text label (e.g., `"Pd(PPh3)4"`)
- **base**: Normalized text label (e.g., `"K2CO3"`)
- **solvent**: Normalized text label (e.g., `"dioxane"`)
- **temperature_C**: Numeric value (°C), if reported
- **time_h**: Numeric value (hours), if reported
- **yield_percent**: Numeric value (0–100), if reported
- **label_3bin**: One of `{lt50, 50to90, ge90}`

---

## Optional Columns

- **scale_mmol**: Numeric value, if reported  
  *(Stored but currently not used in modeling)*

---

## Label Definitions (Current)

### `label_3bin`

- `lt50` → yield < 50%
- `50to90` → 50% ≤ yield < 90%
- `ge90` → yield ≥ 90%

---

## Notes

- `scale_mmol` is currently excluded from modeling. Absolute reaction scale is often less informative than relative catalyst loading or reagent equivalents in methodology-focused publications.

- Binary classification (`label_binary`) is not used in the current workflow. The focus is on 3-bin yield classification.

- All SMILES strings are validated and canonicalized using RDKit prior to feature generation.

- Features are generated from:
  - Molecular fingerprints (ECFP)
  - Encoded reaction conditions (catalyst, base, solvent)
  - Numeric parameters (temperature, time)

---

## Modeling Focus

The current modeling task is **3-class yield classification** using literature-derived reaction data.

The goal is to explore structure–condition–yield relationships under limited dataset conditions and evaluate generalization performance on unseen reactions.

