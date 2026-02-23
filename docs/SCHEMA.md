# Dataset schema (Phase 1)

Each row = one Suzuki reaction outcome.

Required columns (for the current 3-bin workflow)
paper_id: short identifier for source paper (e.g., "paper_A")
reaction_id: unique within paper (e.g., "A_001")
smiles_halide: electrophile SMILES
smiles_boronic: boronic acid/ester SMILES
catalyst: normalized text label (e.g., "Pd(PPh3)4")
base: normalized text label (e.g., "K2CO3")
solvent: normalized text label (e.g., "dioxane")
temperature_C: numeric (float) if reported
time_h: numeric (float) if reported
yield_percent: numeric (float) if reported
label_3bin: one of {lt50, 50to90, ge90}

Optional columns
smiles_product: product SMILES (optional; include if you are running the “with product” experiment)
scale_mmol: numeric if reported (stored but not used in the current model)

Label definitions (current)
label_3bin
lt50: yield < 50%
50to90: 50% ≤ yield < 90%
ge90: yield ≥ 90%

Notes
scale_mmol is currently excluded from modeling because it is not consistently meaningful across methodology papers; catalyst loading / equivalents are likely more informative than absolute scale.
label_binary is not used in the current workflow (focus is 3-bin classification).
