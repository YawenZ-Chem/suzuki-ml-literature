# Dataset schema (Phase 1)

Each row = one Suzuki reaction outcome.

Required columns:
- paper_id: short identifier for source paper (e.g., "paper_A")
- reaction_id: unique within paper (e.g., "A_001")
- smiles_halide: electrophile SMILES
- smiles_boronic: boronic acid/ester SMILES
- smiles_product: product SMILES
- catalyst: normalized text label (e.g., "Pd(PPh3)4")
- base: normalized text label (e.g., "K2CO3")
- solvent: normalized text label (e.g., "dioxane")
- temperature_C: numeric if reported
- time_h: numeric if reported
- scale_mmol: numeric if reported
- yield_percent: numeric if reported; can be blank
- label_3bin: one of {lt1, 1to10, ge10}
- label_binary: one of {fail, success}

Notes:
- label_binary: success := yield >= 1%
- label_3bin: lt1 (<1%), 1to10 (1-10%), ge10 (>=10%)
