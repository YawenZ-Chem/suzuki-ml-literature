# Suzuki Cross-Coupling Yield Classification (Literature-Derived ML Workflow)

## Project Overview

Phase 1 goal: build and document a reproducible machine learning workflow using open-access Suzuki cross-coupling literature data.

This project constructs a structured reaction dataset from peer-reviewed publications and evaluates supervised classification models for reaction yield prediction under limited-data conditions.

---

## Scientific Framing and Inspiration

The selection of input features and modeling structure is inspired by:

**Atz et al.** (DOI: 10.1039/d4md00196f)

In that work, the authors modeled Suzuki reaction outcomes using:

- Halogen (electrophile)
- Boronic acid
- Product
- Catalyst
- Solvent
- Base

They evaluated both binary and three-category classification schemes. This project adopts a similar input structure to enable comparison with literature precedent while adapting the setup to literature-derived data constraints.

---

## Key Differences from Plate-Based HTE Data

Atz et al. used a plate-based experimental setup with consistent reaction conditions (e.g., consistent temperature and time across plates). In contrast, literature-derived data:

- Contains inconsistent reaction times and temperatures
- Often over-reports moderate-to-high yields
- Lacks systematic negative screening

Because of this:

- Reaction temperature and reaction time are included as explicit model inputs.
- The temperature range is restricted to 60–110 °C to represent typical elevated Suzuki conditions and reduce heterogeneity (room-temperature examples are excluded).
- scale_mmol is excluded from modeling because absolute scale is often less informative than relative catalyst loading/equivalents across methodology papers.

---

## Modeling Task

### 3-Class Yield Classification (Adjusted for Literature Bias)

Yield bins (current):

- `lt50` → yield < 50%
- `50to90` → 50% ≤ yield < 90%
- `ge90` → yield ≥ 90%

Rationale: literature methodology papers disproportionately report mid-to-high yields, so bin boundaries are shifted upward to better discriminate among moderate and high-yield transformations.

---

## Dataset Construction

- 4–6 Suzuki methodology publications
- ~130–150 total reactions
- Manual extraction and normalization
- Primary structured dataset:

`data/interim/suzuki_literature_interim_v1.csv`

See `SCHEMA.md` for full dataset specification.

---

## Data Split Strategy

### 1. Internal Validation (Within Pooled Literature Dataset)

To understand model behavior under different levels of distribution shift, three dataset splitting strategies are 
evaluated.

### In-Domain Random Split
A stratified row-level split with label balancing.

- Substrates may appear in both training and test sets
- Measures interpolation within the dataset distribution
- Provides an approximate upper bound on achievable performance

### Reaction-Group Out-of-Distribution (OOD) Split

The primary evaluation metric uses a reaction-group OOD split.

Reactions are grouped using the key:

`smiles_halide | smiles_boronic | smiles_product`

All reactions belonging to the same reaction group are assigned entirely to either the training or test set using 
`GroupShuffleSplit`. This prevents leakage of identical substrate combinations across splits.

Importantly, **papers are not used as grouping units in this split**. Therefore reactions from the same publication may 
appear in both the training and test sets, but the **specific reaction pairs in the test set are never seen during 
training**.

Across all 30 repeated splits we verified that every paper appearing in the test set also appears in the training set. 
Thus this evaluation measures **generalization to unseen reaction pairs rather than unseen publications**.

### Paper-Level Split

For comparison, a paper-level split groups reactions by `paper_id`, assigning entire publications to either training or 
testing sets.

This evaluates **cross-publication generalization**. Because the dataset contains a limited number of papers with 
uneven reaction counts, this split exhibits higher variance and is used primarily as exploratory analysis.

---

### 2. External Validation

An additional evaluation will be performed using a completely separate Suzuki methodology publication that is **not 
included in the pooled training dataset**.

This provides a true external benchmark for assessing cross-paper generalization beyond the internal dataset splits.
 
---
### Fingerprint Resolution Experiment

Morgan fingerprints were generated with radius = 2 for the halide,
boronic acid, and product molecules and concatenated to form the
structural feature vector.

Fingerprint resolution was evaluated using 256, 512, and 1024 bits
under the reaction-group OOD evaluation protocol (30 repeated splits).

| nBits | RandomForest Macro F1 |
|------|----------------------|
| 256 | 0.462 ± 0.091 |
| 512 | 0.456 ± 0.083 |
| 1024 | **0.473 ± 0.102** |

Based on this experiment, **1024-bit fingerprints were selected for
subsequent experiments**.

---

## Pipeline Overview

Literature PDFs  
→ Manual extraction  
→ SMILES validation and canonicalization  
→ Feature generation (ECFP + encoded conditions)  
→ Train/validation split  
→ Random Forest classifier  
→ Evaluation (confusion matrix, classification report)

---

## Limitations

  - Small dataset size (~150 reactions)
  - Class imbalance toward higher yields
  - Heterogeneous experimental conditions
  - No systematic negative screening (unlike HTE data)
  
  This project is primarily focused on workflow reproducibility and small-data modeling exploration.

## Future Directions

  - Expand dataset size (>300 reactions)
  - Compare ECFP vs graph-based models
  - Investigate catalyst-loading normalization
  - Evaluate robustness under stricter cross-paper splits
