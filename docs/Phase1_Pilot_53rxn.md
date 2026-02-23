Computational Methods – Phase I
Baseline Yield Classification Model (n = 53)
Dataset Construction

A curated dataset of 53 Suzuki cross-coupling reactions was assembled from three literature sources. Each reaction included:

Canonical SMILES for halide, boronic partner, and product

Catalyst, base, solvent

Reaction temperature (°C)

Reaction time (hours)

Reported yield (%)

All SMILES were validated and canonicalized using RDKit prior to featurization.

Yield Binning Strategy

Continuous yields were converted into a 3-class classification problem:

Class 0 (Low yield): < 50%

Class 1 (Mid yield): 50–90%

Class 2 (High yield): > 90%

Class 1 was the majority class in the dataset.

Data Splitting and Leakage Prevention

To prevent substrate-level data leakage, an 80/20 grouped split was performed using GroupShuffleSplit.

Grouping key:

smiles_halide | smiles_boronic | smiles_product

Final split:

Training set: 41 reactions

Validation set: 12 reactions

Validation distribution:

Class 0: 2

Class 1: 7

Class 2: 3

Feature Engineering
1. Structural Features

Morgan fingerprints (radius = 2, nBits = 256) were generated for:

Halide

Boronic partner

Product

These were concatenated into a 768-dimensional vector.

2. Reaction Condition Encoding

Categorical variables were one-hot encoded:

Catalyst

Base

Solvent

Unknown validation categories were ignored.

3. Continuous Parameters

Temperature and time were included as scaled continuous features:

temperature_C / 100

time_h / 10

Final feature dimensions:

Without time: 783 features

With time: 784 features

Model Architecture

Model: MLPClassifier (scikit-learn)

Hidden layers: (256, 128)

Activation: ReLU

Max iterations: 300

Random seed: 42

Experimental Comparisons
Experiment 1 — No Reaction Time

Accuracy: ~60%

Observation:

Model failed to detect Class 0

Majority-class bias evident

Experiment 2 — Including Reaction Time

Validation Confusion Matrix:

[[0 2 0]
 [0 7 0]
 [0 1 2]]

Accuracy: 75%

Observations:

One additional high-yield reaction correctly classified

Macro-average recall improved

Class 0 detection remained absent

Interpretation:

Reaction time provides modest predictive signal for distinguishing high-yield outcomes.

Experiment 3 — Random Forest (With Time)

Accuracy: ~60%

Observation:

Underperformed neural network

Likely limited by small dataset size

Key Findings

Grouped splitting successfully prevented substrate-level leakage

Pipeline (featurization → split → train → evaluate) functions correctly

With 41 training samples, neural network likely overparameterized

Model predominantly learns majority-class behavior

Reaction time contributes measurable predictive signal

Low-yield detection remains unresolved

Conceptual Clarification

The current model predicts:

Yield class given structure and chosen reaction conditions.

It does NOT predict:

Optimal reaction conditions for new substrates.

Thus, this model addresses performance classification, not condition recommendation.
