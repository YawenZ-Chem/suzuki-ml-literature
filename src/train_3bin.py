#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def morgan_fp(smiles: str, n_bits: int = 256, radius: int = 2) -> np.ndarray:
    """Return Morgan fingerprint as a (n_bits,) numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def make_fp_matrix(df: pd.DataFrame) -> np.ndarray:
    """Concatenate halide+boronic+product fingerprints -> (n, 768)."""
    X = []
    for _, r in df.iterrows():
        fp_h = morgan_fp(r["smiles_halide"])
        fp_b = morgan_fp(r["smiles_boronic"])
        fp_p = morgan_fp(r["smiles_product"])
        X.append(np.concatenate([fp_h, fp_b, fp_p], axis=0))
    return np.vstack(X)

def assign_label_3bin_v2(y):
    y = float(y)
    if y < 60:
        return 0
    elif y < 85:
        return 1
    else:
        return 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="data/interim/train_all.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    
    ap.add_argument("--label_col", type=str, default="label_3bin")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    label_col = args.label_col

    # ---- basic cleaning ----
    required = [
        "smiles_halide", "smiles_boronic", "smiles_product",
        "catalyst", "base", "solvent",
        "temperature_C", label_col
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    df = df.dropna(subset=["smiles_halide", "smiles_boronic", "smiles_product", label_col])
    df[label_col] = df[label_col].astype(int)

    # fill missing condition fields with explicit tokens (safer than NaN)
    df["catalyst"] = df["catalyst"].fillna("UnknownCatalyst").astype(str)
    df["base"]     = df["base"].fillna("UnknownBase").astype(str)
    df["solvent"]  = df["solvent"].fillna("UnknownSolvent").astype(str)

    # temperature: numeric (if missing, set 0; later you can drop those rows instead)
    df["temperature_C"] = pd.to_numeric(df["temperature_C"], errors="coerce").fillna(0.0)
    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce").fillna(0.0)

    # ---- group key to avoid leakage ----
    df["group_key"] = (
        df["smiles_halide"].astype(str) + "|" +
        df["smiles_boronic"].astype(str) + "|" +
        df["smiles_product"].astype(str)
    )

    # ---- group split ----
    gss = GroupShuffleSplit(test_size=args.test_size, random_state=args.seed)
    train_idx, val_idx = next(gss.split(df, groups=df["group_key"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    
    print(f"Rows (train/val): {len(train_df)} / {len(val_df)}")
    print(f"Unique groups (train/val): {train_df['group_key'].nunique()} / {val_df['group_key'].nunique()}")

    print("Train label counts:\n",
      train_df[label_col].value_counts().sort_index())

    print("Val label counts:\n",
      val_df[label_col].value_counts().sort_index())   

    # ---- featurize ----
    print("Featurizing fingerprints...")
    X_fp_train = make_fp_matrix(train_df)  # (n_train, 768)
    X_fp_val   = make_fp_matrix(val_df)    # (n_val, 768)

    print("Encoding categorical conditions...")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_train = ohe.fit_transform(train_df[["catalyst", "base", "solvent"]])
    X_cat_val   = ohe.transform(val_df[["catalyst", "base", "solvent"]])

    temp_train = train_df[["temperature_C"]].to_numpy(dtype=float) / 100.0
    temp_val   = val_df[["temperature_C"]].to_numpy(dtype=float) / 100.0

    time_train = train_df[["time_h"]].to_numpy(dtype=float) / 10.0   # simple scaling
    time_val   = val_df[["time_h"]].to_numpy(dtype=float) / 10.0

    X_train = np.hstack([X_fp_train, X_cat_train, temp_train, time_train])
    X_val   = np.hstack([X_fp_val, X_cat_val, temp_val, time_val])

    y_train = train_df[label_col].to_numpy()
    y_val   = val_df[label_col].to_numpy()

    print("X_train shape:", X_train.shape, "X_val shape:", X_val.shape)

    # ---- train ----
    clf = RandomForestClassifier(
    n_estimators=300,
    random_state=args.seed,
    class_weight="balanced"
)

    clf.fit(X_train, y_train)

    pred = clf.predict(X_val)

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, pred))
    print("\nClassification report:")
    print(classification_report(y_val, pred, digits=3))

    # ---- external evaluation on foreign dataset ----
    print("\n=== External evaluation: foreign_paper_05.csv ===")

    foreign = pd.read_csv("data/interim/foreign_paper_05.csv")

    missing_cols_foreign = [c for c in required if c not in foreign.columns]
    if missing_cols_foreign:
        raise ValueError(f"Missing columns in foreign CSV: {missing_cols_foreign}")

    foreign = foreign.dropna(subset=["smiles_halide", "smiles_boronic", "smiles_product", "yield_percent"]).copy()

    if label_col == "label_3bin_v2":
        foreign[label_col] = foreign["yield_percent"].apply(assign_label_3bin_v2)
    else:
        foreign = foreign.dropna(subset=[label_col]).copy()
        foreign[label_col] = foreign[label_col].astype(int)

    foreign["catalyst"] = foreign["catalyst"].fillna("UnknownCatalyst").astype(str)
    foreign["base"]     = foreign["base"].fillna("UnknownBase").astype(str)
    foreign["solvent"]  = foreign["solvent"].fillna("UnknownSolvent").astype(str)

    foreign["temperature_C"] = pd.to_numeric(foreign["temperature_C"], errors="coerce").fillna(0.0)
    foreign["time_h"] = pd.to_numeric(foreign["time_h"], errors="coerce").fillna(0.0)

    print(f"Rows (foreign): {len(foreign)}")
    print("Foreign label counts:\n", foreign[label_col].value_counts().sort_index())

    if len(foreign) == 0:
        raise ValueError("Foreign dataset has 0 usable rows after filtering. Check yield_percent or labels.") 

    print("Featurizing foreign fingerprints...")
    X_fp_foreign = make_fp_matrix(foreign)

    print("Encoding foreign categorical conditions...")
    X_cat_foreign = ohe.transform(foreign[["catalyst", "base", "solvent"]])

    temp_foreign = foreign[["temperature_C"]].to_numpy(dtype=float) / 100.0
    time_foreign = foreign[["time_h"]].to_numpy(dtype=float) / 10.0

    X_foreign = np.hstack([X_fp_foreign, X_cat_foreign, temp_foreign, time_foreign])
    y_foreign = foreign[label_col].to_numpy()

    print("X_foreign shape:", X_foreign.shape)

    pred_foreign = clf.predict(X_foreign)

    print("\nForeign confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_foreign, pred_foreign))
    print("\nForeign classification report:")
    print(classification_report(y_foreign, pred_foreign, digits=3))

# ---- save predictions ----
    foreign_out = foreign.copy()
    foreign_out["pred_label"] = pred_foreign

    import os
    os.makedirs("results/exp_011_foreign_dataset", exist_ok=True)

    foreign_out.to_csv(
        "results/exp_011_foreign_dataset/predictions.csv",
        index=False
)

    print("Saved predictions to results/exp_011_foreign_dataset/predictions.csv")

if __name__ == "__main__":
    main()
