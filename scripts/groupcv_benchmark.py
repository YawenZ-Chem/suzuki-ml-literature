#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


def morgan_fp(smiles: str, n_bits: int = 256, radius: int = 2) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def make_fp_matrix(df: pd.DataFrame, n_bits: int = 256, radius: int = 2) -> np.ndarray:
    X = []
    for _, r in df.iterrows():
        fp_h = morgan_fp(r["smiles_halide"], n_bits=n_bits, radius=radius)
        fp_b = morgan_fp(r["smiles_boronic"], n_bits=n_bits, radius=radius)
        fp_p = morgan_fp(r["smiles_product"], n_bits=n_bits, radius=radius)
        X.append(np.concatenate([fp_h, fp_b, fp_p], axis=0))
    return np.vstack(X)


def build_features(train_df: pd.DataFrame, val_df: pd.DataFrame, n_bits: int = 256, radius: int = 2):
    # fingerprints
    X_fp_train = make_fp_matrix(train_df, n_bits=n_bits, radius=radius)
    X_fp_val   = make_fp_matrix(val_df, n_bits=n_bits, radius=radius)

    # OHE categorical conditions
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_train = ohe.fit_transform(train_df[["catalyst", "base", "solvent"]])
    X_cat_val   = ohe.transform(val_df[["catalyst", "base", "solvent"]])

    # scaled continuous
    temp_train = train_df[["temperature_C"]].to_numpy(dtype=float) / 100.0
    temp_val   = val_df[["temperature_C"]].to_numpy(dtype=float) / 100.0

    time_train = train_df[["time_h"]].to_numpy(dtype=float) / 10.0
    time_val   = val_df[["time_h"]].to_numpy(dtype=float) / 10.0

    X_train = np.hstack([X_fp_train, X_cat_train, temp_train, time_train])
    X_val   = np.hstack([X_fp_val, X_cat_val, temp_val, time_val])
    return X_train, X_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="data/interim/suzuki_literature_interim_v1.csv")
    ap.add_argument("--label_col", type=str, default="label_3bin_v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--n_splits", type=int, default=30)
    ap.add_argument("--n_bits", type=int, default=256)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--out", type=str, default="results/exp_004_groupcv_benchmark/metrics.txt")
    ap.add_argument(
    	"--split_mode",
    	type=str,
    	default="ood_reaction_group",
    	choices=["ood_reaction_group", "in_domain_random", "in_domain_paper"],
    	help="Split strategy: OOD reaction-group split (default), in-domain row-random split, or in-domain paper-group split"
)    
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    label_col = args.label_col

    required = [
        "smiles_halide", "smiles_boronic", "smiles_product",
        "catalyst", "base", "solvent",
        "temperature_C", "time_h", label_col
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # cleaning
    df = df.dropna(subset=["smiles_halide", "smiles_boronic", "smiles_product", label_col])
    df[label_col] = df[label_col].astype(int)

    df["catalyst"] = df["catalyst"].fillna("UnknownCatalyst").astype(str)
    df["base"]     = df["base"].fillna("UnknownBase").astype(str)
    df["solvent"]  = df["solvent"].fillna("UnknownSolvent").astype(str)

    df["temperature_C"] = pd.to_numeric(df["temperature_C"], errors="coerce").fillna(0.0)
    df["time_h"]        = pd.to_numeric(df["time_h"], errors="coerce").fillna(0.0)

    # group key (same as your trainer)
    df["group_key"] = (
        df["smiles_halide"].astype(str) + "|" +
        df["smiles_boronic"].astype(str) + "|" +
        df["smiles_product"].astype(str)
    )

    print("Dataset label distribution:")
    print(df[label_col].value_counts().sort_index())

    if args.split_mode == "ood_reaction_group":
        splitter = GroupShuffleSplit(
            n_splits=args.n_splits, test_size=args.test_size, random_state=args.seed
        )
        split_iter = splitter.split(df, groups=df["group_key"])

    elif args.split_mode == "in_domain_random":
        splitter = StratifiedShuffleSplit(
            n_splits=args.n_splits, test_size=args.test_size, random_state=args.seed
        )
        split_iter = splitter.split(df, df[label_col])

    elif args.split_mode == "in_domain_paper":
        splitter = GroupShuffleSplit(
            n_splits=args.n_splits, test_size=args.test_size, random_state=args.seed
        )
        split_iter = splitter.split(df, groups=df["paper_id"].astype(str))

    else:
        raise ValueError(f"Unknown split_mode: {args.split_mode}")

    # Models to compare
    models = {
        "RandomForest_300_bal": RandomForestClassifier(
            n_estimators=300, random_state=args.seed, class_weight="balanced"
        ),
        "LogReg_multinomial": LogisticRegression(
            max_iter=5000, multi_class="multinomial"
        ),
        "LinearSVC": LinearSVC(),  # strong linear baseline for fingerprints
        "MLP_256_128": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=500,
            random_state=args.seed
        ),
    }

    results = {name: {"macro_f1": [], "bal_acc": []} for name in models}

    for i, (train_idx, val_idx) in enumerate(split_iter, 1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        X_train, X_val = build_features(train_df, val_df, n_bits=args.n_bits, radius=args.radius)
        y_train = train_df[label_col].to_numpy()
        y_val   = val_df[label_col].to_numpy()

        for name, clf in models.items():
            clf.fit(X_train, y_train)
            pred = clf.predict(X_val)
            results[name]["macro_f1"].append(f1_score(y_val, pred, average="macro"))
            results[name]["bal_acc"].append(balanced_accuracy_score(y_val, pred))

        if i % 5 == 0:
            print(f"Completed {i}/{args.n_splits} splits...")

    lines = []
    lines.append(f"Input: {args.in_csv}")
    lines.append(f"Label: {label_col}")
    lines.append(f"Split mode: {args.split_mode}")
    lines.append(f"n_splits={args.n_splits} test_size={args.test_size} seed={args.seed}")
    lines.append(f"Fingerprint: Morgan radius={args.radius} n_bits={args.n_bits} (x3 concatenated)")
    lines.append("")
    lines.append("Summary (mean ± std):")
    for name in models:
        mf = np.array(results[name]["macro_f1"])
        ba = np.array(results[name]["bal_acc"])
        lines.append(
            f"- {name}: MacroF1={mf.mean():.3f} ± {mf.std():.3f} | BalAcc={ba.mean():.3f} ± {ba.std():.3f}"
        )

    out_path = args.out
    os_dir = "/".join(out_path.split("/")[:-1])
    if os_dir:
        import os
        os.makedirs(os_dir, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
