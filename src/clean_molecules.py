from pathlib import Path
import pandas as pd

MOLECULES_PATH = Path("data/interim/molecules.csv")

def main():
    df = pd.read_csv(MOLECULES_PATH)

    # Prefer existing "smiles" as raw input
    if "smiles_raw" not in df.columns:
        df["smiles_raw"] = df.get("smiles", "")

    # If smiles_canonical exists, keep it; otherwise create empty placeholder
    if "smiles_canonical" not in df.columns:
        df["smiles_canonical"] = ""

    keep = ["molecule_id", "name_in_paper", "role", "smiles_raw", "smiles_canonical"]
    for col in keep:
        if col not in df.columns:
            df[col] = ""

    df = df[keep].copy()
    df.to_csv(MOLECULES_PATH, index=False)

    print("Cleaned molecules.csv to columns:", keep)

if __name__ == "__main__":
    main()
