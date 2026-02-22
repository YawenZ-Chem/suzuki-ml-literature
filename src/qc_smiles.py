import pandas as pd
from rdkit import Chem

def main(path="data/interim/molecules.csv"):
    df = pd.read_csv(path)

    if "smiles" not in df.columns:
        print("Column 'smiles' not found in CSV.")
        return

    bad = []
    canonical = []

    for i, row in df.iterrows():
        smi = str(row["smiles"]).strip()
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            bad.append((row.get("molecule_id", i), smi))
            canonical.append(None)
        else:
            canonical.append(Chem.MolToSmiles(mol, canonical=True))

    df["smiles_canonical"] = canonical
    df.to_csv(path, index=False)

    if bad:
        print("Invalid SMILES found:")
        for mid, smi in bad:
            print(f"{mid}: {smi}")
    else:
        print("All SMILES valid.")
        print("Canonical SMILES written to:", path)

if __name__ == "__main__":
    main()
