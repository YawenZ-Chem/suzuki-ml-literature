from pathlib import Path
import pandas as pd

MOLECULES_PATH = Path("data/interim/molecules.csv")

def next_product_id(existing_ids):
    i = 1
    while f"P{i}" in existing_ids:
        i += 1
    return f"P{i}"

def main():
    df = pd.read_csv(MOLECULES_PATH)
    existing_ids = set(df["molecule_id"].astype(str))

    # 👇 Replace these with REAL product SMILES from your paper
    new_products = [
        {
            "name_in_paper": "4-methoxybiphenyl",
            "smiles_raw": "COc1ccc(cc1)c2ccccc2"
        }
    ]

    rows = []
    for p in new_products:
        pid = next_product_id(existing_ids)
        existing_ids.add(pid)
        rows.append({
            "molecule_id": pid,
            "name_in_paper": p["name_in_paper"],
            "role": "product",
            "smiles_raw": p["smiles_raw"],
            "smiles_canonical": ""
        })

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(MOLECULES_PATH, index=False)

    print(f"Added {len(rows)} product molecules.")

if __name__ == "__main__":
    main()
