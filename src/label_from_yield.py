import pandas as pd

def label_3bin(y):
    if pd.isna(y):
        return ""
    y = float(y)
    if y < 50:
        return "low"
    elif y < 90:
        return "mid"
    else:
        return "high"

def label_binary(y):
    if pd.isna(y):
        return ""
    y = float(y)
    return "good" if y >= 90 else "not_good"

def main(path_in):
    df = pd.read_csv(path_in)

    if "yield_percent" not in df.columns:
        raise ValueError("yield_percent column missing")

    df["label_3bin"] = df["yield_percent"].apply(label_3bin)
    df["label_binary"] = df["yield_percent"].apply(label_binary)

    if (df["label_3bin"] == "").any():
        print("Some rows have missing yield_percent.")
        raise SystemExit("Fill yield_percent first.")

    df.to_csv(path_in, index=False)
    print("Labels updated successfully.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path_in", required=True)
    args = ap.parse_args()
    main(args.path_in)
