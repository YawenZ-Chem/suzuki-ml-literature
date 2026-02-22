from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from featurize import Vocab, featurize_row
from model import ECFPNN


class SuzukiDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocab):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x_fp, x_cat, y = featurize_row(row, self.vocab)
        return (
            torch.from_numpy(x_fp),
            torch.from_numpy(x_cat),
            torch.tensor(y, dtype=torch.long),
        )


def confusion_matrix(y_true, y_pred, n=3):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--valid_csv", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.valid_csv)

    vocab = Vocab.from_dataframe(train_df)

    train_ds = SuzukiDataset(train_df, vocab)
    valid_ds = SuzukiDataset(valid_df, vocab)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    model = ECFPNN(
        fp_dim=256 * 3,
        n_cat_catalyst=len(vocab.catalyst2id),
        n_cat_base=len(vocab.base2id),
        n_cat_solvent=len(vocab.solvent2id),
        emb_dim=16,
        hidden=256,
        n_classes=3,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x_fp, x_cat, y in train_loader:
            x_fp, x_cat, y = x_fp.to(device), x_cat.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x_fp, x_cat)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * y.size(0)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x_fp, x_cat, y in valid_loader:
                x_fp, x_cat = x_fp.to(device), x_cat.to(device)
                logits = model(x_fp, x_cat)
                pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                y_pred.extend(pred)
                y_true.extend(y.numpy().tolist())

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        acc = (y_true_np == y_pred_np).mean() if len(y_true_np) else float("nan")
        cm = confusion_matrix(y_true_np, y_pred_np, n=3)

        print(f"epoch {epoch:02d} | train_loss={total_loss/len(train_ds):.4f} | valid_acc={acc:.3f}")
        print("confusion matrix (rows=true, cols=pred):\n", cm)

    outdir = Path("models")
    outdir.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": {
                "catalyst": vocab.catalyst2id,
                "base": vocab.base2id,
                "solvent": vocab.solvent2id,
            },
        },
        outdir / "ecfpnn_3bin.pt",
    )
    print("saved:", outdir / "ecfpnn_3bin.pt")


if __name__ == "__main__":
    main()
