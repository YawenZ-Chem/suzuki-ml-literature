from __future__ import annotations
import torch
import torch.nn as nn


class ECFPNN(nn.Module):
    def __init__(
        self,
        fp_dim: int,
        n_cat_catalyst: int,
        n_cat_base: int,
        n_cat_solvent: int,
        emb_dim: int = 16,
        hidden: int = 256,
        n_classes: int = 3,
    ):
        super().__init__()
        self.emb_cat = nn.Embedding(n_cat_catalyst, emb_dim)
        self.emb_base = nn.Embedding(n_cat_base, emb_dim)
        self.emb_sol = nn.Embedding(n_cat_solvent, emb_dim)

        in_dim = fp_dim + 3 * emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x_fp: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        e1 = self.emb_cat(x_cat[:, 0])
        e2 = self.emb_base(x_cat[:, 1])
        e3 = self.emb_sol(x_cat[:, 2])
        x = torch.cat([x_fp, e1, e2, e3], dim=1)
        return self.mlp(x)
