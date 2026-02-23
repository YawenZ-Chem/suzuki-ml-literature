from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

LABEL_MAP_3BIN = {"low": 0, "mid": 1, "high": 2}


def morgan_fp(smiles: str, radius: int = 2, nbits: int = 256) -> np.ndarray:
    """Return Morgan fingerprint as {0,1} numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(bv, arr)
    return arr


@dataclass
class Vocab:
    catalyst2id: Dict[str, int]
    base2id: Dict[str, int]
    solvent2id: Dict[str, int]
    unk_id: int = 0

    @staticmethod
    def build(values: List[str]) -> Dict[str, int]:
        uniq = sorted(set(v.strip() for v in values if v and v.strip()))
        return {"UNK": 0, **{v: i + 1 for i, v in enumerate(uniq)}}

    @classmethod
    def from_dataframe(cls, df) -> "Vocab":
        return cls(
            catalyst2id=cls.build(df["catalyst"].astype(str).tolist()),
            base2id=cls.build(df["base"].astype(str).tolist()),
            solvent2id=cls.build(df["solvent"].astype(str).tolist()),
        )

    def encode(self, kind: str, value: str) -> int:
        value = (value or "").strip()
        if kind == "catalyst":
            return self.catalyst2id.get(value, self.unk_id)
        if kind == "base":
            return self.base2id.get(value, self.unk_id)
        if kind == "solvent":
            return self.solvent2id.get(value, self.unk_id)
        raise KeyError(f"Unknown kind: {kind}")


def featurize_row(row, vocab: Vocab) -> Tuple[np.ndarray, np.ndarray, int]:
    """x_fp: (768,), x_cat: (3,), y: int class 0/1/2."""
    fp_halide = morgan_fp(row["smiles_halide"])
    fp_boronic = morgan_fp(row["smiles_boronic"])
    fp_prod = morgan_fp(row["smiles_product"])
    x_fp = np.concatenate([fp_halide, fp_boronic, fp_prod], axis=0).astype(np.float32)

    x_cat = np.array(
        [
            vocab.encode("catalyst", row["catalyst"]),
            vocab.encode("base", row["base"]),
            vocab.encode("solvent", row["solvent"]),
        ],
        dtype=np.int64,
    )

    label = str(row["label_3bin"]).strip()
    if label not in LABEL_MAP_3BIN:
        raise ValueError(f"label_3bin must be one of {list(LABEL_MAP_3BIN)}, got: {label}")
    y = LABEL_MAP_3BIN[label]
    return x_fp, x_cat, y
