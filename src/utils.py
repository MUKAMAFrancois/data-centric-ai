"""Shared helpers: config loading, seeding, and resumable result logging."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed every RNG we use so that runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch is only needed for the BERT parts (Colab)


def append_result(row: dict, path: str, columns: list | None = None) -> None:
    """Append one experiment row to a CSV, creating it (with header) if needed.

    If `columns` is given, the row is written in exactly that column order and
    missing keys become NaN, so every row in the file has the same schema.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        row = {c: row.get(c, np.nan) for c in columns}
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_exists(path: str, **keys) -> bool:
    """True if a row matching all keys is already logged (lets Colab resume)."""
    path = Path(path)
    if not path.exists():
        return False
    df = pd.read_csv(path)
    mask = np.ones(len(df), dtype=bool)
    for k, v in keys.items():
        if k not in df.columns:
            return False
        col = df[k]
        if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
            mask &= np.isclose(pd.to_numeric(col, errors="coerce"), float(v)).astype(bool)
        else:
            mask &= (col.astype(str) == str(v)).to_numpy()
    return bool(mask.any())
