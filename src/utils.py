"""Shared helpers: config loading, seeding, and resumable result logging."""
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
        pass  # torch isn't needed until Step 5


def append_result(row: dict, path: str) -> None:
    """Append one experiment row to the CSV, creating it if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
        mask &= (df[k].astype(str) == str(v)).to_numpy()
    return bool(mask.any())