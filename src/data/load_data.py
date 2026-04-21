from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_raw_data(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    return pd.read_csv(csv_path)
