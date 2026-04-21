from __future__ import annotations

from pathlib import Path
import pandas as pd


INDEX_LIKE_COLUMNS = {"", "index", "Unnamed: 0"}


def clean_credit_data(df: pd.DataFrame, target: str = "Limit") -> pd.DataFrame:
    df_clean = df.copy()

    cols_to_drop = [col for col in df_clean.columns if col in INDEX_LIKE_COLUMNS]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)

    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    if target and target in df_clean.columns:
        if df_clean[target].isna().any():
            raise ValueError(f"A coluna target '{target}' possui valores nulos.")

    return df_clean


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output
