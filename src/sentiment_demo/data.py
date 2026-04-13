from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"flagged_sentiment"}


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df


def resolve_text_column(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise ValueError(
        f"Neither text column '{preferred}' nor fallback '{fallback}' exists in the dataset."
    )
