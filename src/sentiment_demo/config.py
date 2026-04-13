from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_path: Path = Path("kenya_labeled_with_flagged_sentiment.csv")
    output_dir: Path = Path("artifacts")
    text_column: str = "preprocessed_text"
    fallback_text_column: str = "notes"
    target_column: str = "flagged_sentiment"
    random_state: int = 42
