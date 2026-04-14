import re

import pandas as pd


def clean_text(value: object) -> str:
    """Normalize text for embedding and search."""
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def prepare_lyrics(
    df: pd.DataFrame,
    lyrics_col: str = "lyrics",
    output_col: str = "lyrics_clean",
) -> pd.DataFrame:
    """Add a cleaned lyrics column and drop rows without lyrics."""
    prepared = df.copy()
    prepared[output_col] = prepared[lyrics_col].apply(clean_text)
    prepared = prepared[prepared[output_col] != ""].copy()
    return prepared.reset_index(drop=True)
