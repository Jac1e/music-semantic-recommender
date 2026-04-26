"""Data preprocessing: cleaning, deduplication, feature scaling.

Provides both low-level helpers (``clean_text``, ``prepare_lyrics``) and a
high-level ``preprocess_dataframe`` pipeline that deduplicates rows,
engineers lyrics-derived features, and standardises audio features.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import AUDIO_FEATURE_COLS, SCALED_FEATURE_COLS


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    dedup_col: str = "songNumber",
    audio_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """End-to-end preprocessing pipeline.

    Steps
    -----
    1. Deduplicate rows by *dedup_col* (keep first occurrence).
    2. Add ``lyrics_missing`` (bool) and ``lyrics_length`` (int) columns.
    3. StandardScaler on audio features → ``*_scaled`` columns.
    4. Build a *feature_df* containing only the scaled columns, and a
       *modeling_df* that merges metadata + scaled features + lyrics info.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or merged DataFrame (e.g. ``spotify_with_lyrics.csv``).
    dedup_col : str
        Column used for deduplication.
    audio_features : list[str] or None
        Audio feature columns to scale.  Defaults to
        ``utils.AUDIO_FEATURE_COLS``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]
        ``(cleaned_df, feature_df, modeling_df, scaler)``

        - **cleaned_df** — deduplicated data with ``lyrics_missing`` and
          ``lyrics_length`` columns added.
        - **feature_df** — one ``*_scaled`` column per audio feature.
        - **modeling_df** — cleaned_df merged with the scaled features.
        - **scaler** — the fitted ``StandardScaler`` instance.
    """
    if audio_features is None:
        audio_features = AUDIO_FEATURE_COLS

    # 1. Deduplicate ---------------------------------------------------------
    cleaned = df.drop_duplicates(subset=[dedup_col], keep="first").copy()
    cleaned = cleaned.reset_index(drop=True)

    # 2. Lyrics-derived features ---------------------------------------------
    lyrics_col = "lyrics"
    if lyrics_col in cleaned.columns:
        cleaned["lyrics_missing"] = cleaned[lyrics_col].isna() | (
            cleaned[lyrics_col].astype(str).str.strip() == ""
        )
        cleaned["lyrics_length"] = (
            cleaned[lyrics_col]
            .fillna("")
            .astype(str)
            .str.len()
        )
    else:
        cleaned["lyrics_missing"] = True
        cleaned["lyrics_length"] = 0

    # 3. Scale audio features ------------------------------------------------
    present_features = [c for c in audio_features if c in cleaned.columns]
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(cleaned[present_features].fillna(0))

    scaled_col_names = [f"{c}_scaled" for c in present_features]
    feature_df = pd.DataFrame(scaled_values, columns=scaled_col_names, index=cleaned.index)

    # 4. Build modeling_df ---------------------------------------------------
    modeling_df = pd.concat([cleaned, feature_df], axis=1)

    return cleaned, feature_df, modeling_df, scaler
