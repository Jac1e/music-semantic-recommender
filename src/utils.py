"""Shared helper functions and constants used across the project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    """Return the absolute path to the repository root."""
    return PROJECT_ROOT


# ---------------------------------------------------------------------------
# Feature column constants
# ---------------------------------------------------------------------------

AUDIO_FEATURE_COLS: list[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "explicit",
]

SCALED_FEATURE_COLS: list[str] = [f"{col}_scaled" for col in AUDIO_FEATURE_COLS]

METADATA_COLS: list[str] = [
    "songNumber",
    "artists",
    "album_name",
    "track_name",
    "popularity",
    "duration",
    "key",
    "mode",
    "time_signature",
    "track_genre",
    "lyrics",
]


# ---------------------------------------------------------------------------
# Quick data loaders
# ---------------------------------------------------------------------------

def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both processed CSVs (spotify_with_lyrics and song_feature_vectors).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (spotify_with_lyrics_df, song_feature_vectors_df)
    """
    processed_dir = PROJECT_ROOT / "data" / "processed"
    swl = pd.read_csv(processed_dir / "spotify_with_lyrics.csv")
    sfv = pd.read_csv(processed_dir / "song_feature_vectors.csv")
    return swl, sfv
