"""Tests for data loading and lyrics merging."""

from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.data_loader import load_spotify_with_lyrics, DEFAULT_DATA_PATH


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_returns_dataframe():
    """load_spotify_with_lyrics() should return a DataFrame."""
    df = load_spotify_with_lyrics()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_required_columns_present():
    """The loaded DataFrame must contain key metadata columns."""
    df = load_spotify_with_lyrics()
    required = ["track_name", "artists", "track_genre", "lyrics"]
    for col in required:
        assert col in df.columns, f"Missing required column: {col}"


def test_audio_feature_columns_present():
    """Standard Spotify audio features should be present."""
    df = load_spotify_with_lyrics()
    audio_features = [
        "danceability", "energy", "loudness",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo",
    ]
    for col in audio_features:
        assert col in df.columns, f"Missing audio feature column: {col}"


def test_no_nan_in_key_metadata():
    """track_name and artists should have no NaN values."""
    df = load_spotify_with_lyrics()
    assert df["track_name"].notna().all(), "Found NaN in track_name"
    assert df["artists"].notna().all(), "Found NaN in artists"


def test_default_data_path_exists():
    """The default CSV path should exist on disk."""
    assert DEFAULT_DATA_PATH.exists(), f"Default data path does not exist: {DEFAULT_DATA_PATH}"
