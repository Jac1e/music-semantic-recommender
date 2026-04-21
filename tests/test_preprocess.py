
from __future__ import annotations

import pandas as pd

from src.preprocess import preprocess_dataframe


def make_toy_df():
    return pd.DataFrame(
        {
            "songNumber": [1, 1, 2],
            "artists": ["A", "A", "B"],
            "album_name": ["X", "X", "Y"],
            "track_name": ["Song1", "Song1", "Song2"],
            "popularity": [50, 50, 70],
            "duration": [180000, 180000, 200000],
            "explicit": [0, 0, 1],
            "danceability": [0.5, 0.5, 0.8],
            "energy": [0.6, 0.6, 0.7],
            "key": [1, 1, 5],
            "loudness": [-5.0, -5.0, -4.0],
            "mode": [1, 1, 1],
            "speechiness": [0.03, 0.03, 0.05],
            "acousticness": [0.1, 0.1, 0.2],
            "instrumentalness": [0.0, 0.0, 0.0],
            "liveness": [0.1, 0.1, 0.2],
            "valence": [0.4, 0.4, 0.6],
            "tempo": [120.0, 120.0, 130.0],
            "time_signature": [4, 4, 4],
            "track_genre": ["pop", "pop", "rock"],
            "lyrics": ["hello world", "hello world", ""],
        }
    )


def test_preprocess_removes_duplicates():
    df = make_toy_df()
    cleaned_df, feature_df, modeling_df, scaler = preprocess_dataframe(df)
    assert len(cleaned_df) == 2
    assert len(feature_df) == 2
    assert len(modeling_df) == 2


def test_preprocess_adds_lyrics_columns():
    df = make_toy_df()
    cleaned_df, _, modeling_df, _ = preprocess_dataframe(df)
    assert "lyrics_missing" in cleaned_df.columns
    assert "lyrics_length" in cleaned_df.columns
    assert "lyrics_missing" in modeling_df.columns
    assert "lyrics_length" in modeling_df.columns
