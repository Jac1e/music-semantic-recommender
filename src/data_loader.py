from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_with_lyrics.csv"


def load_spotify_with_lyrics(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the processed Spotify dataset with lyrics."""
    return pd.read_csv(path)
