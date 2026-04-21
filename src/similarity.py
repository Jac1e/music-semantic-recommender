from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .data_loader import PROJECT_ROOT
from .preprocess import clean_text, prepare_lyrics
from .text_embed import load_embedding_model


DEFAULT_NUMERIC_COLUMNS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

DEFAULT_RESULT_COLUMNS = ["track_name", "artists", "track_genre"]
DEFAULT_LYRICS_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "lyrics_embeddings.npy"


@dataclass
class MusicRecommender:
    df: pd.DataFrame
    features: np.ndarray
    model: SentenceTransformer
    numeric_feature_count: int
    result_columns: list[str]

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        numeric_columns: list[str] | None = None,
        result_columns: list[str] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        lyrics_embeddings_path: str | Path = DEFAULT_LYRICS_EMBEDDINGS_PATH,
    ) -> "MusicRecommender":
        numeric_columns = numeric_columns or DEFAULT_NUMERIC_COLUMNS
        result_columns = result_columns or DEFAULT_RESULT_COLUMNS

        prepared = prepare_lyrics(df)
        numeric_data = prepared[numeric_columns].fillna(0)

        scaler = StandardScaler()
        numeric_features = scaler.fit_transform(numeric_data)

        embeddings_path = Path(lyrics_embeddings_path).resolve()
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Precomputed lyrics embeddings not found: {embeddings_path}. "
                "Run the embedding precompute step before fitting MusicRecommender."
            )

        text_features = np.load(embeddings_path)
        if text_features.ndim != 2:
            raise ValueError(
                f"Expected 2D lyrics embeddings, got shape {text_features.shape} at {embeddings_path}."
            )
        if text_features.shape[0] != len(prepared):
            raise ValueError(
                "Lyrics embedding row count does not match prepared dataset rows. "
                f"Embeddings rows={text_features.shape[0]}, prepared rows={len(prepared)}. "
                "Regenerate the .npy file from the same processed dataset."
            )

        model = load_embedding_model(model_name)

        features = np.hstack([numeric_features, text_features])
        return cls(
            df=prepared,
            features=features,
            model=model,
            numeric_feature_count=numeric_features.shape[1],
            result_columns=result_columns,
        )

    def search(self, query: str | None = None, song_idx: int | None = None, k: int = 5) -> pd.DataFrame:
        if query is None and song_idx is None:
            return pd.DataFrame()

        if query is not None:
            query_vec = self.model.encode([clean_text(query)], convert_to_numpy=True)
            query_vec = np.hstack([np.zeros((1, self.numeric_feature_count)), query_vec])
        else:
            if song_idx is None or song_idx < 0 or song_idx >= len(self.df):
                raise IndexError(f"song_idx must be between 0 and {len(self.df) - 1}")
            query_vec = self.features[song_idx].reshape(1, -1)

        scores = cosine_similarity(query_vec, self.features)[0]
        idx = np.argsort(scores)[::-1][:k]

        columns = [col for col in self.result_columns if col in self.df.columns]
        result = self.df.iloc[idx][columns].copy()
        result["score"] = scores[idx]
        return result.reset_index(drop=True)
