from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .preprocess import clean_text, prepare_lyrics
from .text_embed import embed_texts, load_embedding_model


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
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> "MusicRecommender":
        numeric_columns = numeric_columns or DEFAULT_NUMERIC_COLUMNS
        result_columns = result_columns or DEFAULT_RESULT_COLUMNS

        prepared = prepare_lyrics(df)
        numeric_data = prepared[numeric_columns].fillna(0)

        scaler = StandardScaler()
        numeric_features = scaler.fit_transform(numeric_data)

        model = load_embedding_model(model_name)
        text_features = embed_texts(
            prepared["lyrics_clean"].tolist(),
            model=model,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

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
