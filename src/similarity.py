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
# Default set of audio features used to represent each song.
# These are acoustic attributes extracted from the audio signal.
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

DEFAULT_RESULT_COLUMNS = ["track_name", "artists", "track_genre"]# Columns to include in the final recommendation output.
DEFAULT_LYRICS_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "lyrics_embeddings.npy"# Default file path for loading precomputed lyric embeddings.


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
    ) -> "MusicRecommender":# Use default configurations if none are provided
        numeric_columns = numeric_columns or DEFAULT_NUMERIC_COLUMNS
        result_columns = result_columns or DEFAULT_RESULT_COLUMNS

        prepared = prepare_lyrics(df)# Preprocess dataset
        numeric_data = prepared[numeric_columns].fillna(0)# Extract selected audio features and fill missing values

        scaler = StandardScaler() # Standardize audio features to ensure comparable scale
        numeric_features = scaler.fit_transform(numeric_data)

        embeddings_path = Path(lyrics_embeddings_path).resolve()# Resolve path to precomputed lyric embeddings
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Precomputed lyrics embeddings not found: {embeddings_path}. "
                "Run the embedding precompute step before fitting MusicRecommender."
            )

        text_features = np.load(embeddings_path)
        if text_features.ndim != 2:# Embedding should be 2D: each row represents a song, each column is a feature dimention
            raise ValueError(
                f"Expected 2D lyrics embeddings, got shape {text_features.shape} at {embeddings_path}."
            )
        if text_features.shape[0] != len(prepared):# Ensure the number of embeddings matches the number of songs
            raise ValueError(
                "Lyrics embedding row count does not match prepared dataset rows. "
                f"Embeddings rows={text_features.shape[0]}, prepared rows={len(prepared)}. "
                "Regenerate the .npy file from the same processed dataset."
            )

        model = load_embedding_model(model_name)# Load sentence-transformer model for encoding user queries
        # Combine audio features and lyric embeddings into a unified feature matrix, each row represents a song with both acoustic and semantic information
        features = np.hstack([numeric_features, text_features])
        return cls(
            df=prepared,# Preprocessed dataset
            features=features,# Combined feature matrix (audio + text)
            model=model,# Embedding model for query encoding
            numeric_feature_count=numeric_features.shape[1],# Number of audio features
            result_columns=result_columns, # Columns to display in results
        )

    def train_cross_modal_bridge(self):
        """Train a regression model to predict audio features from lyrics embeddings."""
        from sklearn.linear_model import Ridge
        self.bridge_model = Ridge(alpha=1.0)
        
        numeric_features = self.features[:, :self.numeric_feature_count]
        text_features = self.features[:, self.numeric_feature_count:]
        
        # Fit model: predict acoustic properties (Y) from semantic lyrics (X)
        self.bridge_model.fit(text_features, numeric_features)

    def search(
        self, 
        query: str | None = None, 
        song_indices: list[int] | int | None = None, 
        audio_query: str | None = None,
        k: int = 5
    ) -> pd.DataFrame:
        if not query and not song_indices and not audio_query:
            return pd.DataFrame()

        if query or audio_query:
            # 1. Handle Lyrics Query
            if query:
                query_text_vec = self.model.encode([clean_text(query)], convert_to_numpy=True)
            else:
                # model embedding dimension is features.shape[1] - numeric_feature_count
                emb_dim = self.features.shape[1] - self.numeric_feature_count
                query_text_vec = np.zeros((1, emb_dim))

            # 2. Handle Audio Vibe Query
            if audio_query:
                audio_query_text_vec = self.model.encode([clean_text(audio_query)], convert_to_numpy=True)
                if not hasattr(self, 'bridge_model'):
                    self.train_cross_modal_bridge()
                query_audio_pred = self.bridge_model.predict(audio_query_text_vec)
            else:
                query_audio_pred = np.zeros((1, self.numeric_feature_count))

            # Combine them
            query_vec = np.hstack([query_audio_pred, query_text_vec])
        else:
            if isinstance(song_indices, int):
                song_indices = [song_indices]
            
            valid_indices = [i for i in song_indices if 0 <= i < len(self.df)]
            if not valid_indices:
                raise IndexError(f"song_indices must be valid indices between 0 and {len(self.df) - 1}")
            
            # Compute the mean vector of the selected songs
            query_vec = np.mean([self.features[i] for i in valid_indices], axis=0).reshape(1, -1)
        # Compute cosine similarity between the query vector and all song feature vectors
        scores = cosine_similarity(query_vec, self.features)[0]
        idx = np.argsort(scores)[::-1][:k]# Get indices of top-k most similar songs (sorted in descending order)

        columns = [col for col in self.result_columns if col in self.df.columns]
        result = self.df.iloc[idx][columns].copy()# Retrieve the top-k songs from the dataset based on similarity ranking
        result["score"] = scores[idx]# Attach similarity scores to the result
        return result.reset_index(drop=True)
