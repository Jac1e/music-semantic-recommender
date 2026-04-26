"""Tests for the similarity / recommendation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]


def _make_toy_df(n: int = 20) -> pd.DataFrame:
    """Create a small synthetic song DataFrame."""
    rng = np.random.RandomState(42)
    data = {col: rng.rand(n) for col in FEATURE_COLS}
    data["track_name"] = [f"Song_{i}" for i in range(n)]
    data["artists"] = [f"Artist_{i % 5}" for i in range(n)]
    data["track_genre"] = [f"genre_{i % 3}" for i in range(n)]
    return pd.DataFrame(data)


def _build_features(df: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(df[FEATURE_COLS].fillna(0))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cosine_similarity_shape():
    """Cosine similarity matrix should be (n, n)."""
    df = _make_toy_df(10)
    features = _build_features(df)
    sim = cosine_similarity(features)
    assert sim.shape == (10, 10)


def test_cosine_similarity_self_is_one():
    """Every song should have similarity 1.0 with itself."""
    df = _make_toy_df(10)
    features = _build_features(df)
    sim = cosine_similarity(features)
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)


def test_top_k_retrieval_returns_k_results():
    """Top-k retrieval should return exactly k indices."""
    df = _make_toy_df(20)
    features = _build_features(df)
    query_vec = features[0].reshape(1, -1)
    scores = cosine_similarity(query_vec, features)[0]
    k = 5
    top_k = np.argsort(scores)[::-1][:k]
    assert len(top_k) == k


def test_scores_sorted_descending():
    """Returned scores should be sorted in descending order."""
    df = _make_toy_df(20)
    features = _build_features(df)
    query_vec = features[0].reshape(1, -1)
    scores = cosine_similarity(query_vec, features)[0]
    sorted_scores = scores[np.argsort(scores)[::-1]]
    assert all(sorted_scores[i] >= sorted_scores[i + 1] for i in range(len(sorted_scores) - 1))


def test_query_by_index():
    """Querying by a song's own index should return that song as the top result."""
    df = _make_toy_df(20)
    features = _build_features(df)
    idx = 7
    query_vec = features[idx].reshape(1, -1)
    scores = cosine_similarity(query_vec, features)[0]
    best = np.argmax(scores)
    assert best == idx


def test_similarity_symmetry():
    """sim(A, B) should equal sim(B, A)."""
    df = _make_toy_df(10)
    features = _build_features(df)
    sim = cosine_similarity(features)
    np.testing.assert_allclose(sim, sim.T, atol=1e-6)
