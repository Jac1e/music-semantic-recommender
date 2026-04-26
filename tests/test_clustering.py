"""Tests for the clustering module (src/clustering.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.clustering import (
    fit_kmeans,
    fit_gmm,
    cluster,
    ClusterResult,
    kmeans_elbow,
    silhouette_analysis,
    gmm_bic_aic,
    assign_clusters_kmeans,
    assign_clusters_gmm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_features():
    """100 samples, 5 features — 3 well-separated blobs."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(34, 5) + np.array([5, 0, 0, 0, 0])
    c2 = rng.randn(33, 5) + np.array([0, 5, 0, 0, 0])
    c3 = rng.randn(33, 5) + np.array([0, 0, 5, 0, 0])
    return np.vstack([c1, c2, c3])


@pytest.fixture
def toy_df(toy_features):
    return pd.DataFrame({"name": [f"item_{i}" for i in range(len(toy_features))]})


# ---------------------------------------------------------------------------
# fit_kmeans
# ---------------------------------------------------------------------------

def test_fit_kmeans_returns_correct_labels(toy_features):
    km = fit_kmeans(toy_features, n_clusters=3)
    assert len(km.labels_) == len(toy_features)
    assert len(set(km.labels_)) == 3


def test_fit_kmeans_default_clusters(toy_features):
    km = fit_kmeans(toy_features)
    assert km.n_clusters == 8


# ---------------------------------------------------------------------------
# fit_gmm
# ---------------------------------------------------------------------------

def test_fit_gmm_returns_correct_components(toy_features):
    gmm = fit_gmm(toy_features, n_components=3)
    labels = gmm.predict(toy_features)
    assert len(labels) == len(toy_features)
    assert len(set(labels)) <= 3


# ---------------------------------------------------------------------------
# cluster() convenience function
# ---------------------------------------------------------------------------

def test_cluster_kmeans(toy_features):
    cr = cluster(toy_features, method="kmeans", n_clusters=3)
    assert isinstance(cr, ClusterResult)
    assert cr.method == "kmeans"
    assert cr.n_clusters == 3
    assert len(cr.labels) == len(toy_features)
    assert cr.inertia is not None
    assert cr.silhouette is not None


def test_cluster_gmm(toy_features):
    cr = cluster(toy_features, method="gmm", n_clusters=3)
    assert isinstance(cr, ClusterResult)
    assert cr.method == "gmm"
    assert cr.bic is not None
    assert cr.aic is not None
    assert cr.silhouette is not None


def test_cluster_invalid_method(toy_features):
    with pytest.raises(ValueError, match="Unknown method"):
        cluster(toy_features, method="dbscan")


# ---------------------------------------------------------------------------
# ClusterResult.summary()
# ---------------------------------------------------------------------------

def test_summary_kmeans(toy_features):
    cr = cluster(toy_features, method="kmeans", n_clusters=3)
    s = cr.summary()
    assert s["method"] == "kmeans"
    assert s["n_clusters"] == 3
    assert "inertia" in s
    assert "silhouette_score" in s


def test_summary_gmm(toy_features):
    cr = cluster(toy_features, method="gmm", n_clusters=3)
    s = cr.summary()
    assert s["method"] == "gmm"
    assert "bic" in s
    assert "aic" in s


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def test_kmeans_elbow_shape(toy_features):
    k_range = range(2, 6)
    result = kmeans_elbow(toy_features, k_range=k_range)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert list(result.columns) == ["k", "inertia"]


def test_silhouette_analysis_shape(toy_features):
    k_range = range(2, 6)
    result = silhouette_analysis(toy_features, k_range=k_range)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert list(result.columns) == ["k", "silhouette_score"]


def test_gmm_bic_aic_shape(toy_features):
    k_range = range(2, 5)
    result = gmm_bic_aic(toy_features, k_range=k_range)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert set(result.columns) == {"k", "bic", "aic"}


# ---------------------------------------------------------------------------
# assign_clusters helpers
# ---------------------------------------------------------------------------

def test_assign_clusters_kmeans(toy_features, toy_df):
    result_df, km = assign_clusters_kmeans(toy_df, toy_features, n_clusters=3)
    assert "kmeans_cluster" in result_df.columns
    assert len(result_df) == len(toy_df)
    assert len(set(result_df["kmeans_cluster"])) == 3


def test_assign_clusters_gmm(toy_features, toy_df):
    result_df, gmm = assign_clusters_gmm(toy_df, toy_features, n_components=3)
    assert "gmm_cluster" in result_df.columns
    assert len(result_df) == len(toy_df)
    # Check probability columns exist
    assert any(col.startswith("gmm_proba_") for col in result_df.columns)


def test_assign_clusters_gmm_no_proba(toy_features, toy_df):
    result_df, gmm = assign_clusters_gmm(
        toy_df, toy_features, n_components=3, proba_prefix=None,
    )
    assert "gmm_cluster" in result_df.columns
    assert not any(col.startswith("gmm_proba_") for col in result_df.columns)


# ---------------------------------------------------------------------------
# Silhouette on well-separated data should be high
# ---------------------------------------------------------------------------

def test_silhouette_high_for_clear_blobs(toy_features):
    cr = cluster(toy_features, method="kmeans", n_clusters=3)
    assert cr.silhouette is not None
    assert cr.silhouette > 0.5, f"Expected high silhouette for well-separated data, got {cr.silhouette}"
