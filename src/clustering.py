"""Clustering methods for grouping songs in the latent (embedding) space.

Provides K-Means and Gaussian Mixture Model clustering, along with
evaluation utilities (elbow plot, silhouette analysis, BIC comparison)
to help choose the optimal number of clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------

def fit_kmeans(
    features: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 42,
    **kwargs,
) -> KMeans:
    """Fit a K-Means model on the feature matrix.

    Parameters
    ----------
    features : np.ndarray
        2-D array of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed for reproducibility.
    **kwargs
        Extra keyword arguments forwarded to ``sklearn.cluster.KMeans``.

    Returns
    -------
    KMeans
        The fitted KMeans estimator.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, **kwargs)
    km.fit(features)
    return km


# ---------------------------------------------------------------------------
# Gaussian Mixture Model
# ---------------------------------------------------------------------------

def fit_gmm(
    features: np.ndarray,
    n_components: int = 8,
    covariance_type: str = "full",
    random_state: int = 42,
    **kwargs,
) -> GaussianMixture:
    """Fit a Gaussian Mixture Model on the feature matrix.

    Parameters
    ----------
    features : np.ndarray
        2-D array of shape (n_samples, n_features).
    n_components : int
        Number of mixture components (clusters).
    covariance_type : str
        One of ``{"full", "tied", "diag", "spherical"}``.
    random_state : int
        Random seed for reproducibility.
    **kwargs
        Extra keyword arguments forwarded to
        ``sklearn.mixture.GaussianMixture``.

    Returns
    -------
    GaussianMixture
        The fitted GaussianMixture estimator.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        **kwargs,
    )
    gmm.fit(features)
    return gmm


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def kmeans_elbow(
    features: np.ndarray,
    k_range: range | list[int] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute inertia for a range of *k* values (elbow method).

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    k_range : range or list[int], optional
        Cluster counts to evaluate.  Defaults to ``range(2, 11)``.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns ``["k", "inertia"]``.
    """
    if k_range is None:
        k_range = range(2, 11)

    records: list[dict] = []
    for k in k_range:
        km = fit_kmeans(features, n_clusters=k, random_state=random_state)
        records.append({"k": k, "inertia": km.inertia_})
    return pd.DataFrame(records)


def silhouette_analysis(
    features: np.ndarray,
    k_range: range | list[int] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute silhouette scores for a range of *k* values.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    k_range : range or list[int], optional
        Cluster counts to evaluate.  Defaults to ``range(2, 11)``.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns ``["k", "silhouette_score"]``.
    """
    if k_range is None:
        k_range = range(2, 11)

    records: list[dict] = []
    for k in k_range:
        km = fit_kmeans(features, n_clusters=k, random_state=random_state)
        score = silhouette_score(features, km.labels_)
        records.append({"k": k, "silhouette_score": score})
    return pd.DataFrame(records)


def gmm_bic_aic(
    features: np.ndarray,
    k_range: range | list[int] | None = None,
    covariance_type: str = "full",
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute BIC and AIC for a range of component counts.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    k_range : range or list[int], optional
        Component counts to evaluate.  Defaults to ``range(2, 11)``.
    covariance_type : str
        Covariance type for every GMM fit.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns ``["k", "bic", "aic"]``.
    """
    if k_range is None:
        k_range = range(2, 11)

    records: list[dict] = []
    for k in k_range:
        gmm = fit_gmm(
            features,
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        records.append({"k": k, "bic": gmm.bic(features), "aic": gmm.aic(features)})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Cluster-assignment helpers
# ---------------------------------------------------------------------------

def assign_clusters_kmeans(
    df: pd.DataFrame,
    features: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 42,
    label_column: str = "kmeans_cluster",
) -> tuple[pd.DataFrame, KMeans]:
    """Add a K-Means cluster label column to *df*.

    Returns the augmented DataFrame **and** the fitted KMeans model.
    """
    km = fit_kmeans(features, n_clusters=n_clusters, random_state=random_state)
    result = df.copy()
    result[label_column] = km.labels_
    return result, km


def assign_clusters_gmm(
    df: pd.DataFrame,
    features: np.ndarray,
    n_components: int = 8,
    covariance_type: str = "full",
    random_state: int = 42,
    label_column: str = "gmm_cluster",
    proba_prefix: str | None = "gmm_proba_",
) -> tuple[pd.DataFrame, GaussianMixture]:
    """Add a GMM cluster label column (and optional probabilities) to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Song metadata / features DataFrame (must have same row count as
        *features*).
    features : np.ndarray
        Feature matrix used for fitting.
    n_components : int
        Number of Gaussian components.
    covariance_type : str
        Covariance parameterisation.
    random_state : int
        Random seed.
    label_column : str
        Name of the hard-assignment column.
    proba_prefix : str or None
        If not ``None``, add per-component probability columns named
        ``{proba_prefix}0``, ``{proba_prefix}1``, … to the DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, GaussianMixture]
        Augmented DataFrame and fitted GMM.
    """
    gmm = fit_gmm(
        features,
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    result = df.copy()
    result[label_column] = gmm.predict(features)

    if proba_prefix is not None:
        probas = gmm.predict_proba(features)
        for i in range(probas.shape[1]):
            result[f"{proba_prefix}{i}"] = probas[:, i]

    return result, gmm


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Container for clustering results that keeps everything together."""

    labels: np.ndarray
    model: KMeans | GaussianMixture
    method: str  # "kmeans" or "gmm"
    n_clusters: int
    silhouette: float | None = None
    inertia: float | None = None  # K-Means only
    bic: float | None = None      # GMM only
    aic: float | None = None      # GMM only

    def summary(self) -> dict:
        """Return a plain-dict summary of the clustering result."""
        out = {
            "method": self.method,
            "n_clusters": self.n_clusters,
            "silhouette_score": self.silhouette,
        }
        if self.method == "kmeans":
            out["inertia"] = self.inertia
        else:
            out["bic"] = self.bic
            out["aic"] = self.aic
        return out


def cluster(
    features: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 8,
    covariance_type: str = "full",
    random_state: int = 42,
    compute_silhouette: bool = True,
) -> ClusterResult:
    """One-call clustering with automatic metric computation.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    method : str
        ``"kmeans"`` or ``"gmm"``.
    n_clusters : int
        Number of clusters / components.
    covariance_type : str
        GMM covariance type (ignored for K-Means).
    random_state : int
        Random seed.
    compute_silhouette : bool
        Whether to compute the silhouette score (can be slow on very large
        datasets).

    Returns
    -------
    ClusterResult
    """
    if method == "kmeans":
        model = fit_kmeans(features, n_clusters=n_clusters, random_state=random_state)
        labels = model.labels_
        inertia = model.inertia_
        bic = aic = None
    elif method == "gmm":
        model = fit_gmm(
            features,
            n_components=n_clusters,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        labels = model.predict(features)
        inertia = None
        bic = model.bic(features)
        aic = model.aic(features)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'kmeans' or 'gmm'.")

    sil = None
    if compute_silhouette and len(set(labels)) > 1:
        sil = float(silhouette_score(features, labels))

    return ClusterResult(
        labels=labels,
        model=model,
        method=method,
        n_clusters=n_clusters,
        silhouette=sil,
        inertia=inertia,
        bic=bic,
        aic=aic,
    )
