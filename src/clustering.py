"""Clustering methods for grouping songs in the latent (embedding) space.

Provides K-Means and Gaussian Mixture Model clustering, along with
evaluation utilities (elbow plot, silhouette analysis, BIC comparison)
to help choose the optimal number of clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
def fit_kmeans(
    features: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 42,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 10,
    **kwargs,
):
    """Fit a K-Means model from scratch using numpy.

    Runs n_init independent trials and keeps the result with lowest inertia.

    Parameters
    ----------
    features : np.ndarray
        2-D array of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed for reproducibility.
    max_iter : int
        Maximum number of iterations per trial.
    tol : float
        Convergence threshold — stops early if centroids move less than this.
    n_init : int
        Number of independent random restarts. Best result is kept.

    Returns
    -------
    result
        An object with .labels_ , .cluster_centers_ , and .inertia_
        attributes — matching sklearn KMeans interface so the rest of
        the codebase works unchanged.
    """
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class _KMeansResult:
        labels_: np.ndarray
        cluster_centers_: np.ndarray
        inertia_: float
        n_clusters: int

    rng = np.random.RandomState(random_state)

    best_labels = None
    best_centers = None
    best_inertia = float("inf")

    for _ in range(n_init):
        # Step 1: randomly pick k samples as initial centroids
        init_idx = rng.choice(len(features), size=n_clusters, replace=False)
        centers = features[init_idx].copy().astype(float)

        labels = np.zeros(len(features), dtype=int)

        for _ in range(max_iter):
            # Step 2: assign each point to nearest centroid
            # Shape: (n_samples, n_clusters)
            diffs = features[:, np.newaxis, :] - centers[np.newaxis, :, :]
            distances = np.sqrt((diffs ** 2).sum(axis=2))
            new_labels = np.argmin(distances, axis=1)

            # Step 3: recompute centroids as mean of assigned points
            new_centers = np.array([
                features[new_labels == k].mean(axis=0) if (new_labels == k).any()
                else centers[k]  # keep old center if cluster is empty
                for k in range(n_clusters)
            ])

            # Step 4: check convergence
            center_shift = np.sqrt(((new_centers - centers) ** 2).sum(axis=1)).max()
            labels = new_labels
            centers = new_centers

            if center_shift < tol:
                break

        # Compute inertia (sum of squared distances to assigned centroid)
        inertia = sum(
            ((features[labels == k] - centers[k]) ** 2).sum()
            for k in range(n_clusters)
            if (labels == k).any()
        )

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return _KMeansResult(
        labels_=best_labels,
        cluster_centers_=best_centers,
        inertia_=best_inertia,
        n_clusters=n_clusters,
    )
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
) -> tuple[pd.DataFrame, object]:
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
