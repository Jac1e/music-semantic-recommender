from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DEFAULT_FEATURE_COLS = [
    "danceability_scaled",
    "energy_scaled",
    "loudness_scaled",
    "speechiness_scaled",
    "acousticness_scaled",
    "instrumentalness_scaled",
    "liveness_scaled",
    "valence_scaled",
    "tempo_scaled",
    "explicit_scaled",
]


def compute_pca_projection(df, feature_cols=None, n_components=2):
    """
    Project high-dimensional song feature vectors into a lower-dimensional PCA space.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing processed song features.
    feature_cols : list of str, optional
        Feature columns used for PCA. If None, the default scaled audio features are used.
    n_components : int
        Number of PCA components. Usually 2 or 3.

    Returns
    -------
    projected_df : pandas.DataFrame
        A copy of the input dataframe with PCA coordinate columns added.
    explained_variance_ratio : numpy.ndarray
        Variance explained by each PCA component.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    clean_df = df.dropna(subset=feature_cols).copy()
    X = clean_df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    for i in range(n_components):
        clean_df[f"pca_{i + 1}"] = X_pca[:, i]

    return clean_df, pca.explained_variance_ratio_