from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DEFAULT_FEATURE_COLS = [
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


def compute_pca_projection(df, feature_cols=None, n_components=2):
    """
    Project high-dimensional song audio features into a lower-dimensional PCA space.
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