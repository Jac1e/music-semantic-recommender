import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data_loader import load_spotify_with_lyrics

st.set_page_config(page_title="Explore Songs", page_icon="📊", layout="wide")
st.title("📊 Explore the Song Space")
st.markdown("See how songs relate to each other in a 2D space, colored by genre or cluster.")

@st.cache_data
def get_data():
    df = load_spotify_with_lyrics()
    return df

df = get_data()

feature_cols = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

@st.cache_data
def compute_2d_projection(dataframe):
    """
    Placeholder: using sklearn PCA for now.
    TODO: Replace code from src/reduction.py
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    features = scaler.fit_transform(dataframe[feature_cols].fillna(0))

    pca = PCA(n_components=2)
    coords = pca.fit_transform(features)
    return coords, pca.explained_variance_ratio_

coords, variance_ratio = compute_2d_projection(df)
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

# Color by options
color_option = st.radio(
    "Color by:",
    ["Genre", "Cluster (coming soon)"],
    horizontal=True,
)

if color_option == "Genre":
    # Let user filter genres to reduce clutter
    all_genres = sorted(df["track_genre"].unique())
    selected_genres = st.multiselect(
        "Filter genres (leave empty to show all):",
        options=all_genres,
        default=None,
    )

    if selected_genres:
        plot_df = df[df["track_genre"].isin(selected_genres)].copy()
    else:
        # Show a random sample to keep the plot responsive
        plot_df = df.sample(n=min(5000, len(df)), random_state=42).copy()
        st.caption("Showing a random sample of 5,000 songs. Use the genre filter to focus.")

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="track_genre",
        hover_data=["track_name", "artists", "track_genre"],
        title="Songs projected into 2D space (PCA)",
        labels={"x": f"PC1 ({variance_ratio[0]:.1%} variance)",
                "y": f"PC2 ({variance_ratio[1]:.1%} variance)"},
        opacity=0.6,
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Cluster visualization will be available once clustering module is integrated.")

st.markdown("---")
with st.expander("ℹ️ About this visualization"):
    st.markdown(
        f"""
        This plot projects each song's audio features (danceability, energy, 
        valence, tempo, etc.) from 11 dimensions down to 2 using PCA. 
        
        **Current implementation:** sklearn PCA (placeholder)  
        **Coming soon:** PCA/Deep autoencoder
        
        The first two principal components explain 
        {variance_ratio[0]:.1%} and {variance_ratio[1]:.1%} of the total variance respectively.
        """
    )
