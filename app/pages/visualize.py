import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data_loader import load_spotify_with_lyrics
from src.reduction import compute_pca_projection

st.set_page_config(page_title="Explore Songs", page_icon="📊", layout="wide")
st.title("📊 Explore the Song Space")
st.markdown("See how songs relate to each other in a 2D PCA space, colored by genre or cluster.")

@st.cache_data
def get_data():
    df = load_spotify_with_lyrics()
    return df

df = get_data()

@st.cache_data
def get_pca_projection(dataframe):
    projected_df, variance_ratio = compute_pca_projection(dataframe, n_components=2)
    return projected_df, variance_ratio

projected_df, variance_ratio = get_pca_projection(df)

# Use pca_1 and pca_2 as plot coordinates
projected_df["x"] = projected_df["pca_1"]
projected_df["y"] = projected_df["pca_2"]

# Color by options
color_option = st.radio(
    "Color by:",
    ["Genre", "Cluster (coming soon)"],
    horizontal=True,
)

if color_option == "Genre":
    all_genres = sorted(projected_df["track_genre"].dropna().unique())
    selected_genres = st.multiselect(
        "Filter genres (leave empty to show all):",
        options=all_genres,
        default=None,
    )

    if selected_genres:
        plot_df = projected_df[projected_df["track_genre"].isin(selected_genres)].copy()
    else:
        plot_df = projected_df.sample(n=min(5000, len(projected_df)), random_state=42).copy()
        st.caption("Showing a random sample of 5,000 songs. Use the genre filter to focus.")

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="track_genre",
        hover_data=["track_name", "artists", "track_genre"],
        title="Songs Projected into 2D PCA Space",
        labels={
            "x": f"PC1 ({variance_ratio[0]:.1%} variance)",
            "y": f"PC2 ({variance_ratio[1]:.1%} variance)",
        },
        opacity=0.6,
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Cluster visualization will be available once Michael's clustering module is integrated.")

st.markdown("---")
with st.expander("ℹ️ About this visualization"):
    st.markdown(
        f"""
        This plot uses PCA to project each song's processed audio feature vector 
        into a 2D space. Each point represents one song, and songs that appear 
        closer together have more similar audio feature patterns.

        **Current implementation:** PCA from `src/reduction.py`  
        **Feature representation:** scaled Spotify audio features  
        **Future extension:** cluster labels from the clustering module can be added on top of this PCA space.

        The first two principal components explain 
        {variance_ratio[0]:.1%} and {variance_ratio[1]:.1%} of the total variance respectively, 
        or {(variance_ratio[0] + variance_ratio[1]):.1%} combined.
        """
    )
    