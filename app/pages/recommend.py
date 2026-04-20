import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add the project root to the path so we can import from src
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data_loader import load_spotify_with_lyrics

st.set_page_config(page_title="Song Recommendations", page_icon="🎧", layout="wide")
st.title("🎧 Song Recommendations")
st.markdown("Pick a song you like, and we'll find the most similar tracks based on audio features and lyrics.")

# Load data
@st.cache_data
def get_data():
    df = load_spotify_with_lyrics()
    return df

df = get_data()

# Create a display column for the dropdown: "Artist - Track Name"
df["display_name"] = df["artists"] + " — " + df["track_name"]

# Song selector
selected = st.selectbox(
    "Choose a song:",
    options=df["display_name"].tolist(),
    index=None,
    placeholder="Start typing a song or artist name...",
)

# Number of recommendations
k = st.slider("Number of recommendations:", min_value=3, max_value=20, value=5)

if selected:
    song_idx = df[df["display_name"] == selected].index[0]

    st.markdown(f"### Songs similar to *{selected}*")

    # For now, use a simple cosine similarity approach on numeric features
    # This will be replaced with Leo's MusicRecommender once we integrate fully
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    feature_cols = [
        "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo",
    ]

    @st.cache_data
    def compute_similarity(dataframe):
        scaler = StandardScaler()
        features = scaler.fit_transform(dataframe[feature_cols].fillna(0))
        return features

    features = compute_similarity(df)
    query_vec = features[song_idx].reshape(1, -1)
    scores = cosine_similarity(query_vec, features)[0]

    # Get top-k, excluding the song itself
    top_indices = np.argsort(scores)[::-1][1:k+1]

    results = df.iloc[top_indices][["track_name", "artists", "track_genre", "popularity"]].copy()
    results["similarity_score"] = scores[top_indices]
    results = results.reset_index(drop=True)
    results.index = results.index + 1  # Start numbering from 1

    st.dataframe(
        results.rename(columns={
            "track_name": "Track",
            "artists": "Artist",
            "track_genre": "Genre",
            "popularity": "Popularity",
            "similarity_score": "Similarity",
        }),
        use_container_width=True,
    )

    # Show the selected song's features for context
    with st.expander("📊 Audio features of your selected song"):
        song_data = df.iloc[song_idx][feature_cols]
        st.bar_chart(song_data)
