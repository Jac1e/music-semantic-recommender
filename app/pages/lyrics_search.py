import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data_loader import load_spotify_with_lyrics

st.set_page_config(page_title="Lyrics Search", page_icon="🔍", layout="wide")
st.title("🔍 Mood-Based Lyrics Search")
st.markdown("Describe a mood or vibe, and we'll find songs whose lyrics match.")

@st.cache_data
def get_data():
    df = load_spotify_with_lyrics()
    # Only keep rows that have lyrics
    df_with_lyrics = df[df["lyrics"].notna()].reset_index(drop=True)
    return df_with_lyrics

df = get_data()

# Try to load the sentence-transformer model for real semantic search
# If it's not installed, fall back to simple keyword matching
@st.cache_resource
def load_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except ImportError:
        return None

@st.cache_data
def compute_lyrics_embeddings(_model, lyrics_list):
    """Embed all lyrics. This runs once and gets cached."""
    embeddings = _model.encode(lyrics_list, batch_size=32, show_progress_bar=False)
    return embeddings

model = load_model()

query = st.text_input(
    "Describe what you're looking for:",
    placeholder="e.g., sad heartbreak song, upbeat summer vibes, lonely night driving...",
)

k = st.slider("Number of results:", min_value=3, max_value=20, value=5)

if query and model is not None:
    with st.spinner("Searching through lyrics..."):
        # Embed all lyrics (cached after first run)
        lyrics_embeddings = compute_lyrics_embeddings(model, df["lyrics"].tolist())

        # Embed the query
        query_embedding = model.encode([query])

        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_embedding, lyrics_embeddings)[0]

        top_indices = np.argsort(scores)[::-1][:k]

        results = df.iloc[top_indices][["track_name", "artists", "track_genre", "popularity"]].copy()
        results["similarity_score"] = scores[top_indices]
        results = results.reset_index(drop=True)
        results.index = results.index + 1

        st.markdown(f"### Results for: *\"{query}\"*")
        st.dataframe(
            results.rename(columns={
                "track_name": "Track",
                "artists": "Artist",
                "track_genre": "Genre",
                "popularity": "Popularity",
                "similarity_score": "Relevance",
            }),
            use_container_width=True,
        )

        # Show lyrics preview for top results
        with st.expander("📝 Preview lyrics of top results"):
            for i, idx in enumerate(top_indices[:3]):
                song = df.iloc[idx]
                st.markdown(f"**{song['track_name']}** by {song['artists']}")
                lyrics_preview = song["lyrics"][:300] + "..." if len(str(song["lyrics"])) > 300 else song["lyrics"]
                st.text(lyrics_preview)
                st.markdown("---")

elif query and model is None:
    st.warning("sentence-transformers is not installed. Run: pip3 install sentence-transformers")
elif not query:
    st.info("Type a mood or description above to search!")
