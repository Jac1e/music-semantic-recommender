import streamlit as st

st.set_page_config(
    page_title="Music Semantic Recommender",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Music Semantic Recommender")
st.subheader("Discover music through mood, similarity, and exploration")

st.markdown(
    """
    Welcome! This app helps you discover music using machine learning. 
    Here's what you can do:

    - **🔍 Lyrics Search** — Describe a mood or vibe in your own words, 
      and we'll find songs whose lyrics match.
    - **🎧 Recommend** — Pick a song you like, and we'll find the most 
      similar tracks based on audio features and lyrics.
    - **📊 Visualize** — Explore the entire song space in an interactive 
      2D plot, colored by genre or cluster.

    Use the sidebar to navigate between pages.
    """
)

st.markdown("---")
st.caption("Built with Streamlit · NYU Fundamentals of Machine Learning · Spring 2026")
