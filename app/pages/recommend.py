import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
from urllib.parse import quote

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data_loader import load_spotify_with_lyrics

st.set_page_config(page_title="Song Recommendations", page_icon="🎧", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fef9ff 0%, #f0f4ff 100%);
    }
    .rec-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .rec-header h1 {
        font-size: 2.8rem;
        background: linear-gradient(120deg, #e84393 0%, #a855f7 50%, #6c5ce7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .rec-subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #777;
        margin-bottom: 2rem;
    }
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #d8b4fe, #f9a8d4, transparent);
        margin: 2rem 0;
        border: none;
    }
    .selected-song-card {
        background: linear-gradient(135deg, #f3e8ff 0%, #ffe0ec 100%);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        border: 1px solid #d8b4fe;
        text-align: center;
        margin-bottom: 1rem;
    }
    .selected-song-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #333;
    }
    .selected-song-artist {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0.3rem;
    }
    .selected-song-label {
        font-size: 0.85rem;
        color: #a855f7;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .song-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #f0e6ff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .song-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.1);
    }
    .song-rank {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(120deg, #e84393, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-right: 1rem;
    }
    .song-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #333;
    }
    .song-artist {
        font-size: 1rem;
        color: #888;
    }
    .genre-tag {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .genre-tag-pink { background: #ffe0ec; color: #e84393; }
    .genre-tag-purple { background: #f3e8ff; color: #a855f7; }
    .genre-tag-blue { background: #dbeafe; color: #3b82f6; }
    .similarity-bar-container {
        margin-top: 0.5rem;
    }
    .similarity-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #e84393, #a855f7);
    }
    .similarity-label {
        font-size: 0.75rem;
        color: #aaa;
        margin-top: 0.2rem;
    }
    .spotify-section {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #f0e6ff;
        text-align: center;
        margin-top: 0.5rem;
    }
    .compare-header {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin: 1.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="rec-header">
        <h1>🎧 Song Recommendations</h1>
    </div>
    <div class="rec-subtitle">
        Pick a song you love, and we'll find tracks that sound like it
    </div>
    """,
    unsafe_allow_html=True,
)

# Load data
@st.cache_data
def get_data():
    df = load_spotify_with_lyrics()
    return df

df = get_data()
df["display_name"] = df["artists"] + " — " + df["track_name"]

feature_cols = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

radar_features = [
    "danceability", "energy", "acousticness",
    "valence", "instrumentalness", "speechiness",
]
radar_labels = ["Dance", "Energy", "Acoustic", "Valence", "Instrum.", "Speech"]
tag_colors = ["pink", "purple", "blue"]


def make_radar_chart(song, features, labels, color="#a855f7", fill_color="rgba(168, 85, 247, 0.15)"):
    values = [song[f] for f in features]
    values.append(values[0])
    labels_plot = list(labels) + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels_plot,
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=color, width=2),
        marker=dict(size=5, color="#e84393"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                showticklabels=False, gridcolor="#f0e6ff",
            ),
            angularaxis=dict(gridcolor="#f0e6ff", linecolor="#f0e6ff"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_comparison_radar(selected_song, rec_song, features, labels):
    """Radar chart comparing the selected song vs a recommended song."""
    sel_values = [selected_song[f] for f in features] + [selected_song[features[0]]]
    rec_values = [rec_song[f] for f in features] + [rec_song[features[0]]]
    labels_plot = list(labels) + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=sel_values, theta=labels_plot,
        fill="toself",
        fillcolor="rgba(232, 67, 147, 0.1)",
        line=dict(color="#e84393", width=2),
        name="Your song",
    ))
    fig.add_trace(go.Scatterpolar(
        r=rec_values, theta=labels_plot,
        fill="toself",
        fillcolor="rgba(168, 85, 247, 0.1)",
        line=dict(color="#a855f7", width=2),
        name="Recommended",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                showticklabels=False, gridcolor="#f0e6ff",
            ),
            angularaxis=dict(gridcolor="#f0e6ff", linecolor="#f0e6ff"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.2,
            xanchor="center", x=0.5, font=dict(size=10),
        ),
        margin=dict(l=40, r=40, t=20, b=40),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def get_spotify_search_url(track_name, artist):
    artist_clean = artist.split(";")[0].strip()
    search_query = quote(f"{track_name} {artist_clean}")
    return f"https://open.spotify.com/search/{search_query}"


# Song selector
selected = st.selectbox(
    "Choose a song:",
    options=df["display_name"].tolist(),
    index=None,
    placeholder="Start typing a song or artist name...",
)

col_slider, col_spacer = st.columns([1, 3])
with col_slider:
    k = st.slider("Number of recommendations:", min_value=3, max_value=15, value=5)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

if selected:
    song_idx = df[df["display_name"] == selected].index[0]
    selected_song = df.iloc[song_idx]
    spotify_url_selected = get_spotify_search_url(selected_song["track_name"], selected_song["artists"])

    # Selected song display
    col_sel_info, col_sel_radar = st.columns([3, 2])

    with col_sel_info:
        st.markdown(
            f"""
            <div class="selected-song-card">
                <div class="selected-song-label">🎵 YOUR SELECTED SONG</div>
                <div class="selected-song-title">{selected_song['track_name']}</div>
                <div class="selected-song-artist">{selected_song['artists']}</div>
                <span class="genre-tag genre-tag-purple" style="margin-top: 0.8rem;">
                    {selected_song['track_genre']}
                </span>
                <div style="margin-top: 0.8rem;">
                    <a href="{spotify_url_selected}" target="_blank"
                       style="display: inline-block;
                              background: linear-gradient(120deg, #1DB954, #1ed760);
                              color: white;
                              padding: 0.4rem 1.2rem;
                              border-radius: 50px;
                              text-decoration: none;
                              font-weight: 600;
                              font-size: 0.85rem;">
                        🎧 Listen on Spotify
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_sel_radar:
        fig = make_radar_chart(
            selected_song, radar_features, radar_labels,
            color="#e84393", fill_color="rgba(232, 67, 147, 0.15)",
        )
        st.plotly_chart(fig, use_container_width=True, key="selected_radar")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Compute recommendations
    @st.cache_data
    def compute_similarity(dataframe):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(dataframe[feature_cols].fillna(0))
        return features

    features = compute_similarity(df)
    from sklearn.metrics.pairwise import cosine_similarity
    query_vec = features[song_idx].reshape(1, -1)
    scores = cosine_similarity(query_vec, features)[0]

    top_indices = np.argsort(scores)[::-1][1:k*3]
    results = df.iloc[top_indices].copy()
    results["similarity_score"] = scores[top_indices]
    # Remove the selected song itself (same name + artist under different genres)
    results = results[
        ~((results["track_name"] == selected_song["track_name"]) &
          (results["artists"] == selected_song["artists"]))
    ]
    results = results.drop_duplicates(subset=["track_name", "artists"], keep="first")
    results = results.head(k)

    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <span style="font-size: 1.3rem; color: #333;">
                Songs similar to
                <span style="color: #a855f7; font-weight: 700;">
                    {selected_song['track_name']}
                </span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for rank, (_, song) in enumerate(results.iterrows()):
        score = song["similarity_score"]
        tag_color = tag_colors[rank % len(tag_colors)]
        bar_width = int(score * 100)
        spotify_url = get_spotify_search_url(song["track_name"], song["artists"])

        col_info, col_radar, col_spotify = st.columns([3, 2, 2])

        with col_info:
            st.markdown(
                f"""
                <div class="song-card">
                    <div style="display: flex; align-items: flex-start;">
                        <div class="song-rank">#{rank + 1}</div>
                        <div>
                            <div class="song-title">{song['track_name']}</div>
                            <div class="song-artist">{song['artists']}</div>
                            <span class="genre-tag genre-tag-{tag_color}">
                                {song['track_genre']}
                            </span>
                            <div style="font-size: 0.85rem; color: #aaa; margin-top: 0.3rem;">
                                Popularity: {song['popularity']}/100
                            </div>
                            <div class="similarity-bar-container">
                                <div class="similarity-bar" style="width: {bar_width}%;"></div>
                                <div class="similarity-label">
                                    Similarity: {score:.1%}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_radar:
            fig = make_comparison_radar(
                selected_song, song, radar_features, radar_labels,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"compare_{rank}")

        with col_spotify:
            st.markdown(
                f"""
                <div class="spotify-section">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎧</div>
                    <a href="{spotify_url}" target="_blank"
                       style="display: inline-block;
                              background: linear-gradient(120deg, #1DB954, #1ed760);
                              color: white;
                              padding: 0.5rem 1.5rem;
                              border-radius: 50px;
                              text-decoration: none;
                              font-weight: 600;
                              font-size: 0.9rem;">
                        Listen on Spotify
                    </a>
                    <div style="font-size: 0.75rem; color: #aaa; margin-top: 0.5rem;">
                        Opens Spotify search
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem; color: #aaa;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎧</div>
            <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">
                Search for a song above to get started
            </div>
            <div style="font-size: 0.9rem;">
                We'll find tracks with similar audio characteristics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
