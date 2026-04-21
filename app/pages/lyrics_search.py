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

st.set_page_config(page_title="Lyrics Search", page_icon="🔍", layout="wide")

# ── Mood theme configuration ──
MOOD_THEMES = {
    "sad":        {"emoji": "🌧️", "bg1": "#1a1a2e", "bg2": "#16213e", "accent": "#667eea",
                   "card_bg": "rgba(102,126,234,0.1)", "card_border": "rgba(102,126,234,0.25)",
                   "text": "#c8d6e5", "title": "#667eea", "tag_bg": "rgba(102,126,234,0.2)",
                   "tag_text": "#a4b6f5", "bar": "#667eea", "dark": True},
    "heartbreak": {"emoji": "💔", "bg1": "#1a1a2e", "bg2": "#2d1f3d", "accent": "#ee9ca7",
                   "card_bg": "rgba(238,156,167,0.1)", "card_border": "rgba(238,156,167,0.25)",
                   "text": "#e8d5d8", "title": "#ee9ca7", "tag_bg": "rgba(238,156,167,0.2)",
                   "tag_text": "#f0b8c0", "bar": "#ee9ca7", "dark": True},
    "night":      {"emoji": "🌙", "bg1": "#0a0a1a", "bg2": "#1a1a3e", "accent": "#4ca1af",
                   "card_bg": "rgba(76,161,175,0.08)", "card_border": "rgba(76,161,175,0.25)",
                   "text": "#b8d4da", "title": "#4ca1af", "tag_bg": "rgba(76,161,175,0.2)",
                   "tag_text": "#7ec8d4", "bar": "#4ca1af", "dark": True},
    "lonely":     {"emoji": "🥀", "bg1": "#1a1a2e", "bg2": "#1e1e3a", "accent": "#a18cd1",
                   "card_bg": "rgba(161,140,209,0.1)", "card_border": "rgba(161,140,209,0.25)",
                   "text": "#cfc5e5", "title": "#a18cd1", "tag_bg": "rgba(161,140,209,0.2)",
                   "tag_text": "#c4b5e0", "bar": "#a18cd1", "dark": True},
    "dark":       {"emoji": "🖤", "bg1": "#0d0d0d", "bg2": "#1a1a1a", "accent": "#888",
                   "card_bg": "rgba(255,255,255,0.05)", "card_border": "rgba(255,255,255,0.1)",
                   "text": "#aaa", "title": "#ccc", "tag_bg": "rgba(255,255,255,0.1)",
                   "tag_text": "#bbb", "bar": "#888", "dark": True},
    "miss":       {"emoji": "💭", "bg1": "#1a1a2e", "bg2": "#1e1e3a", "accent": "#a18cd1",
                   "card_bg": "rgba(161,140,209,0.1)", "card_border": "rgba(161,140,209,0.25)",
                   "text": "#cfc5e5", "title": "#a18cd1", "tag_bg": "rgba(161,140,209,0.2)",
                   "tag_text": "#c4b5e0", "bar": "#a18cd1", "dark": True},
    "happy":      {"emoji": "☀️", "bg1": "#fffdf7", "bg2": "#fff5e6", "accent": "#f5576c",
                   "card_bg": "rgba(245,87,108,0.06)", "card_border": "rgba(245,87,108,0.2)",
                   "text": "#555", "title": "#f5576c", "tag_bg": "rgba(245,87,108,0.12)",
                   "tag_text": "#f5576c", "bar": "#f5576c", "dark": False},
    "summer":     {"emoji": "🌊", "bg1": "#f0fffe", "bg2": "#e6fff9", "accent": "#00b894",
                   "card_bg": "rgba(0,184,148,0.06)", "card_border": "rgba(0,184,148,0.2)",
                   "text": "#555", "title": "#00b894", "tag_bg": "rgba(0,184,148,0.12)",
                   "tag_text": "#00b894", "bar": "#00b894", "dark": False},
    "energy":     {"emoji": "⚡", "bg1": "#fffdf5", "bg2": "#fff3e0", "accent": "#fa709a",
                   "card_bg": "rgba(250,112,154,0.06)", "card_border": "rgba(250,112,154,0.2)",
                   "text": "#555", "title": "#fa709a", "tag_bg": "rgba(250,112,154,0.12)",
                   "tag_text": "#fa709a", "bar": "#fa709a", "dark": False},
    "party":      {"emoji": "🪩", "bg1": "#1a0a2e", "bg2": "#2d1050", "accent": "#f093fb",
                   "card_bg": "rgba(240,147,251,0.1)", "card_border": "rgba(240,147,251,0.25)",
                   "text": "#e0c5f0", "title": "#f093fb", "tag_bg": "rgba(240,147,251,0.2)",
                   "tag_text": "#f0b8fb", "bar": "#f093fb", "dark": True},
    "calm":       {"emoji": "🍃", "bg1": "#f5fffe", "bg2": "#e8f8f5", "accent": "#55a68a",
                   "card_bg": "rgba(85,166,138,0.06)", "card_border": "rgba(85,166,138,0.2)",
                   "text": "#555", "title": "#55a68a", "tag_bg": "rgba(85,166,138,0.12)",
                   "tag_text": "#55a68a", "bar": "#55a68a", "dark": False},
    "peace":      {"emoji": "🕊️", "bg1": "#f5fffe", "bg2": "#e8f8f5", "accent": "#55a68a",
                   "card_bg": "rgba(85,166,138,0.06)", "card_border": "rgba(85,166,138,0.2)",
                   "text": "#555", "title": "#55a68a", "tag_bg": "rgba(85,166,138,0.12)",
                   "tag_text": "#55a68a", "bar": "#55a68a", "dark": False},
    "ocean":      {"emoji": "🌊", "bg1": "#f0f5ff", "bg2": "#e0ecff", "accent": "#3b82f6",
                   "card_bg": "rgba(59,130,246,0.06)", "card_border": "rgba(59,130,246,0.2)",
                   "text": "#555", "title": "#3b82f6", "tag_bg": "rgba(59,130,246,0.12)",
                   "tag_text": "#3b82f6", "bar": "#3b82f6", "dark": False},
    "love":       {"emoji": "💕", "bg1": "#fff5f7", "bg2": "#ffe8ee", "accent": "#e84393",
                   "card_bg": "rgba(232,67,147,0.06)", "card_border": "rgba(232,67,147,0.2)",
                   "text": "#555", "title": "#e84393", "tag_bg": "rgba(232,67,147,0.12)",
                   "tag_text": "#e84393", "bar": "#e84393", "dark": False},
    "chill":      {"emoji": "😌", "bg1": "#f8f5ff", "bg2": "#f0e6ff", "accent": "#a855f7",
                   "card_bg": "rgba(168,85,247,0.06)", "card_border": "rgba(168,85,247,0.2)",
                   "text": "#555", "title": "#a855f7", "tag_bg": "rgba(168,85,247,0.12)",
                   "tag_text": "#a855f7", "bar": "#a855f7", "dark": False},
    "angry":      {"emoji": "🔥", "bg1": "#1a0a0a", "bg2": "#2d1010", "accent": "#ff6b6b",
                   "card_bg": "rgba(255,107,107,0.1)", "card_border": "rgba(255,107,107,0.25)",
                   "text": "#e5c5c5", "title": "#ff6b6b", "tag_bg": "rgba(255,107,107,0.2)",
                   "tag_text": "#ff9b9b", "bar": "#ff6b6b", "dark": True},
    "rebel":      {"emoji": "🔥", "bg1": "#1a0a0a", "bg2": "#2d1010", "accent": "#ff6b6b",
                   "card_bg": "rgba(255,107,107,0.1)", "card_border": "rgba(255,107,107,0.25)",
                   "text": "#e5c5c5", "title": "#ff6b6b", "tag_bg": "rgba(255,107,107,0.2)",
                   "tag_text": "#ff9b9b", "bar": "#ff6b6b", "dark": True},
    "dream":      {"emoji": "✨", "bg1": "#1a1a2e", "bg2": "#2d1f3d", "accent": "#c9a0ff",
                   "card_bg": "rgba(201,160,255,0.1)", "card_border": "rgba(201,160,255,0.25)",
                   "text": "#d5c5f0", "title": "#c9a0ff", "tag_bg": "rgba(201,160,255,0.2)",
                   "tag_text": "#d5b8ff", "bar": "#c9a0ff", "dark": True},
    "danc":       {"emoji": "💃", "bg1": "#1a0a2e", "bg2": "#2d1050", "accent": "#f093fb",
                   "card_bg": "rgba(240,147,251,0.1)", "card_border": "rgba(240,147,251,0.25)",
                   "text": "#e0c5f0", "title": "#f093fb", "tag_bg": "rgba(240,147,251,0.2)",
                   "tag_text": "#f0b8fb", "bar": "#f093fb", "dark": True},
    "rain":       {"emoji": "🌧️", "bg1": "#0f1923", "bg2": "#162433", "accent": "#66a6ff",
                   "card_bg": "rgba(102,166,255,0.08)", "card_border": "rgba(102,166,255,0.2)",
                   "text": "#b0ccee", "title": "#66a6ff", "tag_bg": "rgba(102,166,255,0.15)",
                   "tag_text": "#88bbff", "bar": "#66a6ff", "dark": True},
    "drive":      {"emoji": "🚗", "bg1": "#1a1510", "bg2": "#2d2518", "accent": "#fcb69f",
                   "card_bg": "rgba(252,182,159,0.1)", "card_border": "rgba(252,182,159,0.2)",
                   "text": "#e0d0c5", "title": "#fcb69f", "tag_bg": "rgba(252,182,159,0.15)",
                   "tag_text": "#fcc5b0", "bar": "#fcb69f", "dark": True},
    "morning":    {"emoji": "🌅", "bg1": "#fffdf5", "bg2": "#fff5e0", "accent": "#f0932b",
                   "card_bg": "rgba(240,147,43,0.06)", "card_border": "rgba(240,147,43,0.2)",
                   "text": "#555", "title": "#f0932b", "tag_bg": "rgba(240,147,43,0.12)",
                   "tag_text": "#f0932b", "bar": "#f0932b", "dark": False},
    "workout":    {"emoji": "💪", "bg1": "#fffdf5", "bg2": "#fff3e0", "accent": "#e17055",
                   "card_bg": "rgba(225,112,85,0.06)", "card_border": "rgba(225,112,85,0.2)",
                   "text": "#555", "title": "#e17055", "tag_bg": "rgba(225,112,85,0.12)",
                   "tag_text": "#e17055", "bar": "#e17055", "dark": False},
    "nostalgi":   {"emoji": "📷", "bg1": "#1a1815", "bg2": "#2d2820", "accent": "#daa520",
                   "card_bg": "rgba(218,165,32,0.1)", "card_border": "rgba(218,165,32,0.2)",
                   "text": "#d5cdb8", "title": "#daa520", "tag_bg": "rgba(218,165,32,0.15)",
                   "tag_text": "#e0c050", "bar": "#daa520", "dark": True},
}

# Default theme (before search or no keyword match)
DEFAULT_THEME = {
    "emoji": "🎵", "bg1": "#fef9ff", "bg2": "#f0f4ff", "accent": "#a855f7",
    "card_bg": "rgba(168,85,247,0.04)", "card_border": "rgba(168,85,247,0.15)",
    "text": "#555", "title": "#a855f7", "tag_bg": "rgba(168,85,247,0.1)",
    "tag_text": "#a855f7", "bar": "#a855f7", "dark": False,
}


def get_theme(query_text):
    if not query_text:
        return DEFAULT_THEME
    q = query_text.lower()
    for keyword, theme in MOOD_THEMES.items():
        if keyword in q:
            return theme
    return DEFAULT_THEME


# ── Load data (before theme so query can be read) ──
@st.cache_data
def get_data():
    df = load_spotify_with_lyrics()
    df_with_lyrics = df[df["lyrics"].notna()].reset_index(drop=True)
    return df_with_lyrics

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
    embeddings = _model.encode(lyrics_list, batch_size=32, show_progress_bar=False)
    return embeddings

df = get_data()
model = load_model()

# ── Get query early so theme can react ──
# We use session state to persist the query across reruns
if "lyrics_query" not in st.session_state:
    st.session_state.lyrics_query = ""

# Temporary placeholder for query input — actual widget below
query = st.session_state.get("lyrics_query", "")
theme = get_theme(query)
t = theme  # shorthand

# ── Dynamic CSS ──
header_text_color = "#fff" if t["dark"] else t["title"]
subtitle_color = t["text"]
spotify_btn = "linear-gradient(120deg, #1DB954, #1ed760)"
chip_styles = ""
if t["dark"]:
    chip_styles = f"""
    .chip-pink {{ background: rgba(238,156,167,0.2); color: #f0b8c0; }}
    .chip-purple {{ background: rgba(161,140,209,0.2); color: #c4b5e0; }}
    .chip-blue {{ background: rgba(102,166,255,0.2); color: #88bbff; }}
    .chip-green {{ background: rgba(67,233,123,0.2); color: #70e8a0; }}
    .chip-orange {{ background: rgba(252,182,159,0.2); color: #fcc5b0; }}
    """
else:
    chip_styles = """
    .chip-pink { background: #ffe0ec; color: #e84393; }
    .chip-purple { background: #f3e8ff; color: #a855f7; }
    .chip-blue { background: #dbeafe; color: #3b82f6; }
    .chip-green { background: #d1fae5; color: #059669; }
    .chip-orange { background: #ffedd5; color: #ea580c; }
    """

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, {t['bg1']} 0%, {t['bg2']} 100%);
        transition: background 0.5s ease;
    }}
    .stApp header {{
        background: transparent !important;
    }}
    .search-header h1 {{
        font-size: 2.8rem;
        color: {header_text_color};
        font-weight: 800;
    }}
    .search-subtitle {{
        text-align: center; font-size: 1.1rem;
        color: {subtitle_color}; margin-bottom: 2rem;
    }}
    .gradient-divider {{
        height: 2px;
        background: linear-gradient(90deg, transparent, {t['accent']}, transparent);
        margin: 2rem 0; border: none;
    }}
    .song-card {{
        background: {t['card_bg']};
        border-radius: 16px; padding: 1.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid {t['card_border']};
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    .song-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 20px {t['card_bg']};
    }}
    .song-rank {{
        font-size: 1.8rem; font-weight: 800;
        color: {t['accent']}; margin-right: 1rem;
    }}
    .song-title {{ font-size: 1.2rem; font-weight: 700; color: {t['text']}; }}
    .song-artist {{ font-size: 1rem; color: {t['text']}; opacity: 0.7; }}
    .genre-tag {{
        display: inline-block; padding: 0.2rem 0.8rem;
        border-radius: 50px; font-size: 0.8rem; font-weight: 600;
        margin-top: 0.5rem;
        background: {t['tag_bg']}; color: {t['tag_text']};
    }}
    .relevance-bar {{
        height: 6px; border-radius: 3px;
        background: {t['bar']};
    }}
    .relevance-label {{
        font-size: 0.75rem; color: {t['text']}; opacity: 0.5;
        margin-top: 0.2rem;
    }}
    .lyrics-preview {{
        background: {t['card_bg']};
        border-left: 3px solid {t['accent']};
        border-radius: 0 12px 12px 0;
        padding: 1.2rem 1.5rem; margin: 0.5rem 0;
        font-style: italic; color: {t['text']}; line-height: 1.8;
    }}
    .lyrics-song-label {{
        font-weight: 700; color: {t['text']};
        font-style: normal; margin-bottom: 0.5rem;
    }}
    .spotify-section {{
        background: {t['card_bg']};
        border-radius: 12px; padding: 1rem;
        border: 1px solid {t['card_border']};
        text-align: center; margin-top: 0.5rem;
    }}
    .suggestion-chips {{ text-align: center; margin: 1rem 0 2rem 0; }}
    .chip {{
        display: inline-block; padding: 0.4rem 1rem;
        margin: 0.3rem; border-radius: 50px;
        font-size: 0.85rem; transition: transform 0.2s;
    }}
    .chip:hover {{ transform: scale(1.05); }}
    {chip_styles}
    .empty-state {{
        text-align: center; padding: 3rem;
        color: {t['text']}; opacity: 0.6;
    }}
    .empty-state-icon {{ font-size: 4rem; margin-bottom: 1rem; }}
    .mood-banner {{
        text-align: center; padding: 2rem;
        border-radius: 20px; margin: 0 auto 1.5rem auto;
        max-width: 700px;
        background: linear-gradient(135deg,
            {t['accent']}33, {t['accent']}11);
        border: 1px solid {t['card_border']};
    }}
    .mood-emoji {{ font-size: 3rem; margin-bottom: 0.5rem; }}
    .mood-query {{
        font-size: 1.5rem; font-weight: 700;
        color: {t['accent']};
    }}
    .results-label {{
        text-align: center; margin-bottom: 1.5rem;
        font-size: 1.3rem; color: {t['text']};
    }}
    .pop-label {{
        font-size: 0.85rem; color: {t['text']}; opacity: 0.5;
        margin-top: 0.3rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──
st.markdown(
    f"""
    <div class="search-header" style="text-align:center; padding:2rem 0 1rem 0;">
        <h1>🔍 Mood-Based Lyrics Search</h1>
    </div>
    <div class="search-subtitle">
        Describe a feeling, a scene, or a vibe — we'll find songs whose lyrics match
    </div>
    """,
    unsafe_allow_html=True,
)

# Suggestion chips
st.markdown(
    """
    <div class="suggestion-chips">
        <span class="chip chip-pink">💔 sad heartbreak</span>
        <span class="chip chip-purple">🌙 late night lonely</span>
        <span class="chip chip-blue">🌊 calm ocean peace</span>
        <span class="chip chip-green">☀️ happy summer day</span>
        <span class="chip chip-orange">🔥 angry rebellious energy</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Search input ──
col_search, col_slider = st.columns([3, 1])
with col_search:
    query = st.text_input(
        "What are you in the mood for?",
        placeholder="e.g., dancing in the rain, missing someone far away...",
        label_visibility="collapsed",
        key="lyrics_query",
    )
with col_slider:
    k = st.slider("Results", min_value=3, max_value=15, value=5, label_visibility="collapsed")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# Re-derive theme based on actual query input
theme = get_theme(query)
t = theme


def make_radar_chart(song, features, labels):
    values = [song[f] for f in features]
    values.append(values[0])
    labels_plot = list(labels) + [labels[0]]

    line_color = t["accent"]
    # Convert hex accent to rgba for fill
    hex_color = t["accent"].lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    fill_color = f"rgba({r},{g},{b},0.15)"
    grid_color = t["card_border"]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels_plot, fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2),
        marker=dict(size=5, color=line_color),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            showticklabels=False, gridcolor=grid_color),
            angularaxis=dict(gridcolor=grid_color, linecolor=grid_color,
                             tickfont=dict(color=t["text"], size=10)),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def get_spotify_search_url(track_name, artist):
    artist_clean = artist.split(";")[0].strip()
    search_query = quote(f"{track_name} {artist_clean}")
    return f"https://open.spotify.com/search/{search_query}"


radar_features = [
    "danceability", "energy", "acousticness",
    "valence", "instrumentalness", "speechiness",
]
radar_labels = ["Dance", "Energy", "Acoustic", "Valence", "Instrum.", "Speech"]

if query and model is not None:
    with st.spinner("🎵 Searching through 29,000+ lyrics..."):
        lyrics_embeddings = compute_lyrics_embeddings(model, df["lyrics"].tolist())
        query_embedding = model.encode([query])
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_embedding, lyrics_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:k]

    # Mood banner
    st.markdown(
        f"""
        <div class="mood-banner">
            <div class="mood-emoji">{t['emoji']}</div>
            <div class="mood-query">"{query}"</div>
        </div>
        <div class="results-label">Top matches for your mood</div>
        """,
        unsafe_allow_html=True,
    )

    for rank, idx in enumerate(top_indices):
        song = df.iloc[idx]
        score = scores[idx]
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
                            <span class="genre-tag">{song['track_genre']}</span>
                            <div class="pop-label">Popularity: {song['popularity']}/100</div>
                            <div style="margin-top: 0.5rem;">
                                <div class="relevance-bar" style="width: {bar_width}%;"></div>
                                <div class="relevance-label">Relevance: {score:.1%}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_radar:
            fig = make_radar_chart(song, radar_features, radar_labels)
            st.plotly_chart(fig, use_container_width=True, key=f"radar_{rank}")

        with col_spotify:
            st.markdown(
                f"""
                <div class="spotify-section">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎧</div>
                    <a href="{spotify_url}" target="_blank"
                       style="display: inline-block;
                              background: {spotify_btn};
                              color: white;
                              padding: 0.5rem 1.5rem;
                              border-radius: 50px;
                              text-decoration: none;
                              font-weight: 600;
                              font-size: 0.9rem;">
                        Listen on Spotify
                    </a>
                    <div style="font-size: 0.75rem; color: {t['text']}; opacity: 0.4;
                                margin-top: 0.5rem;">
                        Opens Spotify search
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander(f"📝 Lyrics preview — {song['track_name']}"):
            lyrics_preview = song["lyrics"][:400] + "..." if len(str(song["lyrics"])) > 400 else song["lyrics"]
            lyrics_display = str(lyrics_preview).replace("\\n", "\n")
            st.markdown(
                f"""
                <div class="lyrics-preview">
                    <div class="lyrics-song-label">🎵 {song['track_name']} — {song['artists']}</div>
                    {lyrics_display.replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

elif query and model is None:
    st.warning("sentence-transformers is not installed. Run: pip3 install sentence-transformers")

else:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">🎵</div>
            <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">
                Type a mood or feeling to discover matching songs
            </div>
            <div style="font-size: 0.9rem;">
                Try something like "nostalgic childhood memories" or "confident boss energy"
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )