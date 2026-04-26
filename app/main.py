import streamlit as st
import random

st.set_page_config(
    page_title="Music Semantic Recommender",
    page_icon="🎵",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
    <style>
    /* Overall background */
    .stApp {
        background: linear-gradient(180deg, #fef9ff 0%, #f0f4ff 100%);
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 3rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 3.5rem;
        color: #a855f7;
        font-weight: 800;
        margin-bottom: 0;
    }
    .main-subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #777;
        margin-bottom: 0.5rem;
    }
    .music-notes {
        text-align: center;
        font-size: 2rem;
        letter-spacing: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Mood ticker */
    .mood-ticker {
        text-align: center;
        padding: 1rem;
        margin: 0.5rem auto 2rem auto;
        max-width: 500px;
        background: white;
        border-radius: 50px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #f0e6ff;
    }
    .mood-ticker-label {
        color: #aaa;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }
    .mood-ticker-text {
        color: #a855f7;
        font-size: 1.1rem;
        font-style: italic;
        font-weight: 500;
    }

    /* Feature cards */
    .card-pink {
        background: linear-gradient(135deg, #fff0f6 0%, #ffe0ec 100%);
        border: 1px solid #ffcad9;
        border-radius: 20px;
        padding: 2rem;
        height: 280px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card-pink:hover {
        transform: translateY(-6px);
        box-shadow: 0 8px 25px rgba(232, 67, 147, 0.15);
    }
    .card-purple {
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
        border: 1px solid #d8b4fe;
        border-radius: 20px;
        padding: 2rem;
        height: 280px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card-purple:hover {
        transform: translateY(-6px);
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.15);
    }
    .card-blue {
        background: linear-gradient(135deg, #e8f4fd 0%, #dbeafe 100%);
        border: 1px solid #93c5fd;
        border-radius: 20px;
        padding: 2rem;
        height: 280px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card-blue:hover {
        transform: translateY(-6px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
    }

    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
    }
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .card-desc {
        font-size: 0.95rem;
        color: #666;
        line-height: 1.6;
    }

    /* Divider */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #d8b4fe, #f9a8d4, transparent);
        margin: 2.5rem 0;
        border: none;
    }

    /* Stats */
    .stat-box {
        text-align: center;
        padding: 1.5rem 1rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        border: 1px solid #f0e6ff;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #e84393;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #888;
        margin-top: 0.3rem;
    }

    /* How it works */
    .how-it-works {
        text-align: center;
        padding: 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        border: 1px solid #f0e6ff;
        max-width: 700px;
        margin: 0 auto;
    }
    .how-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.8rem;
    }
    .how-desc {
        font-size: 0.95rem;
        color: #666;
        line-height: 1.7;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #aaa;
        padding: 2rem 0;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>Music Semantic Recommender</h1>
    </div>
    <div class="music-notes">🎵 🎧 🎶 🎤 🎸 💿</div>
    <div class="main-subtitle">
        Discover your next favorite song through mood, similarity, and exploration
    </div>
    """,
    unsafe_allow_html=True,
)

# Rotating mood suggestions
moods = [
    "chill acoustic rainy day ☔",
    "upbeat summer road trip 🚗",
    "late night study session 🌙",
    "energetic workout mix 💪",
    "sad heartbreak ballad 💔",
    "happy Sunday morning ☀️",
    "dreamy stargazing vibes ✨",
    "party dance floor energy 🪩",
]
selected_mood = random.choice(moods)
st.markdown(
    f"""
    <div class="mood-ticker">
        <div class="mood-ticker-label">✨ try searching for</div>
        <div class="mood-ticker-text">"{selected_mood}"</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Feature cards
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown(
        """
        <a href="lyrics_search" target="_self" style="text-decoration: none; color: inherit; display: block;">
            <div class="card-pink">
                <div class="card-icon">🔍</div>
                <div class="card-title">Mood Search</div>
                <div class="card-desc">
                    Describe a vibe in your own words — <em>"mellow jazz for 
                    cooking dinner"</em> — and we'll find songs whose lyrics 
                    match your mood.
                </div>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <a href="recommend" target="_self" style="text-decoration: none; color: inherit; display: block;">
            <div class="card-purple">
                <div class="card-icon">🎧</div>
                <div class="card-title">Similar Songs</div>
                <div class="card-desc">
                    Pick a song you love, and we'll find tracks that sound 
                    like it — matched by danceability, energy, tempo, valence, 
                    and more.
                </div>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <a href="visualize" target="_self" style="text-decoration: none; color: inherit; display: block;">
            <div class="card-blue">
                <div class="card-icon">📊</div>
                <div class="card-title">Explore & Visualize</div>
                <div class="card-desc">
                    See how 52,000 songs relate to each other in an interactive 
                    2D map. Filter by genre, discover clusters, and find hidden 
                    connections.
                </div>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# Stats row
s1, s2, s3, s4 = st.columns(4, gap="medium")

with s1:
    st.markdown(
        """
        <div class="stat-box">
            <div class="stat-number">52K</div>
            <div class="stat-label">🎵 Songs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with s2:
    st.markdown(
        """
        <div class="stat-box">
            <div class="stat-number">29.7K</div>
            <div class="stat-label">📝 With Lyrics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with s3:
    st.markdown(
        """
        <div class="stat-box">
            <div class="stat-number">114</div>
            <div class="stat-label">🎸 Genres</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with s4:
    st.markdown(
        """
        <div class="stat-box">
            <div class="stat-number">10K+</div>
            <div class="stat-label">🎤 Artists</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# How it works
st.markdown(
    """
    <div class="how-it-works">
        <div class="how-title">✨ How It Works</div>
        <div class="how-desc">
            We use machine learning to turn songs into mathematical vectors — 
            capturing everything from audio characteristics to lyrical themes. 
            Songs that are similar end up close together in this space, 
            letting us find connections that go beyond simple genre labels.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div class="footer">
        Built using Streamlit · NYU Fundamentals of Machine Learning · Spring 2026<br>
        <small>Use the sidebar to navigate between pages →</small>
    </div>
    """,
    unsafe_allow_html=True,
)