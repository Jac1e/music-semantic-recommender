import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.reduction import compute_pca_projection

st.set_page_config(page_title="Explore Songs", page_icon="📊", layout="wide")

# Custom CSS matching pastel theme
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fef9ff 0%, #f0f4ff 100%);
    }
    .viz-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .viz-header h1 {
        font-size: 2.8rem;
        background: linear-gradient(120deg, #e84393 0%, #a855f7 50%, #6c5ce7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .viz-subtitle {
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
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #f0e6ff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    .info-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .info-desc {
        font-size: 0.9rem;
        color: #666;
        line-height: 1.6;
    }
    .stat-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .stat-pill-purple {
        background: #f3e8ff;
        color: #a855f7;
    }
    .stat-pill-pink {
        background: #ffe0ec;
        color: #e84393;
    }
    .stat-pill-blue {
        background: #dbeafe;
        color: #3b82f6;
    }
    .control-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #f0e6ff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .control-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .chart-container {
        background: white;
        border-radius: 20px;
        padding: 1rem;
        border: 1px solid #f0e6ff;
        box-shadow: 0 2px 15px rgba(0,0,0,0.04);
    }
    .coming-soon {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(135deg, #f3e8ff 0%, #ffe0ec 100%);
        border-radius: 20px;
        border: 1px solid #d8b4fe;
    }
    .coming-soon-icon {
        font-size: 3rem;
        margin-bottom: 0.8rem;
    }
    .coming-soon-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.3rem;
    }
    .coming-soon-desc {
        font-size: 0.9rem;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="viz-header">
        <h1>📊 Explore the Song Space</h1>
    </div>
    <div class="viz-subtitle">
        See how 52,000 songs relate to each other, colored by genre or cluster
    </div>
    """,
    unsafe_allow_html=True,
)

# Load processed data
@st.cache_data
def get_data():
    df = pd.read_csv(project_root / "data" / "processed" / "song_feature_vectors.csv")
    return df

df = get_data()

# Use reduction module
@st.cache_data
def get_pca_projection(dataframe):
    projected_df, variance_ratio = compute_pca_projection(dataframe, n_components=2)
    return projected_df, variance_ratio

projected_df, variance_ratio = get_pca_projection(df)
projected_df["x"] = projected_df["pca_1"]
projected_df["y"] = projected_df["pca_2"]

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# Controls
col_controls, col_chart = st.columns([1, 3])

with col_controls:
    st.markdown(
        """
        <div class="control-card">
            <div class="control-label">🎨 Color by</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    color_option = st.radio(
        "Color by:",
        ["Genre", "Cluster (coming soon)"],
        label_visibility="collapsed",
    )

    if color_option == "Genre":
        st.markdown(
            """
            <div class="control-card">
                <div class="control-label">🎸 Filter genres</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        all_genres = sorted(projected_df["track_genre"].dropna().unique())
        selected_genres = st.multiselect(
            "Filter genres:",
            options=all_genres,
            default=None,
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div class="control-card">
                <div class="control-label">📊 Sample size</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        sample_size = st.slider(
            "Sample size:",
            min_value=1000,
            max_value=20000,
            value=5000,
            step=1000,
            label_visibility="collapsed",
        )

    st.markdown(
        f"""
        <div class="info-card" style="margin-top: 1rem;">
            <div class="info-title">ℹ️ About this plot</div>
            <div class="info-desc">
                Each dot is a song, projected from scaled audio features 
                down to 2D using PCA via reduction module.
            </div>
            <div style="margin-top: 0.8rem;">
                <span class="stat-pill stat-pill-purple">
                    PC1: {variance_ratio[0]:.1%}
                </span>
                <span class="stat-pill stat-pill-pink">
                    PC2: {variance_ratio[1]:.1%}
                </span>
                <span class="stat-pill stat-pill-blue">
                    Total: {sum(variance_ratio):.1%}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_chart:
    if color_option == "Genre":
        if selected_genres:
            plot_df = projected_df[projected_df["track_genre"].isin(selected_genres)].copy()
            subtitle = f"Showing {len(plot_df):,} songs in {len(selected_genres)} genres"
        else:
            plot_df = projected_df.sample(n=min(sample_size, len(projected_df)), random_state=42).copy()
            subtitle = f"Showing a random sample of {len(plot_df):,} songs"

        pastel_colors = [
            "#e84393", "#a855f7", "#6c5ce7", "#3b82f6", "#00b894",
            "#00cec9", "#f0932b", "#f5576c", "#fd79a8", "#74b9ff",
            "#55efc4", "#ffeaa7", "#fab1a0", "#81ecec", "#dfe6e9",
            "#ff7675", "#a29bfe", "#fad390", "#f8c291", "#6a89cc",
            "#82ccdd", "#b8e994", "#78e08f", "#e55039", "#4a69bd",
        ]

        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="track_genre",
            hover_data=["track_name", "artists", "track_genre"],
            labels={
                "x": f"PC1 ({variance_ratio[0]:.1%} variance)",
                "y": f"PC2 ({variance_ratio[1]:.1%} variance)",
            },
            opacity=0.7,
            color_discrete_sequence=pastel_colors,
        )
        fig.update_layout(
            height=650,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(254,249,255,0.5)",
            font=dict(family="sans-serif", color="#555"),
            legend=dict(
                title=dict(text="Genre", font=dict(size=12, color="#333")),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#f0e6ff",
                borderwidth=1,
                font=dict(size=10),
            ),
            xaxis=dict(gridcolor="#f0e6ff", zerolinecolor="#e0d0f0"),
            yaxis=dict(gridcolor="#f0e6ff", zerolinecolor="#e0d0f0"),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        fig.update_traces(marker=dict(size=5, line=dict(width=0)))

        st.markdown(
            f'<div style="text-align:center; color:#888; font-size:0.85rem; margin-bottom:0.5rem;">{subtitle}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div class="coming-soon">
                <div class="coming-soon-icon">🔮</div>
                <div class="coming-soon-title">Cluster visualization coming soon</div>
                <div class="coming-soon-desc">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 0.85rem; color: #888;">
            Built with PCA dimensionality reduction · Plotly interactive charts ·
            52,000 songs from Spotify
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)