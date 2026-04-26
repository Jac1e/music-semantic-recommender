import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.reduction import compute_pca_projection
from src.clustering import (
    cluster,
    kmeans_elbow,
    silhouette_analysis,
    gmm_bic_aic,
)

st.set_page_config(page_title="Explore Songs", page_icon="📊", layout="wide")

# ── Custom CSS ──
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fef9ff 0%, #f0f4ff 100%);
    }
    .viz-header {
        text-align: center;
        padding: 2rem 0 0.5rem 0;
    }
    .viz-header h1 {
        font-size: 2.8rem;
        color: #a855f7 !important;
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
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        border: 1px solid #f0e6ff;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #a855f7;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.3rem;
    }
    .info-box {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #f0e6ff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──
st.markdown(
    """
    <div class="viz-header">
        <h1>📊 Explore the Song Space</h1>
    </div>
    <div class="viz-subtitle">
        See how songs relate to each other — color by genre or discover hidden clusters
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load data ──
@st.cache_resource
def get_data():
    df = pd.read_csv(project_root / "data" / "processed" / "song_feature_vectors.csv")
    return df


@st.cache_resource
def get_pca_projection(dataframe, n_components):
    projected_df, variance_ratio = compute_pca_projection(dataframe, n_components=n_components)
    return projected_df, variance_ratio


@st.cache_resource
def run_clustering(features, method, n_clusters, covariance_type="full"):
    result = cluster(
        features,
        method=method,
        n_clusters=n_clusters,
        covariance_type=covariance_type,
        random_state=42,
        compute_silhouette=True,
    )
    return result


@st.cache_data
def run_elbow(features, k_range_list):
    return kmeans_elbow(features, k_range=k_range_list)


@st.cache_data
def run_silhouette(features, k_range_list):
    return silhouette_analysis(features, k_range=k_range_list)


@st.cache_data
def run_bic_aic(features, k_range_list, covariance_type):
    return gmm_bic_aic(features, k_range=k_range_list, covariance_type=covariance_type)


df = get_data()

# ── Sidebar controls ──
st.sidebar.markdown("### ⚙️ Visualization Settings")

n_dims = st.sidebar.radio("Dimensions", [2, 3], index=0, horizontal=True)

color_mode = st.sidebar.radio(
    "Color by",
    ["Genre", "K-Means Cluster", "GMM Cluster"],
    index=0,
)

# Cluster-specific controls
n_clusters = 8
cov_type = "full"
if color_mode in ("K-Means Cluster", "GMM Cluster"):
    n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=15, value=8)
    if color_mode == "GMM Cluster":
        cov_type = st.sidebar.selectbox(
            "Covariance type",
            ["full", "tied", "diag", "spherical"],
            index=0,
        )
    show_eval = st.sidebar.checkbox("Show evaluation charts", value=False)
else:
    show_eval = False

# Sampling controls
sample_size = st.sidebar.slider("Max points to plot", min_value=1000, max_value=20000, value=5000, step=1000)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── PCA projection ──
projected_df, variance_ratio = get_pca_projection(df, n_components=n_dims)

# Subsample for plotting
if color_mode == "Genre":
    all_genres = sorted(projected_df["track_genre"].dropna().unique())
    selected_genres = st.multiselect(
        "Filter genres (leave empty to show all):",
        options=all_genres,
        default=None,
    )
    if selected_genres:
        plot_df = projected_df[projected_df["track_genre"].isin(selected_genres)].copy()
    else:
        plot_df = projected_df.sample(n=min(sample_size, len(projected_df)), random_state=42).copy()
        st.caption(f"Showing a random sample of {min(sample_size, len(projected_df)):,} songs. Use the genre filter to focus.")
else:
    plot_df = projected_df.sample(n=min(sample_size, len(projected_df)), random_state=42).copy()

# ── Build feature matrix for clustering ──
pca_cols = [f"pca_{i+1}" for i in range(n_dims)]
features_for_clustering = plot_df[pca_cols].values

# ── Color logic ──
if color_mode == "Genre":
    color_col = "track_genre"
elif color_mode in ("K-Means Cluster", "GMM Cluster"):
    method = "kmeans" if color_mode == "K-Means Cluster" else "gmm"
    with st.spinner(f"Running {color_mode} with k={n_clusters}..."):
        cr = run_clustering(features_for_clustering, method, n_clusters, cov_type)
    plot_df["cluster"] = cr.labels.astype(str)
    color_col = "cluster"

    # ── Metric cards ──
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{n_clusters}</div>
                <div class="metric-label">Clusters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc2:
        sil_display = f"{cr.silhouette:.3f}" if cr.silhouette is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{sil_display}</div>
                <div class="metric-label">Silhouette Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc3:
        if method == "kmeans":
            extra_label = "Inertia"
            extra_value = f"{cr.inertia:,.0f}" if cr.inertia is not None else "N/A"
        else:
            extra_label = "BIC"
            extra_value = f"{cr.bic:,.0f}" if cr.bic is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{extra_value}</div>
                <div class="metric-label">{extra_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── Plot ──
hover_cols = ["track_name", "artists", "track_genre"]
if "cluster" in plot_df.columns:
    hover_cols.append("cluster")

if n_dims == 2:
    fig = px.scatter(
        plot_df,
        x="pca_1",
        y="pca_2",
        color=color_col,
        hover_data=hover_cols,
        labels={
            "pca_1": f"PC1 ({variance_ratio[0]:.1%} variance)",
            "pca_2": f"PC2 ({variance_ratio[1]:.1%} variance)",
        },
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set2 if color_col == "cluster" else None,
    )
    fig.update_layout(
        height=650,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#555"),
        legend=dict(
            title=color_col.replace("_", " ").title(),
            font=dict(size=10, color="#555"),
        ),
    )
    fig.update_xaxes(gridcolor="#f0e6ff", zerolinecolor="#e0d0f0")
    fig.update_yaxes(gridcolor="#f0e6ff", zerolinecolor="#e0d0f0")
else:
    fig = px.scatter_3d(
        plot_df,
        x="pca_1",
        y="pca_2",
        z="pca_3",
        color=color_col,
        hover_data=hover_cols,
        labels={
            "pca_1": f"PC1 ({variance_ratio[0]:.1%})",
            "pca_2": f"PC2 ({variance_ratio[1]:.1%})",
            "pca_3": f"PC3 ({variance_ratio[2]:.1%})",
        },
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set2 if color_col == "cluster" else None,
    )
    fig.update_layout(
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#555"),
        legend=dict(
            title=color_col.replace("_", " ").title(),
            font=dict(size=10, color="#555"),
        ),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#f0e6ff"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#f0e6ff"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#f0e6ff"),
        ),
    )

st.plotly_chart(fig, use_container_width=True)

# ── Evaluation charts (optional) ──
if show_eval and color_mode in ("K-Means Cluster", "GMM Cluster"):
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📈 Cluster Evaluation")

    k_range = list(range(2, 13))

    if color_mode == "K-Means Cluster":
        eval_col1, eval_col2 = st.columns(2)

        with eval_col1:
            with st.spinner("Computing elbow curve..."):
                elbow_df = run_elbow(features_for_clustering, k_range)
            fig_elbow = px.line(
                elbow_df, x="k", y="inertia",
                markers=True,
                title="Elbow Method (Inertia vs. k)",
                labels={"k": "Number of Clusters (k)", "inertia": "Inertia"},
            )
            fig_elbow.update_traces(line_color="#a855f7", marker_color="#e84393")
            fig_elbow.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#555"),
            )
            fig_elbow.update_xaxes(gridcolor="#f0e6ff")
            fig_elbow.update_yaxes(gridcolor="#f0e6ff")
            st.plotly_chart(fig_elbow, use_container_width=True)

        with eval_col2:
            with st.spinner("Computing silhouette scores..."):
                sil_df = run_silhouette(features_for_clustering, k_range)
            fig_sil = px.line(
                sil_df, x="k", y="silhouette_score",
                markers=True,
                title="Silhouette Score vs. k",
                labels={"k": "Number of Clusters (k)", "silhouette_score": "Silhouette Score"},
            )
            fig_sil.update_traces(line_color="#a855f7", marker_color="#e84393")
            fig_sil.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#555"),
            )
            fig_sil.update_xaxes(gridcolor="#f0e6ff")
            fig_sil.update_yaxes(gridcolor="#f0e6ff")
            st.plotly_chart(fig_sil, use_container_width=True)

    else:  # GMM
        with st.spinner("Computing BIC/AIC curves..."):
            bic_aic_df = run_bic_aic(features_for_clustering, k_range, cov_type)

        fig_bic = go.Figure()
        fig_bic.add_trace(go.Scatter(
            x=bic_aic_df["k"], y=bic_aic_df["bic"],
            mode="lines+markers", name="BIC",
            line=dict(color="#a855f7", width=2),
            marker=dict(color="#e84393", size=8),
        ))
        fig_bic.add_trace(go.Scatter(
            x=bic_aic_df["k"], y=bic_aic_df["aic"],
            mode="lines+markers", name="AIC",
            line=dict(color="#6c5ce7", width=2, dash="dash"),
            marker=dict(color="#6c5ce7", size=8),
        ))
        fig_bic.update_layout(
            title="BIC & AIC vs. Number of Components",
            xaxis_title="Number of Components (k)",
            yaxis_title="Score",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color="#555"),
        )
        fig_bic.update_xaxes(gridcolor="#f0e6ff")
        fig_bic.update_yaxes(gridcolor="#f0e6ff")
        st.plotly_chart(fig_bic, use_container_width=True)

# ── About section ──
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
with st.expander("ℹ️ About this visualization"):
    total_var = sum(variance_ratio[:n_dims])
    dim_labels = " + ".join([f"PC{i+1} ({variance_ratio[i]:.1%})" for i in range(n_dims)])
    st.markdown(
        f"""
        This plot uses **PCA** to project each song's processed audio feature vector
        into a **{n_dims}D** space. Each point represents one song, and songs that appear
        closer together have more similar audio feature patterns.

        **Principal components:** {dim_labels} = **{total_var:.1%}** total variance explained.

        **Clustering modes:**
        - **K-Means** — partitions songs into *k* hard clusters by minimizing within-cluster distances.
        - **GMM** — models each cluster as a Gaussian distribution, allowing soft (probabilistic) assignments.

        **Evaluation metrics:**
        - **Silhouette Score** — measures how similar a song is to its own cluster vs. neighboring clusters (higher is better, max = 1.0).
        - **Inertia** — sum of squared distances to cluster centres (lower is tighter; used for the elbow method).
        - **BIC / AIC** — Bayesian / Akaike information criteria for GMM model selection (lower is better).
        """
    )

# ── Footer ──
st.markdown(
    """
    <div style="text-align: center; color: #aaa; padding: 1rem 0; font-size: 0.85rem;">
        Built using Streamlit · Clustering powered by scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)
