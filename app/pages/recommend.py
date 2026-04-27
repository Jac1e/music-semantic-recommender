import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
from urllib.parse import quote

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data_loader import load_spotify_with_lyrics
from src.clustering import fit_kmeans, kmeans_elbow, silhouette_analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
        color: #a855f7 !important;
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
    .map-section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid #f0e6ff;
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.06);
        margin: 1rem 0;
    }
    .map-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #333;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .map-subtitle {
        font-size: 0.9rem;
        color: #888;
        text-align: center;
        margin-bottom: 1rem;
    }
    .cluster-metric {
        background: linear-gradient(135deg, #f3e8ff 0%, #ffe0ec 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e8d5f5;
    }
    .cluster-metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #a855f7;
    }
    .cluster-metric-label {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.2rem;
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
selected_songs = st.multiselect(
    "Choose up to 5 songs to blend their vibes:",
    options=df["display_name"].tolist(),
    max_selections=5,
    placeholder="Start typing song or artist names...",
)

col_toggle, col_vibe = st.columns([1, 1])
with col_toggle:
    st.markdown("<br>", unsafe_allow_html=True)
    use_pure_audio = st.toggle("Ignore lyrics", value=True)
with col_vibe:
    vibe_modifier = st.text_input("Input a musical vibe", placeholder="e.g., calm piano, high energy...", key="vibe_modifier")

col_slider, col_spacer = st.columns([1, 3])
with col_slider:
    k = st.slider("Number of recommendations:", min_value=3, max_value=15, value=5)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

if selected_songs or vibe_modifier:
    # Selected songs display
    selected_song_rows = []
    if selected_songs:
        st.markdown("### 🎵 Your Selected Vibes")
        cols = st.columns(len(selected_songs))
        
        for i, song_name in enumerate(selected_songs):
            song_idx = df[df["display_name"] == song_name].index[0]
            song_data = df.iloc[song_idx]
            selected_song_rows.append(song_data)
            
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="selected-song-card" style="padding: 1rem;">
                        <div class="selected-song-title" style="font-size: 1.1rem;">{song_data['track_name']}</div>
                        <div class="selected-song-artist" style="font-size: 0.9rem;">{song_data['artists']}</div>
                        <span class="genre-tag genre-tag-purple" style="font-size: 0.7rem;">
                            {song_data['track_genre']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Compute average features for the combined vibe radar
        avg_selected_song = pd.Series(dtype="object")
        for feat in radar_features:
            avg_selected_song[feat] = np.mean([song[feat] for song in selected_song_rows])
        avg_selected_song["track_name"] = "Combined Vibe"
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# Compute recommendations using MusicRecommender (audio + lyrics combined)
    from src.similarity import MusicRecommender
    from src.preprocess import clean_text

    @st.cache_resource
    def get_recommender():
        raw_df = load_spotify_with_lyrics()
        return MusicRecommender.fit(raw_df)

    @st.cache_resource
    def compute_similarity_fallback(dataframe):
        """Fallback: audio-only similarity if MusicRecommender drops the song."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(dataframe[feature_cols].fillna(0))
        return features

    try:
        if use_pure_audio:
            raise ValueError("Forced pure audio mode.")
            
        recommender = get_recommender()
        rec_df = recommender.df
        
        rec_song_indices = []
        for song_data in selected_song_rows:
            match = rec_df[
                (rec_df["track_name"] == song_data["track_name"]) &
                (rec_df["artists"] == song_data["artists"])
            ]
            if len(match) > 0:
                rec_song_indices.append(match.index[0])
            else:
                raise ValueError("A song was not found in recommender (missing lyrics).")

        if rec_song_indices:
            query_vec = np.mean([recommender.features[i] for i in rec_song_indices], axis=0).reshape(1, -1)
        else:
            query_vec = None
            
        if vibe_modifier:
            audio_query_text_vec = recommender.model.encode([clean_text(vibe_modifier)], convert_to_numpy=True)
            if not hasattr(recommender, 'bridge_model'):
                recommender.train_cross_modal_bridge()
            query_audio_pred = recommender.bridge_model.predict(audio_query_text_vec)
            
            # Pad text features with zeros because vibe modifier only specifies audio structure
            vibe_vec = np.hstack([query_audio_pred, np.zeros((1, recommender.features.shape[1] - recommender.numeric_feature_count))])
            
            if query_vec is not None:
                query_vec = np.mean(np.vstack([query_vec, vibe_vec]), axis=0).reshape(1, -1)
            else:
                query_vec = vibe_vec

        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_vec, recommender.features)[0]
        top_indices = np.argsort(scores)[::-1][:k*3]
        
        columns = ["track_name", "artists", "track_genre"]
        raw_results = recommender.df.iloc[top_indices][columns].copy()
        raw_results["score"] = scores[top_indices]

        # Remove the selected songs themselves and deduplicate
        selected_names = [s["track_name"] for s in selected_song_rows]
        selected_artists = [s["artists"] for s in selected_song_rows]
        
        results = raw_results[
            ~raw_results.apply(lambda row: row["track_name"] in selected_names and row["artists"] in selected_artists, axis=1)
        ].copy()
        
        results = results.drop_duplicates(subset=["track_name", "artists"], keep="first")
        results = results.head(k)
        results = results.rename(columns={"score": "similarity_score"})

        # Merge back full song data for radar charts
        results = results.merge(
            df[["track_name", "artists", "track_genre", "popularity"] + radar_features].drop_duplicates(
                subset=["track_name", "artists"], keep="first"
            ),
            on=["track_name", "artists"],
            how="left",
            suffixes=("", "_dup"),
        )

    except Exception:
        # Fallback to audio-only similarity
        from sklearn.metrics.pairwise import cosine_similarity
        features = compute_similarity_fallback(df)
        
        if selected_song_rows:
            song_indices = [df[df["display_name"] == s["display_name"]].index[0] for s in selected_song_rows]
            query_vec = np.mean(features[song_indices], axis=0).reshape(1, -1)
        else:
            query_vec = None
            
        if vibe_modifier:
            recommender = get_recommender()
            audio_query_text_vec = recommender.model.encode([clean_text(vibe_modifier)], convert_to_numpy=True)
            if not hasattr(recommender, 'bridge_model'):
                recommender.train_cross_modal_bridge()
            query_audio_pred = recommender.bridge_model.predict(audio_query_text_vec)
            
            if query_vec is not None:
                query_vec = np.mean(np.vstack([query_vec, query_audio_pred]), axis=0).reshape(1, -1)
            else:
                query_vec = query_audio_pred
        
        scores = cosine_similarity(query_vec, features)[0]

        top_indices = np.argsort(scores)[::-1][:max(1, k+len(selected_songs))*3]
        results = df.iloc[top_indices].copy()
        results["similarity_score"] = scores[top_indices]
        
        selected_names = [s["track_name"] for s in selected_song_rows]
        selected_artists = [s["artists"] for s in selected_song_rows]
        
        results = results[
            ~results.apply(lambda row: row["track_name"] in selected_names and row["artists"] in selected_artists, axis=1)
        ]
        results = results.drop_duplicates(subset=["track_name", "artists"], keep="first")
        results = results.head(k)
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <span style="font-size: 1.3rem; color: #333;">
                Songs matching your
                <span style="color: #a855f7; font-weight: 700;">
                    Combined Vibe
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
            if len(selected_song_rows) > 0:
                fig = make_comparison_radar(
                    avg_selected_song, song, radar_features, radar_labels,
                )
            else:
                # If no seed songs were provided, just display the radar chart for the song
                # Reusing make_radar_chart from lyrics_search.py logic, but customized here:
                values = [song[f] for f in radar_features]
                values.append(values[0])
                labels_plot = list(radar_labels) + [radar_labels[0]]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values, theta=labels_plot, fill="toself",
                    fillcolor="rgba(168, 85, 247, 0.15)",
                    line=dict(color="#a855f7", width=2),
                    marker=dict(size=5, color="#a855f7"),
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="#f0e6ff"),
                        angularaxis=dict(gridcolor="#f0e6ff", linecolor="#f0e6ff"),
                    ),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=20, b=20),
                    height=200,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
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

    # ── Recommendation Map ──
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="map-title">🗺️ Recommendation Map</div>
        <div class="map-subtitle">See how your selections and recommendations relate in audio feature space</div>
        """,
        unsafe_allow_html=True,
    )

    # Build combined dataframe of input songs + recommendations
    map_rows = []
    for song_data in selected_song_rows:
        row = {feat: song_data[feat] for feat in feature_cols}
        row["track_name"] = song_data["track_name"]
        row["artists"] = song_data["artists"]
        row["track_genre"] = song_data["track_genre"]
        row["role"] = "🎵 Your Pick"
        row["similarity_score"] = 1.0
        map_rows.append(row)

    for rank, (_, song) in enumerate(results.iterrows()):
        row = {feat: song[feat] for feat in feature_cols if feat in song.index}
        row["track_name"] = song["track_name"]
        row["artists"] = song["artists"]
        row["track_genre"] = song["track_genre"]
        row["role"] = "💜 Recommended"
        row["similarity_score"] = song.get("similarity_score", 0)
        map_rows.append(row)

    if len(map_rows) >= 2:
        map_df = pd.DataFrame(map_rows)
        # Ensure all feature cols are present
        available_feats = [f for f in feature_cols if f in map_df.columns]
        feat_matrix = map_df[available_feats].fillna(0).values

        # PCA projection to 2D
        scaler_map = StandardScaler()
        feat_scaled = scaler_map.fit_transform(feat_matrix)
        pca_map = PCA(n_components=2)
        coords = pca_map.fit_transform(feat_scaled)
        map_df["PC1"] = coords[:, 0]
        map_df["PC2"] = coords[:, 1]
        map_df["label"] = map_df["track_name"] + " — " + map_df["artists"]
        map_df["size"] = map_df["role"].apply(lambda r: 18 if "Your Pick" in r else 12)

        var1 = pca_map.explained_variance_ratio_[0]
        var2 = pca_map.explained_variance_ratio_[1]

        # Compute centroid of input songs
        input_mask = map_df["role"].str.contains("Your Pick")
        centroid_x = map_df.loc[input_mask, "PC1"].mean()
        centroid_y = map_df.loc[input_mask, "PC2"].mean()

        fig_map = px.scatter(
            map_df,
            x="PC1",
            y="PC2",
            color="role",
            size="size",
            hover_data=["track_name", "artists", "track_genre", "similarity_score"],
            text="track_name",
            color_discrete_map={
                "🎵 Your Pick": "#e84393",
                "💜 Recommended": "#a855f7",
            },
            labels={
                "PC1": f"PC1 ({var1:.0%} var)",
                "PC2": f"PC2 ({var2:.0%} var)",
            },
        )

        # Add connecting lines from each recommendation to the input centroid
        rec_mask = map_df["role"].str.contains("Recommended")
        for _, rec_row in map_df[rec_mask].iterrows():
            fig_map.add_trace(go.Scatter(
                x=[centroid_x, rec_row["PC1"]],
                y=[centroid_y, rec_row["PC2"]],
                mode="lines",
                line=dict(color="rgba(168, 85, 247, 0.2)", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig_map.update_traces(
            textposition="top center",
            textfont=dict(size=9, color="#555"),
            selector=dict(mode="markers+text"),
        )
        fig_map.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#555"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
            margin=dict(l=60, r=60, t=40, b=60),
        )
        fig_map.update_xaxes(gridcolor="#f0e6ff", zerolinecolor="#e0d0f0")
        fig_map.update_yaxes(gridcolor="#f0e6ff", zerolinecolor="#e0d0f0")

        st.plotly_chart(fig_map, use_container_width=True)

        # ── Cluster Analysis ──
        with st.expander("📊 Cluster Analysis"):
            max_k = min(len(map_df) - 1, 8)
            if max_k >= 2:
                k_range = list(range(2, max_k + 1))

                col_elbow, col_sil = st.columns(2)

                with col_elbow:
                    with st.spinner("Computing elbow curve..."):
                        elbow_records = []
                        for kk in k_range:
                            km = fit_kmeans(feat_scaled, n_clusters=kk, random_state=42)
                            elbow_records.append({"k": kk, "inertia": km.inertia_})
                        elbow_df = pd.DataFrame(elbow_records)

                    fig_elbow = px.line(
                        elbow_df, x="k", y="inertia",
                        markers=True,
                        title="Elbow Method (Inertia vs k)",
                        labels={"k": "Number of Clusters (k)", "inertia": "Inertia"},
                    )
                    fig_elbow.update_traces(line_color="#a855f7", marker_color="#e84393")
                    fig_elbow.update_layout(
                        height=320,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#555"),
                    )
                    fig_elbow.update_xaxes(gridcolor="#f0e6ff")
                    fig_elbow.update_yaxes(gridcolor="#f0e6ff")
                    st.plotly_chart(fig_elbow, use_container_width=True)

                with col_sil:
                    with st.spinner("Computing silhouette scores..."):
                        from sklearn.metrics import silhouette_score as sk_silhouette
                        sil_records = []
                        for kk in k_range:
                            km = fit_kmeans(feat_scaled, n_clusters=kk, random_state=42)
                            score = sk_silhouette(feat_scaled, km.labels_)
                            sil_records.append({"k": kk, "silhouette_score": score})
                        sil_df = pd.DataFrame(sil_records)

                    fig_sil = px.line(
                        sil_df, x="k", y="silhouette_score",
                        markers=True,
                        title="Silhouette Score vs k",
                        labels={"k": "Number of Clusters (k)", "silhouette_score": "Silhouette Score"},
                    )
                    fig_sil.update_traces(line_color="#a855f7", marker_color="#e84393")
                    fig_sil.update_layout(
                        height=320,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#555"),
                    )
                    fig_sil.update_xaxes(gridcolor="#f0e6ff")
                    fig_sil.update_yaxes(gridcolor="#f0e6ff")
                    st.plotly_chart(fig_sil, use_container_width=True)

                # Optimal k suggestion via largest second derivative of inertia
                inertias = elbow_df["inertia"].values
                if len(inertias) >= 3:
                    second_deriv = np.diff(inertias, 2)
                    best_k_idx = np.argmax(second_deriv) + 2  # offset by 2 for the range start
                    best_k = k_range[best_k_idx] if best_k_idx < len(k_range) else k_range[-1]
                else:
                    best_k = 2

                # Run clustering at best k for the summary
                best_km = fit_kmeans(feat_scaled, n_clusters=best_k, random_state=42)
                map_df["cluster"] = best_km.labels_
                best_sil = sk_silhouette(feat_scaled, best_km.labels_) if len(set(best_km.labels_)) > 1 else 0.0

                # Metric cards
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.markdown(
                        f'<div class="cluster-metric"><div class="cluster-metric-value">{best_k}</div><div class="cluster-metric-label">Suggested Clusters</div></div>',
                        unsafe_allow_html=True,
                    )
                with mc2:
                    st.markdown(
                        f'<div class="cluster-metric"><div class="cluster-metric-value">{best_sil:.3f}</div><div class="cluster-metric-label">Silhouette Score</div></div>',
                        unsafe_allow_html=True,
                    )
                with mc3:
                    total_var = var1 + var2
                    st.markdown(
                        f'<div class="cluster-metric"><div class="cluster-metric-value">{total_var:.0%}</div><div class="cluster-metric-label">Variance Explained (2 PCs)</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Per-song summary table
                summary_rows = []
                for _, row in map_df.iterrows():
                    # Find top 2 dominant audio features
                    feat_vals = {f: row.get(f, 0) for f in available_feats}
                    sorted_feats = sorted(feat_vals.items(), key=lambda x: abs(x[1]), reverse=True)
                    top_feats = ", ".join([f"{name}: {val:.2f}" for name, val in sorted_feats[:2]])

                    # Distance to centroid
                    dist = np.sqrt((row["PC1"] - centroid_x)**2 + (row["PC2"] - centroid_y)**2)

                    summary_rows.append({
                        "Song": row["track_name"],
                        "Artist": row["artists"],
                        "Role": row["role"],
                        "Cluster": int(row["cluster"]),
                        "Distance to Input": round(dist, 2),
                        "Top Features": top_feats,
                    })

                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Song": st.column_config.TextColumn(width="medium"),
                        "Role": st.column_config.TextColumn(width="small"),
                        "Cluster": st.column_config.NumberColumn(width="small"),
                        "Distance to Input": st.column_config.NumberColumn(width="small", format="%.2f"),
                    },
                )
            else:
                st.info("Need at least 3 total songs (input + recommended) for cluster analysis. Try increasing the number of recommendations.")

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
