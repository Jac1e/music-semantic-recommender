# Music Semantic Recommender

## Project Overview

Music Semantic Recommender is a web-based music exploration and recommendation system that helps users discover songs through both audio similarity and lyrical meaning. Given a song title or a natural-language description such as “sad but energetic” or “calm late-night vibe,” the application returns relevant tracks and visualizes how they relate to one another in a lower-dimensional latent space. The goal of the project is to make music recommendation more interpretable and interactive by combining audio-feature similarity with lyric-based semantic search.

## Dataset

This project uses two datasets:

1. **Spotify dataset (`spotify52kData.csv`)**
   - Contains approximately 52,000 songs.
   - Includes song metadata and audio features such as:
     - `artists`
     - `album_name`
     - `track_name`
     - `popularity`
     - `duration_ms`
     - `explicit`
     - `danceability`
     - `energy`
     - `loudness`
     - `speechiness`
     - `acousticness`
     - `instrumentalness`
     - `liveness`
     - `valence`
     - `tempo`
     - `track_genre`

2. **Genius lyrics dataset**
   - Provides song lyrics used to augment the Spotify dataset.
   - Lyrics are merged into the Spotify table by song title to support semantic search and lyric-based retrieval.

After merging and preprocessing, the cleaned dataset is stored in `data/processed/`.

## Project Goals

The system is designed to support several user-facing tasks:

- find songs that are similar to a selected input song
- explore clusters of related songs
- visualize songs in a 2D or 3D latent space
- search for songs using lyric meaning or mood-based text prompts

## Methods

The project uses methods drawn from machine learning and representation learning:

- **Preprocessing and feature engineering**
  - Remove duplicates
  - Clean merged Spotify + lyrics data
  - Standardize numerical song attributes
  - Build final song feature vectors

- **Similarity-based retrieval**
  - Cosine similarity
  - k-nearest neighbors (KNN)

- **Visualization**
  - Principal component analysis (PCA)
  - Optional deep autoencoder for nonlinear latent-space projection

- **Clustering**
  - k-means clustering
  - Optional Gaussian mixture model (GMM)

- **Lyric-based semantic search**
  - Text embeddings for lyrics and user prompts
  - Semantic similarity between lyric representations and text queries

## Repository Structure

```text
music-semantic-recommender/
├── README.md
│   Project overview, dataset details, and setup and usage guide.
├── requirements.txt
│   Packages needed for the project.
├── environment.yml
│   Environment configuration.
├── app/
│   Frontend code.
│   ├── main.py
│   │   Entry point for the Streamlit application.
│   └── pages/
│       ├── recommend.py
│       │   Similarity-based recommendation interface.
│       ├── visualize.py
│       │   Song visualization in latent space.
│       └── lyrics_search.py
│           Lyric-based semantic retrieval.
├── src/
│   Machine learning and data-processing code.
│   ├── data_loader.py
│   │   Augments the Spotify dataset with lyrics data.
│   ├── preprocess.py
│   │   Cleans data and removes duplicates.
│   ├── similarity.py
│   │   Computes cosine similarity and retrieval.
│   ├── clustering.py
│   │   Performs k-means clustering to group similar songs.
│   ├── reduction.py
│   │   Performs PCA for visualization.
│   ├── text_embed.py
│   │   Embeds lyrics and text prompts.
│   └── utils.py
│       Shared helper functions used across the project.
├── data/
│   Raw and processed datasets.
│   ├── raw/
│   │   Original Spotify and lyrics files.
│   └── processed/
│       Cleaned and merged data.
└── tests/
    Tests for the modules.
    ├── test_preprocess.py
    │   Tests cleaning.
    ├── test_similarity.py
    │   Tests retrieval and cosine similarity.
    └── test_merge.py
        Tests lyrics merging.
```

## Team Responsibilities

- **Huajie Zeng** — Data preprocessing and song vector representation, including removing duplicates, standardizing numerical audio features, and constructing final song feature vectors. Also contributed test coverage for preprocessing, similarity, and data merging.
- **Jennifer Ran** — Streamlit frontend development across all four pages, dynamic mood theming, UI/UX design, integration of all backend modules, duplicate fix in recommendations, and k-means from scratch implementation.
- **Angela Gu** — PCA dimensionality reduction pipeline in reduction.py and pre-scaled audio feature vectors used for visualization.
- **Michael Wang** — Clustering module (k-means and GMM), cluster evaluation utilities (elbow method, silhouette analysis, BIC/AIC),clustering integration in the visualization page, advanced recommendation features, and preprocessing pipeline expansion.
- **Leo Li** — Lyric text embeddings, sentence-transformer model integration, precomputed lyrics embeddings pipeline, and MusicRecommender backend combining audio and lyric features for semantic similarity search.

## Setup

### Option 1: pip

```bash
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate music-semantic-recommender
```

## Running the App

From the project root directory, run:

```bash
streamlit run app/main.py
```

## Precompute Lyrics Embeddings

To avoid recomputing lyric embeddings at runtime, precompute them once and save to `data/processed/lyrics_embeddings.npy`:

```bash
python -m src.precompute_lyrics_embeddings
```

Optional flags:

```bash
python -m src.precompute_lyrics_embeddings --batch-size 64 --hide-progress
python -m src.precompute_lyrics_embeddings --output data/processed/lyrics_embeddings.npy
```

## Expected Inputs and Outputs

### Inputs

- a song title selected by the user
- optional filters or settings
- a natural-language lyric or mood query

### Outputs

- recommended similar songs
- nearest-neighbor retrieval results
- cluster assignments
- low-dimensional visualizations
- lyric-based semantic search results

## Current Status

The application is fully implemented. All backend modules are integrated into a working Streamlit app with four pages: a landing page, mood-based lyrics search, song recommendations, and an interactive visualization.

## Notes

- The lyrics merge is based primarily on song title and may introduce mismatches for songs with identical names by different artists.
- Processed data files should be stored in `data/processed/`.
- Team members should keep module ownership clear, but the final application should be fully integrated and documented.
