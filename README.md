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
- **Huajie Zeng** — Data preprocessing and song vector representation, including removing duplicates, standardizing numerical features, and constructing final song feature vectors.
- **Jennifer Ran** — Cosine similarity and KNN retrieval to return the top-k most similar tracks for a given input song.
- **Angela Gu** — PCA or deep autoencoder to project high-dimensional song features into a 2D or 3D latent space for visualization.
- **Michael Wang** — k-means or Gaussian mixture model to group songs into clusters in the latent space.
- **Leo Li** — Lyric embeddings, mood-related semantic extraction, and semantic search functionality.

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
This repository currently contains the project structure, preprocessing pipeline, and placeholders or modules for retrieval, clustering, visualization, and semantic lyric search. The final system will integrate these components into a single interactive Streamlit application.

## Notes
- The lyrics merge is based primarily on song title and may introduce mismatches for songs with identical names by different artists.
- Processed data files should be stored in `data/processed/`.
- Team members should keep module ownership clear, but the final application should be fully integrated and documented.
