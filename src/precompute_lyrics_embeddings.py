from __future__ import annotations
"""Offline utility for generating lyrics embeddings.

Notes:
- This script is intentionally offline/precompute-only and does not modify app search logic.
- The saved matrix row order matches the processed dataset rows after filtering to non-null
    lyrics. Consumers should use the same filtering when aligning song rows to embeddings.
- Output defaults to data/processed/lyrics_embeddings.npy and is overwritten on rerun.
"""

import argparse
from pathlib import Path

import numpy as np

from src.data_loader import PROJECT_ROOT, load_spotify_with_lyrics
from src.text_embed import DEFAULT_MODEL_NAME, embed_texts, load_embedding_model

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "lyrics_embeddings.npy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute lyrics embeddings and save them to a .npy file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for embeddings (.npy).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Sentence-transformers model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--hide-progress",
        action="store_true",
        help="Disable progress bar while generating embeddings.",
    )
    return parser


def load_lyrics_texts() -> list[str]:
    """Load lyrics in the same row order used by current app-level lyrics filtering."""
    df = load_spotify_with_lyrics()
    if "lyrics" not in df.columns:
        raise ValueError("Expected a 'lyrics' column in processed dataset.")

    lyrics_series = df[df["lyrics"].notna()]["lyrics"].astype(str)
    lyrics = lyrics_series.tolist()
    if not lyrics:
        raise ValueError("No non-null lyrics found in processed dataset.")
    return lyrics


def precompute_lyrics_embeddings(
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    show_progress_bar: bool = True,
) -> tuple[int, int, Path]:
    """Generate and persist embeddings; returns (rows, dimensions, output_path)."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lyrics = load_lyrics_texts()
    model = load_embedding_model(model_name=model_name)
    embeddings = embed_texts(
        lyrics,
        model=model,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    np.save(output_path, embeddings)
    rows, dims = embeddings.shape
    return rows, dims, output_path


def main() -> None:
    args = build_parser().parse_args()

    rows, dims, output_path = precompute_lyrics_embeddings(
        output_path=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        show_progress_bar=not args.hide_progress,
    )

    print(f"Saved lyrics embeddings to: {output_path}")
    print(f"Rows embedded: {rows}")
    print(f"Embedding dimensions: {dims}")


if __name__ == "__main__":
    main()
