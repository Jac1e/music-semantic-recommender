from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load the sentence-transformers model used for lyric embeddings."""
    return SentenceTransformer(model_name)


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress_bar: bool = True,
):
    """Embed a list of texts as a NumPy array."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
