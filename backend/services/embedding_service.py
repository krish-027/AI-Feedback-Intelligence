import os
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class EmbeddingService(Embeddings):
    """
    Sentence Transformer embedding service compatible with LangChain.

    The class implements:
    - embed_documents(): embeddings for multiple documents
    - embed_query(): embedding for one query

    By normalizing vectors, cosine similarity can later be implemented
    through FAISS inner-product search.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize the embedding model.

        Args:
            model_name:
                Hugging Face Sentence Transformer model name.
                If omitted, EMBEDDING_MODEL from the environment is used,
                otherwise the project default is used.

            batch_size:
                Number of texts encoded in one batch.
        """

        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL")
            or self.DEFAULT_MODEL_NAME
        )

        self.batch_size = batch_size

        print(f"Loading embedding model: {self.model_name}")

        self.model = SentenceTransformer(self.model_name)

        # Determine embedding dimensionality once during initialization.
        dimension = self.model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                "Could not determine embedding dimension "
                f"for model: {self.model_name}"
            )

        self._dimension: int = dimension

        self._dimension: int = dimension

        print(
                f"Embedding model loaded successfully "
                f"(dimension={self._dimension})"
            )

    @property
    def dimension(self) -> int:
        """
        Return the dimensionality of generated embeddings.
        """

        return self._dimension

    def _validate_texts(self, texts: Sequence[str]) -> list[str]:
        """
        Validate and clean input texts before encoding.
        """

        if not texts:
            raise ValueError("At least one text must be provided.")

        cleaned_texts: list[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Text at index {index} must be a string, "
                    f"got {type(text).__name__}."
                )

            cleaned_text = text.strip()

            if not cleaned_text:
                raise ValueError(
                    f"Text at index {index} is empty."
                )

            cleaned_texts.append(cleaned_text)

        return cleaned_texts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate normalized embeddings for multiple documents.

        Args:
            texts:
                List of document texts.

        Returns:
            List of embeddings, where each embedding is a list of floats.
        """

        cleaned_texts = self._validate_texts(texts)

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Generate a normalized embedding for one query.
        """

        cleaned_texts = self._validate_texts([text])

        embedding = self.model.encode(
            cleaned_texts[0],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embedding = np.asarray(embedding, dtype=np.float32)

        return embedding.tolist()


# ---------------------------------------------------------------------------
# Simple factory function
# ---------------------------------------------------------------------------

_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """
    Return a shared EmbeddingService instance.

    Loading a Sentence Transformer model is relatively expensive, so we
    avoid loading the model repeatedly whenever different parts of the
    application request the embedding service.
    """

    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService()

    return _embedding_service