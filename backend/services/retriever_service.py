from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from backend.services.vector_store_service import (
    FAISSVectorStoreService,
)


class RetrieverService:
    """
    Retrieval layer over the Stage 6 FAISS vector store.

    The service loads the persisted FAISS index and exposes similarity
    and MMR retrieval methods while validating the returned metadata.
    """

    DEFAULT_VECTOR_STORE_PATH = Path(
        "backend/vector_store/customer_feedback"
    )

    VALID_CATEGORIES = {
        "Excellent",
        "Good",
        "Need Improvements",
        "Poor",
    }

    REQUIRED_SPLIT = "reference"

    def __init__(
        self,
        vector_store_path: Path | str | None = None,
    ) -> None:
        """
        Initialize the retriever service.

        Args:
            vector_store_path:
                Location of the persisted FAISS vector store.
                If omitted, the production Stage 6 location is used.
        """

        self.vector_store_path = Path(
            vector_store_path
            if vector_store_path is not None
            else self.DEFAULT_VECTOR_STORE_PATH
        )

        self.vector_store_service = FAISSVectorStoreService(
            vector_store_path=self.vector_store_path
        )

        self.vector_store: Any = None

    def load(self) -> None:
        """
        Load the persisted FAISS vector store.

        Raises:
            FileNotFoundError:
                If the persisted FAISS index does not exist.
            RuntimeError:
                If the vector store cannot be loaded.
        """

        if not self.vector_store_path.exists():
            raise FileNotFoundError(
                "FAISS vector store does not exist at: "
                f"{self.vector_store_path}. "
                "Build the Stage 6 vector store first."
            )

        loaded_store = self.vector_store_service.load()

        if loaded_store is None:
            raise RuntimeError(
                "FAISS vector store could not be loaded."
            )

        self.vector_store = loaded_store

    def _ensure_loaded(self) -> None:
        """Load the vector store lazily when required."""

        if self.vector_store is None:
            self.load()

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """
        Retrieve the most similar reference documents.

        Args:
            query:
                Natural-language feedback/query text.
            k:
                Number of results requested.

        Returns:
            A list of validated reference documents.
        """

        self._validate_query(query)
        self._validate_k(k)

        self._ensure_loaded()

        results = self.vector_store.similarity_search(
            query,
            k=k,
        )

        return self._validate_results(results)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve similar reference documents along with FAISS scores.

        Args:
            query:
                Natural-language feedback/query text.
            k:
                Number of results requested.

        Returns:
            A list containing:
                (Document, score)
        """

        self._validate_query(query)
        self._validate_k(k)

        self._ensure_loaded()

        results = self.vector_store.similarity_search_with_score(
            query,
            k=k,
        )

        validated_results: list[tuple[Document, float]] = []

        for document, score in results:
            self._validate_document(document)

            validated_results.append(
                (
                    document,
                    float(score),
                )
            )

        return validated_results

    def mmr_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 12,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """
        Retrieve diverse and relevant reference documents using MMR.

        Maximum Marginal Relevance balances:
        - relevance to the query
        - diversity among retrieved documents

        Args:
            query:
                Natural-language feedback/query text.
            k:
                Number of final documents to return.
            fetch_k:
                Number of candidates considered before MMR selection.
            lambda_mult:
                Relevance/diversity trade-off.

                1.0 -> maximum relevance
                0.0 -> maximum diversity

        Returns:
            A list of validated reference documents.
        """

        self._validate_query(query)
        self._validate_k(k)

        if fetch_k < k:
            raise ValueError(
                "fetch_k must be greater than or equal to k."
            )

        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError(
                "lambda_mult must be between 0.0 and 1.0."
            )

        self._ensure_loaded()

        results = self.vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

        return self._validate_results(results)

    def _validate_query(
        self,
        query: str,
    ) -> None:
        """Validate the retrieval query."""

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

    def _validate_k(
        self,
        k: int,
    ) -> None:
        """Validate the number of retrieval results."""

        if not isinstance(k, int):
            raise TypeError(
                "k must be an integer."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0."
            )

    def _validate_results(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Validate retrieval results.

        Only reference documents with valid category metadata are allowed
        into the downstream RAG pipeline.
        """

        validated_documents: list[Document] = []

        for document in documents:
            self._validate_document(document)
            validated_documents.append(document)

        return validated_documents

    def _validate_document(
        self,
        document: Document,
    ) -> None:
        """Validate metadata on an individual retrieved document."""

        if not isinstance(document, Document):
            raise TypeError(
                "Retrieved object is not a LangChain Document."
            )

        metadata: dict[str, Any] = document.metadata

        split = metadata.get("split")

        if split != self.REQUIRED_SPLIT:
            raise ValueError(
                "Retrieved document is not from the reference split. "
                f"Expected '{self.REQUIRED_SPLIT}', got '{split}'."
            )

        category = metadata.get("category")

        if category not in self.VALID_CATEGORIES:
            raise ValueError(
                "Retrieved document has an invalid category: "
                f"{category}"
            )

        feedback_id = metadata.get("feedback_id")

        if not feedback_id:
            raise ValueError(
                "Retrieved document is missing feedback_id."
            )


def get_retriever_service(
    vector_store_path: Path | str | None = None,
) -> RetrieverService:
    """
    Return a configured RetrieverService.

    This helper keeps service construction consistent across the project.
    """

    return RetrieverService(
        vector_store_path=vector_store_path
    )