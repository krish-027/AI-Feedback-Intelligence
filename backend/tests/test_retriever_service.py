
from pathlib import Path

import pytest
from langchain_core.documents import Document

from backend.services.retriever_service import (
    RetrieverService,
)


REAL_VECTOR_STORE_PATH = Path(
    "backend/vector_store/customer_feedback"
)


def test_retriever_loads_vector_store() -> None:
    """The retriever should load the persisted Stage 6 FAISS index."""

    service = RetrieverService(
        vector_store_path=REAL_VECTOR_STORE_PATH
    )

    service.load()

    assert service.vector_store is not None


def test_similarity_search_returns_reference_documents() -> None:
    """Similarity search should return valid reference feedback."""

    service = RetrieverService(
        vector_store_path=REAL_VECTOR_STORE_PATH
    )

    results = service.similarity_search(
        "The staff was very helpful and polite.",
        k=3,
    )

    assert len(results) == 3

    for document in results:
        assert document.metadata["split"] == "reference"

        assert document.metadata["category"] in {
            "Excellent",
            "Good",
            "Need Improvements",
            "Poor",
        }

        assert document.metadata["feedback_id"]


def test_similarity_search_with_score_returns_scores() -> None:
    """Similarity search with scores should return documents and numeric scores."""

    service = RetrieverService(
        vector_store_path=REAL_VECTOR_STORE_PATH
    )

    results = service.similarity_search_with_score(
        "The waiting time was too long.",
        k=3,
    )

    assert len(results) == 3

    for document, score in results:
        assert isinstance(document, Document)
        assert isinstance(score, float)

        assert document.metadata["split"] == "reference"


def test_mmr_search_returns_reference_documents() -> None:
    """MMR retrieval should return valid and diverse reference documents."""

    service = RetrieverService(
        vector_store_path=REAL_VECTOR_STORE_PATH
    )

    results = service.mmr_search(
        "The waiting time was too long and the staff was not helpful.",
        k=3,
        fetch_k=10,
        lambda_mult=0.5,
    )

    assert len(results) == 3

    for document in results:
        assert document.metadata["split"] == "reference"

        assert document.metadata["category"] in {
            "Excellent",
            "Good",
            "Need Improvements",
            "Poor",
        }

        assert document.metadata["feedback_id"]


def test_empty_query_is_rejected() -> None:
    """An empty query should raise ValueError."""

    service = RetrieverService(
        vector_store_path=REAL_VECTOR_STORE_PATH
    )

    with pytest.raises(ValueError):
        service.similarity_search(
            "",
            k=3,
        )


def test_invalid_k_is_rejected() -> None:
    """A non-positive k value should raise ValueError."""

    service = RetrieverService(
        vector_store_path=REAL_VECTOR_STORE_PATH
    )

    with pytest.raises(ValueError):
        service.similarity_search(
            "The staff was helpful.",
            k=0,
        )


def test_invalid_mmr_parameters_are_rejected() -> None:
    """Invalid MMR configuration should raise ValueError."""

    service = RetrieverService(
        vector_store_path=REAL_VECTOR_STORE_PATH
    )

    with pytest.raises(ValueError):
        service.mmr_search(
            "The service was good.",
            k=3,
            fetch_k=2,
        )

    with pytest.raises(ValueError):
        service.mmr_search(
            "The service was good.",
            k=3,
            fetch_k=10,
            lambda_mult=1.5,
        )


def test_non_reference_document_is_rejected() -> None:
    """The retriever must reject documents outside the reference split."""

    service = RetrieverService(
        vector_store_path=REAL_VECTOR_STORE_PATH
    )

    evaluation_document = Document(
        page_content="This is an evaluation document.",
        metadata={
            "split": "evaluation",
            "category": "Good",
            "feedback_id": "FB-EVAL001",
        },
    )

    with pytest.raises(ValueError):
        service._validate_document(
            evaluation_document
        )


def test_invalid_category_is_rejected() -> None:
    """The retriever must reject documents with unknown categories."""

    service = RetrieverService(
        vector_store_path=REAL_VECTOR_STORE_PATH
    )

    invalid_document = Document(
        page_content="This document has an invalid category.",
        metadata={
            "split": "reference",
            "category": "Maybe",
            "feedback_id": "FB-TEST001",
        },
    )

    with pytest.raises(ValueError):
        service._validate_document(
            invalid_document
        )


def test_missing_feedback_id_is_rejected() -> None:
    """The retriever must reject documents without feedback_id."""

    service = RetrieverService(
        vector_store_path=REAL_VECTOR_STORE_PATH
    )

    invalid_document = Document(
        page_content="Missing feedback identifier.",
        metadata={
            "split": "reference",
            "category": "Good",
        },
    )

    with pytest.raises(ValueError):
        service._validate_document(
            invalid_document
        )