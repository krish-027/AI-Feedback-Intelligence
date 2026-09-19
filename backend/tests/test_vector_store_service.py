from pathlib import Path

from langchain_core.documents import Document

from backend.services.vector_store_service import (
    FAISSVectorStoreService,
    SentenceTransformerEmbeddings,
)


def test_embedding_dimension() -> None:
    """The configured Sentence Transformer should produce 384-dimensional vectors."""

    embedding_service = SentenceTransformerEmbeddings()

    assert embedding_service.get_dimension() == 384


def test_pii_is_removed_before_embedding() -> None:
    """Customer PII should be removed from the text used for embeddings."""

    service = FAISSVectorStoreService()

    document = Document(
        page_content=(
            "Customer Name: Rahul Sharma\n"
            "Account Number: 1234567890\n"
            "Contact Number: 9876543210\n"
            "The staff was very helpful and polite."
        ),
        metadata={
            "category": "Good",
            "feedback_id": "FB-TEST001",
            "split": "reference",
        },
    )

    prepared_documents = service._prepare_embedding_documents(
        [document]
    )

    assert len(prepared_documents) == 1

    embedding_text = prepared_documents[0].page_content

    assert "Rahul Sharma" not in embedding_text
    assert "1234567890" not in embedding_text
    assert "9876543210" not in embedding_text

    assert "staff was very helpful" in embedding_text


def test_embedding_document_preserves_category() -> None:
    """Embedding preparation should preserve the feedback category metadata."""

    service = FAISSVectorStoreService()

    document = Document(
        page_content="The staff was very helpful.",
        metadata={
            "category": "Good",
            "feedback_id": "FB-TEST002",
            "split": "reference",
        },
    )

    prepared_documents = service._prepare_embedding_documents(
        [document]
    )

    assert len(prepared_documents) == 1

    metadata = prepared_documents[0].metadata

    assert metadata["category"] == "Good"
    assert metadata["feedback_id"] == "FB-TEST002"
    assert metadata["split"] == "reference"


def test_manifest_contains_expected_splits() -> None:
    """The dataset manifest should contain the 60/20/20 dataset split."""

    service = FAISSVectorStoreService()

    records = service._load_manifest()

    assert len(records) == 100

    splits = {
        record["split"]
        for record in records
    }

    assert splits == {
        "reference",
        "development",
        "final_benchmark",
    }

    reference_count = sum(
        1
        for record in records
        if record["split"] == "reference"
    )

    development_count = sum(
        1
        for record in records
        if record["split"] == "development"
    )

    final_benchmark_count = sum(
        1
        for record in records
        if record["split"] == "final_benchmark"
    )

    assert reference_count == 60
    assert development_count == 20
    assert final_benchmark_count == 20


def test_build_faiss_vector_store(tmp_path: Path) -> None:
    """Build a real FAISS index from the 60 reference documents."""

    test_vector_store_path = (
        tmp_path / "test_customer_feedback"
    )

    service = FAISSVectorStoreService(
        vector_store_path=test_vector_store_path
    )

    vector_store = service.build()

    assert vector_store is not None

    assert (
        test_vector_store_path / "index.faiss"
    ).exists()

    assert (
        test_vector_store_path / "index.pkl"
    ).exists()


def test_retrieval_returns_reference_metadata(
    tmp_path: Path,
) -> None:
    """Retrieved documents should come only from the reference split."""

    test_vector_store_path = (
        tmp_path / "test_customer_feedback"
    )

    service = FAISSVectorStoreService(
        vector_store_path=test_vector_store_path
    )

    service.build()

    results = service.similarity_search(
        "The staff was very helpful.",
        k=3,
    )

    assert len(results) > 0

    for document in results:
        assert document.metadata["split"] == "reference"

        assert document.metadata["category"] in {
            "Excellent",
            "Good",
            "Need Improvements",
            "Poor",
        }

        assert document.metadata["feedback_id"]


def test_vector_store_can_be_loaded_after_persistence(
    tmp_path: Path,
) -> None:
    """A persisted FAISS index should be loadable again."""

    test_vector_store_path = (
        tmp_path / "test_customer_feedback"
    )

    service = FAISSVectorStoreService(
        vector_store_path=test_vector_store_path
    )

    service.build()

    new_service = FAISSVectorStoreService(
        vector_store_path=test_vector_store_path
    )

    loaded = new_service.load()

    assert loaded is not None

    results = new_service.similarity_search(
        "The staff was very helpful.",
        k=2,
    )

    assert len(results) > 0