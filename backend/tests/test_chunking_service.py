from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from backend.services.chunking_service import (
    DocumentChunkingService,
    get_chunking_service,
)
from backend.services.document_service import get_document_service


def test_short_document_stays_single_chunk() -> None:
    """A short feedback document should remain one chunk."""

    service = DocumentChunkingService(
        chunk_size=900,
        chunk_overlap=150,
    )

    document = Document(
        page_content=(
            "The staff was very helpful and the overall experience "
            "was good."
        ),
        metadata={
            "feedback_id": "TEST-001",
            "source": "test.pdf",
        },
    )

    chunks = service.chunk_document(document)

    assert len(chunks) == 1

    assert chunks[0].page_content == document.page_content.strip()

    assert chunks[0].metadata["feedback_id"] == "TEST-001"
    assert chunks[0].metadata["source"] == "test.pdf"

    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_count"] == 1
    assert chunks[0].metadata["chunk_size"] > 0

    assert chunks[0].metadata["chunk_id"]


def test_long_document_is_split() -> None:
    """A sufficiently long document should produce multiple chunks."""

    service = DocumentChunkingService(
        chunk_size=100,
        chunk_overlap=20,
    )

    text = (
        "The customer service team was helpful. "
        "The waiting time was longer than expected. "
        "The staff eventually resolved the issue. "
        "The final experience was satisfactory. "
        "I would suggest improving response time."
    )

    document = Document(
        page_content=text,
        metadata={
            "feedback_id": "TEST-002",
        },
    )

    chunks = service.chunk_document(document)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.page_content) > 0


def test_chunk_metadata_is_preserved() -> None:
    """Every chunk should retain metadata from the original document."""

    service = DocumentChunkingService(
        chunk_size=100,
        chunk_overlap=20,
    )

    text = "Customer feedback sentence. " * 20

    document = Document(
        page_content=text,
        metadata={
            "feedback_id": "TEST-003",
            "category": "Need Improvements",
            "source": "feedback_003.pdf",
            "page": 1,
        },
    )

    chunks = service.chunk_document(document)

    assert len(chunks) > 1

    for index, chunk in enumerate(chunks):
        assert chunk.metadata["feedback_id"] == "TEST-003"
        assert chunk.metadata["category"] == "Need Improvements"
        assert chunk.metadata["source"] == "feedback_003.pdf"
        assert chunk.metadata["page"] == 1

        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["chunk_count"] == len(chunks)
        assert chunk.metadata["chunk_id"]


def test_chunk_ids_are_unique() -> None:
    """Each chunk should receive a unique deterministic identifier."""

    service = DocumentChunkingService(
        chunk_size=100,
        chunk_overlap=20,
    )

    text = "Customer feedback sentence. " * 20

    document = Document(
        page_content=text,
        metadata={
            "feedback_id": "TEST-004",
        },
    )

    chunks = service.chunk_document(document)

    chunk_ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_empty_document_returns_no_chunks() -> None:
    """An empty document should not create meaningless vector entries."""

    service = DocumentChunkingService()

    document = Document(
        page_content="   ",
        metadata={
            "feedback_id": "TEST-005",
        },
    )

    chunks = service.chunk_document(document)

    assert chunks == []


def test_overlap_exists_between_chunks() -> None:
    """
    Consecutive chunks should share content when the requested overlap
    is large enough to retain at least one logical text segment.
    """

    service = DocumentChunkingService(
        chunk_size=120,
        chunk_overlap=70,
    )

    text = (
        "Customer service was generally satisfactory. "
        "However, the waiting time was longer than expected. "
        "The representative eventually resolved the issue. "
        "The customer appreciated the final resolution. "
        "The customer suggested improving response time."
    ) * 3

    document = Document(
        page_content=text,
        metadata={
            "feedback_id": "TEST-006",
        },
    )

    chunks = service.chunk_document(document)

    assert len(chunks) > 1

    found_overlap = False

    for first, second in zip(chunks, chunks[1:]):
        first_sentences = [
            sentence.strip()
            for sentence in first.page_content.split(".")
            if sentence.strip()
        ]

        for sentence in first_sentences:
            if len(sentence) >= 20 and sentence in second.page_content:
                found_overlap = True
                break

        if found_overlap:
            break

    assert found_overlap


def test_real_feedback_pdf_can_be_chunked() -> None:
    """
    Run the Stage 4 PDF ingestion service followed by Stage 5 chunking.

    This test uses the first actual feedback PDF from the project dataset.
    """

    pdf_path = Path(
        "Sample_Data/customer_feedback_forms/Customer_Feedback_001.pdf"
    )

    assert pdf_path.exists(), (
        f"Expected PDF not found: {pdf_path}"
    )

    document_service = get_document_service()
    chunking_service = get_chunking_service()

    documents = document_service.load_pdf(pdf_path)

    assert documents is not None

    # Stage 4 may return a single Document or multiple page Documents
    # depending on the configured loading behavior.
    if isinstance(documents, Document):
        source_documents = [documents]
    else:
        source_documents = list(documents)

    assert len(source_documents) > 0

    chunks = chunking_service.chunk_documents(
        source_documents
    )

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.page_content.strip()

        assert "chunk_index" in chunk.metadata
        assert "chunk_count" in chunk.metadata
        assert "chunk_id" in chunk.metadata