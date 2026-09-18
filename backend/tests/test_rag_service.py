import pytest
from typing import cast

from langchain_core.documents import Document

from backend.models.classification import (
    FeedbackClassification,
)
from backend.services.rag_service import (
    RAGService,
)
from backend.services.rag_chain_service import (
    RAGChainService,
)


class FakeRAGChainService:
    """Fake Stage 9 RAG chain used for facade testing."""

    def __init__(
        self,
        result: dict,
    ) -> None:
        self.result = result
        self.last_feedback = None

    def classify(
        self,
        feedback: str,
    ) -> dict:
        """Return predefined chain output."""

        self.last_feedback = feedback

        return self.result


def create_reference_documents() -> list[Document]:
    """Create deterministic reference documents."""

    return [
        Document(
            page_content=(
                "The employee was polite and helpful."
            ),
            metadata={
                "feedback_id": "FB-REF001",
                "category": "Good",
                "split": "reference",
            },
        ),
        Document(
            page_content=(
                "The waiting time was too long."
            ),
            metadata={
                "feedback_id": "FB-REF002",
                "category": "Need Improvements",
                "split": "reference",
            },
        ),
    ]


def create_classification() -> FeedbackClassification:
    """Create a deterministic classification."""

    return FeedbackClassification(
        category="Need Improvements",
        confidence=0.89,
        explanation=(
            "The customer identified a meaningful "
            "waiting-time issue."
        ),
        flagged_keywords=[
            "waiting time",
        ],
    )


def create_fake_chain(
    documents: list[Document] | None = None,
    classification: FeedbackClassification | None = None,
) -> FakeRAGChainService:
    """Create a fake Stage 9 chain result."""

    if documents is None:
        documents = create_reference_documents()

    if classification is None:
        classification = create_classification()

    return FakeRAGChainService(
        {
            "feedback": (
                "The waiting time was too long."
            ),
            "retrieved_documents": documents,
            "retrieved_examples": (
                "Reference examples"
            ),
            "classification": classification,
        }
    )

def as_rag_chain_service(
    fake_chain: FakeRAGChainService,
) -> RAGChainService:
    """Provide the test double with the production dependency type."""

    return cast(RAGChainService, fake_chain)


def test_classify_feedback_returns_classification() -> None:
    """Facade should return the structured classification."""

    fake_chain = create_fake_chain()

    service = RAGService(
            rag_chain_service=as_rag_chain_service(fake_chain)
    )

    result = service.classify_feedback(
        "The waiting time was too long."
    )

    assert isinstance(
        result["classification"],
        FeedbackClassification,
    )

    assert (
        result["classification"].category
        == "Need Improvements"
    )

    assert (
        result["classification"].confidence
        == 0.89
    )


def test_classify_feedback_calls_rag_chain() -> None:
    """The facade should pass feedback to the underlying RAG chain."""

    fake_chain = create_fake_chain()

    service = RAGService(
            rag_chain_service=as_rag_chain_service(fake_chain)
    )

    feedback = (
        "The employee was helpful."
    )

    service.classify_feedback(
        feedback
    )

    assert fake_chain.last_feedback == feedback


def test_classify_feedback_strips_surrounding_whitespace() -> None:
    """Leading/trailing whitespace should be removed."""

    fake_chain = create_fake_chain()

    service = RAGService(
           rag_chain_service=as_rag_chain_service(fake_chain)
    )

    result = service.classify_feedback(
        "   The employee was helpful.   "
    )

    assert (
        result["feedback"]
        == "The employee was helpful."
    )


def test_retrieved_documents_are_serialized() -> None:
    """LangChain Documents should become plain dictionaries."""

    fake_chain = create_fake_chain()

    service = RAGService(
           rag_chain_service=as_rag_chain_service(fake_chain)
    )

    result = service.classify_feedback(
        "The waiting time was too long."
    )

    examples = result[
        "retrieved_examples"
    ]

    assert len(examples) == 2

    first = examples[0]

    assert first["feedback_id"] == "FB-REF001"
    assert first["category"] == "Good"
    assert first["split"] == "reference"
    assert (
        first["feedback"]
        == "The employee was polite and helpful."
    )


def test_retrieved_count_is_correct() -> None:
    """The result should report the number of retrieved examples."""

    fake_chain = create_fake_chain()

    service = RAGService(
           rag_chain_service=as_rag_chain_service(fake_chain)
    )

    result = service.classify_feedback(
        "The waiting time was too long."
    )

    assert result["retrieved_count"] == 2


def test_retrieved_examples_can_be_excluded() -> None:
    """Callers should be able to omit retrieved example payloads."""

    fake_chain = create_fake_chain()

    service = RAGService(
           rag_chain_service=as_rag_chain_service(fake_chain)
    )

    result = service.classify_feedback(
        "The waiting time was too long.",
        include_retrieved_examples=False,
    )

    assert result["retrieved_examples"] == []

    assert result["retrieved_count"] == 2


def test_classification_dictionary_is_validated() -> None:
    """Dictionary classifications should be converted to the schema."""

    fake_chain = FakeRAGChainService(
        {
            "feedback": "The service was good.",
            "retrieved_documents": [],
            "classification": {
                "category": "Good",
                "confidence": 0.91,
                "explanation": (
                    "The customer reported a positive experience."
                ),
                "flagged_keywords": [
                    "good",
                ],
            },
        }
    )

    service = RAGService(
            rag_chain_service=as_rag_chain_service(fake_chain)
    )

    result = service.classify(
        "The service was good."
    )

    assert isinstance(
        result["classification"],
        FeedbackClassification,
    )

    assert (
        result["classification"].category
        == "Good"
    )


def test_empty_feedback_is_rejected() -> None:
    """Empty feedback should raise ValueError."""

    fake_chain = create_fake_chain()

    service = RAGService(
           rag_chain_service=as_rag_chain_service(fake_chain)
    )

    with pytest.raises(ValueError):
        service.classify_feedback("")


def test_non_string_feedback_is_rejected() -> None:
    """Feedback must be a string."""

    fake_chain = create_fake_chain()

    service = RAGService(
            rag_chain_service=as_rag_chain_service(fake_chain)
    )

    with pytest.raises(TypeError):
        service.classify_feedback(
            123  # type: ignore[arg-type]
        )


def test_evaluation_document_is_rejected() -> None:
    """Evaluation documents must never be exposed as RAG examples."""

    evaluation_document = Document(
        page_content="Evaluation document.",
        metadata={
            "feedback_id": "FB-EVAL001",
            "category": "Good",
            "split": "evaluation",
        },
    )

    fake_chain = create_fake_chain(
        documents=[
            evaluation_document
        ]
    )

    service = RAGService(
            rag_chain_service=as_rag_chain_service(fake_chain)
    )

    with pytest.raises(ValueError):
        service.classify_feedback(
            "The employee was helpful."
        )


def test_invalid_category_is_rejected() -> None:
    """Unknown categories should be rejected."""

    invalid_document = Document(
        page_content="Invalid category.",
        metadata={
            "feedback_id": "FB-REF999",
            "category": "Average",
            "split": "reference",
        },
    )

    fake_chain = create_fake_chain(
        documents=[
            invalid_document
        ]
    )

    service = RAGService(
            rag_chain_service=as_rag_chain_service(fake_chain)
    )

    with pytest.raises(ValueError):
        service.classify_feedback(
            "The service was average."
        )


def test_missing_feedback_id_is_rejected() -> None:
    """Retrieved documents must have a feedback_id."""

    invalid_document = Document(
        page_content="Missing identifier.",
        metadata={
            "category": "Good",
            "split": "reference",
        },
    )

    fake_chain = create_fake_chain(
        documents=[
            invalid_document
        ]
    )

    service = RAGService(
            rag_chain_service=as_rag_chain_service(fake_chain)
    )

    with pytest.raises(ValueError):
        service.classify_feedback(
            "The employee was helpful."
        )


def test_invalid_classification_type_is_rejected() -> None:
    """Unexpected classification objects should be rejected."""

    fake_chain = FakeRAGChainService(
        {
            "feedback": "Some feedback.",
            "retrieved_documents": [],
            "classification": "invalid",
        }
    )

    service = RAGService(
            rag_chain_service=as_rag_chain_service(fake_chain)
    )

    with pytest.raises(TypeError):
        service.classify_feedback(
            "Some feedback."
        )


def test_classify_alias_matches_classify_feedback() -> None:
    """The classify alias should behave like classify_feedback."""

    fake_chain = create_fake_chain()

    service = RAGService(
            rag_chain_service=as_rag_chain_service(fake_chain)
    )

    result = service.classify(
        "The waiting time was too long."
    )

    assert (
        result["classification"].category
        == "Need Improvements"
    )