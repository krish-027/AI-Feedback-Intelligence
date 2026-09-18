from typing import cast

import pytest

from langchain_core.documents import Document

from backend.models.classification import (
    FeedbackClassification,
)
from backend.services.rag_chain_service import (
    RAGChainService,
)
from backend.services.gemini_service import (
    GeminiService,
)
from backend.services.retriever_service import (
    RetrieverService,
)


class FakeRetrieverService:
    """Fake retriever used to test the RAG chain."""

    def __init__(
        self,
        documents: list[Document],
    ) -> None:
        self.documents = documents
        self.last_query = None
        self.last_k = None
        self.last_fetch_k = None
        self.last_lambda_mult = None

    def mmr_search(
        self,
        query: str,
        k: int,
        fetch_k: int,
        lambda_mult: float,
    ) -> list[Document]:
        """Return predefined documents and record parameters."""

        self.last_query = query
        self.last_k = k
        self.last_fetch_k = fetch_k
        self.last_lambda_mult = lambda_mult

        return self.documents


class FakeStructuredLLM:
    """
    Fake structured LLM used to avoid external Gemini calls.

    The __call__ method is intentionally implemented so that LangChain
    can treat this object as a callable Runnable when it is composed
    with the "|" operator.
    """

    def __init__(
        self,
        result: FeedbackClassification,
    ) -> None:
        self.result = result
        self.last_input = None

    def __call__(
        self,
        input_data,
    ) -> FeedbackClassification:
        """
        Record the input and return the predefined classification.

        LangChain's runnable composition accepts callable objects, so
        implementing __call__ makes this fake compatible with LCEL.
        """

        self.last_input = input_data

        return self.result


class FakeGeminiService:
    """
    Fake Gemini service exposing a callable structured LLM.

    This mimics the important interface consumed by RAGChainService
    without making a real Gemini API request.
    """

    def __init__(
        self,
        result: FeedbackClassification,
    ) -> None:
        self.structured_llm = FakeStructuredLLM(
            result
        )


def create_reference_documents() -> list[Document]:
    """Create deterministic reference documents for testing."""

    return [
        Document(
            page_content=(
                "The employee was polite and resolved "
                "my issue quickly."
            ),
            metadata={
                "category": "Good",
                "split": "reference",
                "feedback_id": "FB-REF001",
            },
        ),
        Document(
            page_content=(
                "The waiting time was longer than expected."
            ),
            metadata={
                "category": "Need Improvements",
                "split": "reference",
                "feedback_id": "FB-REF002",
            },
        ),
        Document(
            page_content=(
                "The staff was excellent and extremely helpful."
            ),
            metadata={
                "category": "Excellent",
                "split": "reference",
                "feedback_id": "FB-REF003",
            },
        ),
    ]


def create_rag_service(
    documents: list[Document] | None = None,
    classification: FeedbackClassification | None = None,
    k: int = 3,
    fetch_k: int = 8,
    lambda_mult: float = 0.5,
) -> tuple[
    RAGChainService,
    FakeRetrieverService,
    FakeStructuredLLM,
]:
    """Create a RAG chain with fake dependencies."""

    if documents is None:
        documents = create_reference_documents()

    if classification is None:
        classification = FeedbackClassification(
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

    retriever = FakeRetrieverService(
        documents
    )

    gemini = FakeGeminiService(
        classification
    )

    service = RAGChainService(
        retriever_service=cast(RetrieverService, retriever),
        gemini_service=cast(GeminiService, gemini),
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )

    return (
        service,
        retriever,
        gemini.structured_llm,
    )


def test_rag_chain_returns_structured_classification() -> None:
    """The RAG chain should return a valid classification object."""

    service, _, _ = create_rag_service()

    result = service.classify(
        "The waiting time was longer than expected."
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


def test_rag_chain_retrieves_with_configured_mmr() -> None:
    """The RAG chain should pass its MMR configuration to the retriever."""

    service, retriever, _ = create_rag_service(
        k=3,
        fetch_k=8,
        lambda_mult=0.7,
    )

    feedback = (
        "The service was slow but the employee was polite."
    )

    service.classify(
        feedback
    )

    assert retriever.last_query == feedback
    assert retriever.last_k == 3
    assert retriever.last_fetch_k == 8
    assert retriever.last_lambda_mult == 0.7


def test_rag_chain_formats_retrieved_examples() -> None:
    """Retrieved documents should be converted into labeled examples."""

    service, _, _ = create_rag_service()

    result = service.classify(
        "The waiting time was too long."
    )

    formatted_examples = (
        result["retrieved_examples"]
    )

    assert "Reference Example 1" in formatted_examples
    assert "Verified Category: Good" in formatted_examples

    assert (
        "Verified Category: Need Improvements"
        in formatted_examples
    )

    assert (
        "Verified Category: Excellent"
        in formatted_examples
    )

    assert (
        "The waiting time was longer than expected."
        in formatted_examples
    )


def test_rag_chain_passes_prompt_to_structured_llm() -> None:
    """The Gemini runnable should receive the final prompt."""

    service, _, fake_llm = create_rag_service()

    service.classify(
        "The employee was helpful but the waiting time was long."
    )

    prompt_value = fake_llm.last_input

    assert prompt_value is not None

    messages = prompt_value.to_messages()

    assert len(messages) == 2

    system_message = messages[0]
    human_message = messages[1]

    assert (
        "customer feedback classification assistant"
        in system_message.content
    )

    assert (
        "The employee was helpful but the waiting time was long."
        in human_message.content
    )

    assert "Reference Example 1" in human_message.content
    assert "Verified Category:" in human_message.content

    # Stage 8 schema and the corrected prompt should agree.
    assert "explanation" in human_message.content


def test_rag_chain_returns_retrieved_documents() -> None:
    """The final result should preserve the actual retrieved Documents."""

    service, _, _ = create_rag_service()

    result = service.classify(
        "The staff was very helpful."
    )

    documents = result[
        "retrieved_documents"
    ]

    assert len(documents) == 3

    for document in documents:
        assert document.metadata["split"] == "reference"
        assert document.metadata["feedback_id"]


def test_non_reference_document_is_rejected() -> None:
    """Evaluation documents must never enter the RAG context."""

    evaluation_document = Document(
        page_content="Evaluation document.",
        metadata={
            "category": "Good",
            "split": "evaluation",
            "feedback_id": "FB-EVAL001",
        },
    )

    service, _, _ = create_rag_service(
        documents=[evaluation_document]
    )

    with pytest.raises(ValueError):
        service.classify(
            "The staff was helpful."
        )


def test_invalid_category_is_rejected() -> None:
    """Invalid retrieved categories should be rejected."""

    invalid_document = Document(
        page_content="Invalid category document.",
        metadata={
            "category": "Average",
            "split": "reference",
            "feedback_id": "FB-REF999",
        },
    )

    service, _, _ = create_rag_service(
        documents=[invalid_document]
    )

    with pytest.raises(ValueError):
        service.classify(
            "The service was average."
        )


def test_missing_feedback_id_is_rejected() -> None:
    """Retrieved documents must contain a feedback_id."""

    invalid_document = Document(
        page_content="Missing feedback id.",
        metadata={
            "category": "Good",
            "split": "reference",
        },
    )

    service, _, _ = create_rag_service(
        documents=[invalid_document]
    )

    with pytest.raises(ValueError):
        service.classify(
            "The employee was helpful."
        )


def test_empty_feedback_is_rejected() -> None:
    """Blank customer feedback should be rejected."""

    service, _, _ = create_rag_service()

    with pytest.raises(ValueError):
        service.classify("")


def test_non_string_feedback_is_rejected() -> None:
    """Feedback must be supplied as a string."""

    service, _, _ = create_rag_service()

    with pytest.raises(TypeError):
        service.classify(123)  # type: ignore[arg-type]


def test_invalid_k_is_rejected() -> None:
    """k must be a positive integer."""

    with pytest.raises(ValueError):
        create_rag_service(
            k=0
        )


def test_invalid_fetch_k_is_rejected() -> None:
    """fetch_k cannot be smaller than k."""

    with pytest.raises(ValueError):
        create_rag_service(
            k=5,
            fetch_k=3,
        )


def test_invalid_lambda_mult_is_rejected() -> None:
    """MMR lambda_mult must remain between 0 and 1."""

    with pytest.raises(ValueError):
        create_rag_service(
            lambda_mult=1.5
        )


def test_final_result_preserves_feedback() -> None:
    """The original feedback should be preserved in the result."""

    service, _, _ = create_rag_service()

    feedback = (
        "The staff explained everything clearly."
    )

    result = service.classify(
        feedback
    )

    assert result["feedback"] == feedback