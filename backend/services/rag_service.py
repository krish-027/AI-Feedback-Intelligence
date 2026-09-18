from typing import Any

from langchain_core.documents import Document

from backend.models.classification import (
    FeedbackClassification,
)
from backend.services.rag_chain_service import (
    RAGChainService,
    get_rag_chain_service,
)


class RAGService:
    """
    Application-level facade over RAGChainService.

    This class intentionally hides LangChain-specific orchestration
    details from the API and frontend layers.
    """

    def __init__(
        self,
        rag_chain_service: RAGChainService | None = None,
    ) -> None:
        """
        Initialize the RAG service.

        Args:
            rag_chain_service:
                Optional RAGChainService instance.

                Dependency injection is useful for testing and avoids
                making real Gemini/FAISS calls during unit tests.
        """

        self.rag_chain_service = (
            rag_chain_service
            if rag_chain_service is not None
            else get_rag_chain_service()
        )

    def classify_feedback(
        self,
        feedback: str,
        include_retrieved_examples: bool = True,
    ) -> dict[str, Any]:
        """
        Classify customer feedback through the complete RAG pipeline.

        Args:
            feedback:
                Current customer feedback text.

            include_retrieved_examples:
                Whether retrieved historical examples should be included
                in the returned result.

        Returns:
            A JSON-friendly dictionary containing:

            feedback
            classification
            retrieved_examples
            retrieved_count
        """

        self._validate_feedback(feedback)

        chain_result = (
            self.rag_chain_service.classify(
                feedback.strip()
            )
        )

        classification = self._validate_classification(
            chain_result.get("classification")
        )

        retrieved_documents = (
            chain_result.get("retrieved_documents", [])
        )

        serialized_examples = (
            self._serialize_retrieved_documents(
                retrieved_documents
            )
            if include_retrieved_examples
            else []
        )

        return {
            "feedback": feedback.strip(),
            "classification": classification,
            "retrieved_examples": serialized_examples,
            "retrieved_count": len(
                retrieved_documents
            ),
        }

    def classify(
        self,
        feedback: str,
        include_retrieved_examples: bool = True,
    ) -> dict[str, Any]:
        """
        Alias for classify_feedback().

        This provides a concise interface for callers that simply want
        to classify a feedback item.
        """

        return self.classify_feedback(
            feedback=feedback,
            include_retrieved_examples=include_retrieved_examples,
        )

    def _validate_feedback(
        self,
        feedback: str,
    ) -> None:
        """Validate customer feedback text."""

        if not isinstance(
            feedback,
            str,
        ):
            raise TypeError(
                "feedback must be a string."
            )

        if not feedback.strip():
            raise ValueError(
                "feedback cannot be empty."
            )

    def _validate_classification(
        self,
        classification: Any,
    ) -> FeedbackClassification:
        """
        Validate the classification returned by the RAG chain.
        """

        if isinstance(
            classification,
            FeedbackClassification,
        ):
            return classification

        if isinstance(
            classification,
            dict,
        ):
            return FeedbackClassification.model_validate(
                classification
            )

        raise TypeError(
            "RAG chain returned an invalid classification type: "
            f"{type(classification).__name__}"
        )

    def _serialize_retrieved_documents(
        self,
        documents: list[Document],
    ) -> list[dict[str, Any]]:
        """
        Convert LangChain Documents into JSON-friendly dictionaries.

        Only metadata useful to the application is exposed.

        The full metadata dictionary is intentionally not returned so
        that internal vector-store implementation details do not leak
        into the API layer.
        """

        serialized_examples: list[
            dict[str, Any]
        ] = []

        for document in documents:
            self._validate_retrieved_document(
                document
            )

            metadata = document.metadata

            serialized_examples.append(
                {
                    "feedback_id": metadata[
                        "feedback_id"
                    ],
                    "category": metadata[
                        "category"
                    ],
                    "split": metadata[
                        "split"
                    ],
                    "feedback": document.page_content.strip(),
                }
            )

        return serialized_examples

    def _validate_retrieved_document(
        self,
        document: Document,
    ) -> None:
        """
        Validate a retrieved document before exposing it downstream.
        """

        if not isinstance(
            document,
            Document,
        ):
            raise TypeError(
                "RAG chain returned an object that is not "
                "a LangChain Document."
            )

        metadata = document.metadata

        if metadata.get("split") != "reference":
            raise ValueError(
                "RAG service received a non-reference document."
            )

        if metadata.get("category") not in {
            "Excellent",
            "Good",
            "Need Improvements",
            "Poor",
        }:
            raise ValueError(
                "RAG service received a document with "
                "an invalid category."
            )

        if not metadata.get("feedback_id"):
            raise ValueError(
                "RAG service received a document without feedback_id."
            )


def get_rag_service(
    rag_chain_service: RAGChainService | None = None,
) -> RAGService:
    """
    Return a configured RAGService instance.
    """

    return RAGService(
        rag_chain_service=rag_chain_service
    )