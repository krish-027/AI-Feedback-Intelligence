from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

from backend.models.classification import (
    FeedbackClassification,
)
from backend.prompts.classification_prompt import (
    get_classification_prompt,
)
from backend.services.gemini_service import (
    GeminiService,
    get_gemini_service,
)
from backend.services.retriever_service import (
    RetrieverService,
    get_retriever_service,
)


class RAGChainService:
    """
    Main RAG chain for customer-feedback classification.

    The chain retrieves verified reference examples from FAISS,
    inserts them into the existing classification prompt, and sends the
    resulting prompt to Gemini for structured classification.
    """

    DEFAULT_K = 4
    DEFAULT_FETCH_K = 12
    DEFAULT_LAMBDA_MULT = 0.5

    def __init__(
        self,
        retriever_service: RetrieverService | None = None,
        gemini_service: GeminiService | None = None,
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mult: float = DEFAULT_LAMBDA_MULT,
    ) -> None:
        """
        Initialize the RAG chain.

        Args:
            retriever_service:
                Retriever used to find historical reference examples.

            gemini_service:
                Gemini service providing the structured-output model.

            k:
                Number of reference documents returned to the prompt.

            fetch_k:
                Number of candidates considered by MMR before selecting k.

            lambda_mult:
                MMR relevance/diversity trade-off.
        """

        self.retriever_service = (
            retriever_service
            if retriever_service is not None
            else get_retriever_service()
        )

        self.gemini_service = (
            gemini_service
            if gemini_service is not None
            else get_gemini_service()
        )

        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult

        self._validate_configuration()

        self.prompt = get_classification_prompt()

        self.chain = self._build_chain()

    def classify(
        self,
        feedback: str,
    ) -> dict[str, Any]:
        """
        Run the complete RAG classification pipeline.

        Args:
            feedback:
                Current customer feedback text.

        Returns:
            Dictionary containing:
                feedback
                retrieved_documents
                retrieved_examples
                classification
        """

        self._validate_feedback(feedback)

        result = self.chain.invoke(
            {
                "feedback": feedback.strip(),
            }
        )

        return result

    def invoke(
        self,
        feedback: str,
    ) -> dict[str, Any]:
        """
        Alias for classify(), following LangChain's invoke terminology.
        """

        return self.classify(feedback)

    def _build_chain(self):
        """
        Build the LCEL RAG chain.

        The chain keeps retrieved documents in the final result so later
        evaluation and presentation layers can inspect the RAG context.
        """

        retrieval_runnable = RunnableLambda(
            self._retrieve_documents
        )

        formatting_runnable = RunnableLambda(
            self._format_retrieved_examples
        )

        prompt_input_runnable = RunnableLambda(
            self._build_prompt_input
        )

        classification_runnable = (
            prompt_input_runnable
            | self.prompt
            | self.gemini_service.structured_llm
        )

        chain = (
            RunnablePassthrough.assign(
                retrieved_documents=retrieval_runnable
            )
            .assign(
                retrieved_examples=formatting_runnable
            )
            .assign(
                classification=classification_runnable
            )
            | RunnableLambda(
                self._build_final_result
            )
        )

        return chain

    def _retrieve_documents(
        self,
        inputs: dict[str, Any],
    ) -> list[Document]:
        """
        Retrieve relevant historical reference examples using MMR.
        """

        feedback = inputs["feedback"]

        documents = self.retriever_service.mmr_search(
            query=feedback,
            k=self.k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )

        self._validate_retrieved_documents(
            documents
        )

        return documents

    def _format_retrieved_examples(
        self,
        inputs: dict[str, Any],
    ) -> str:
        """
        Convert retrieved LangChain Documents into the text format
        expected by classification_prompt.py.
        """

        documents: list[Document] = inputs[
            "retrieved_documents"
        ]

        if not documents:
            return (
                "No reference examples were retrieved."
            )

        formatted_examples: list[str] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            category = document.metadata.get(
                "category",
                "Unknown",
            )

            feedback_text = (
                document.page_content.strip()
            )

            formatted_examples.append(
                f"""Reference Example {index}
Verified Category: {category}
Feedback:
{feedback_text}"""
            )

        return "\n\n".join(
            formatted_examples
        )

    def _build_prompt_input(
        self,
        inputs: dict[str, Any],
    ) -> dict[str, str]:
        """
        Select the variables required by the classification prompt.
        """

        return {
            "feedback": inputs["feedback"],
            "retrieved_examples": inputs[
                "retrieved_examples"
            ],
        }

    def _build_final_result(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Construct the final RAG chain result.

        This keeps both the model classification and the retrieved
        context available to downstream services.
        """

        classification = result.get(
            "classification"
        )

        if not isinstance(
            classification,
            FeedbackClassification,
        ):
            classification = (
                FeedbackClassification.model_validate(
                    classification
                )
            )

        return {
            "feedback": result["feedback"],
            "retrieved_documents": result[
                "retrieved_documents"
            ],
            "retrieved_examples": result[
                "retrieved_examples"
            ],
            "classification": classification,
        }

    def _validate_retrieved_documents(
        self,
        documents: list[Document],
    ) -> None:
        """
        Ensure that every retrieved document is a valid reference
        document before it reaches the LLM.
        """

        for document in documents:
            if not isinstance(
                document,
                Document,
            ):
                raise TypeError(
                    "Retriever returned an object that "
                    "is not a LangChain Document."
                )

            if document.metadata.get(
                "split"
            ) != "reference":
                raise ValueError(
                    "RAG chain received a non-reference "
                    "document."
                )

            if document.metadata.get(
                "category"
            ) not in {
                "Excellent",
                "Good",
                "Need Improvements",
                "Poor",
            }:
                raise ValueError(
                    "RAG chain received a document with "
                    "an invalid category."
                )

            if not document.metadata.get(
                "feedback_id"
            ):
                raise ValueError(
                    "RAG chain received a document "
                    "without feedback_id."
                )

    def _validate_feedback(
        self,
        feedback: str,
    ) -> None:
        """Validate the current feedback text."""

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

    def _validate_configuration(self) -> None:
        """Validate RAG configuration."""

        if not isinstance(
            self.k,
            int,
        ):
            raise TypeError(
                "k must be an integer."
            )

        if self.k <= 0:
            raise ValueError(
                "k must be greater than 0."
            )

        if not isinstance(
            self.fetch_k,
            int,
        ):
            raise TypeError(
                "fetch_k must be an integer."
            )

        if self.fetch_k < self.k:
            raise ValueError(
                "fetch_k must be greater than or equal to k."
            )

        if not 0.0 <= self.lambda_mult <= 1.0:
            raise ValueError(
                "lambda_mult must be between 0.0 and 1.0."
            )


def get_rag_chain_service(
    retriever_service: RetrieverService | None = None,
    gemini_service: GeminiService | None = None,
) -> RAGChainService:
    """
    Return a configured RAGChainService.
    """

    return RAGChainService(
        retriever_service=retriever_service,
        gemini_service=gemini_service,
    )