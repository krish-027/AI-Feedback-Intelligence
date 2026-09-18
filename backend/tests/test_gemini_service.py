import pytest

from backend.models.classification import (
    FeedbackClassification,
)

from backend.services.gemini_service import (
    GeminiService,
)


class FakeStructuredLLM:
    """
    Fake structured LLM used to test GeminiService without
    making an external API request.
    """

    def __init__(self, result):
        self.result = result
        self.received_messages = None

    def invoke(self, messages):
        """Return the predefined test result."""

        self.received_messages = messages

        return self.result


class FakeLLM:
    """
    Fake base LLM that returns a fake structured runnable.
    """

    def __init__(self, result):
        self.result = result
        self.structured_llm = FakeStructuredLLM(
            result
        )

    def with_structured_output(
        self,
        schema,
        method=None,
    ):
        """
        Return the fake structured LLM.

        The production Gemini service explicitly requests native JSON
        Schema structured output rather than function calling.
        """

        assert schema is FeedbackClassification
        assert method == "json_schema"

        return self.structured_llm


def create_service(
    result,
) -> tuple[GeminiService, FakeLLM]:
    """Create a GeminiService using the fake LLM."""

    fake_llm = FakeLLM(result)

    service = GeminiService(
        api_key="test-key",
        llm=fake_llm,
    )

    return service, fake_llm


def test_installed_gemini_supports_json_schema() -> None:
    """The installed LangChain Gemini integration must support native JSON Schema."""

    import inspect

    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
    )

    signature = inspect.signature(
        ChatGoogleGenerativeAI.with_structured_output
    )

    assert "method" in signature.parameters

    method_parameter = signature.parameters["method"]

    assert method_parameter.default in {
        "json_schema",
        None,
    }


def test_thinking_budget_configuration() -> None:
    """Thinking should be disabled by default for classification."""

    service, _ = create_service(
        FeedbackClassification(
            category="Good",
            confidence=0.9,
            explanation="Test classification.",
            flagged_keywords=[],
        )
    )

    assert service.thinking_budget == 0
    assert service.max_output_tokens == 1024


def test_feedback_classification_accepts_valid_output() -> None:
    """A valid classification should pass schema validation."""

    result = FeedbackClassification(
        category="Good",
        confidence=0.91,
        explanation="The customer reported a positive experience.",
        flagged_keywords=[
            "positive experience",
        ],
    )

    assert result.category == "Good"
    assert result.confidence == 0.91
    assert result.flagged_keywords == [
        "positive experience",
    ]


def test_feedback_classification_rejects_invalid_category() -> None:
    """An unknown category should be rejected."""

    with pytest.raises(ValueError):
        FeedbackClassification(
            category="Average",  # type: ignore[arg-type]
            confidence=0.80,
            explanation="Invalid category.",
            flagged_keywords=[],
        )


def test_feedback_classification_rejects_invalid_confidence() -> None:
    """Confidence must remain between 0 and 1."""

    with pytest.raises(ValueError):
        FeedbackClassification(
            category="Good",
            confidence=1.5,
            explanation="Invalid confidence.",
            flagged_keywords=[],
        )


def test_gemini_service_returns_structured_result() -> None:
    """The service should return a FeedbackClassification instance."""

    result = FeedbackClassification(
        category="Need Improvements",
        confidence=0.87,
        explanation=(
            "The feedback identifies a specific service gap."
        ),
        flagged_keywords=[
            "waiting time",
        ],
    )

    service, _ = create_service(
        result
    )

    response = service.classify(
        system_prompt="Classify customer feedback.",
        user_prompt=(
            "The waiting time was longer than expected."
        ),
    )

    assert isinstance(
        response,
        FeedbackClassification,
    )

    assert response.category == "Need Improvements"
    assert response.confidence == 0.87


def test_gemini_service_accepts_dictionary_result() -> None:
    """Dictionary output should be converted into the schema."""

    result = {
        "category": "Poor",
        "confidence": 0.96,
        "explanation": (
            "The customer expressed strong dissatisfaction."
        ),
        "flagged_keywords": [
            "very disappointed",
        ],
    }

    service, _ = create_service(
        result
    )

    response = service.classify(
        system_prompt="Classify customer feedback.",
        user_prompt=(
            "I am very disappointed with the service."
        ),
    )

    assert isinstance(
        response,
        FeedbackClassification,
    )

    assert response.category == "Poor"
    assert response.confidence == 0.96


def test_gemini_service_rejects_invalid_result_type() -> None:
    """Unexpected LLM output types should raise TypeError."""

    service, _ = create_service(
        "not structured output"
    )

    with pytest.raises(TypeError):
        service.classify(
            system_prompt="Classify customer feedback.",
            user_prompt="The service was okay.",
        )


def test_empty_system_prompt_is_rejected() -> None:
    """An empty system prompt should raise ValueError."""

    result = FeedbackClassification(
        category="Good",
        confidence=0.90,
        explanation="Positive feedback.",
        flagged_keywords=[],
    )

    service, _ = create_service(
        result
    )

    with pytest.raises(ValueError):
        service.classify(
            system_prompt="",
            user_prompt="The staff was helpful.",
        )


def test_empty_user_prompt_is_rejected() -> None:
    """An empty user prompt should raise ValueError."""

    result = FeedbackClassification(
        category="Good",
        confidence=0.90,
        explanation="Positive feedback.",
        flagged_keywords=[],
    )

    service, _ = create_service(
        result
    )

    with pytest.raises(ValueError):
        service.classify(
            system_prompt="Classify this feedback.",
            user_prompt="",
        )


def test_invalid_temperature_is_rejected() -> None:
    """Temperature outside the supported range should be rejected."""

    with pytest.raises(ValueError):
        GeminiService(
            api_key="test-key",
            temperature=3.0,
            llm=FakeLLM(
                FeedbackClassification(
                    category="Good",
                    confidence=0.9,
                    explanation="Test.",
                    flagged_keywords=[],
                )
            ),
        )


def test_invalid_max_output_tokens_is_rejected() -> None:
    """Maximum output tokens must be positive."""

    with pytest.raises(ValueError):
        GeminiService(
            api_key="test-key",
            max_output_tokens=0,
            llm=FakeLLM(
                FeedbackClassification(
                    category="Good",
                    confidence=0.9,
                    explanation="Test.",
                    flagged_keywords=[],
                )
            ),
        )


def test_messages_are_sent_to_llm() -> None:
    """The service should send both system and user messages."""

    result = FeedbackClassification(
        category="Excellent",
        confidence=0.98,
        explanation=(
            "The customer expressed strong satisfaction."
        ),
        flagged_keywords=[
            "excellent",
        ],
    )

    service, fake_llm = create_service(
        result
    )

    response = service.classify(
        system_prompt=(
            "You classify banking customer feedback."
        ),
        user_prompt=(
            "The service was excellent."
        ),
    )

    assert response.category == "Excellent"

    assert fake_llm.structured_llm.received_messages is not None

    assert len(
        fake_llm.structured_llm.received_messages
    ) == 2

    assert (
        fake_llm
        .structured_llm
        .received_messages[0]
        .content
        == "You classify banking customer feedback."
    )

    assert (
        fake_llm
        .structured_llm
        .received_messages[1]
        .content
        == "The service was excellent."
    )