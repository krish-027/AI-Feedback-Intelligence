import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.models.classification import (
    FeedbackClassification,
)


class GeminiService:
    """
    Service responsible for interacting with Gemini.

    The service returns validated FeedbackClassification objects rather
    than raw model-generated text.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_OUTPUT_TOKENS = 1024
    DEFAULT_THINKING_BUDGET = 0

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        llm: Any | None = None,
    ) -> None:
        """
        Initialize the Gemini service.

        Args:
            api_key:
                Gemini API key. If omitted, GEMINI_API_KEY is read from
                backend/.env.

            model_name:
                Gemini model identifier.

            temperature:
                Generation temperature.

            max_output_tokens:
                Maximum number of output tokens.

            llm:
                Optional preconfigured LLM/runnable used mainly for tests.
                When supplied, no Gemini API client is created.
        """

        self._load_environment()

        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("GEMINI_API_KEY")
        )

        self.model_name = (
            model_name
            if model_name is not None
            else os.getenv(
                "GEMINI_MODEL",
                self.DEFAULT_MODEL,
            )
        )

        self.temperature = (
            temperature
            if temperature is not None
            else self._get_float_environment_value(
                "GEMINI_TEMPERATURE",
                self.DEFAULT_TEMPERATURE,
            )
        )

        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else self._get_int_environment_value(
                "GEMINI_MAX_OUTPUT_TOKENS",
                self.DEFAULT_MAX_OUTPUT_TOKENS,
            )
        )

        self.thinking_budget = (
            self._get_int_environment_value(
                "GEMINI_THINKING_BUDGET",
                self.DEFAULT_THINKING_BUDGET,
            )
        )

        if self.thinking_budget < 0:
            raise ValueError(
                "thinking_budget must be greater than or equal to 0."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0."
            )

        if self.max_output_tokens <= 0:
            raise ValueError(
                "max_output_tokens must be greater than 0."
            )

        self.llm = llm

        if self.llm is None:
            self.llm = self._create_llm()

        self.structured_llm = (
            self.llm.with_structured_output(
                FeedbackClassification,
                method="json_schema",
            )
        )

    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> FeedbackClassification:
        """
        Send a classification request to Gemini.

        Args:
            system_prompt:
                System-level instructions for the model.

            user_prompt:
                Feedback/classification input.

        Returns:
            A validated FeedbackClassification object.
        """

        self._validate_prompt(
            system_prompt,
            "system_prompt",
        )

        self._validate_prompt(
            user_prompt,
            "user_prompt",
        )

        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=user_prompt
            ),
        ]

        result = self.structured_llm.invoke(
            messages
        )

        return self._validate_result(result)

    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """
        Create the Gemini chat model.
        """

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Add it to backend/.env or pass api_key explicitly."
            )

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_budget=self.thinking_budget,
        )

    def _validate_result(
        self,
        result: Any,
    ) -> FeedbackClassification:
        """
        Validate and normalize the structured model output.
        """

        if isinstance(
            result,
            FeedbackClassification,
        ):
            return result

        if isinstance(
            result,
            dict,
        ):
            return FeedbackClassification.model_validate(
                result
            )

        raise TypeError(
            "Gemini returned an unexpected result type: "
            f"{type(result).__name__}"
        )

    def _validate_prompt(
        self,
        prompt: str,
        prompt_name: str,
    ) -> None:
        """Validate a prompt string."""

        if not isinstance(prompt, str):
            raise TypeError(
                f"{prompt_name} must be a string."
            )

        if not prompt.strip():
            raise ValueError(
                f"{prompt_name} cannot be empty."
            )

    def _get_float_environment_value(
        self,
        name: str,
        default: float,
    ) -> float:
        """Read a floating-point configuration value."""

        value = os.getenv(name)

        if value is None:
            return default

        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a valid floating-point number."
            ) from exc

    def _get_int_environment_value(
        self,
        name: str,
        default: int,
    ) -> int:
        """Read an integer configuration value."""

        value = os.getenv(name)

        if value is None:
            return default

        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a valid integer."
            ) from exc

    def _load_environment(self) -> None:
        """
        Load variables from backend/.env.

        Loading is explicit here so that the Gemini service can be used
        independently from FastAPI or Streamlit.
        """

        backend_directory = (
            Path(__file__).resolve().parents[1]
        )

        env_path = backend_directory / ".env"

        load_dotenv(
            dotenv_path=env_path
        )


def get_gemini_service(
    api_key: str | None = None,
    model_name: str | None = None,
) -> GeminiService:
    """
    Return a configured GeminiService.
    """

    return GeminiService(
        api_key=api_key,
        model_name=model_name,
    )