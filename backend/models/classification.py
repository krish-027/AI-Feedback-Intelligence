from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


FeedbackCategory = Literal[
    "Excellent",
    "Good",
    "Need Improvements",
    "Poor",
]


class FeedbackClassification(BaseModel):
    """
    Structured classification returned by the Gemini model.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    category: FeedbackCategory = Field(
        description=(
            "One of: Excellent, Good, Need Improvements, Poor."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Model confidence in the classification, "
            "represented as a value from 0.0 to 1.0."
        ),
    )

    explanation: str = Field(
        min_length=1,
        description=(
            "A concise plain-language explanation "
            "supporting the classification."
        ),
    )

    flagged_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Important words or phrases from the feedback "
            "that influenced the classification."
        ),
    )