from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FeedbackCategory(str, Enum):
    """Allowed customer feedback categories defined by the Cognizant use case."""

    EXCELLENT = "Excellent"
    GOOD = "Good"
    NEED_IMPROVEMENTS = "Need Improvements"
    POOR = "Poor"


class FeedbackClassification(BaseModel):
    """
    Structured output produced by the LLM for one customer feedback form.
    """

    model_config = ConfigDict(extra="forbid")

    category: FeedbackCategory = Field(
        description=(
            "The classification category for the customer feedback. "
            "Must be exactly one of: Excellent, Good, Need Improvements, Poor."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "The model's confidence in the classification, expressed as "
            "a decimal value between 0 and 1."
        )
    )

    rationale: str = Field(
        min_length=1,
        description=(
            "A concise, evidence-based explanation of why the feedback "
            "was assigned to the selected category."
        )
    )

    flagged_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Important words or short phrases from the feedback that "
            "contributed to the classification decision."
        )
    )