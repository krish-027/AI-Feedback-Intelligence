from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


FeedbackCategory = Literal[
    "Excellent",
    "Good",
    "Need Improvements",
    "Poor",
]


class FeedbackClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: FeedbackCategory
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)
    flagged_keywords: list[str] = Field(default_factory=list)


class FeedbackAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback: str = Field(
        min_length=1,
        description="Customer feedback text to classify.",
    )

    filename: str = Field(
        default="manual_input.txt",
        min_length=1,
        max_length=255,
        description="Original filename associated with the feedback.",
    )


class FeedbackAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    filename: str
    feedback: str
    classification: FeedbackClassificationResponse
    retrieved_examples: list[dict]
    retrieved_count: int = Field(ge=0)