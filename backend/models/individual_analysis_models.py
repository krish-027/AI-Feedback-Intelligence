from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RetrievedExampleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    category: str
    feedback: str
    similarity_score: float
    retrieval_rank: int = Field(ge=1)


class IndividualFeedbackAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    filename: str
    feedback: str
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    flagged_keywords: list[str]
    created_at: datetime
    retrieved_examples: list[RetrievedExampleResponse]
    retrieved_count: int = Field(ge=0)