from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FeedbackListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    filename: str
    category: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    created_at: datetime


class FeedbackListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[FeedbackListItem]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)