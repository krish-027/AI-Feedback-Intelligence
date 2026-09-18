from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CategoryCount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    count: int = Field(ge=0)
    percentage: float = Field(ge=0.0, le=100.0)


class ConfidenceBand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    band: str
    min_confidence: float = Field(ge=0.0, le=1.0)
    max_confidence: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    percentage: float = Field(ge=0.0, le=100.0)


class KeywordCount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str
    count: int = Field(ge=0)


class CategoryTrend(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    category: str
    count: int = Field(ge=0)


class RecentFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    filename: str
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime


class AttentionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    poor_feedback_count: int = Field(ge=0)
    need_improvements_count: int = Field(ge=0)
    low_confidence_count: int = Field(ge=0)
    low_confidence_threshold: float = Field(ge=0.0, le=1.0)


class AnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_feedback: int = Field(ge=0)

    average_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    category_distribution: list[CategoryCount]

    confidence_distribution: list[ConfidenceBand]

    flagged_keywords: list[KeywordCount]

    attention: AttentionSummary

    category_trends: list[CategoryTrend]

    recent_feedback: list[RecentFeedback]