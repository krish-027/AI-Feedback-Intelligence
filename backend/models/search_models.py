from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description="Customer feedback text used for similarity search.",
    )

    k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of similar feedback examples to retrieve.",
    )


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    category: str
    split: str
    feedback: str
    score: float


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[SearchResult]
    result_count: int = Field(ge=0)