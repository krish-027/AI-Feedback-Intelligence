from pydantic import BaseModel, ConfigDict, Field


class UploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    filename: str
    status: str
    message: str
    extracted_text: str
    category: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    explanation: str
    flagged_keywords: list[str]
    retrieved_count: int = Field(
        ge=0,
    )