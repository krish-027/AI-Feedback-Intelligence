from pydantic import BaseModel
from typing import Optional


class FeedbackResponse(BaseModel):
    id: str
    filename: str
    category: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None