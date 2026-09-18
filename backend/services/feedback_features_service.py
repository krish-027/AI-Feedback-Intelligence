from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackFeatures:
    """Structured fields extracted from one customer feedback form."""

    greeting_rating: int
    listening_rating: int
    explanation_rating: int
    professionalism_rating: int
    resolution_rating: int
    overall_satisfaction: int
    recommendation: str

    average_service_rating: float
    minimum_service_rating: int
    low_rating_count: int
    high_rating_count: int

    def to_prompt_text(self) -> str:
        """Convert the extracted features into a compact prompt section."""

        return (
            "Structured Feedback Summary:\n"
            f"- Greeting rating: {self.greeting_rating}/5\n"
            f"- Listening rating: {self.listening_rating}/5\n"
            f"- Explanation rating: {self.explanation_rating}/5\n"
            f"- Professionalism rating: {self.professionalism_rating}/5\n"
            f"- Resolution rating: {self.resolution_rating}/5\n"
            f"- Overall satisfaction: {self.overall_satisfaction}/5\n"
            f"- Recommendation: {self.recommendation}\n"
            f"- Average service rating: {self.average_service_rating:.2f}/5\n"
            f"- Minimum service rating: {self.minimum_service_rating}/5\n"
            f"- Number of service ratings <= 2: {self.low_rating_count}\n"
            f"- Number of service ratings >= 4: {self.high_rating_count}"
        )


class FeedbackFeaturesService:
    """Parse structured fields from the standardized feedback text."""

    RATING_PATTERNS = {
        "greeting_rating": r"Employee greeted you politely\s+(\d+)",
        "listening_rating": r"Employee listened carefully to your concerns\s+(\d+)",
        "explanation_rating": r"Employee explained banking services clearly\s+(\d+)",
        "professionalism_rating": r"Employee behaved professionally\s+(\d+)",
        "resolution_rating": r"Employee resolved your issue efficiently\s+(\d+)",
        "overall_satisfaction": r"Overall satisfaction with employee behavior\s+(\d+)",
    }

    RECOMMENDATION_PATTERN = r"Recommendation:\s*(Yes|No)"

    def extract(self, feedback: str) -> FeedbackFeatures:
        """Extract structured ratings and derived indicators."""

        if not isinstance(feedback, str):
            raise TypeError("feedback must be a string.")

        if not feedback.strip():
            raise ValueError("feedback must not be empty.")

        values: dict[str, int] = {}

        for field_name, pattern in self.RATING_PATTERNS.items():
            match = re.search(
                pattern,
                feedback,
                flags=re.IGNORECASE,
            )

            if not match:
                raise ValueError(
                    f"Could not extract required field: {field_name}"
                )

            value = int(match.group(1))

            if not 1 <= value <= 5:
                raise ValueError(
                    f"{field_name} must be between 1 and 5."
                )

            values[field_name] = value

        recommendation_match = re.search(
            self.RECOMMENDATION_PATTERN,
            feedback,
            flags=re.IGNORECASE,
        )

        if not recommendation_match:
            raise ValueError(
                "Could not extract recommendation."
            )

        recommendation = recommendation_match.group(1).capitalize()

        service_ratings = [
            values["greeting_rating"],
            values["listening_rating"],
            values["explanation_rating"],
            values["professionalism_rating"],
            values["resolution_rating"],
        ]

        average_service_rating = (
            sum(service_ratings) / len(service_ratings)
        )

        return FeedbackFeatures(
            greeting_rating=values["greeting_rating"],
            listening_rating=values["listening_rating"],
            explanation_rating=values["explanation_rating"],
            professionalism_rating=values["professionalism_rating"],
            resolution_rating=values["resolution_rating"],
            overall_satisfaction=values["overall_satisfaction"],
            recommendation=recommendation,
            average_service_rating=average_service_rating,
            minimum_service_rating=min(service_ratings),
            low_rating_count=sum(
                rating <= 2
                for rating in service_ratings
            ),
            high_rating_count=sum(
                rating >= 4
                for rating in service_ratings
            ),
        )


_feedback_features_service: FeedbackFeaturesService | None = None


def get_feedback_features_service() -> FeedbackFeaturesService:
    """Return the shared feedback feature-extraction service."""

    global _feedback_features_service

    if _feedback_features_service is None:
        _feedback_features_service = FeedbackFeaturesService()

    return _feedback_features_service