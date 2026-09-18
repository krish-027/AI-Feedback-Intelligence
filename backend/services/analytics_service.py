import json
from collections import Counter
from datetime import datetime

from sqlalchemy.orm import Session

from backend.models.analytics_models import (
    AnalyticsResponse,
    AttentionSummary,
    CategoryCount,
    CategoryTrend,
    ConfidenceBand,
    KeywordCount,
    RecentFeedback,
)
from backend.models.database_models import FeedbackRecord


CATEGORIES = [
    "Excellent",
    "Good",
    "Need Improvements",
    "Poor",
]

LOW_CONFIDENCE_THRESHOLD = 0.70
MEDIUM_CONFIDENCE_THRESHOLD = 0.90
DEFAULT_KEYWORD_LIMIT = 15
DEFAULT_RECENT_LIMIT = 10


class AnalyticsService:

    def get_dashboard_analytics(
        self,
        db: Session,
        keyword_limit: int = DEFAULT_KEYWORD_LIMIT,
        recent_limit: int = DEFAULT_RECENT_LIMIT,
        category: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> AnalyticsResponse:

        if keyword_limit < 1:
            raise ValueError("keyword_limit must be at least 1.")

        if recent_limit < 1:
            raise ValueError("recent_limit must be at least 1.")

        if category is not None and category not in CATEGORIES:
            raise ValueError(
                f"Invalid category. Expected one of: {', '.join(CATEGORIES)}"
            )

        if start_date is not None and end_date is not None:
            if start_date > end_date:
                raise ValueError(
                    "start_date cannot be later than end_date."
                )

        records = self._get_records(
            db=db,
            category=category,
            start_date=start_date,
            end_date=end_date,
        )

        total_feedback = len(records)

        average_confidence = self._calculate_average_confidence(
            records
        )

        category_distribution = self._calculate_category_distribution(
            records,
            total_feedback,
        )

        confidence_distribution = self._calculate_confidence_distribution(
            records,
            total_feedback,
        )

        flagged_keywords = self._calculate_flagged_keywords(
            records,
            keyword_limit,
        )

        attention = self._calculate_attention_summary(
            records
        )

        category_trends = self._calculate_category_trends(
            records
        )

        recent_feedback = self._calculate_recent_feedback(
            records,
            recent_limit,
        )

        return AnalyticsResponse(
            total_feedback=total_feedback,
            average_confidence=average_confidence,
            category_distribution=category_distribution,
            confidence_distribution=confidence_distribution,
            flagged_keywords=flagged_keywords,
            attention=attention,
            category_trends=category_trends,
            recent_feedback=recent_feedback,
        )

    def _get_records(
        self,
        db: Session,
        category: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[FeedbackRecord]:

        query = db.query(FeedbackRecord)

        if category is not None:
            query = query.filter(
                FeedbackRecord.category == category
            )

        if start_date is not None:
            query = query.filter(
                FeedbackRecord.created_at >= start_date
            )

        if end_date is not None:
            query = query.filter(
                FeedbackRecord.created_at <= end_date
            )

        return (
            query
            .order_by(FeedbackRecord.created_at.desc())
            .all()
        )

    def _calculate_average_confidence(
        self,
        records: list[FeedbackRecord],
    ) -> float:

        if not records:
            return 0.0

        return sum(
            record.confidence
            for record in records
        ) / len(records)

    def _calculate_category_distribution(
        self,
        records: list[FeedbackRecord],
        total_feedback: int,
    ) -> list[CategoryCount]:

        counts = Counter(
            record.category
            for record in records
        )

        return [
            CategoryCount(
                category=category,
                count=counts.get(category, 0),
                percentage=(
                    counts.get(category, 0) / total_feedback * 100
                    if total_feedback
                    else 0.0
                ),
            )
            for category in CATEGORIES
        ]

    def _calculate_confidence_distribution(
        self,
        records: list[FeedbackRecord],
        total_feedback: int,
    ) -> list[ConfidenceBand]:

        bands = [
            (
                "Low",
                0.0,
                LOW_CONFIDENCE_THRESHOLD,
            ),
            (
                "Medium",
                LOW_CONFIDENCE_THRESHOLD,
                MEDIUM_CONFIDENCE_THRESHOLD,
            ),
            (
                "High",
                MEDIUM_CONFIDENCE_THRESHOLD,
                1.0,
            ),
        ]

        result = []

        for band, minimum, maximum in bands:

            count = sum(
                minimum <= record.confidence < maximum
                for record in records
            )

            if band == "High":
                count = sum(
                    minimum <= record.confidence <= maximum
                    for record in records
                )

            percentage = (
                count / total_feedback * 100
                if total_feedback
                else 0.0
            )

            result.append(
                ConfidenceBand(
                    band=band,
                    min_confidence=minimum,
                    max_confidence=maximum,
                    count=count,
                    percentage=percentage,
                )
            )

        return result

    def _calculate_flagged_keywords(
        self,
        records: list[FeedbackRecord],
        keyword_limit: int,
    ) -> list[KeywordCount]:

        counter = Counter()

        for record in records:

            if not record.flagged_keywords:
                continue

            try:
                keywords = json.loads(
                    record.flagged_keywords
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

            for keyword in keywords:

                if not isinstance(keyword, str):
                    continue

                keyword = keyword.strip()

                if keyword:
                    counter[keyword] += 1

        return [
            KeywordCount(
                keyword=keyword,
                count=count,
            )
            for keyword, count in counter.most_common(
                keyword_limit
            )
        ]

    def _calculate_attention_summary(
        self,
        records: list[FeedbackRecord],
    ) -> AttentionSummary:

        poor_count = sum(
            record.category == "Poor"
            for record in records
        )

        need_improvements_count = sum(
            record.category == "Need Improvements"
            for record in records
        )

        low_confidence_count = sum(
            record.confidence < LOW_CONFIDENCE_THRESHOLD
            for record in records
        )

        return AttentionSummary(
            poor_feedback_count=poor_count,
            need_improvements_count=need_improvements_count,
            low_confidence_count=low_confidence_count,
            low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
        )

    def _calculate_category_trends(
        self,
        records: list[FeedbackRecord],
    ) -> list[CategoryTrend]:

        counts: Counter[tuple[str, str]] = Counter()

        for record in records:

            date = self._format_date(
                record.created_at
            )

            counts[(date, record.category)] += 1

        result = []

        for (date, category), count in sorted(
            counts.items()
        ):

            result.append(
                CategoryTrend(
                    date=date,
                    category=category,
                    count=count,
                )
            )

        return result

    def _calculate_recent_feedback(
        self,
        records: list[FeedbackRecord],
        recent_limit: int,
    ) -> list[RecentFeedback]:

        return [
            RecentFeedback(
                feedback_id=record.feedback_id,
                filename=record.filename,
                category=record.category,
                confidence=record.confidence,
                created_at=record.created_at,
            )
            for record in records[:recent_limit]
        ]

    @staticmethod
    def _format_date(
        value: datetime,
    ) -> str:

        return value.strftime("%Y-%m-%d")


def get_analytics_service() -> AnalyticsService:
    return AnalyticsService()