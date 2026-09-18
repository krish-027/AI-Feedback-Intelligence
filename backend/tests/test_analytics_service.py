import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.models.database_models import Base, FeedbackRecord
from backend.services.analytics_service import (
    AnalyticsService,
    LOW_CONFIDENCE_THRESHOLD,
)


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        yield session

    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def analytics_service():
    return AnalyticsService()


def create_record(
    db_session,
    feedback_id,
    filename,
    category,
    confidence,
    flagged_keywords,
    created_at,
):
    record = FeedbackRecord(
        feedback_id=feedback_id,
        filename=filename,
        feedback_text=f"Feedback for {feedback_id}",
        category=category,
        confidence=confidence,
        explanation=f"Explanation for {feedback_id}",
        flagged_keywords=json.dumps(flagged_keywords),
        created_at=created_at,
    )

    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)

    return record


def test_empty_database(
    db_session,
    analytics_service,
):
    result = analytics_service.get_dashboard_analytics(
        db=db_session,
    )

    assert result.total_feedback == 0
    assert result.average_confidence == 0.0

    assert len(result.category_distribution) == 4
    assert all(
        item.count == 0
        for item in result.category_distribution
    )

    assert len(result.confidence_distribution) == 3
    assert all(
        item.count == 0
        for item in result.confidence_distribution
    )

    assert result.flagged_keywords == []
    assert result.attention.poor_feedback_count == 0
    assert result.attention.need_improvements_count == 0
    assert result.attention.low_confidence_count == 0
    assert result.category_trends == []
    assert result.recent_feedback == []


def test_dashboard_analytics(
    db_session,
    analytics_service,
):
    create_record(
        db_session,
        "FB-001",
        "feedback_001.pdf",
        "Excellent",
        0.95,
        ["helpful", "excellent"],
        datetime(2026, 9, 15, 10, 0, 0),
    )

    create_record(
        db_session,
        "FB-002",
        "feedback_002.pdf",
        "Good",
        0.85,
        ["helpful"],
        datetime(2026, 9, 15, 11, 0, 0),
    )

    create_record(
        db_session,
        "FB-003",
        "feedback_003.pdf",
        "Need Improvements",
        0.65,
        ["delay", "waiting"],
        datetime(2026, 9, 16, 10, 0, 0),
    )

    create_record(
        db_session,
        "FB-004",
        "feedback_004.pdf",
        "Poor",
        0.55,
        ["delay", "poor service"],
        datetime(2026, 9, 16, 11, 0, 0),
    )

    result = analytics_service.get_dashboard_analytics(
        db=db_session,
    )

    assert result.total_feedback == 4

    expected_average = (0.95 + 0.85 + 0.65 + 0.55) / 4

    assert result.average_confidence == round(
        expected_average,
        4,
    )

    distribution = {
        item.category: item
        for item in result.category_distribution
    }

    assert distribution["Excellent"].count == 1
    assert distribution["Good"].count == 1
    assert distribution["Need Improvements"].count == 1
    assert distribution["Poor"].count == 1

    assert distribution["Excellent"].percentage == 25.0
    assert distribution["Good"].percentage == 25.0
    assert distribution["Need Improvements"].percentage == 25.0
    assert distribution["Poor"].percentage == 25.0

    confidence_distribution = {
        item.band: item
        for item in result.confidence_distribution
    }

    assert confidence_distribution["Low"].count == 2
    assert confidence_distribution["Medium"].count == 1
    assert confidence_distribution["High"].count == 1

    keywords = {
        item.keyword: item.count
        for item in result.flagged_keywords
    }

    assert keywords["helpful"] == 2
    assert keywords["delay"] == 2
    assert keywords["excellent"] == 1
    assert keywords["waiting"] == 1
    assert keywords["poor service"] == 1

    assert result.attention.poor_feedback_count == 1
    assert result.attention.need_improvements_count == 1
    assert result.attention.low_confidence_count == 2
    assert (
        result.attention.low_confidence_threshold
        == LOW_CONFIDENCE_THRESHOLD
    )

    trends = {
        (item.date, item.category): item.count
        for item in result.category_trends
    }

    assert trends[("2026-09-15", "Excellent")] == 1
    assert trends[("2026-09-15", "Good")] == 1
    assert trends[("2026-09-16", "Need Improvements")] == 1
    assert trends[("2026-09-16", "Poor")] == 1


def test_recent_feedback_limit(
    db_session,
    analytics_service,
):
    for index in range(1, 6):
        create_record(
            db_session,
            f"FB-{index:03d}",
            f"feedback_{index:03d}.pdf",
            "Good",
            0.85,
            [],
            datetime(2026, 9, 15, index, 0, 0),
        )

    result = analytics_service.get_dashboard_analytics(
        db=db_session,
        recent_limit=3,
    )

    assert len(result.recent_feedback) == 3

    assert result.recent_feedback[0].feedback_id == "FB-005"
    assert result.recent_feedback[1].feedback_id == "FB-004"
    assert result.recent_feedback[2].feedback_id == "FB-003"


def test_keyword_limit(
    db_session,
    analytics_service,
):
    create_record(
        db_session,
        "FB-001",
        "feedback_001.pdf",
        "Poor",
        0.50,
        ["delay", "waiting", "slow"],
        datetime(2026, 9, 15, 10, 0, 0),
    )

    result = analytics_service.get_dashboard_analytics(
        db=db_session,
        keyword_limit=2,
    )

    assert len(result.flagged_keywords) == 2


def test_invalid_keyword_json_is_ignored(
    db_session,
    analytics_service,
):
    record = FeedbackRecord(
        feedback_id="FB-INVALID",
        filename="feedback_invalid.pdf",
        feedback_text="Test feedback",
        category="Good",
        confidence=0.85,
        explanation="Test explanation",
        flagged_keywords="not-valid-json",
        created_at=datetime(2026, 9, 15, 10, 0, 0),
    )

    db_session.add(record)
    db_session.commit()

    result = analytics_service.get_dashboard_analytics(
        db=db_session,
    )

    assert result.total_feedback == 1
    assert result.flagged_keywords == []


def test_invalid_limits_raise_error(
    db_session,
    analytics_service,
):
    with pytest.raises(ValueError):
        analytics_service.get_dashboard_analytics(
            db=db_session,
            keyword_limit=0,
        )

    with pytest.raises(ValueError):
        analytics_service.get_dashboard_analytics(
            db=db_session,
            recent_limit=0,
        )