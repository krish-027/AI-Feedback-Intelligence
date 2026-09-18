import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import get_db
from backend.main import app
from backend.models.database_models import Base, FeedbackRecord


if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "Real PostgreSQL integration tests disabled. "
        "Set RUN_INTEGRATION_TESTS=1 to run them.",
        allow_module_level=True,
    )


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/customer_feedback_db",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

TestingSessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.rollback()

        db.query(FeedbackRecord).filter(
            FeedbackRecord.feedback_id.like(
                "ANALYTICS-INTEGRATION-%"
            )
        ).delete(
            synchronize_session=False
        )

        db.commit()
        db.close()


@pytest.fixture
def api_client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def add_feedback(
    db,
    feedback_id,
    filename,
    category,
    confidence,
    explanation,
    flagged_keywords,
    created_at,
):
    record = FeedbackRecord(
        feedback_id=feedback_id,
        filename=filename,
        feedback_text=f"Integration feedback for {feedback_id}.",
        category=category,
        confidence=confidence,
        explanation=explanation,
        flagged_keywords=flagged_keywords,
        created_at=created_at,
    )

    db.add(record)
    db.commit()

    return record


def test_analytics_endpoint_with_postgresql(
    api_client,
    db_session,
):
    base_time = datetime.now(timezone.utc).replace(
        tzinfo=None
    )

    add_feedback(
        db=db_session,
        feedback_id="ANALYTICS-INTEGRATION-1",
        filename="excellent.pdf",
        category="Excellent",
        confidence=0.96,
        explanation="Strongly positive feedback.",
        flagged_keywords='["helpful", "excellent"]',
        created_at=base_time,
    )

    response = api_client.get(
        "/api/analytics",
        params={"category": "Excellent"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_feedback"] == 1

    assert data["average_confidence"] == pytest.approx(
        0.96,
        abs=0.0001,
    )

    distribution = {
        item["category"]: item
        for item in data["category_distribution"]
    }

    assert distribution["Excellent"]["count"] == 1
    assert distribution["Excellent"]["percentage"] == pytest.approx(
        100.0
    )

    assert distribution["Good"]["count"] == 0
    assert distribution["Need Improvements"]["count"] == 0
    assert distribution["Poor"]["count"] == 0

    keywords = {
        item["keyword"]: item["count"]
        for item in data["flagged_keywords"]
    }

    assert keywords["helpful"] == 1
    assert keywords["excellent"] == 1

    attention = data["attention"]

    assert attention["poor_feedback_count"] == 0
    assert attention["need_improvements_count"] == 0
    assert attention["low_confidence_count"] == 0

    assert len(data["category_trends"]) == 1

    assert len(data["recent_feedback"]) == 1

    assert data["recent_feedback"][0]["feedback_id"] == (
        "ANALYTICS-INTEGRATION-1"
    )


def test_category_filter_with_postgresql(
    api_client,
    db_session,
):
    base_time = datetime.now(timezone.utc).replace(
        tzinfo=None
    )

    add_feedback(
        db=db_session,
        feedback_id="ANALYTICS-INTEGRATION-2",
        filename="poor-1.pdf",
        category="Poor",
        confidence=0.60,
        explanation="Negative feedback.",
        flagged_keywords='["delay"]',
        created_at=base_time,
    )

    add_feedback(
        db=db_session,
        feedback_id="ANALYTICS-INTEGRATION-3",
        filename="poor-2.pdf",
        category="Poor",
        confidence=0.55,
        explanation="Negative feedback.",
        flagged_keywords='["service"]',
        created_at=base_time - timedelta(days=1),
    )

    add_feedback(
        db=db_session,
        feedback_id="ANALYTICS-INTEGRATION-4",
        filename="good-1.pdf",
        category="Good",
        confidence=0.92,
        explanation="Positive feedback.",
        flagged_keywords='["helpful"]',
        created_at=base_time - timedelta(days=2),
    )

    response = api_client.get(
        "/api/analytics",
        params={"category": "Poor"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_feedback"] == 2

    distribution = {
        item["category"]: item
        for item in data["category_distribution"]
    }

    assert distribution["Poor"]["count"] == 2
    assert distribution["Poor"]["percentage"] == pytest.approx(
        100.0
    )

    assert distribution["Good"]["count"] == 0
    assert distribution["Excellent"]["count"] == 0
    assert distribution["Need Improvements"]["count"] == 0

    assert data["average_confidence"] == pytest.approx(
        (0.60 + 0.55) / 2,
        abs=0.0001,
    )

    assert data["attention"]["poor_feedback_count"] == 2
    assert data["attention"]["need_improvements_count"] == 0
    assert data["attention"]["low_confidence_count"] == 2

    keywords = {
        item["keyword"]: item["count"]
        for item in data["flagged_keywords"]
    }

    assert keywords["delay"] == 1
    assert keywords["service"] == 1

    assert len(data["recent_feedback"]) == 2

    assert all(
        item["category"] == "Poor"
        for item in data["recent_feedback"]
    )