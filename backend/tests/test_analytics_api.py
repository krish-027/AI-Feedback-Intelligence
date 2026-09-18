from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import get_db
from backend.main import app
from backend.models.database_models import Base, FeedbackRecord


DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

TestingSessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


@pytest.fixture
def db_session():
    connection = engine.connect()
    transaction = connection.begin()

    Base.metadata.create_all(bind=connection)

    session = TestingSessionLocal(bind=connection)

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def add_feedback(
    db_session,
    feedback_id,
    filename,
    category,
    confidence,
    explanation,
    flagged_keywords="[]",
    created_at=None,
):
    record = FeedbackRecord(
        feedback_id=feedback_id,
        filename=filename,
        feedback_text=f"Feedback text for {feedback_id}.",
        category=category,
        confidence=confidence,
        explanation=explanation,
        flagged_keywords=flagged_keywords,
        created_at=created_at or datetime.utcnow(),
    )

    db_session.add(record)
    db_session.commit()

    return record


def test_analytics_status(client):
    response = client.get("/api/analytics/status")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Analytics route is ready"
    }


def test_get_analytics_empty_database(client):
    response = client.get("/api/analytics")

    assert response.status_code == 200

    data = response.json()

    assert data["total_feedback"] == 0
    assert data["average_confidence"] == 0.0

    assert len(data["category_distribution"]) == 4

    assert all(
        item["count"] == 0
        for item in data["category_distribution"]
    )

    assert data["confidence_distribution"]
    assert data["flagged_keywords"] == []

    assert data["attention"]["poor_feedback_count"] == 0
    assert data["attention"]["need_improvements_count"] == 0
    assert data["attention"]["low_confidence_count"] == 0

    assert data["category_trends"] == []
    assert data["recent_feedback"] == []


def test_get_analytics_with_feedback(
    client,
    db_session,
):
    add_feedback(
        db_session=db_session,
        feedback_id="ANALYTICS-1",
        filename="excellent.pdf",
        category="Excellent",
        confidence=0.96,
        explanation="Strongly positive feedback.",
        flagged_keywords='["helpful", "excellent"]',
    )

    add_feedback(
        db_session=db_session,
        feedback_id="ANALYTICS-2",
        filename="good.pdf",
        category="Good",
        confidence=0.88,
        explanation="Generally positive feedback.",
        flagged_keywords='["good"]',
    )

    add_feedback(
        db_session=db_session,
        feedback_id="ANALYTICS-3",
        filename="improvement.pdf",
        category="Need Improvements",
        confidence=0.65,
        explanation="Customer identified areas for improvement.",
        flagged_keywords='["delay", "waiting"]',
    )

    add_feedback(
        db_session=db_session,
        feedback_id="ANALYTICS-4",
        filename="poor.pdf",
        category="Poor",
        confidence=0.52,
        explanation="Strong negative feedback.",
        flagged_keywords='["poor", "delay"]',
    )

    response = client.get("/api/analytics")

    assert response.status_code == 200

    data = response.json()

    assert data["total_feedback"] == 4

    expected_average = (
        0.96 + 0.88 + 0.65 + 0.52
    ) / 4

    assert data["average_confidence"] == pytest.approx(
        expected_average
    )

    distribution = {
        item["category"]: item
        for item in data["category_distribution"]
    }

    assert distribution["Excellent"]["count"] == 1
    assert distribution["Good"]["count"] == 1
    assert distribution["Need Improvements"]["count"] == 1
    assert distribution["Poor"]["count"] == 1

    assert distribution["Excellent"]["percentage"] == pytest.approx(
        25.0
    )

    assert distribution["Poor"]["percentage"] == pytest.approx(
        25.0
    )

    attention = data["attention"]

    assert attention["poor_feedback_count"] == 1
    assert attention["need_improvements_count"] == 1
    assert attention["low_confidence_count"] == 2
    assert attention["low_confidence_threshold"] == pytest.approx(
        0.70
    )

    keywords = {
        item["keyword"]: item["count"]
        for item in data["flagged_keywords"]
    }

    assert keywords["delay"] == 2
    assert keywords["good"] == 1
    assert keywords["poor"] == 1


def test_get_analytics_respects_limits(
    client,
    db_session,
):
    for index in range(5):
        add_feedback(
            db_session=db_session,
            feedback_id=f"LIMIT-{index}",
            filename=f"feedback-{index}.pdf",
            category="Good",
            confidence=0.90,
            explanation="Positive feedback.",
            flagged_keywords=f'["keyword-{index}"]',
        )

    response = client.get(
        "/api/analytics",
        params={
            "keyword_limit": 2,
            "recent_limit": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["flagged_keywords"]) <= 2
    assert len(data["recent_feedback"]) == 3


def test_get_analytics_rejects_invalid_limits(client):
    response = client.get(
        "/api/analytics",
        params={
            "keyword_limit": 0,
        },
    )

    assert response.status_code == 422

    response = client.get(
        "/api/analytics",
        params={
            "recent_limit": 0,
        },
    )

    assert response.status_code == 422


def test_get_analytics_with_category_filter(
    client,
    db_session,
):
    add_feedback(
        db_session=db_session,
        feedback_id="ANALYTICS-FILTER-1",
        filename="poor.pdf",
        category="Poor",
        confidence=0.91,
        explanation="Strong negative sentiment.",
        flagged_keywords='["poor", "delay"]',
    )

    add_feedback(
        db_session=db_session,
        feedback_id="ANALYTICS-FILTER-2",
        filename="good.pdf",
        category="Good",
        confidence=0.95,
        explanation="Positive customer feedback.",
        flagged_keywords='["good"]',
    )

    response = client.get(
        "/api/analytics",
        params={"category": "Poor"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_feedback"] == 1

    distribution = {
        item["category"]: item
        for item in data["category_distribution"]
    }

    assert distribution["Excellent"]["count"] == 0
    assert distribution["Good"]["count"] == 0
    assert distribution["Need Improvements"]["count"] == 0
    assert distribution["Poor"]["count"] == 1

    assert distribution["Poor"]["percentage"] == pytest.approx(
        100.0
    )

    assert data["attention"]["poor_feedback_count"] == 1
    assert data["attention"]["need_improvements_count"] == 0

    assert len(data["recent_feedback"]) == 1

    assert data["recent_feedback"][0]["feedback_id"] == (
        "ANALYTICS-FILTER-1"
    )


def test_get_analytics_with_each_valid_category(
    client,
    db_session,
):
    categories = [
        "Excellent",
        "Good",
        "Need Improvements",
        "Poor",
    ]

    for index, category in enumerate(categories):
        add_feedback(
            db_session=db_session,
            feedback_id=f"CATEGORY-{index}",
            filename=f"{category}.pdf",
            category=category,
            confidence=0.90,
            explanation=f"Feedback classified as {category}.",
        )

    for category in categories:
        response = client.get(
            "/api/analytics",
            params={"category": category},
        )

        assert response.status_code == 200

        data = response.json()

        assert data["total_feedback"] == 1

        distribution = {
            item["category"]: item
            for item in data["category_distribution"]
        }

        assert distribution[category]["count"] == 1

        for other_category in categories:
            if other_category != category:
                assert distribution[other_category]["count"] == 0


def test_get_analytics_rejects_invalid_category(client):
    response = client.get(
        "/api/analytics",
        params={"category": "Invalid Category"},
    )

    assert response.status_code == 400

    data = response.json()

    assert "Invalid category" in data["detail"]


def test_get_analytics_with_start_date(
    client,
    db_session,
):
    base_date = datetime(2026, 9, 10, 12, 0, 0)

    add_feedback(
        db_session=db_session,
        feedback_id="DATE-1",
        filename="old.pdf",
        category="Good",
        confidence=0.90,
        explanation="Older feedback.",
        created_at=base_date,
    )

    add_feedback(
        db_session=db_session,
        feedback_id="DATE-2",
        filename="new.pdf",
        category="Poor",
        confidence=0.60,
        explanation="Newer feedback.",
        created_at=base_date + timedelta(days=5),
    )

    response = client.get(
        "/api/analytics",
        params={
            "start_date": "2026-09-12",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_feedback"] == 1
    assert data["recent_feedback"][0]["feedback_id"] == "DATE-2"


def test_get_analytics_with_end_date(
    client,
    db_session,
):
    base_date = datetime(2026, 9, 10, 12, 0, 0)

    add_feedback(
        db_session=db_session,
        feedback_id="END-DATE-1",
        filename="included.pdf",
        category="Good",
        confidence=0.90,
        explanation="Included feedback.",
        created_at=base_date,
    )

    add_feedback(
        db_session=db_session,
        feedback_id="END-DATE-2",
        filename="excluded.pdf",
        category="Poor",
        confidence=0.60,
        explanation="Later feedback.",
        created_at=base_date + timedelta(days=5),
    )

    response = client.get(
        "/api/analytics",
        params={
            "end_date": "2026-09-10",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_feedback"] == 1
    assert data["recent_feedback"][0]["feedback_id"] == (
        "END-DATE-1"
    )


def test_get_analytics_with_date_range(
    client,
    db_session,
):
    base_date = datetime(2026, 9, 1, 12, 0, 0)

    add_feedback(
        db_session=db_session,
        feedback_id="RANGE-1",
        filename="before.pdf",
        category="Good",
        confidence=0.90,
        explanation="Before range.",
        created_at=base_date,
    )

    add_feedback(
        db_session=db_session,
        feedback_id="RANGE-2",
        filename="inside-1.pdf",
        category="Excellent",
        confidence=0.95,
        explanation="Inside range.",
        created_at=base_date + timedelta(days=5),
    )

    add_feedback(
        db_session=db_session,
        feedback_id="RANGE-3",
        filename="inside-2.pdf",
        category="Poor",
        confidence=0.55,
        explanation="Inside range.",
        created_at=base_date + timedelta(days=8),
    )

    add_feedback(
        db_session=db_session,
        feedback_id="RANGE-4",
        filename="after.pdf",
        category="Good",
        confidence=0.90,
        explanation="After range.",
        created_at=base_date + timedelta(days=15),
    )

    response = client.get(
        "/api/analytics",
        params={
            "start_date": "2026-09-05",
            "end_date": "2026-09-10",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_feedback"] == 2

    returned_ids = {
        item["feedback_id"]
        for item in data["recent_feedback"]
    }

    assert returned_ids == {
        "RANGE-2",
        "RANGE-3",
    }


def test_get_analytics_with_category_and_date_range(
    client,
    db_session,
):
    base_date = datetime(2026, 9, 1, 12, 0, 0)

    add_feedback(
        db_session=db_session,
        feedback_id="COMBINED-1",
        filename="poor-inside.pdf",
        category="Poor",
        confidence=0.55,
        explanation="Poor feedback inside range.",
        created_at=base_date + timedelta(days=5),
    )

    add_feedback(
        db_session=db_session,
        feedback_id="COMBINED-2",
        filename="good-inside.pdf",
        category="Good",
        confidence=0.90,
        explanation="Good feedback inside range.",
        created_at=base_date + timedelta(days=5),
    )

    add_feedback(
        db_session=db_session,
        feedback_id="COMBINED-3",
        filename="poor-outside.pdf",
        category="Poor",
        confidence=0.50,
        explanation="Poor feedback outside range.",
        created_at=base_date + timedelta(days=20),
    )

    response = client.get(
        "/api/analytics",
        params={
            "category": "Poor",
            "start_date": "2026-09-05",
            "end_date": "2026-09-10",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_feedback"] == 1

    assert data["category_distribution"][3]["category"] == "Poor"
    assert data["category_distribution"][3]["count"] == 1

    assert data["recent_feedback"][0]["feedback_id"] == (
        "COMBINED-1"
    )


def test_get_analytics_rejects_invalid_date_range(
    client,
):
    response = client.get(
        "/api/analytics",
        params={
            "start_date": "2026-09-20",
            "end_date": "2026-09-10",
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert "start_date cannot be later than end_date" in (
        data["detail"]
    )