import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import get_db
from backend.main import app
from backend.models.database_models import Base, FeedbackRecord, RetrievalRecord


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

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
TestingSessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

client = TestClient(app)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


def test_classify_and_retrieve_from_postgresql():
    payload = {
        "feedback": (
            "The support team was helpful and resolved my issue, "
            "but I had to wait longer than expected before receiving assistance."
        ),
        "filename": "integration_test_feedback.txt",
    }

    classify_response = client.post(
        "/api/analysis/classify",
        json=payload,
    )

    assert classify_response.status_code == 200

    classify_data = classify_response.json()

    assert classify_data["feedback_id"].startswith("API-")
    assert classify_data["filename"] == payload["filename"]
    assert classify_data["feedback"] == payload["feedback"]

    classification = classify_data["classification"]

    assert classification["category"] in {
        "Excellent",
        "Good",
        "Need Improvements",
        "Poor",
    }
    assert 0.0 <= classification["confidence"] <= 1.0
    assert classification["explanation"]
    assert isinstance(classification["flagged_keywords"], list)

    assert classify_data["retrieved_count"] == len(
        classify_data["retrieved_examples"]
    )
    assert classify_data["retrieved_count"] > 0

    feedback_id = classify_data["feedback_id"]

    db = TestingSessionLocal()

    try:
        feedback_record = (
            db.query(FeedbackRecord)
            .filter(FeedbackRecord.feedback_id == feedback_id)
            .one_or_none()
        )

        assert feedback_record is not None
        assert feedback_record.filename == payload["filename"]
        assert feedback_record.feedback_text == payload["feedback"]
        assert feedback_record.category == classification["category"]
        assert feedback_record.confidence == classification["confidence"]
        assert feedback_record.explanation == classification["explanation"]

        retrieval_records = (
            db.query(RetrievalRecord)
            .filter(
                RetrievalRecord.feedback_record_id == feedback_record.id
            )
            .order_by(RetrievalRecord.retrieval_rank)
            .all()
        )

        assert len(retrieval_records) == classify_data["retrieved_count"]

        for expected_rank, record in enumerate(retrieval_records, start=1):
            assert record.retrieval_rank == expected_rank
            assert record.retrieved_feedback_id
            assert record.retrieved_category in {
                "Excellent",
                "Good",
                "Need Improvements",
                "Poor",
            }
            assert record.retrieved_feedback
            assert isinstance(record.similarity_score, float)

    finally:
        db.close()

    retrieve_response = client.get(
        f"/api/analysis/{feedback_id}",
    )

    assert retrieve_response.status_code == 200

    retrieve_data = retrieve_response.json()

    assert retrieve_data["feedback_id"] == feedback_id
    assert retrieve_data["filename"] == payload["filename"]
    assert retrieve_data["feedback"] == payload["feedback"]
    assert retrieve_data["category"] == classification["category"]
    assert retrieve_data["confidence"] == classification["confidence"]
    assert retrieve_data["explanation"] == classification["explanation"]

    assert retrieve_data["retrieved_count"] == classify_data["retrieved_count"]
    assert len(retrieve_data["retrieved_examples"]) == (
        classify_data["retrieved_count"]
    )

    assert all(
        example["retrieval_rank"] >= 1
        for example in retrieve_data["retrieved_examples"]
    )