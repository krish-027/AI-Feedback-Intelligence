import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.models.database_models import Base, FeedbackRecord, RetrievalRecord
from backend.services.feedback_database_service import (
    FeedbackDatabaseService,
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
def database_service():
    return FeedbackDatabaseService()


@pytest.fixture
def test_feedback_id():
    return "FB-DB-TEST-001"


def test_create_feedback_record(
    db_session: Session,
    database_service: FeedbackDatabaseService,
    test_feedback_id: str,
):
    record = database_service.create_feedback_record(
        db=db_session,
        feedback_id=test_feedback_id,
        filename="feedback_test_001.pdf",
        feedback_text="The service was helpful but slow.",
        category="Need Improvements",
        confidence=0.90,
        explanation="The service was helpful, but the response time needs improvement.",
        flagged_keywords=[
            "slow",
            "response time",
        ],
    )

    assert isinstance(record, FeedbackRecord)
    assert record.id is not None
    assert record.feedback_id == test_feedback_id
    assert record.filename == "feedback_test_001.pdf"
    assert record.feedback_text == "The service was helpful but slow."
    assert record.category == "Need Improvements"
    assert record.confidence == 0.90
    assert (
        record.explanation
        == "The service was helpful, but the response time needs improvement."
    )
    assert record.flagged_keywords is not None
    assert "slow" in record.flagged_keywords
    assert "response time" in record.flagged_keywords
    assert record.created_at is not None


def test_get_feedback_record(
    db_session: Session,
    database_service: FeedbackDatabaseService,
    test_feedback_id: str,
):
    database_service.create_feedback_record(
        db=db_session,
        feedback_id=test_feedback_id,
        filename="feedback_test_001.pdf",
        feedback_text="The employee was polite.",
        category="Good",
        confidence=0.85,
        explanation="The customer reported a generally positive experience.",
        flagged_keywords=["polite"],
    )

    record = database_service.get_feedback_record(
        db=db_session,
        feedback_id=test_feedback_id,
    )

    assert record is not None
    assert isinstance(record, FeedbackRecord)
    assert record.feedback_id == test_feedback_id
    assert record.filename == "feedback_test_001.pdf"
    assert record.category == "Good"
    assert record.confidence == 0.85


def test_get_feedback_record_returns_none_for_unknown_id(
    db_session: Session,
    database_service: FeedbackDatabaseService,
):
    record = database_service.get_feedback_record(
        db=db_session,
        feedback_id="FB-UNKNOWN-999",
    )

    assert record is None


def test_get_all_feedback_records(
    db_session: Session,
    database_service: FeedbackDatabaseService,
    test_feedback_id: str,
):
    database_service.create_feedback_record(
        db=db_session,
        feedback_id=test_feedback_id,
        filename="feedback_test_001.pdf",
        feedback_text="The service was excellent.",
        category="Excellent",
        confidence=0.95,
        explanation="The feedback was strongly positive.",
        flagged_keywords=["excellent"],
    )

    records = database_service.get_all_feedback_records(
        db=db_session,
    )

    assert len(records) == 1
    assert isinstance(records[0], FeedbackRecord)
    assert records[0].feedback_id == test_feedback_id
    assert records[0].filename == "feedback_test_001.pdf"
    assert records[0].category == "Excellent"


def test_create_retrieval_records(
    db_session: Session,
    database_service: FeedbackDatabaseService,
    test_feedback_id: str,
):
    feedback_record = database_service.create_feedback_record(
        db=db_session,
        feedback_id=test_feedback_id,
        filename="feedback_test_001.pdf",
        feedback_text="The service was helpful but slow.",
        category="Need Improvements",
        confidence=0.90,
        explanation="The service was helpful, but the response time needs improvement.",
        flagged_keywords=["slow"],
    )

    retrieved_examples = [
        {
            "feedback_id": "FB-REF-001",
            "category": "Need Improvements",
            "feedback": "The customer experienced a long waiting time.",
            "score": 1.25,
        },
        {
            "feedback_id": "FB-REF-002",
            "category": "Good",
            "feedback": "The service was good overall.",
            "score": 1.42,
        },
    ]

    records = database_service.create_retrieval_records(
        db=db_session,
        feedback_record_id=feedback_record.id,
        retrieved_examples=retrieved_examples,
    )

    assert len(records) == 2

    assert isinstance(records[0], RetrievalRecord)
    assert records[0].feedback_record_id == feedback_record.id
    assert records[0].retrieved_feedback_id == "FB-REF-001"
    assert records[0].retrieved_category == "Need Improvements"
    assert records[0].similarity_score == 1.25
    assert records[0].retrieval_rank == 1

    assert records[1].retrieved_feedback_id == "FB-REF-002"
    assert records[1].retrieved_category == "Good"
    assert records[1].similarity_score == 1.42
    assert records[1].retrieval_rank == 2


def test_get_retrieval_records(
    db_session: Session,
    database_service: FeedbackDatabaseService,
    test_feedback_id: str,
):
    feedback_record = database_service.create_feedback_record(
        db=db_session,
        feedback_id=test_feedback_id,
        filename="feedback_test_001.pdf",
        feedback_text="The service was helpful but slow.",
        category="Need Improvements",
        confidence=0.90,
        explanation="The service was helpful, but the response time needs improvement.",
        flagged_keywords=["slow"],
    )

    retrieved_examples = [
        {
            "feedback_id": "FB-REF-001",
            "category": "Need Improvements",
            "feedback": "The customer experienced a long waiting time.",
            "score": 1.25,
        },
        {
            "feedback_id": "FB-REF-002",
            "category": "Good",
            "feedback": "The service was good overall.",
            "score": 1.42,
        },
        {
            "feedback_id": "FB-REF-003",
            "category": "Need Improvements",
            "feedback": "The issue took too long to resolve.",
            "score": 1.50,
        },
    ]

    database_service.create_retrieval_records(
        db=db_session,
        feedback_record_id=feedback_record.id,
        retrieved_examples=retrieved_examples,
    )

    records = database_service.get_retrieval_records(
        db=db_session,
        feedback_record_id=feedback_record.id,
    )

    assert len(records) == 3

    assert records[0].retrieval_rank == 1
    assert records[0].retrieved_feedback_id == "FB-REF-001"

    assert records[1].retrieval_rank == 2
    assert records[1].retrieved_feedback_id == "FB-REF-002"

    assert records[2].retrieval_rank == 3
    assert records[2].retrieved_feedback_id == "FB-REF-003"


def test_get_retrieval_records_returns_empty_for_feedback_without_retrievals(
    db_session: Session,
    database_service: FeedbackDatabaseService,
    test_feedback_id: str,
):
    feedback_record = database_service.create_feedback_record(
        db=db_session,
        feedback_id=test_feedback_id,
        filename="feedback_test_001.pdf",
        feedback_text="The service was good.",
        category="Good",
        confidence=0.85,
        explanation="The customer reported a positive experience.",
        flagged_keywords=["good"],
    )

    records = database_service.get_retrieval_records(
        db=db_session,
        feedback_record_id=feedback_record.id,
    )

    assert records == []


def test_retrieval_records_are_deleted_with_feedback_record(
    db_session: Session,
    database_service: FeedbackDatabaseService,
    test_feedback_id: str,
):
    feedback_record = database_service.create_feedback_record(
        db=db_session,
        feedback_id=test_feedback_id,
        filename="feedback_test_001.pdf",
        feedback_text="The service was helpful but slow.",
        category="Need Improvements",
        confidence=0.90,
        explanation="The service was helpful, but the response time needs improvement.",
        flagged_keywords=["slow"],
    )

    database_service.create_retrieval_records(
        db=db_session,
        feedback_record_id=feedback_record.id,
        retrieved_examples=[
            {
                "feedback_id": "FB-REF-001",
                "category": "Need Improvements",
                "feedback": "The customer experienced a long waiting time.",
                "score": 1.25,
            }
        ],
    )

    assert len(
        database_service.get_retrieval_records(
            db=db_session,
            feedback_record_id=feedback_record.id,
        )
    ) == 1

    db_session.delete(feedback_record)
    db_session.commit()

    remaining_retrievals = (
        db_session.query(RetrievalRecord)
        .all()
    )

    assert remaining_retrievals == []


def test_get_feedback_records_paginated(
    db_session: Session,
    database_service: FeedbackDatabaseService,
):
    from datetime import datetime, timedelta

    base_time = datetime(
        2026,
        9,
        18,
        10,
        0,
    )

    for index in range(1, 6):
        record = FeedbackRecord(
            feedback_id=f"FB-PAGE-{index:03d}",
            filename=f"feedback_{index:03d}.pdf",
            feedback_text=f"Feedback {index}",
            category="Good",
            confidence=0.80 + (index * 0.01),
            explanation=f"Explanation {index}",
            flagged_keywords="[]",
            created_at=base_time + timedelta(
                minutes=index
            ),
        )

        db_session.add(record)

    db_session.commit()

    records, total = (
        database_service.get_feedback_records_paginated(
            db=db_session,
            limit=2,
            offset=0,
        )
    )

    assert total == 5
    assert len(records) == 2

    assert records[0].feedback_id == "FB-PAGE-005"
    assert records[1].feedback_id == "FB-PAGE-004"


def test_get_feedback_records_paginated_with_offset(
    db_session: Session,
    database_service: FeedbackDatabaseService,
):
    from datetime import datetime, timedelta

    base_time = datetime(
        2026,
        9,
        18,
        10,
        0,
    )

    for index in range(1, 6):
        record = FeedbackRecord(
            feedback_id=f"FB-PAGE-{index:03d}",
            filename=f"feedback_{index:03d}.pdf",
            feedback_text=f"Feedback {index}",
            category="Good",
            confidence=0.80,
            explanation=f"Explanation {index}",
            flagged_keywords="[]",
            created_at=base_time + timedelta(
                minutes=index
            ),
        )

        db_session.add(record)

    db_session.commit()

    records, total = (
        database_service.get_feedback_records_paginated(
            db=db_session,
            limit=2,
            offset=2,
        )
    )

    assert total == 5
    assert len(records) == 2

    assert records[0].feedback_id == "FB-PAGE-003"
    assert records[1].feedback_id == "FB-PAGE-002"


def test_get_feedback_records_paginated_beyond_end(
    db_session: Session,
    database_service: FeedbackDatabaseService,
):
    record = FeedbackRecord(
        feedback_id="FB-PAGE-001",
        filename="feedback_001.pdf",
        feedback_text="Test feedback",
        category="Good",
        confidence=0.85,
        explanation="Test explanation",
        flagged_keywords="[]",
    )

    db_session.add(record)
    db_session.commit()

    records, total = (
        database_service.get_feedback_records_paginated(
            db=db_session,
            limit=10,
            offset=100,
        )
    )

    assert total == 1
    assert records == []


def test_get_feedback_records_paginated_rejects_invalid_limit(
    db_session: Session,
    database_service: FeedbackDatabaseService,
):
    with pytest.raises(ValueError):
        database_service.get_feedback_records_paginated(
            db=db_session,
            limit=0,
            offset=0,
        )


def test_get_feedback_records_paginated_rejects_negative_offset(
    db_session: Session,
    database_service: FeedbackDatabaseService,
):
    with pytest.raises(ValueError):
        database_service.get_feedback_records_paginated(
            db=db_session,
            limit=10,
            offset=-1,
        )