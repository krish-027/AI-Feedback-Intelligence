from fastapi.testclient import TestClient

from backend.main import app


class FakeRAGService:
    def classify_feedback(
        self,
        feedback: str,
        include_retrieved_examples: bool = True,
    ):
        assert include_retrieved_examples is True

        return {
            "feedback": feedback,
            "classification": FakeClassification(),
            "retrieved_examples": [
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
            ],
            "retrieved_count": 2,
        }


class FakeClassification:
    category = "Need Improvements"
    confidence = 0.90
    explanation = (
        "The employee was helpful, but the long waiting time "
        "indicates an area that needs improvement."
    )
    flagged_keywords = [
        "long waiting time",
        "resolution",
    ]


class FakeFeedbackRecord:
    id = 101


class FakeFeedbackDatabaseService:
    def __init__(self):
        self.created_feedback = None
        self.created_retrievals = None

    def create_feedback_record(
        self,
        db,
        feedback_id,
        filename,
        feedback_text,
        category,
        confidence,
        explanation,
        flagged_keywords,
    ):
        self.created_feedback = {
            "db": db,
            "feedback_id": feedback_id,
            "filename": filename,
            "feedback_text": feedback_text,
            "category": category,
            "confidence": confidence,
            "explanation": explanation,
            "flagged_keywords": flagged_keywords,
        }

        return FakeFeedbackRecord()

    def create_retrieval_records(
        self,
        db,
        feedback_record_id,
        retrieved_examples,
    ):
        self.created_retrievals = {
            "db": db,
            "feedback_record_id": feedback_record_id,
            "retrieved_examples": retrieved_examples,
        }

        return []


def test_root_endpoint():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert "service" in data


def test_analysis_status():
    client = TestClient(app)

    response = client.get("/api/analysis/status")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Analysis route is ready"
    }


def test_classify_feedback(monkeypatch):
    fake_rag_service = FakeRAGService()
    fake_database_service = FakeFeedbackDatabaseService()

    monkeypatch.setattr(
        "backend.routes.analysis.get_rag_service",
        lambda: fake_rag_service,
    )

    monkeypatch.setattr(
        "backend.routes.analysis.get_feedback_database_service",
        lambda: fake_database_service,
    )

    client = TestClient(app)

    response = client.post(
        "/api/analysis/classify",
        json={
            "feedback": (
                "The employee was helpful, but I had to wait "
                "a long time for my issue to be resolved."
            ),
            "filename": "feedback_001.pdf",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["feedback_id"].startswith("API-")
    assert len(data["feedback_id"]) == 16

    assert data["filename"] == "feedback_001.pdf"

    assert data["feedback"] == (
        "The employee was helpful, but I had to wait "
        "a long time for my issue to be resolved."
    )

    assert data["classification"]["category"] == "Need Improvements"
    assert data["classification"]["confidence"] == 0.90

    assert data["classification"]["explanation"] == (
        "The employee was helpful, but the long waiting time "
        "indicates an area that needs improvement."
    )

    assert data["classification"]["flagged_keywords"] == [
        "long waiting time",
        "resolution",
    ]

    assert data["retrieved_count"] == 2
    assert len(data["retrieved_examples"]) == 2

    assert data["retrieved_examples"][0]["feedback_id"] == "FB-REF-001"
    assert data["retrieved_examples"][0]["category"] == "Need Improvements"
    assert data["retrieved_examples"][0]["score"] == 1.25

    assert data["retrieved_examples"][1]["feedback_id"] == "FB-REF-002"
    assert data["retrieved_examples"][1]["category"] == "Good"
    assert data["retrieved_examples"][1]["score"] == 1.42

    assert fake_database_service.created_feedback is not None

    created_feedback = fake_database_service.created_feedback

    assert created_feedback["feedback_id"] == data["feedback_id"]
    assert created_feedback["filename"] == "feedback_001.pdf"
    assert created_feedback["feedback_text"] == data["feedback"]
    assert created_feedback["category"] == "Need Improvements"
    assert created_feedback["confidence"] == 0.90
    assert created_feedback["explanation"] == (
        "The employee was helpful, but the long waiting time "
        "indicates an area that needs improvement."
    )
    assert created_feedback["flagged_keywords"] == [
        "long waiting time",
        "resolution",
    ]

    assert fake_database_service.created_retrievals is not None

    created_retrievals = fake_database_service.created_retrievals

    assert created_retrievals["feedback_record_id"] == 101
    assert created_retrievals["retrieved_examples"] == [
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


def test_get_feedback_analysis(monkeypatch):
    from datetime import datetime

    class FakeFeedbackRecord:
        id = 101
        feedback_id = "API-TEST-001"
        filename = "feedback_001.pdf"
        feedback_text = (
            "The employee was helpful, but I had to wait "
            "a long time for my issue to be resolved."
        )
        category = "Need Improvements"
        confidence = 0.90
        explanation = (
            "The employee was helpful, but the long waiting time "
            "indicates an area that needs improvement."
        )
        flagged_keywords = '["long waiting time", "resolution"]'
        created_at = datetime(
            2026,
            9,
            18,
            10,
            30,
        )

    class FakeRetrievalRecord:
        def __init__(
            self,
            feedback_id,
            category,
            feedback,
            score,
            rank,
        ):
            self.retrieved_feedback_id = feedback_id
            self.retrieved_category = category
            self.retrieved_feedback = feedback
            self.similarity_score = score
            self.retrieval_rank = rank

    class FakeFeedbackDatabaseService:
        def get_feedback_record(
            self,
            db,
            feedback_id,
        ):
            assert feedback_id == "API-TEST-001"
            return FakeFeedbackRecord()

        def get_retrieval_records(
            self,
            db,
            feedback_record_id,
        ):
            assert feedback_record_id == 101

            return [
                FakeRetrievalRecord(
                    "FB-REF-001",
                    "Need Improvements",
                    "The customer experienced a long waiting time.",
                    1.25,
                    1,
                ),
                FakeRetrievalRecord(
                    "FB-REF-002",
                    "Good",
                    "The service was good overall.",
                    1.42,
                    2,
                ),
            ]

    monkeypatch.setattr(
        "backend.routes.analysis.get_feedback_database_service",
        lambda: FakeFeedbackDatabaseService(),
    )

    client = TestClient(app)

    response = client.get(
        "/api/analysis/API-TEST-001"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["feedback_id"] == "API-TEST-001"
    assert data["filename"] == "feedback_001.pdf"

    assert data["feedback"] == (
        "The employee was helpful, but I had to wait "
        "a long time for my issue to be resolved."
    )

    assert data["category"] == "Need Improvements"
    assert data["confidence"] == 0.90

    assert data["explanation"] == (
        "The employee was helpful, but the long waiting time "
        "indicates an area that needs improvement."
    )

    assert data["flagged_keywords"] == [
        "long waiting time",
        "resolution",
    ]

    assert data["created_at"] == "2026-09-18T10:30:00"

    assert data["retrieved_count"] == 2

    assert data["retrieved_examples"][0] == {
        "feedback_id": "FB-REF-001",
        "category": "Need Improvements",
        "feedback": (
            "The customer experienced a long waiting time."
        ),
        "similarity_score": 1.25,
        "retrieval_rank": 1,
    }

    assert data["retrieved_examples"][1] == {
        "feedback_id": "FB-REF-002",
        "category": "Good",
        "feedback": "The service was good overall.",
        "similarity_score": 1.42,
        "retrieval_rank": 2,
    }


def test_get_feedback_analysis_not_found(monkeypatch):
    class FakeFeedbackDatabaseService:
        def get_feedback_record(
            self,
            db,
            feedback_id,
        ):
            return None

    monkeypatch.setattr(
        "backend.routes.analysis.get_feedback_database_service",
        lambda: FakeFeedbackDatabaseService(),
    )

    client = TestClient(app)

    response = client.get(
        "/api/analysis/API-NOT-FOUND"
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": (
            "Feedback record 'API-NOT-FOUND' was not found."
        )
    }


def test_list_feedback(monkeypatch):
    from datetime import datetime

    class FakeFeedbackRecord:
        def __init__(
            self,
            feedback_id,
            filename,
            category,
            confidence,
            created_at,
        ):
            self.feedback_id = feedback_id
            self.filename = filename
            self.category = category
            self.confidence = confidence
            self.created_at = created_at

    class FakeFeedbackDatabaseService:
        def get_feedback_records_paginated(
            self,
            db,
            limit,
            offset,
        ):
            assert limit == 20
            assert offset == 0

            return (
                [
                    FakeFeedbackRecord(
                        "API-001",
                        "feedback_001.pdf",
                        "Poor",
                        0.95,
                        datetime(
                            2026,
                            9,
                            18,
                            12,
                            0,
                        ),
                    ),
                    FakeFeedbackRecord(
                        "API-002",
                        "feedback_002.pdf",
                        "Good",
                        0.88,
                        datetime(
                            2026,
                            9,
                            18,
                            11,
                            0,
                        ),
                    ),
                ],
                2,
            )

    monkeypatch.setattr(
        "backend.routes.analysis.get_feedback_database_service",
        lambda: FakeFeedbackDatabaseService(),
    )

    client = TestClient(app)

    response = client.get(
        "/api/analysis"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2
    assert data["limit"] == 20
    assert data["offset"] == 0
    assert len(data["items"]) == 2

    assert data["items"][0] == {
        "feedback_id": "API-001",
        "filename": "feedback_001.pdf",
        "category": "Poor",
        "confidence": 0.95,
        "created_at": "2026-09-18T12:00:00",
    }

    assert data["items"][1] == {
        "feedback_id": "API-002",
        "filename": "feedback_002.pdf",
        "category": "Good",
        "confidence": 0.88,
        "created_at": "2026-09-18T11:00:00",
    }


def test_list_feedback_with_pagination(
    monkeypatch,
):
    class FakeFeedbackDatabaseService:
        def get_feedback_records_paginated(
            self,
            db,
            limit,
            offset,
        ):
            assert limit == 5
            assert offset == 10

            return ([], 25)

    monkeypatch.setattr(
        "backend.routes.analysis.get_feedback_database_service",
        lambda: FakeFeedbackDatabaseService(),
    )

    client = TestClient(app)

    response = client.get(
        "/api/analysis?limit=5&offset=10"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["items"] == []
    assert data["total"] == 25
    assert data["limit"] == 5
    assert data["offset"] == 10


def test_list_feedback_rejects_invalid_limit():
    client = TestClient(app)

    response = client.get(
        "/api/analysis?limit=0"
    )

    assert response.status_code == 422


def test_list_feedback_rejects_invalid_offset():
    client = TestClient(app)

    response = client.get(
        "/api/analysis?offset=-1"
    )

    assert response.status_code == 422