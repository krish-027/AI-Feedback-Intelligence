from io import BytesIO
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from backend.main import app


class FakeDocument:
    page_content = (
        "The employee was helpful, but I had to wait "
        "a long time for my issue to be resolved."
    )


class FakeDocumentService:
    def load_pdf(self, pdf_path):
        return [FakeDocument()]


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


class FakeRAGService:
    def classify_feedback(
        self,
        feedback,
        include_retrieved_examples=True,
    ):
        assert include_retrieved_examples is True

        return {
            "feedback": feedback,
            "classification": FakeClassification(),
            "retrieved_examples": [
                {
                    "feedback_id": "FB-REF-001",
                    "category": "Need Improvements",
                    "feedback": (
                        "The customer experienced a long waiting time."
                    ),
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


def test_upload_status():
    client = TestClient(app)

    response = client.get(
        "/api/upload/status"
    )

    assert response.status_code == 200

    assert response.json() == {
        "message": "Upload route is ready"
    }


def test_upload_pdf(monkeypatch):
    fake_database_service = FakeFeedbackDatabaseService()

    monkeypatch.setattr(
        "backend.routes.upload.get_document_service",
        lambda: FakeDocumentService(),
    )

    monkeypatch.setattr(
        "backend.routes.upload.get_rag_service",
        lambda: FakeRAGService(),
    )

    monkeypatch.setattr(
        "backend.routes.upload.get_feedback_database_service",
        lambda: fake_database_service,
    )

    client = TestClient(app)

    response = client.post(
        "/api/upload/pdf",
        files={
            "file": (
                "feedback.pdf",
                BytesIO(b"%PDF-1.4 fake pdf content"),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["feedback_id"].startswith("PDF-")
    assert len(data["feedback_id"]) == 16

    assert data["filename"] == "feedback.pdf"
    assert data["status"] == "success"

    assert data["message"] == (
        "PDF uploaded, analyzed, and stored successfully."
    )

    assert data["extracted_text"] == (
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

    assert data["retrieved_count"] == 2

    assert fake_database_service.created_feedback is not None

    created_feedback = fake_database_service.created_feedback

    assert created_feedback["feedback_id"] == data["feedback_id"]
    assert created_feedback["filename"] == "feedback.pdf"
    assert created_feedback["feedback_text"] == (
        "The employee was helpful, but I had to wait "
        "a long time for my issue to be resolved."
    )
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
            "feedback": (
                "The customer experienced a long waiting time."
            ),
            "score": 1.25,
        },
        {
            "feedback_id": "FB-REF-002",
            "category": "Good",
            "feedback": "The service was good overall.",
            "score": 1.42,
        },
    ]


def test_upload_rejects_non_pdf():
    client = TestClient(app)

    response = client.post(
        "/api/upload/pdf",
        files={
            "file": (
                "feedback.txt",
                BytesIO(b"plain text"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Only PDF files are supported."
    }


def test_upload_rejects_empty_pdf():
    client = TestClient(app)

    response = client.post(
        "/api/upload/pdf",
        files={
            "file": (
                "feedback.pdf",
                BytesIO(b""),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "The uploaded PDF is empty."
    }


def test_upload_handles_document_service_error(
    monkeypatch,
):
    class FailingDocumentService:
        def load_pdf(self, pdf_path):
            raise RuntimeError(
                "Test document processing failure"
            )

    monkeypatch.setattr(
        "backend.routes.upload.get_document_service",
        lambda: FailingDocumentService(),
    )

    client = TestClient(app)

    response = client.post(
        "/api/upload/pdf",
        files={
            "file": (
                "feedback.pdf",
                BytesIO(b"%PDF-1.4 fake pdf content"),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 500

    assert (
        "PDF processing and analysis failed"
        in response.json()["detail"]
    )

    assert (
        "Test document processing failure"
        in response.json()["detail"]
    )