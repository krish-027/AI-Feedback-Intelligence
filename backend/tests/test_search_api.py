from fastapi.testclient import TestClient

from langchain_core.documents import Document

from backend.main import app


class FakeRetrieverService:
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ):
        documents = [
            (
                Document(
                    page_content=(
                        "The customer waited a long time "
                        "before the issue was resolved."
                    ),
                    metadata={
                        "feedback_id": "FB-TEST001",
                        "category": "Need Improvements",
                        "split": "reference",
                    },
                ),
                0.42,
            ),
            (
                Document(
                    page_content=(
                        "The employee was helpful and "
                        "professional."
                    ),
                    metadata={
                        "feedback_id": "FB-TEST002",
                        "category": "Good",
                        "split": "reference",
                    },
                ),
                0.58,
            ),
        ]

        return documents[:k]


def test_search_status():
    client = TestClient(app)

    response = client.get("/api/search/status")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Search route is ready",
    }


def test_search_feedback(monkeypatch):
    fake_service = FakeRetrieverService()

    monkeypatch.setattr(
        "backend.routes.search.get_retriever_service",
        lambda: fake_service,
    )

    client = TestClient(app)

    response = client.post(
        "/api/search",
        json={
            "query": (
                "The customer had to wait "
                "for their issue to be resolved."
            ),
            "k": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"].startswith(
        "The customer had to wait"
    )
    assert data["result_count"] == 2
    assert len(data["results"]) == 2

    assert data["results"][0]["feedback_id"] == "FB-TEST001"
    assert data["results"][0]["category"] == "Need Improvements"
    assert data["results"][0]["split"] == "reference"
    assert data["results"][0]["score"] == 0.42

    assert data["results"][1]["feedback_id"] == "FB-TEST002"
    assert data["results"][1]["category"] == "Good"


def test_search_rejects_empty_query():
    client = TestClient(app)

    response = client.post(
        "/api/search",
        json={
            "query": "",
            "k": 4,
        },
    )

    assert response.status_code == 422


def test_search_rejects_invalid_k():
    client = TestClient(app)

    response = client.post(
        "/api/search",
        json={
            "query": "customer service",
            "k": 0,
        },
    )

    assert response.status_code == 422


def test_search_handles_retriever_error(monkeypatch):
    class FailingRetrieverService:
        def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
        ):
            raise RuntimeError("Test retrieval failure")

    monkeypatch.setattr(
        "backend.routes.search.get_retriever_service",
        lambda: FailingRetrieverService(),
    )

    client = TestClient(app)

    response = client.post(
        "/api/search",
        json={
            "query": "customer service",
            "k": 4,
        },
    )

    assert response.status_code == 500
    assert "Feedback search failed" in response.json()["detail"]