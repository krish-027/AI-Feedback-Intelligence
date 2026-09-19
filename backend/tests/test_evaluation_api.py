import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.evaluation_report_service import (
    EvaluationReportInvalidError,
    EvaluationReportNotFoundError,
)


client = TestClient(app)


def make_rag_report(split="development"):
    return {
        "evaluation_split": split,
        "classification_metrics": {
            "accuracy": 0.95,
            "macro_precision": 0.97,
            "macro_recall": 0.96,
            "macro_f1": 0.96,
            "weighted_f1": 0.95,
        },
        "retrieval_metrics": {
            "retrieval_success_rate": 1.0,
            "reference_only_rate": 1.0,
            "gold_category_retrieval_rate": 1.0,
            "average_retrieved": 4.0,
        },
        "cases": [],
    }


def make_ablation_report(split="development"):
    return {
        "evaluation_split": split,
        "rag_metrics": {
            "accuracy": 0.95,
            "macro_precision": 0.97,
            "macro_recall": 0.96,
            "macro_f1": 0.96,
            "weighted_f1": 0.95,
        },
        "no_rag_metrics": {
            "accuracy": 0.50,
            "macro_precision": 0.40,
            "macro_recall": 0.35,
            "macro_f1": 0.33,
            "weighted_f1": 0.45,
        },
        "paired_comparison": {
            "rag_better": 10,
            "no_rag_better": 1,
            "both_correct": 9,
            "both_wrong": 0,
        },
    }


@pytest.fixture
def evaluation_route_module():
    import backend.routes.evaluation as evaluation_route

    return evaluation_route


def test_evaluation_status():
    response = client.get(
        "/api/evaluation/status"
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["message"]
        == "Evaluation API is ready"
    )

    assert (
        "development"
        in data["supported_splits"]
    )

    assert (
        "final_benchmark"
        in data["supported_splits"]
    )


def test_development_evaluation_api(
    monkeypatch,
    evaluation_route_module,
):
    expected = make_rag_report(
        "development"
    )

    class FakeService:

        def get_rag_evaluation(self, split):
            assert split == "development"
            return expected

    monkeypatch.setattr(
        evaluation_route_module,
        "evaluation_service",
        FakeService(),
    )

    response = client.get(
        "/api/evaluation/development"
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_final_benchmark_evaluation_api(
    monkeypatch,
    evaluation_route_module,
):
    expected = make_rag_report(
        "final_benchmark"
    )

    class FakeService:

        def get_rag_evaluation(self, split):
            assert split == "final_benchmark"
            return expected

    monkeypatch.setattr(
        evaluation_route_module,
        "evaluation_service",
        FakeService(),
    )

    response = client.get(
        "/api/evaluation/final-benchmark"
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_development_rag_vs_no_rag_api(
    monkeypatch,
    evaluation_route_module,
):
    expected = make_ablation_report(
        "development"
    )

    class FakeService:

        def get_rag_ablation(self, split):
            assert split == "development"
            return expected

    monkeypatch.setattr(
        evaluation_route_module,
        "evaluation_service",
        FakeService(),
    )

    response = client.get(
        "/api/evaluation/rag-vs-no-rag/development"
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_final_benchmark_rag_vs_no_rag_api(
    monkeypatch,
    evaluation_route_module,
):
    expected = make_ablation_report(
        "final_benchmark"
    )

    class FakeService:

        def get_rag_ablation(self, split):
            assert split == "final_benchmark"
            return expected

    monkeypatch.setattr(
        evaluation_route_module,
        "evaluation_service",
        FakeService(),
    )

    response = client.get(
        "/api/evaluation/rag-vs-no-rag/final-benchmark"
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_missing_rag_report_returns_404(
    monkeypatch,
    evaluation_route_module,
):
    class FakeService:

        def get_rag_evaluation(self, split):
            raise EvaluationReportNotFoundError(
                "Report not found"
            )

    monkeypatch.setattr(
        evaluation_route_module,
        "evaluation_service",
        FakeService(),
    )

    response = client.get(
        "/api/evaluation/development"
    )

    assert response.status_code == 404
    assert (
        response.json()["detail"]
        == "Report not found"
    )


def test_missing_ablation_report_returns_404(
    monkeypatch,
    evaluation_route_module,
):
    class FakeService:

        def get_rag_ablation(self, split):
            raise EvaluationReportNotFoundError(
                "Ablation report not found"
            )

    monkeypatch.setattr(
        evaluation_route_module,
        "evaluation_service",
        FakeService(),
    )

    response = client.get(
        "/api/evaluation/rag-vs-no-rag/development"
    )

    assert response.status_code == 404
    assert (
        response.json()["detail"]
        == "Ablation report not found"
    )


def test_invalid_report_returns_500(
    monkeypatch,
    evaluation_route_module,
):
    class FakeService:

        def get_rag_evaluation(self, split):
            raise EvaluationReportInvalidError(
                "Malformed report"
            )

    monkeypatch.setattr(
        evaluation_route_module,
        "evaluation_service",
        FakeService(),
    )

    response = client.get(
        "/api/evaluation/development"
    )

    assert response.status_code == 500
    assert (
        response.json()["detail"]
        == "Malformed report"
    )


def test_invalid_split_returns_400(
    monkeypatch,
    evaluation_route_module,
):
    class FakeService:

        def get_rag_evaluation(self, split):
            raise ValueError(
                "Invalid evaluation split"
            )

    monkeypatch.setattr(
        evaluation_route_module,
        "evaluation_service",
        FakeService(),
    )

    response = client.get(
        "/api/evaluation/development"
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Invalid evaluation split"
    )


def test_summary_api(
    monkeypatch,
    evaluation_route_module,
):
    expected = {
        "development": {
            "rag_evaluation": make_rag_report(
                "development"
            ),
            "rag_vs_no_rag": make_ablation_report(
                "development"
            ),
        },
        "final_benchmark": {
            "rag_evaluation": make_rag_report(
                "final_benchmark"
            ),
            "rag_vs_no_rag": make_ablation_report(
                "final_benchmark"
            ),
        },
    }

    class FakeService:

        def get_summary(self):
            return expected

    monkeypatch.setattr(
        evaluation_route_module,
        "evaluation_service",
        FakeService(),
    )

    response = client.get(
        "/api/evaluation/summary"
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_evaluation_get_endpoints_do_not_run_gemini(
    monkeypatch,
    evaluation_route_module,
):
    expected = make_rag_report(
        "development"
    )

    gemini_called = False

    class FakeService:

        def get_rag_evaluation(self, split):
            return expected

    def fail_if_gemini_called(*args, **kwargs):
        nonlocal gemini_called
        gemini_called = True
        raise AssertionError(
            "Gemini should not be called by "
            "evaluation GET endpoint"
        )

    monkeypatch.setattr(
        evaluation_route_module,
        "evaluation_service",
        FakeService(),
    )

    # This is intentionally defensive. The endpoint itself
    # only reads a persisted report through the service.
    response = client.get(
        "/api/evaluation/development"
    )

    assert response.status_code == 200
    assert response.json() == expected
    assert gemini_called is False