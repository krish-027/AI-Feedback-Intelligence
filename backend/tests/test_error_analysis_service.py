import json

from backend.evaluation.error_analysis_service import (
    ErrorAnalysisService,
)


def create_test_report(tmp_path):
    """Create a small fake Stage 11 evaluation report."""

    report_path = tmp_path / "rag_evaluation_test.json"

    report = {
        "cases": [
            {
                "feedback_id": "FB001",
                "gold_category": "Good",
                "predicted_category": "Good",
                "correct": True,
                "reference_only": True,
                "gold_category_retrieved": True,
                "retrieved_count": 4,
            },
            {
                "feedback_id": "FB002",
                "gold_category": "Good",
                "predicted_category": "Need Improvements",
                "correct": False,
                "reference_only": True,
                "gold_category_retrieved": True,
                "retrieved_count": 4,
            },
            {
                "feedback_id": "FB003",
                "gold_category": "Poor",
                "predicted_category": "Need Improvements",
                "correct": False,
                "reference_only": True,
                "gold_category_retrieved": False,
                "retrieved_count": 2,
            },
            {
                "feedback_id": "FB004",
                "gold_category": "Need Improvements",
                "predicted_category": "Need Improvements",
                "correct": True,
                "reference_only": True,
                "gold_category_retrieved": True,
                "retrieved_count": 4,
            },
        ]
    }

    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    return report_path


def test_incorrect_cases(tmp_path):
    """Incorrect cases should be identified correctly."""

    service = ErrorAnalysisService(
        create_test_report(tmp_path)
    )

    incorrect = service.get_incorrect_cases()

    assert len(incorrect) == 2
    assert incorrect[0]["feedback_id"] == "FB002"
    assert incorrect[1]["feedback_id"] == "FB003"


def test_confusion_matrix(tmp_path):
    """Confusion matrix should count actual/predicted pairs."""

    service = ErrorAnalysisService(
        create_test_report(tmp_path)
    )

    matrix = service.confusion_matrix()

    assert matrix["Good"]["Good"] == 1
    assert matrix["Good"]["Need Improvements"] == 1
    assert matrix["Poor"]["Need Improvements"] == 1
    assert matrix["Need Improvements"]["Need Improvements"] == 1


def test_error_patterns(tmp_path):
    """Common classification error patterns should be counted."""

    service = ErrorAnalysisService(
        create_test_report(tmp_path)
    )

    patterns = service.error_patterns()

    assert patterns["Good -> Need Improvements"] == 1
    assert patterns["Poor -> Need Improvements"] == 1


def test_per_category_metrics(tmp_path):
    """Per-category precision, recall and F1 should be calculated."""

    service = ErrorAnalysisService(
        create_test_report(tmp_path)
    )

    metrics = service.per_category_metrics()

    assert metrics["Good"]["support"] == 2
    assert metrics["Need Improvements"]["support"] == 1
    assert metrics["Poor"]["support"] == 1

    assert metrics["Good"]["recall"] == 0.5
    assert metrics["Need Improvements"]["recall"] == 1.0
    assert metrics["Poor"]["recall"] == 0.0


def test_retrieval_diagnostics(tmp_path):
    """Retrieval statistics should be calculated correctly."""

    service = ErrorAnalysisService(
        create_test_report(tmp_path)
    )

    diagnostics = service.retrieval_diagnostics()

    assert diagnostics["total_cases"] == 4
    assert diagnostics["reference_only_rate"] == 1.0
    assert diagnostics["gold_category_retrieval_rate"] == 0.75
    assert diagnostics["average_retrieved_count"] == 3.5


def test_generate_report(tmp_path):
    """Complete Stage 12.1 report should contain all major sections."""

    service = ErrorAnalysisService(
        create_test_report(tmp_path)
    )

    report = service.generate_report()

    assert report["total_cases"] == 4
    assert report["correct_cases"] == 2
    assert report["incorrect_cases"] == 2
    assert report["accuracy"] == 0.5

    assert "confusion_matrix" in report
    assert "per_category_metrics" in report
    assert "error_patterns" in report
    assert "retrieval_diagnostics" in report
    assert "incorrect_case_details" in report