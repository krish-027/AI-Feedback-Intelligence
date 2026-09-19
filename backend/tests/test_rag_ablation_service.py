import csv
import json
from pathlib import Path
from typing import Any, cast

from backend.evaluation.rag_ablation_service import (
    RAGAblationService,
)
from backend.models.classification import (
    FeedbackClassification,
)


class FakeDocument:
    def __init__(self, text: str) -> None:
        self.page_content = text


class FakeDocumentService:
    def load_pdf(self, path: Path):
        return [
            FakeDocument(
                f"Feedback extracted from {path}"
            )
        ]


class FakeGeminiService:
    def __init__(
        self,
        predictions: list[FeedbackClassification],
    ) -> None:
        self.predictions = list(predictions)
        self.calls: list[dict[str, str]] = []

    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> FeedbackClassification:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

        if not self.predictions:
            raise AssertionError(
                "FakeGeminiService received more calls "
                "than expected."
            )

        return self.predictions.pop(0)


def create_manifest(tmp_path: Path) -> Path:
    """
    Create a small development manifest for testing.
    """

    manifest_path = (
        tmp_path / "dataset_manifest.csv"
    )

    rows = [
        {
            "feedback_id": "FB001",
            "filename": "development_001.pdf",
            "relative_path": "development_001.pdf",
            "category": "Good",
            "split": "development",
        },
        {
            "feedback_id": "FB002",
            "filename": "development_002.pdf",
            "relative_path": "development_002.pdf",
            "category": "Poor",
            "split": "development",
        },
        {
            "feedback_id": "FB003",
            "filename": "reference_003.pdf",
            "relative_path": "reference_003.pdf",
            "category": "Good",
            "split": "reference",
        },
    ]

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "feedback_id",
                "filename",
                "relative_path",
                "category",
                "split",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def create_rag_report(tmp_path: Path) -> Path:
    """
    Create a mock RAG evaluation report containing
    successfully classified development cases.
    """

    report_path = (
        tmp_path
        / "rag_evaluation_development_20260919_000000.json"
    )

    report = {
        "evaluation_split": "development",
        "cases": [
            {
                "feedback_id": "FB001",
                "gold_category": "Good",
                "predicted_category": "Good",
                "confidence": 0.90,
                "status": "success",
            },
            {
                "feedback_id": "FB002",
                "gold_category": "Poor",
                "predicted_category": "Need Improvements",
                "confidence": 0.90,
                "status": "success",
            },
        ],
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report_path


def create_service(
    tmp_path: Path,
) -> tuple[
    RAGAblationService,
    FakeGeminiService,
]:
    manifest_path = create_manifest(
        tmp_path
    )

    rag_report_path = create_rag_report(
        tmp_path
    )

    fake_gemini = FakeGeminiService(
        [
            FeedbackClassification(
                category="Need Improvements",
                confidence=0.80,
                explanation="Baseline prediction.",
                flagged_keywords=[],
            ),
            FeedbackClassification(
                category="Poor",
                confidence=0.91,
                explanation="Baseline prediction.",
                flagged_keywords=[],
            ),
        ]
    )

    service = RAGAblationService(
        manifest_path=manifest_path,
        split="development",
        rag_report_path=rag_report_path,
        document_service=FakeDocumentService(),
        gemini_service=cast(
            Any,
            fake_gemini,
        ),
    )

    return service, fake_gemini


def test_only_successful_rag_cases_are_loaded(
    tmp_path: Path,
):
    service, _ = create_service(tmp_path)

    cases = (
        service._load_successful_rag_cases()
    )

    assert len(cases) == 2

    assert (
        cases[0]["feedback_id"]
        == "FB001"
    )

    assert (
        cases[1]["feedback_id"]
        == "FB002"
    )

    assert (
        cases[0]["gold_category"]
        == "Good"
    )

    assert (
        cases[0]["predicted_category"]
        == "Good"
    )


def test_no_rag_classification_is_called(
    tmp_path: Path,
):
    service, fake_gemini = create_service(
        tmp_path
    )

    report = service.evaluate()

    assert (
        report["evaluation_split"]
        == "development"
    )

    assert len(report["results"]) == 2

    assert len(fake_gemini.calls) == 2

    assert (
        "Current customer feedback:"
        in fake_gemini.calls[0]["user_prompt"]
    )

    assert (
        "Reference Example"
        not in fake_gemini.calls[0]["user_prompt"]
    )


def test_rag_vs_no_rag_metrics(
    tmp_path: Path,
):
    service, _ = create_service(tmp_path)

    report = service.evaluate()

    # RAG:
    # FB001: Good -> Good = correct
    # FB002: Poor -> Need Improvements = wrong
    assert (
        report["rag_metrics"]["accuracy"]
        == 0.5
    )

    # No-RAG:
    # FB001: Good -> Need Improvements = wrong
    # FB002: Poor -> Poor = correct
    assert (
        report["no_rag_metrics"]["accuracy"]
        == 0.5
    )

    assert (
        report["accuracy_difference"]
        == 0.0
    )

    assert (
        report["rag_better_cases"]
        == 1
    )

    assert (
        report["no_rag_better_cases"]
        == 1
    )

    assert (
        report["both_correct_cases"]
        == 0
    )

    assert (
        report["both_wrong_cases"]
        == 0
    )


def test_better_case_accounting(
    tmp_path: Path,
):
    service, fake_gemini = create_service(
        tmp_path
    )

    fake_gemini.predictions = [
        FeedbackClassification(
            category="Good",
            confidence=0.90,
            explanation="Correct baseline prediction.",
            flagged_keywords=[],
        ),
        FeedbackClassification(
            category="Poor",
            confidence=0.90,
            explanation="Correct baseline prediction.",
            flagged_keywords=[],
        ),
    ]

    report = service.evaluate()

    # FB001:
    # RAG correct, No-RAG correct
    #
    # FB002:
    # RAG wrong, No-RAG correct
    assert (
        report["rag_better_cases"]
        == 0
    )

    assert (
        report["no_rag_better_cases"]
        == 1
    )

    assert (
        report["both_correct_cases"]
        == 1
    )

    assert (
        report["both_wrong_cases"]
        == 0
    )


def test_report_can_be_saved(
    tmp_path: Path,
):
    service, _ = create_service(tmp_path)

    report = service.evaluate()

    json_path = service.save_report(
        report
    )

    csv_path = service.save_csv(
        report
    )

    assert json_path.exists()
    assert csv_path.exists()

    saved_report = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        saved_report["evaluation_split"]
        == "development"
    )

    assert (
        len(saved_report["results"])
        == 2
    )

    assert (
        "rag_metrics"
        in saved_report
    )

    assert (
        "no_rag_metrics"
        in saved_report
    )