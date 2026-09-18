import json
from typing import Any, cast

from backend.models.classification import (
    FeedbackClassification,
)
from backend.evaluation.rag_ablation_service import (
    RAGAblationService,
)


class FakeDocument:
    """Minimal fake LangChain-style document."""

    def __init__(self, text: str) -> None:
        self.page_content = text


class FakeDocumentService:
    """Return controlled feedback text without reading real PDFs."""

    def load_pdf(self, path):
        return [
            FakeDocument(
                f"Feedback extracted from {path}"
            )
        ]


class FakeGeminiService:
    """
    Fake Gemini service that produces predefined no-RAG predictions.
    """

    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.calls = []

    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> FeedbackClassification:
        """Return the next predefined classification."""

        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

        return self.predictions.pop(0)


def create_manifest(tmp_path):
    """Create a small evaluation manifest."""

    manifest_path = (
        tmp_path / "dataset_manifest.csv"
    )

    manifest_path.write_text(
        "feedback_id,relative_path,category,split\n"
        "FB001,evaluation_001.pdf,Good,evaluation\n"
        "FB002,evaluation_002.pdf,Poor,evaluation\n"
        "FB003,reference_003.pdf,Good,reference\n",
        encoding="utf-8",
    )

    return manifest_path


def create_rag_report(tmp_path):
    """Create a fake Stage 11 report."""

    report_path = (
        tmp_path / "rag_evaluation_20260918.json"
    )

    report = {
        "cases": [
            {
                "feedback_id": "FB001",
                "gold_category": "Good",
                "predicted_category": "Good",
                "confidence": 0.90,
            },
            {
                "feedback_id": "FB002",
                "gold_category": "Poor",
                "predicted_category": "Need Improvements",
                "confidence": 0.90,
            },
        ]
    }

    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    return report_path


def create_service(tmp_path):
    """Create the ablation service with fake dependencies."""

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
        rag_report_path=rag_report_path,
        output_directory=tmp_path / "results",
        document_service=FakeDocumentService(),
        gemini_service=cast(Any, fake_gemini),
    )

    return service, fake_gemini


def test_only_successful_rag_cases_are_loaded(tmp_path):
    """Only successfully classified Stage 11 cases should be loaded."""

    service, fake_gemini = create_service(
        tmp_path
    )

    cases = service._load_successful_rag_cases()

    assert len(cases) == 2
    assert cases[0]["feedback_id"] == "FB001"
    assert cases[1]["feedback_id"] == "FB002"


def test_no_rag_classification_is_called(tmp_path):
    """No-RAG baseline should call Gemini once for each evaluation case."""

    service, fake_gemini = create_service(
        tmp_path
    )

    report = service.evaluate()

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


def test_rag_vs_no_rag_metrics(tmp_path):
    """RAG and no-RAG metrics should be calculated correctly."""

    service, fake_gemini = create_service(
        tmp_path
    )

    report = service.evaluate()

    # RAG:
    # Good -> Good      correct
    # Poor -> Need Imp. wrong
    assert (
        report["rag_metrics"]["accuracy"]
        == 0.5
    )

    # No-RAG:
    # Good -> Need Imp. wrong
    # Poor -> Poor       correct
    assert (
        report["no_rag_metrics"]["accuracy"]
        == 0.5
    )

    assert (
        report["accuracy_difference"]
        == 0.0
    )

    assert report["rag_better_cases"] == 1
    assert report["no_rag_better_cases"] == 1
    assert report["both_correct_cases"] == 0
    assert report["both_wrong_cases"] == 0


def test_better_case_accounting(tmp_path):
    """
    Cases where only one system is correct should be counted separately.
    """

    service, fake_gemini = create_service(
        tmp_path
    )

    # Modify the fake Gemini predictions so no-RAG makes the
    # first case correct and the second case correct.
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

    assert report["rag_better_cases"] == 0
    assert report["no_rag_better_cases"] == 1
    assert report["both_correct_cases"] == 1
    assert report["both_wrong_cases"] == 0


def test_report_can_be_saved(tmp_path):
    """JSON and CSV reports should be generated."""

    service, _ = create_service(
        tmp_path
    )

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
        saved_report["evaluation_cases"]
        == 2
    )
    assert "rag_metrics" in saved_report
    assert "no_rag_metrics" in saved_report