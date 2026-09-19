import csv
import json
from pathlib import Path
from typing import cast

import pytest

from langchain_core.documents import Document

from backend.evaluation.rag_evaluation_service import (
    RAGEvaluationService,
)
from backend.services.rag_service import RAGService
from backend.models.classification import (
    FeedbackClassification,
)


class FakeDocumentService:
    def __init__(
        self,
        documents_by_path: dict[str, list[Document]],
    ) -> None:
        self.documents_by_path = documents_by_path

    def load_pdf(
        self,
        pdf_path: Path,
    ) -> list[Document]:
        return self.documents_by_path[
            str(pdf_path)
        ]


class FakeRAGService:
    def __init__(
        self,
        predictions: dict[str, FeedbackClassification],
    ) -> None:
        self.predictions = predictions
        self.calls = []

    def classify_feedback(
        self,
        feedback: str,
        include_retrieved_examples: bool = True,
    ) -> dict:
        self.calls.append(
            {
                "feedback": feedback,
                "include_retrieved_examples": (
                    include_retrieved_examples
                ),
            }
        )

        key = feedback.strip()

        classification = self.predictions[
            key
        ]

        if key == "Development feedback 1":
            retrieved = [
                {
                    "feedback_id": "FB-REF001",
                    "category": "Good",
                    "split": "reference",
                    "feedback": "Reference feedback 1",
                },
                {
                    "feedback_id": "FB-REF002",
                    "category": "Need Improvements",
                    "split": "reference",
                    "feedback": "Reference feedback 2",
                },
            ]

        else:
            retrieved = [
                {
                    "feedback_id": "FB-REF003",
                    "category": "Poor",
                    "split": "reference",
                    "feedback": "Reference feedback 3",
                }
            ]

        return {
            "feedback": feedback,
            "classification": classification,
            "retrieved_examples": retrieved,
            "retrieved_count": len(retrieved),
        }


def create_manifest(
    path: Path,
) -> None:
    rows = [
        {
            "relative_path": "development_001.pdf",
            "split": "development",
            "category": "Need Improvements",
            "feedback_id": "FB-DEV001",
        },
        {
            "relative_path": "development_002.pdf",
            "split": "development",
            "category": "Poor",
            "feedback_id": "FB-DEV002",
        },
        {
            "relative_path": "final_001.pdf",
            "split": "final_benchmark",
            "category": "Good",
            "feedback_id": "FB-FINAL001",
        },
        {
            "relative_path": "reference_001.pdf",
            "split": "reference",
            "category": "Good",
            "feedback_id": "FB-REF999",
        },
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "relative_path",
                "split",
                "category",
                "feedback_id",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def create_test_services(
    manifest_path: Path,
) -> RAGEvaluationService:
    classification_1 = FeedbackClassification(
        category="Need Improvements",
        confidence=0.90,
        explanation="A meaningful service issue was identified.",
        flagged_keywords=[
            "waiting",
        ],
    )

    classification_2 = FeedbackClassification(
        category="Poor",
        confidence=0.95,
        explanation="The customer expressed strong dissatisfaction.",
        flagged_keywords=[
            "terrible",
        ],
    )

    predictions = {
        "Development feedback 1": classification_1,
        "Development feedback 2": classification_2,
    }

    rag_service = FakeRAGService(
        predictions
    )

    documents = {
        "development_001.pdf": [
            Document(
                page_content="Development feedback 1"
            )
        ],
        "development_002.pdf": [
            Document(
                page_content="Development feedback 2"
            )
        ],
    }

    document_service = FakeDocumentService(
        documents
    )

    return RAGEvaluationService(
        manifest_path=manifest_path,
        output_directory=(
            manifest_path.parent / "results"
        ),
        rag_service=cast(
            RAGService,
            rag_service,
        ),
        document_service=document_service,
    )


def test_manifest_selects_only_development_records(
    tmp_path: Path,
) -> None:
    manifest = (
        tmp_path / "dataset_manifest.csv"
    )

    create_manifest(
        manifest
    )

    evaluator = create_test_services(
        manifest
    )

    records = (
        evaluator._load_evaluation_records(
            split="development"
        )
    )

    assert len(records) == 2

    assert all(
        record["split"] == "development"
        for record in records
    )


def test_manifest_can_select_final_benchmark_records(
    tmp_path: Path,
) -> None:
    manifest = (
        tmp_path / "dataset_manifest.csv"
    )

    create_manifest(
        manifest
    )

    evaluator = create_test_services(
        manifest
    )

    records = (
        evaluator._load_evaluation_records(
            split="final_benchmark"
        )
    )

    assert len(records) == 1

    assert records[0]["split"] == "final_benchmark"

    assert (
        records[0]["feedback_id"]
        == "FB-FINAL001"
    )


def test_evaluation_calculates_classification_metrics(
    tmp_path: Path,
) -> None:
    manifest = (
        tmp_path / "dataset_manifest.csv"
    )

    create_manifest(
        manifest
    )

    evaluator = create_test_services(
        manifest
    )

    report = evaluator.evaluate(
        split="development",
        save_report=False,
    )

    metrics = report[
        "classification_metrics"
    ]

    assert (
        report["evaluation_split"]
        == "development"
    )

    assert metrics[
        "evaluated_cases"
    ] == 2

    assert metrics[
        "correct_cases"
    ] == 2

    assert metrics[
        "accuracy"
    ] == 1.0

    assert metrics[
        "macro_f1"
    ] == 0.5

    assert metrics[
        "weighted_f1"
    ] == 1.0


def test_evaluation_calculates_retrieval_metrics(
    tmp_path: Path,
) -> None:
    manifest = (
        tmp_path / "dataset_manifest.csv"
    )

    create_manifest(
        manifest
    )

    evaluator = create_test_services(
        manifest
    )

    report = evaluator.evaluate(
        split="development",
        save_report=False,
    )

    metrics = report[
        "retrieval_metrics"
    ]

    assert metrics[
        "retrieval_success_rate"
    ] == 1.0

    assert metrics[
        "reference_only_rate"
    ] == 1.0

    assert metrics[
        "gold_category_retrieval_rate"
    ] == 1.0

    assert metrics[
        "average_retrieved_count"
    ] == 1.5


def test_case_results_contain_gold_and_prediction(
    tmp_path: Path,
) -> None:
    manifest = (
        tmp_path / "dataset_manifest.csv"
    )

    create_manifest(
        manifest
    )

    evaluator = create_test_services(
        manifest
    )

    report = evaluator.evaluate(
        split="development",
        save_report=False,
    )

    first_case = report[
        "cases"
    ][0]

    assert (
        first_case["feedback_id"]
        == "FB-DEV001"
    )

    assert (
        first_case["gold_category"]
        == "Need Improvements"
    )

    assert (
        first_case["predicted_category"]
        == "Need Improvements"
    )

    assert first_case[
        "correct"
    ] is True

    assert (
        first_case["status"]
        == "success"
    )


def test_evaluation_does_not_include_reference_or_final_benchmark(
    tmp_path: Path,
) -> None:
    manifest = (
        tmp_path / "dataset_manifest.csv"
    )

    create_manifest(
        manifest
    )

    evaluator = create_test_services(
        manifest
    )

    report = evaluator.evaluate(
        split="development",
        save_report=False,
    )

    paths = [
        case["relative_path"]
        for case in report["cases"]
    ]

    assert (
        "reference_001.pdf"
        not in paths
    )

    assert (
        "final_001.pdf"
        not in paths
    )

    assert len(paths) == 2


def test_invalid_evaluation_split_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = (
        tmp_path / "dataset_manifest.csv"
    )

    create_manifest(
        manifest
    )

    evaluator = create_test_services(
        manifest
    )

    with pytest.raises(ValueError):
        evaluator._load_evaluation_records(
            split="evaluation"
        )


def test_invalid_manifest_category_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = (
        tmp_path / "dataset_manifest.csv"
    )

    with manifest.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "relative_path",
                "split",
                "category",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "relative_path": "development_001.pdf",
                "split": "development",
                "category": "Average",
            }
        )

    evaluator = RAGEvaluationService(
        manifest_path=manifest
    )

    with pytest.raises(ValueError):
        evaluator._load_evaluation_records(
            split="development"
        )


def test_report_can_be_saved(
    tmp_path: Path,
) -> None:
    manifest = (
        tmp_path / "dataset_manifest.csv"
    )

    create_manifest(
        manifest
    )

    output_directory = (
        tmp_path / "results"
    )

    test_evaluator = create_test_services(
        manifest
    )

    evaluator = RAGEvaluationService(
        manifest_path=manifest,
        output_directory=output_directory,
        rag_service=test_evaluator.rag_service,
        document_service=test_evaluator.document_service,
    )

    evaluator.evaluate(
        split="development",
        save_report=True,
    )

    json_files = list(
        output_directory.glob(
            "*.json"
        )
    )

    csv_files = list(
        output_directory.glob(
            "*.csv"
        )
    )

    assert len(json_files) == 1
    assert len(csv_files) == 1

    with json_files[0].open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_report = json.load(
            file
        )

    assert (
        saved_report[
            "evaluation_split"
        ]
        == "development"
    )

    assert (
        saved_report[
            "classification_metrics"
        ]["accuracy"]
        == 1.0
    )


def test_safe_divide_handles_zero() -> None:
    assert (
        RAGEvaluationService._safe_divide(
            1,
            0,
        )
        == 0.0
    )