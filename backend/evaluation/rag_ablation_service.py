from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.models.classification import FeedbackClassification
from backend.prompts.classification_prompt import CLASSIFICATION_POLICY
from backend.services.document_service import get_document_service
from backend.services.gemini_service import (
    GeminiService,
    get_gemini_service,
)


CATEGORIES = [
    "Excellent",
    "Good",
    "Need Improvements",
    "Poor",
]

DEFAULT_MANIFEST_PATH = Path(
    "Sample_Data/evaluation/dataset_manifest.csv"
)

DEFAULT_RESULTS_DIRECTORY = Path(
    "backend/evaluation/results"
)


class RAGAblationService:
    """Compare the existing RAG results against a no-RAG baseline."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        rag_report_path: str | Path | None = None,
        output_directory: str | Path = DEFAULT_RESULTS_DIRECTORY,
        document_service: Any | None = None,
        gemini_service: GeminiService | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.output_directory = Path(output_directory)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Dataset manifest not found: {self.manifest_path}"
            )

        self.rag_report_path = (
            Path(rag_report_path)
            if rag_report_path is not None
            else self.find_latest_rag_report(
                self.output_directory
            )
        )

        if not self.rag_report_path.exists():
            raise FileNotFoundError(
                f"RAG evaluation report not found: "
                f"{self.rag_report_path}"
            )

        self.document_service = (
            document_service
            if document_service is not None
            else get_document_service()
        )

        self.gemini_service = (
            gemini_service
            if gemini_service is not None
            else get_gemini_service()
        )

    @staticmethod
    def find_latest_rag_report(
        results_directory: str | Path,
    ) -> Path:
        """Return the most recently generated Stage 11 RAG report."""

        directory = Path(results_directory)

        reports = sorted(
            directory.glob("rag_evaluation_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if not reports:
            raise FileNotFoundError(
                f"No Stage 11 RAG evaluation reports found in "
                f"{directory}"
            )

        return reports[0]

    def _load_manifest(self) -> list[dict[str, str]]:
        """Load the dataset manifest."""

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            return list(csv.DictReader(file))

    def _load_evaluation_manifest(
        self,
    ) -> dict[str, dict[str, str]]:
        """
        Load only evaluation-split records keyed by feedback_id.
        """

        rows = self._load_manifest()

        evaluation_rows = {
            row["feedback_id"]: row
            for row in rows
            if row.get("split") == "evaluation"
        }

        return evaluation_rows

    def _load_rag_report(self) -> dict[str, Any]:
        """Load the Stage 11 RAG evaluation report."""

        with self.rag_report_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(file)

        if not isinstance(report, dict):
            raise ValueError(
                "RAG evaluation report must contain a JSON object."
            )

        return report

    def _load_successful_rag_cases(
        self,
    ) -> list[dict[str, Any]]:
        """
        Extract successful RAG evaluation cases from Stage 11.

        Evaluation cases without a predicted category are ignored because
        they were not successfully classified.
        """

        report = self._load_rag_report()

        cases = report.get("cases")

        if not isinstance(cases, list):
            raise ValueError(
                "RAG evaluation report does not contain a 'cases' list."
            )

        successful_cases = [
            case
            for case in cases
            if case.get("gold_category")
            and case.get("predicted_category")
        ]

        if not successful_cases:
            raise ValueError(
                "The RAG evaluation report contains no successfully "
                "classified cases."
            )

        return successful_cases

    def _load_feedback_text(
        self,
        relative_path: str,
    ) -> str:
        """
        Extract evaluation feedback text using the existing document service.
        """

        documents = self.document_service.load_pdf(
            Path(relative_path)
        )

        if isinstance(documents, list):
            document_list = documents
        else:
            document_list = [documents]

        text_parts: list[str] = []

        for document in document_list:
            page_content = getattr(
                document,
                "page_content",
                None,
            )

            if page_content:
                text_parts.append(
                    page_content.strip()
                )

        feedback = "\n\n".join(
            part
            for part in text_parts
            if part
        )

        if not feedback.strip():
            raise ValueError(
                f"No usable feedback text extracted from "
                f"{relative_path}"
            )

        return feedback

    def _classify_without_rag(
        self,
        feedback: str,
    ) -> FeedbackClassification:
        """
        Classify feedback directly with Gemini without retrieved examples.

        The same classification policy used by the RAG system is retained.
        Only the historical reference examples are removed.
        """

        user_prompt = f"""
Current customer feedback:

{feedback}

Classify this feedback according to the classification policy.

Return:
- category
- confidence
- explanation
- flagged_keywords

The explanation must be concise and evidence-based.

Do not use any external information.
Do not invent facts that are not present in the feedback.
"""

        return self.gemini_service.classify(
            system_prompt=CLASSIFICATION_POLICY,
            user_prompt=user_prompt,
        )

    @staticmethod
    def _safe_divide(
        numerator: float,
        denominator: float,
    ) -> float:
        """Safely divide two values."""

        if denominator == 0:
            return 0.0

        return numerator / denominator

    @classmethod
    def _classification_metrics(
        cls,
        gold_labels: list[str],
        predicted_labels: list[str],
    ) -> dict[str, Any]:
        """
        Calculate accuracy, macro F1, weighted F1 and per-class metrics
        without requiring scikit-learn.
        """

        if len(gold_labels) != len(predicted_labels):
            raise ValueError(
                "Gold and predicted label lists must have equal length."
            )

        total = len(gold_labels)

        correct = sum(
            1
            for gold, predicted in zip(
                gold_labels,
                predicted_labels,
            )
            if gold == predicted
        )

        per_class: dict[str, dict[str, float]] = {}

        for category in CATEGORIES:
            true_positive = 0
            false_positive = 0
            false_negative = 0
            support = 0

            for gold, predicted in zip(
                gold_labels,
                predicted_labels,
            ):
                if gold == category:
                    support += 1

                if (
                    gold == category
                    and predicted == category
                ):
                    true_positive += 1

                elif (
                    gold != category
                    and predicted == category
                ):
                    false_positive += 1

                elif (
                    gold == category
                    and predicted != category
                ):
                    false_negative += 1

            precision = cls._safe_divide(
                true_positive,
                true_positive + false_positive,
            )

            recall = cls._safe_divide(
                true_positive,
                true_positive + false_negative,
            )

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_class[category] = {
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        macro_f1 = (
            sum(
                metrics["f1"]
                for metrics in per_class.values()
            )
            / len(CATEGORIES)
        )

        weighted_f1 = cls._safe_divide(
            sum(
                metrics["f1"] * metrics["support"]
                for metrics in per_class.values()
            ),
            total,
        )

        return {
            "total_cases": total,
            "correct_cases": correct,
            "accuracy": cls._safe_divide(
                correct,
                total,
            ),
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class": per_class,
        }

    def evaluate(self) -> dict[str, Any]:
        """
        Run the no-RAG baseline on exactly the same successful cases used
        by the Stage 11 RAG evaluation.
        """

        evaluation_manifest = (
            self._load_evaluation_manifest()
        )

        rag_cases = (
            self._load_successful_rag_cases()
        )

        results: list[dict[str, Any]] = []

        for index, rag_case in enumerate(
            rag_cases,
            start=1,
        ):
            feedback_id = rag_case["feedback_id"]

            manifest_row = evaluation_manifest.get(
                feedback_id
            )

            if manifest_row is None:
                raise ValueError(
                    f"Feedback ID {feedback_id} from the RAG report "
                    f"is not present in the evaluation manifest."
                )

            relative_path = manifest_row[
                "relative_path"
            ]

            gold_category = rag_case[
                "gold_category"
            ]

            rag_prediction = rag_case[
                "predicted_category"
            ]

            print(
                f"[{index}/{len(rag_cases)}] "
                f"{feedback_id}"
            )

            feedback = self._load_feedback_text(
                relative_path
            )

            no_rag_classification = (
                self._classify_without_rag(
                    feedback
                )
            )

            no_rag_prediction = (
                no_rag_classification.category
            )

            rag_correct = (
                rag_prediction
                == gold_category
            )

            no_rag_correct = (
                no_rag_prediction
                == gold_category
            )

            results.append(
                {
                    "feedback_id": feedback_id,
                    "relative_path": relative_path,
                    "gold_category": gold_category,
                    "rag_prediction": rag_prediction,
                    "rag_correct": rag_correct,
                    "rag_confidence": rag_case.get(
                        "confidence"
                    ),
                    "no_rag_prediction": (
                        no_rag_prediction
                    ),
                    "no_rag_correct": no_rag_correct,
                    "no_rag_confidence": (
                        no_rag_classification.confidence
                    ),
                    "no_rag_explanation": (
                        no_rag_classification.explanation
                    ),
                    "rag_better": (
                        rag_correct
                        and not no_rag_correct
                    ),
                    "no_rag_better": (
                        no_rag_correct
                        and not rag_correct
                    ),
                    "same_correctness": (
                        rag_correct
                        == no_rag_correct
                    ),
                }
            )

        gold_labels = [
            result["gold_category"]
            for result in results
        ]

        rag_predictions = [
            result["rag_prediction"]
            for result in results
        ]

        no_rag_predictions = [
            result["no_rag_prediction"]
            for result in results
        ]

        rag_metrics = self._classification_metrics(
            gold_labels,
            rag_predictions,
        )

        no_rag_metrics = self._classification_metrics(
            gold_labels,
            no_rag_predictions,
        )

        rag_better_cases = sum(
            1
            for result in results
            if result["rag_better"]
        )

        no_rag_better_cases = sum(
            1
            for result in results
            if result["no_rag_better"]
        )

        both_correct_cases = sum(
            1
            for result in results
            if result["rag_correct"]
            and result["no_rag_correct"]
        )

        both_wrong_cases = sum(
            1
            for result in results
            if not result["rag_correct"]
            and not result["no_rag_correct"]
        )

        report = {
            "stage": "12.2",
            "experiment": "RAG vs No-RAG Ablation",
            "source_rag_report": str(
                self.rag_report_path
            ),
            "evaluation_cases": len(results),
            "rag_metrics": rag_metrics,
            "no_rag_metrics": no_rag_metrics,
            "accuracy_difference": (
                rag_metrics["accuracy"]
                - no_rag_metrics["accuracy"]
            ),
            "macro_f1_difference": (
                rag_metrics["macro_f1"]
                - no_rag_metrics["macro_f1"]
            ),
            "weighted_f1_difference": (
                rag_metrics["weighted_f1"]
                - no_rag_metrics["weighted_f1"]
            ),
            "rag_better_cases": rag_better_cases,
            "no_rag_better_cases": (
                no_rag_better_cases
            ),
            "both_correct_cases": (
                both_correct_cases
            ),
            "both_wrong_cases": (
                both_wrong_cases
            ),
            "results": results,
        }

        return report

    def save_report(
        self,
        report: dict[str, Any],
    ) -> Path:
        """Save the Stage 12.2 JSON report."""

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            self.output_directory
            / "stage12_rag_ablation.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
            )

        return output_path

    @staticmethod
    def _prediction_confusion(
        results: list[dict[str, Any]],
        prediction_key: str,
    ) -> dict[str, dict[str, int]]:
        """Create a confusion matrix for a prediction column."""

        matrix = {
            actual: {
                predicted: 0
                for predicted in CATEGORIES
            }
            for actual in CATEGORIES
        }

        for result in results:
            actual = result["gold_category"]
            predicted = result[prediction_key]

            if (
                actual in CATEGORIES
                and predicted in CATEGORIES
            ):
                matrix[actual][predicted] += 1

        return matrix

    def save_csv(
        self,
        report: dict[str, Any],
    ) -> Path:
        """Save case-level Stage 12.2 results as CSV."""

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            self.output_directory
            / "stage12_rag_ablation.csv"
        )

        fieldnames = [
            "feedback_id",
            "relative_path",
            "gold_category",
            "rag_prediction",
            "rag_correct",
            "rag_confidence",
            "no_rag_prediction",
            "no_rag_correct",
            "no_rag_confidence",
            "no_rag_explanation",
            "rag_better",
            "no_rag_better",
            "same_correctness",
        ]

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for result in report["results"]:
                writer.writerow(
                    {
                        field: result.get(field)
                        for field in fieldnames
                    }
                )

        return output_path

    def run_and_save(self) -> dict[str, Any]:
        """Run the ablation and save JSON and CSV reports."""

        report = self.evaluate()

        json_path = self.save_report(
            report
        )

        csv_path = self.save_csv(
            report
        )

        print("\n" + "=" * 60)
        print("STAGE 12.2 - RAG VS NO-RAG")
        print("=" * 60)

        print(
            f"Evaluation cases : "
            f"{report['evaluation_cases']}"
        )

        print(
            f"RAG accuracy     : "
            f"{report['rag_metrics']['accuracy']:.4f}"
        )

        print(
            f"No-RAG accuracy  : "
            f"{report['no_rag_metrics']['accuracy']:.4f}"
        )

        print(
            f"Accuracy delta   : "
            f"{report['accuracy_difference']:+.4f}"
        )

        print(
            f"RAG macro F1     : "
            f"{report['rag_metrics']['macro_f1']:.4f}"
        )

        print(
            f"No-RAG macro F1  : "
            f"{report['no_rag_metrics']['macro_f1']:.4f}"
        )

        print(
            f"Macro F1 delta   : "
            f"{report['macro_f1_difference']:+.4f}"
        )

        print(
            f"RAG better cases : "
            f"{report['rag_better_cases']}"
        )

        print(
            f"No-RAG better    : "
            f"{report['no_rag_better_cases']}"
        )

        print(
            f"Both correct     : "
            f"{report['both_correct_cases']}"
        )

        print(
            f"Both wrong       : "
            f"{report['both_wrong_cases']}"
        )

        print(
            f"\nJSON report      : {json_path}"
        )

        print(
            f"CSV report       : {csv_path}"
        )

        return report


def main() -> None:
    """Run the Stage 12.2 RAG ablation."""

    service = RAGAblationService()

    service.run_and_save()


if __name__ == "__main__":
    main()