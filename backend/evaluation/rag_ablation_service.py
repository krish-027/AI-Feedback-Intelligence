from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
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

VALID_SPLITS = {
    "development",
    "final_benchmark",
}

DEFAULT_MANIFEST_PATH = Path(
    "Sample_Data/evaluation/dataset_manifest.csv"
)

DEFAULT_RESULTS_DIRECTORY = Path(
    "backend/evaluation/results"
)


class RAGAblationService:
    """
    Compare RAG classification results against a no-RAG baseline.

    The RAG predictions come from an already-generated RAG evaluation
    report. The no-RAG baseline classifies the exact same successfully
    evaluated cases using the same classification policy but without
    retrieved reference examples.
    """

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        split: str = "development",
        rag_report_path: str | Path | None = None,
        output_directory: str | Path = DEFAULT_RESULTS_DIRECTORY,
        document_service: Any | None = None,
        gemini_service: GeminiService | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Unsupported evaluation split: {split}. "
                f"Expected one of: {sorted(VALID_SPLITS)}"
            )

        self.split = split

        self.manifest_path = self._resolve_path(
            manifest_path
        )

        self.output_directory = self._resolve_path(
            output_directory
        )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                "Dataset manifest not found: "
                f"{self.manifest_path}"
            )

        if rag_report_path is not None:
            self.rag_report_path = self._resolve_path(
                rag_report_path
            )
        else:
            self.rag_report_path = (
                self.find_latest_rag_report(
                    self.output_directory,
                    split=self.split,
                )
            )

        if not self.rag_report_path.exists():
            raise FileNotFoundError(
                "RAG evaluation report not found: "
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
    def _resolve_path(
        path: str | Path,
    ) -> Path:
        """
        Resolve a project-relative path.

        Absolute paths are returned unchanged. Relative paths are first
        checked from the current working directory and then from the
        project root.
        """

        candidate = Path(path)

        if candidate.is_absolute():
            return candidate

        if candidate.exists():
            return candidate

        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        project_candidate = (
            project_root / candidate
        )

        return project_candidate

    @classmethod
    def find_latest_rag_report(
        cls,
        results_directory: str | Path,
        split: str,
    ) -> Path:
        """
        Find the latest timestamped RAG evaluation report for a split.
        """

        if split not in VALID_SPLITS:
            raise ValueError(
                f"Unsupported evaluation split: {split}. "
                f"Expected one of: {sorted(VALID_SPLITS)}"
            )

        directory = cls._resolve_path(
            results_directory
        )

        pattern = (
            f"rag_evaluation_{split}_*.json"
        )

        reports = sorted(
            directory.glob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if not reports:
            raise FileNotFoundError(
                "No RAG evaluation reports found for "
                f"split '{split}' in {directory}"
            )

        return reports[0]

    def _load_manifest(
        self,
    ) -> list[dict[str, str]]:
        """
        Load the complete dataset manifest.
        """

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            return list(
                csv.DictReader(file)
            )

    def _load_split_manifest(
        self,
    ) -> dict[str, dict[str, str]]:
        """
        Load records belonging to the configured evaluation split.
        """

        rows = self._load_manifest()

        split_rows: dict[
            str,
            dict[str, str],
        ] = {}

        for row in rows:
            feedback_id = row.get(
                "feedback_id"
            )
            row_split = row.get("split")

            if (
                feedback_id
                and row_split == self.split
            ):
                split_rows[feedback_id] = row

        return split_rows

    def _load_evaluation_manifest(
        self,
    ) -> dict[str, dict[str, str]]:
        """
        Backward-compatible alias for loading the configured
        evaluation split.
        """

        return self._load_split_manifest()

    def _load_rag_report(
        self,
    ) -> dict[str, Any]:
        """
        Load and validate the RAG evaluation report.
        """

        with self.rag_report_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(file)

        if not isinstance(report, dict):
            raise ValueError(
                "RAG evaluation report must contain "
                "a JSON object."
            )

        report_split = report.get(
            "evaluation_split"
        )

        if (
            report_split is not None
            and report_split != self.split
        ):
            raise ValueError(
                "RAG evaluation report split does not "
                f"match requested split '{self.split}'. "
                f"Report split: '{report_split}'."
            )

        return report

    def _load_successful_rag_cases(
        self,
    ) -> list[dict[str, Any]]:
        """
        Extract successfully classified RAG cases.

        Only cases with:
        - feedback_id
        - gold_category
        - predicted_category
        - status == success

        are used for the paired ablation.
        """

        report = self._load_rag_report()

        cases = report.get("cases")

        if not isinstance(cases, list):
            raise ValueError(
                "RAG evaluation report does not contain "
                "a valid 'cases' list."
            )

        successful_cases: list[
            dict[str, Any]
        ] = []

        for case in cases:
            if not isinstance(case, dict):
                continue

            if (
                case.get("feedback_id")
                and case.get("gold_category")
                and case.get("predicted_category")
                and case.get("status") == "success"
            ):
                successful_cases.append(
                    case
                )

        if not successful_cases:
            raise ValueError(
                "The RAG evaluation report contains "
                "no successfully classified cases."
            )

        return successful_cases

    def _load_feedback_text(
        self,
        relative_path: str,
    ) -> str:
        """
        Extract feedback text using the existing document service.
        """

        pdf_path = self._resolve_feedback_path(
            relative_path
        )

        documents = (
            self.document_service.load_pdf(
                pdf_path
            )
        )

        if isinstance(
            documents,
            list,
        ):
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
                cleaned = str(
                    page_content
                ).strip()

                if cleaned:
                    text_parts.append(
                        cleaned
                    )

        feedback = "\n\n".join(
            text_parts
        ).strip()

        if not feedback:
            raise ValueError(
                "No usable feedback text extracted "
                f"from {relative_path}"
            )

        return feedback

    def _resolve_feedback_path(
        self,
        relative_path: str,
    ) -> Path:
        """
        Resolve a manifest relative path to the actual PDF.

        Several common project locations are checked so that the
        evaluation service remains robust to the dataset layout.
        """

        raw_path = Path(relative_path)

        if raw_path.is_absolute():
            return raw_path

        candidates = [
            raw_path,
            self.manifest_path.parent
            / raw_path,
            self.manifest_path.parent.parent
            / raw_path,
            self.manifest_path.parent.parent.parent
            / raw_path,
        ]

        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        candidates.extend(
            [
                project_root
                / raw_path,
                project_root
                / "Sample_Data"
                / raw_path,
                project_root
                / "Sample_Data"
                / "feedback"
                / raw_path,
                project_root
                / "Sample_Data"
                / "evaluation"
                / raw_path,
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Let the document service raise the useful file-related
        # error if the path really does not exist.
        return (
            project_root / raw_path
        )

    def _classify_without_rag(
        self,
        feedback: str,
    ) -> FeedbackClassification:
        """
        Classify feedback using the same classification policy as RAG,
        but without retrieved examples.
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

Do not use external information.
Do not invent facts that are not present in the feedback.
Do not assume a particular industry unless the feedback itself provides
that context.
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
        """
        Safely divide two numbers.
        """

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
        Calculate classification metrics without requiring an external
        metric library.
        """

        if len(gold_labels) != len(
            predicted_labels
        ):
            raise ValueError(
                "Gold and predicted label lists "
                "must have equal length."
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

        per_class: dict[
            str,
            dict[str, float],
        ] = {}

        confusion_matrix = {
            actual: {
                predicted: 0
                for predicted in CATEGORIES
            }
            for actual in CATEGORIES
        }

        for gold, predicted in zip(
            gold_labels,
            predicted_labels,
        ):
            if (
                gold in CATEGORIES
                and predicted in CATEGORIES
            ):
                confusion_matrix[
                    gold
                ][predicted] += 1

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
                true_positive
                + false_positive,
            )

            recall = cls._safe_divide(
                true_positive,
                true_positive
                + false_negative,
            )

            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            per_class[category] = {
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        macro_precision = (
            sum(
                metrics["precision"]
                for metrics in per_class.values()
            )
            / len(CATEGORIES)
        )

        macro_recall = (
            sum(
                metrics["recall"]
                for metrics in per_class.values()
            )
            / len(CATEGORIES)
        )

        macro_f1 = (
            sum(
                metrics["f1"]
                for metrics in per_class.values()
            )
            / len(CATEGORIES)
        )

        weighted_f1 = cls._safe_divide(
            sum(
                metrics["f1"]
                * metrics["support"]
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
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class": per_class,
            "confusion_matrix": confusion_matrix,
        }

    @staticmethod
    def _build_paired_comparison(
        *,
        rag_metrics: dict[str, Any],
        no_rag_metrics: dict[str, Any],
        rag_better_cases: int,
        no_rag_better_cases: int,
        both_correct_cases: int,
        both_wrong_cases: int,
        no_rag_failed_cases: int,
    ) -> dict[str, Any]:
        """
        Build the API-facing paired comparison section.
        """

        return {
            "accuracy_difference": (
                rag_metrics["accuracy"]
                - no_rag_metrics["accuracy"]
            ),
            "macro_precision_difference": (
                rag_metrics["macro_precision"]
                - no_rag_metrics[
                    "macro_precision"
                ]
            ),
            "macro_recall_difference": (
                rag_metrics["macro_recall"]
                - no_rag_metrics[
                    "macro_recall"
                ]
            ),
            "macro_f1_difference": (
                rag_metrics["macro_f1"]
                - no_rag_metrics["macro_f1"]
            ),
            "weighted_f1_difference": (
                rag_metrics["weighted_f1"]
                - no_rag_metrics["weighted_f1"]
            ),
            "rag_better_cases": (
                rag_better_cases
            ),
            "no_rag_better_cases": (
                no_rag_better_cases
            ),
            "both_correct_cases": (
                both_correct_cases
            ),
            "both_wrong_cases": (
                both_wrong_cases
            ),
            "no_rag_failed_cases": (
                no_rag_failed_cases
            ),
        }

    def evaluate(
        self,
    ) -> dict[str, Any]:
        """
        Run the no-RAG baseline against the successfully classified
        RAG cases for the configured split.
        """

        split_manifest = (
            self._load_split_manifest()
        )

        rag_cases = (
            self._load_successful_rag_cases()
        )

        results: list[
            dict[str, Any]
        ] = []

        no_rag_failed_cases = 0

        total_cases = len(rag_cases)

        for index, rag_case in enumerate(
            rag_cases,
            start=1,
        ):
            feedback_id = rag_case[
                "feedback_id"
            ]

            manifest_row = split_manifest.get(
                feedback_id
            )

            if manifest_row is None:
                raise ValueError(
                    f"Feedback ID {feedback_id} from "
                    f"the RAG report is not present "
                    f"in the {self.split} manifest."
                )

            relative_path = (
                manifest_row["relative_path"]
            )

            gold_category = rag_case[
                "gold_category"
            ]

            rag_prediction = rag_case[
                "predicted_category"
            ]

            print(
                f"[{index}/{total_cases}] "
                f"{feedback_id}"
            )

            feedback = (
                self._load_feedback_text(
                    relative_path
                )
            )

            try:
                no_rag_classification = (
                    self._classify_without_rag(
                        feedback
                    )
                )
            except Exception as exc:
                no_rag_failed_cases += 1

                results.append(
                    {
                        "feedback_id": feedback_id,
                        "relative_path": relative_path,
                        "gold_category": gold_category,
                        "rag_prediction": (
                            rag_prediction
                        ),
                        "rag_correct": (
                            rag_prediction
                            == gold_category
                        ),
                        "rag_confidence": (
                            rag_case.get(
                                "confidence"
                            )
                        ),
                        "no_rag_prediction": None,
                        "no_rag_correct": None,
                        "no_rag_confidence": None,
                        "no_rag_explanation": None,
                        "no_rag_status": "error",
                        "no_rag_error": str(exc),
                    }
                )

                continue

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
                    "rag_prediction": (
                        rag_prediction
                    ),
                    "rag_correct": rag_correct,
                    "rag_confidence": (
                        rag_case.get(
                            "confidence"
                        )
                    ),
                    "no_rag_prediction": (
                        no_rag_prediction
                    ),
                    "no_rag_correct": (
                        no_rag_correct
                    ),
                    "no_rag_confidence": (
                        no_rag_classification.confidence
                    ),
                    "no_rag_explanation": (
                        no_rag_classification.explanation
                    ),
                    "no_rag_status": "success",
                    "no_rag_error": None,
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

        successful_pairs = [
            result
            for result in results
            if result["no_rag_status"]
            == "success"
        ]

        if not successful_pairs:
            raise ValueError(
                "No successful RAG vs No-RAG "
                "comparisons were produced."
            )

        gold_labels = [
            result["gold_category"]
            for result in successful_pairs
        ]

        rag_predictions = [
            result["rag_prediction"]
            for result in successful_pairs
        ]

        no_rag_predictions = [
            result["no_rag_prediction"]
            for result in successful_pairs
        ]

        rag_metrics = (
            self._classification_metrics(
                gold_labels,
                rag_predictions,
            )
        )

        no_rag_metrics = (
            self._classification_metrics(
                gold_labels,
                no_rag_predictions,
            )
        )

        rag_better_cases = sum(
            1
            for result in successful_pairs
            if result["rag_correct"]
            and not result["no_rag_correct"]
        )

        no_rag_better_cases = sum(
            1
            for result in successful_pairs
            if result["no_rag_correct"]
            and not result["rag_correct"]
        )

        both_correct_cases = sum(
            1
            for result in successful_pairs
            if result["rag_correct"]
            and result["no_rag_correct"]
        )

        both_wrong_cases = sum(
            1
            for result in successful_pairs
            if not result["rag_correct"]
            and not result["no_rag_correct"]
        )

        paired_comparison = (
            self._build_paired_comparison(
                rag_metrics=rag_metrics,
                no_rag_metrics=no_rag_metrics,
                rag_better_cases=(
                    rag_better_cases
                ),
                no_rag_better_cases=(
                    no_rag_better_cases
                ),
                both_correct_cases=(
                    both_correct_cases
                ),
                both_wrong_cases=(
                    both_wrong_cases
                ),
                no_rag_failed_cases=(
                    no_rag_failed_cases
                ),
            )
        )

        report = {
            "evaluation_timestamp": (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
            "evaluation_split": self.split,
            "experiment": (
                "RAG vs No-RAG Ablation"
            ),
            "manifest_path": str(
                self.manifest_path
            ),
            "source_rag_report": str(
                self.rag_report_path
            ),
            "total_evaluation_records": (
                len(rag_cases)
            ),
            "paired_evaluation_cases": (
                len(successful_pairs)
            ),
            "no_rag_failed_cases": (
                no_rag_failed_cases
            ),
            "evaluation_cases": (
                len(successful_pairs)
            ),
            "rag_metrics": rag_metrics,
            "no_rag_metrics": no_rag_metrics,
            "paired_comparison": (
                paired_comparison
            ),

            # Backward-compatible top-level
            # values used by existing tests
            # and reports.
            "accuracy_difference": (
                paired_comparison[
                    "accuracy_difference"
                ]
            ),
            "macro_precision_difference": (
                paired_comparison[
                    "macro_precision_difference"
                ]
            ),
            "macro_recall_difference": (
                paired_comparison[
                    "macro_recall_difference"
                ]
            ),
            "macro_f1_difference": (
                paired_comparison[
                    "macro_f1_difference"
                ]
            ),
            "weighted_f1_difference": (
                paired_comparison[
                    "weighted_f1_difference"
                ]
            ),
            "rag_better_cases": (
                rag_better_cases
            ),
            "no_rag_better_cases": (
                no_rag_better_cases
            ),
            "both_correct_cases": (
                both_correct_cases
            ),
            "both_wrong_cases": (
                both_wrong_cases
            ),

            "results": successful_pairs,
        }

        return report

    def save_report(
        self,
        report: dict[str, Any],
    ) -> Path:
        """
        Save the ablation report as JSON.
        """

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            self.output_directory
            / (
                f"rag_ablation_"
                f"{self.split}.json"
            )
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return output_path

    def save_csv(
        self,
        report: dict[str, Any],
    ) -> Path:
        """
        Save case-level ablation results as CSV.
        """

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            self.output_directory
            / (
                f"rag_ablation_"
                f"{self.split}.csv"
            )
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
            "no_rag_status",
            "no_rag_error",
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
                extrasaction="ignore",
            )

            writer.writeheader()

            for result in report[
                "results"
            ]:
                writer.writerow(
                    {
                        field: result.get(
                            field
                        )
                        for field in fieldnames
                    }
                )

        return output_path

    def run_and_save(
        self,
    ) -> dict[str, Any]:
        """
        Run the ablation and save JSON and CSV reports.
        """

        report = self.evaluate()

        json_path = self.save_report(
            report
        )

        csv_path = self.save_csv(
            report
        )

        paired_cases = report[
            "paired_evaluation_cases"
        ]

        successful_comparisons = report[
            "paired_evaluation_cases"
        ]

        print(
            "\n"
            + "=" * 60
        )
        print(
            "RAG VS NO-RAG EVALUATION"
        )
        print(
            "=" * 60
        )

        print(
            f"Evaluation split       : "
            f"{report['evaluation_split']}"
        )

        print(
            f"Paired evaluation cases: "
            f"{paired_cases}"
        )

        print(
            f"Successful comparisons : "
            f"{successful_comparisons}"
        )

        print(
            f"No-RAG failed cases    : "
            f"{report['no_rag_failed_cases']}"
        )

        print()

        print(
            f"RAG accuracy           : "
            f"{report['rag_metrics']['accuracy']:.4f}"
        )

        print(
            f"No-RAG accuracy        : "
            f"{report['no_rag_metrics']['accuracy']:.4f}"
        )

        print(
            f"Accuracy delta         : "
            f"{report['accuracy_difference']:+.4f}"
        )

        print(
            f"RAG macro precision    : "
            f"{report['rag_metrics']['macro_precision']:.4f}"
        )

        print(
            f"No-RAG macro precision : "
            f"{report['no_rag_metrics']['macro_precision']:.4f}"
        )

        print(
            f"Precision delta        : "
            f"{report['macro_precision_difference']:+.4f}"
        )

        print(
            f"RAG macro recall       : "
            f"{report['rag_metrics']['macro_recall']:.4f}"
        )

        print(
            f"No-RAG macro recall    : "
            f"{report['no_rag_metrics']['macro_recall']:.4f}"
        )

        print(
            f"Recall delta           : "
            f"{report['macro_recall_difference']:+.4f}"
        )

        print(
            f"RAG macro F1           : "
            f"{report['rag_metrics']['macro_f1']:.4f}"
        )

        print(
            f"No-RAG macro F1        : "
            f"{report['no_rag_metrics']['macro_f1']:.4f}"
        )

        print(
            f"Macro F1 delta         : "
            f"{report['macro_f1_difference']:+.4f}"
        )

        print(
            f"RAG weighted F1        : "
            f"{report['rag_metrics']['weighted_f1']:.4f}"
        )

        print(
            f"No-RAG weighted F1     : "
            f"{report['no_rag_metrics']['weighted_f1']:.4f}"
        )

        print(
            f"Weighted F1 delta      : "
            f"{report['weighted_f1_difference']:+.4f}"
        )

        print()

        print(
            f"RAG better cases       : "
            f"{report['rag_better_cases']}"
        )

        print(
            f"No-RAG better cases    : "
            f"{report['no_rag_better_cases']}"
        )

        print(
            f"Both correct            : "
            f"{report['both_correct_cases']}"
        )

        print(
            f"Both wrong              : "
            f"{report['both_wrong_cases']}"
        )

        print()

        print(
            f"JSON report             : "
            f"{json_path}"
        )

        print(
            f"CSV report              : "
            f"{csv_path}"
        )

        return report


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run a RAG vs No-RAG ablation "
            "for customer feedback classification."
        )
    )

    parser.add_argument(
        "--split",
        choices=sorted(VALID_SPLITS),
        default="development",
        help=(
            "Evaluation split to use. "
            "Default: development"
        ),
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=(
            "Path to dataset_manifest.csv."
        ),
    )

    parser.add_argument(
        "--rag-report-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit RAG evaluation "
            "report path. If omitted, the latest "
            "report for the selected split is used."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
        help=(
            "Directory where JSON and CSV reports "
            "will be saved."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Run the ablation from the command line.
    """

    args = parse_args()

    service = RAGAblationService(
        manifest_path=args.manifest_path,
        split=args.split,
        rag_report_path=args.rag_report_path,
        output_directory=args.output_directory,
    )

    service.run_and_save()


if __name__ == "__main__":
    main()