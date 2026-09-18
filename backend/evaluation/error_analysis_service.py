from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


CATEGORIES = [
    "Excellent",
    "Good",
    "Need Improvements",
    "Poor",
]


class ErrorAnalysisService:
    """Analyze an existing Stage 11 RAG evaluation report."""

    def __init__(self, report_path: str | Path) -> None:
        self.report_path = Path(report_path)

        if not self.report_path.exists():
            raise FileNotFoundError(
                f"Evaluation report not found: {self.report_path}"
            )

    def load_report(self) -> dict[str, Any]:
        """Load the JSON evaluation report."""

        with self.report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)

        if not isinstance(report, dict):
            raise ValueError("Evaluation report must contain a JSON object.")

        return report

    def get_cases(self) -> list[dict[str, Any]]:
        """Extract evaluated cases from the report."""

        report = self.load_report()

        cases = report.get("cases")

        if cases is None:
            cases = report.get("results")

        if not isinstance(cases, list):
            raise ValueError(
                "Evaluation report does not contain a 'cases' or 'results' list."
            )

        return cases

    @staticmethod
    def _gold_category(case: dict[str, Any]) -> str | None:
        """Return the manually labelled category from an evaluation case."""

        return case.get("gold_category")

    @staticmethod
    def _predicted_category(case: dict[str, Any]) -> str | None:
        """Return the model-predicted category from an evaluation case."""

        return case.get("predicted_category")

    def get_incorrect_cases(self) -> list[dict[str, Any]]:
        """Return all evaluation cases where prediction differs from gold label."""

        cases = self.get_cases()

        return [
            case
            for case in cases
            if self._gold_category(case) != self._predicted_category(case)
        ]

    def confusion_matrix(self) -> dict[str, dict[str, int]]:
        """
        Build a four-category confusion matrix.

        Rows represent the gold/manual category.
        Columns represent the predicted category.
        """

        matrix = {
            actual: {predicted: 0 for predicted in CATEGORIES}
            for actual in CATEGORIES
        }

        for case in self.get_cases():
            actual = self._gold_category(case)
            predicted = self._predicted_category(case)

            if actual not in CATEGORIES or predicted not in CATEGORIES:
                continue

            matrix[actual][predicted] += 1

        return matrix

    def per_category_metrics(self) -> dict[str, dict[str, float]]:
        """Calculate precision, recall, and F1 for every category."""

        cases = self.get_cases()
        metrics: dict[str, dict[str, float]] = {}

        for category in CATEGORIES:
            true_positive = 0
            false_positive = 0
            false_negative = 0

            for case in cases:
                actual = self._gold_category(case)
                predicted = self._predicted_category(case)

                if actual == category and predicted == category:
                    true_positive += 1

                elif actual != category and predicted == category:
                    false_positive += 1

                elif actual == category and predicted != category:
                    false_negative += 1

            precision_denominator = true_positive + false_positive
            recall_denominator = true_positive + false_negative

            precision = (
                true_positive / precision_denominator
                if precision_denominator
                else 0.0
            )

            recall = (
                true_positive / recall_denominator
                if recall_denominator
                else 0.0
            )

            f1_denominator = precision + recall

            f1 = (
                2 * precision * recall / f1_denominator
                if f1_denominator
                else 0.0
            )

            metrics[category] = {
                "support": sum(
                    1
                    for case in cases
                    if self._gold_category(case) == category
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        return metrics

    def error_patterns(self) -> dict[str, int]:
        """
        Count common misclassification patterns.

        Example:
        'Good -> Need Improvements': 3
        """

        patterns = Counter()

        for case in self.get_incorrect_cases():
            gold = self._gold_category(case)
            predicted = self._predicted_category(case)

            if gold and predicted:
                patterns[f"{gold} -> {predicted}"] += 1

        return dict(patterns.most_common())

    def retrieval_diagnostics(self) -> dict[str, Any]:
        """Summarize retrieval-related information from evaluation cases."""

        cases = self.get_cases()

        total = len(cases)

        reference_only_count = sum(
            1
            for case in cases
            if case.get("reference_only", False)
        )

        gold_category_retrieved_count = sum(
            1
            for case in cases
            if case.get("gold_category_retrieved", False)
        )

        retrieved_counts = [
            int(case.get("retrieved_count", 0))
            for case in cases
        ]

        return {
            "total_cases": total,
            "reference_only_cases": reference_only_count,
            "reference_only_rate": (
                reference_only_count / total if total else 0.0
            ),
            "gold_category_retrieved_cases": gold_category_retrieved_count,
            "gold_category_retrieval_rate": (
                gold_category_retrieved_count / total if total else 0.0
            ),
            "average_retrieved_count": (
                sum(retrieved_counts) / len(retrieved_counts)
                if retrieved_counts
                else 0.0
            ),
        }

    def generate_report(self) -> dict[str, Any]:
        """Generate the complete Stage 12 error-analysis report."""

        cases = self.get_cases()

        correct_cases = sum(
            1
            for case in cases
            if case.get("correct") is True
        )

        total_cases = len(cases)

        accuracy = (
            correct_cases / total_cases
            if total_cases
            else 0.0
        )

        return {
            "source_report": str(self.report_path),
            "total_cases": total_cases,
            "correct_cases": correct_cases,
            "incorrect_cases": total_cases - correct_cases,
            "accuracy": accuracy,
            "confusion_matrix": self.confusion_matrix(),
            "per_category_metrics": self.per_category_metrics(),
            "error_patterns": self.error_patterns(),
            "retrieval_diagnostics": self.retrieval_diagnostics(),
            "incorrect_case_details": self.get_incorrect_cases(),
        }


def find_latest_evaluation_report(
    results_directory: str | Path = "backend/evaluation/results",
) -> Path:
    """Find the most recent Stage 11 JSON evaluation report."""

    directory = Path(results_directory)

    reports = sorted(
        directory.glob("rag_evaluation_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not reports:
        raise FileNotFoundError(
            f"No RAG evaluation reports found in {directory}"
        )

    return reports[0]


def main() -> None:
    """Run Stage 12.1 error analysis on the latest Stage 11 report."""

    report_path = find_latest_evaluation_report()

    service = ErrorAnalysisService(report_path)
    report = service.generate_report()

    output_directory = Path("backend/evaluation/results")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory / "stage12_error_analysis.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("=" * 60)
    print("STAGE 12.1 - ERROR ANALYSIS")
    print("=" * 60)

    print(f"Source report: {report_path}")
    print(f"Total cases: {report['total_cases']}")
    print(f"Correct cases: {report['correct_cases']}")
    print(f"Incorrect cases: {report['incorrect_cases']}")
    print(f"Accuracy: {report['accuracy']:.4f}")

    print("\nError Patterns:")
    for pattern, count in report["error_patterns"].items():
        print(f"  {pattern}: {count}")

    print("\nConfusion Matrix:")
    matrix = report["confusion_matrix"]

    print(
        f"{'Actual':<20}"
        f"{'Excellent':<12}"
        f"{'Good':<12}"
        f"{'Need Imp.':<12}"
        f"{'Poor':<12}"
    )

    for actual in CATEGORIES:
        print(
            f"{actual:<20}"
            f"{matrix[actual]['Excellent']:<12}"
            f"{matrix[actual]['Good']:<12}"
            f"{matrix[actual]['Need Improvements']:<12}"
            f"{matrix[actual]['Poor']:<12}"
        )

    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()