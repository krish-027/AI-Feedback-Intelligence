from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvaluationReportNotFoundError(Exception):
    """Raised when the requested evaluation report does not exist."""


class EvaluationReportInvalidError(Exception):
    """Raised when an evaluation report exists but is malformed."""


class EvaluationReportService:
    """
    Loads and validates persisted RAG evaluation and RAG-vs-No-RAG reports.

    This service only reads saved reports. It does not execute evaluation
    or make Gemini calls.
    """

    VALID_SPLITS = {"development", "final_benchmark"}

    def __init__(self, results_dir: Path | None = None):
        project_root = Path(__file__).resolve().parents[2]

        self.results_dir = (
            results_dir
            if results_dir is not None
            else project_root / "backend" / "evaluation" / "results"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_rag_evaluation(self, split: str) -> dict[str, Any]:
        """Return the latest saved RAG evaluation report for a split."""
        split = self._validate_split(split)

        report_path = self._find_latest_rag_report(split)

        if report_path is None:
            raise EvaluationReportNotFoundError(
                f"No RAG evaluation report found for split '{split}'."
            )

        report = self._load_json(report_path)

        self._validate_rag_report(report, split)

        return report

    def get_rag_ablation(self, split: str) -> dict[str, Any]:
        """Return the saved RAG-vs-No-RAG report for a split."""
        split = self._validate_split(split)

        report_path = self.results_dir / f"rag_ablation_{split}.json"

        if not report_path.exists():
            raise EvaluationReportNotFoundError(
                f"No RAG-vs-No-RAG report found for split '{split}'."
            )

        report = self._load_json(report_path)

        self._validate_ablation_report(report, split)

        return report

    def get_summary(self) -> dict[str, Any]:
        """
        Return all available saved evaluation reports.

        Missing reports are represented as None rather than causing the
        entire summary request to fail.
        """
        return {
            "development": {
                "rag_evaluation": self._safe_get_rag("development"),
                "rag_vs_no_rag": self._safe_get_ablation("development"),
            },
            "final_benchmark": {
                "rag_evaluation": self._safe_get_rag("final_benchmark"),
                "rag_vs_no_rag": self._safe_get_ablation("final_benchmark"),
            },
        }

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def _find_latest_rag_report(self, split: str) -> Path | None:
        """
        Find the newest persisted RAG evaluation JSON for the requested split.

        Evaluation files are timestamped:
        rag_evaluation_<split>_<timestamp>.json
        """
        pattern = f"rag_evaluation_{split}_*.json"

        reports = [
            path
            for path in self.results_dir.glob(pattern)
            if path.is_file()
        ]

        if not reports:
            return None

        return max(reports, key=lambda path: path.stat().st_mtime)

    def _load_json(self, path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationReportInvalidError(
                f"Unable to read evaluation report '{path.name}'."
            ) from exc

        if not isinstance(data, dict):
            raise EvaluationReportInvalidError(
                f"Evaluation report '{path.name}' must contain a JSON object."
            )

        return data

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_split(self, split: str) -> str:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid evaluation split '{split}'. "
                f"Expected one of: {sorted(self.VALID_SPLITS)}."
            )

        return split

    def _validate_rag_report(
        self,
        report: dict[str, Any],
        split: str,
    ) -> None:
        if report.get("evaluation_split") != split:
            raise EvaluationReportInvalidError(
                f"RAG evaluation report contains split "
                f"'{report.get('evaluation_split')}', expected '{split}'."
            )

        required_sections = {
            "classification_metrics",
            "retrieval_metrics",
            "cases",
        }

        missing = required_sections - report.keys()

        if missing:
            raise EvaluationReportInvalidError(
                f"RAG evaluation report is missing sections: "
                f"{sorted(missing)}."
            )

        self._validate_metrics(
            report["classification_metrics"],
            "classification_metrics",
        )

        self._validate_metrics(
            report["retrieval_metrics"],
            "retrieval_metrics",
        )

    def _validate_ablation_report(
        self,
        report: dict[str, Any],
        split: str,
    ) -> None:
        if report.get("evaluation_split") != split:
            raise EvaluationReportInvalidError(
                f"RAG-vs-No-RAG report contains split "
                f"'{report.get('evaluation_split')}', expected '{split}'."
            )

        required_sections = {
            "rag_metrics",
            "no_rag_metrics",
            "paired_comparison",
        }

        missing = required_sections - report.keys()

        if missing:
            raise EvaluationReportInvalidError(
                f"RAG-vs-No-RAG report is missing sections: "
                f"{sorted(missing)}."
            )

        self._validate_metrics(
            report["rag_metrics"],
            "rag_metrics",
        )

        self._validate_metrics(
            report["no_rag_metrics"],
            "no_rag_metrics",
        )

    @staticmethod
    def _validate_metrics(
        metrics: Any,
        section_name: str,
    ) -> None:
        if not isinstance(metrics, dict):
            raise EvaluationReportInvalidError(
                f"'{section_name}' must be a JSON object."
            )

        bounded_metrics = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "retrieval_success_rate",
            "reference_only_rate",
            "gold_category_retrieval_rate",
        }

        for key in bounded_metrics:
            if key not in metrics:
                continue

            value = metrics[key]

            if not isinstance(value, (int, float)):
                raise EvaluationReportInvalidError(
                    f"Metric '{key}' in '{section_name}' must be numeric."
                )

            if not 0.0 <= float(value) <= 1.0:
                raise EvaluationReportInvalidError(
                    f"Metric '{key}' in '{section_name}' must be "
                    f"between 0 and 1."
                )

    # ------------------------------------------------------------------
    # Safe summary helpers
    # ------------------------------------------------------------------

    def _safe_get_rag(self, split: str) -> dict[str, Any] | None:
        try:
            return self.get_rag_evaluation(split)
        except EvaluationReportNotFoundError:
            return None

    def _safe_get_ablation(self, split: str) -> dict[str, Any] | None:
        try:
            return self.get_rag_ablation(split)
        except EvaluationReportNotFoundError:
            return None