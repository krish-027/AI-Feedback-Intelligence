
from __future__ import annotations

import csv
import importlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from backend.services.document_service import (
    get_document_service,
)
from backend.services.rag_service import (
    RAGService,
    get_rag_service,
)


VALID_CATEGORIES = (
    "Excellent",
    "Good",
    "Need Improvements",
    "Poor",
)

VALID_EVALUATION_SPLITS = (
    "development",
    "final_benchmark",
)

DEFAULT_MANIFEST_PATH = Path(
    "Sample_Data/evaluation/dataset_manifest.csv"
)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "backend/evaluation/results"
)

DEFAULT_EVALUATION_SPLIT = "development"


@dataclass
class EvaluationCaseResult:
    feedback_id: str
    relative_path: str
    gold_category: str

    predicted_category: str | None
    confidence: float | None
    explanation: str | None
    flagged_keywords: list[str]

    retrieved_feedback_ids: list[str]
    retrieved_categories: list[str]

    gold_category_retrieved: bool
    retrieved_only_reference: bool

    correct: bool | None
    status: str
    error: str | None


class RAGEvaluationService:
    def __init__(
        self,
        manifest_path: Path | str | None = None,
        output_directory: Path | str | None = None,
        rag_service: RAGService | None = None,
        document_service: Any | None = None,
    ) -> None:
        self._load_environment()

        self.manifest_path = Path(
            manifest_path
            if manifest_path is not None
            else DEFAULT_MANIFEST_PATH
        )

        self.output_directory = Path(
            output_directory
            if output_directory is not None
            else DEFAULT_OUTPUT_DIRECTORY
        )

        self.rag_service = (
            rag_service
            if rag_service is not None
            else get_rag_service()
        )

        self.document_service = (
            document_service
            if document_service is not None
            else get_document_service()
        )

    def evaluate(
        self,
        split: str = DEFAULT_EVALUATION_SPLIT,
        include_ragas: bool = False,
        ragas_sample_size: int | None = None,
        save_report: bool = True,
    ) -> dict[str, Any]:
        self._validate_evaluation_split(split)

        records = self._load_evaluation_records(
            split=split
        )

        if not records:
            raise ValueError(
                f"No records were found for evaluation split: {split}"
            )

        case_results: list[EvaluationCaseResult] = []

        for index, record in enumerate(
            records,
            start=1,
        ):
            print(
                f"Evaluating {index}/{len(records)}: "
                f"{record['relative_path']}"
            )

            result = self._evaluate_single_case(
                record
            )

            case_results.append(result)

        classification_metrics = (
            self._calculate_classification_metrics(
                case_results
            )
        )

        retrieval_metrics = (
            self._calculate_retrieval_metrics(
                case_results
            )
        )

        report: dict[str, Any] = {
            "evaluation_timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "manifest_path": str(
                self.manifest_path
            ),
            "evaluation_split": split,
            "total_evaluation_records": len(
                records
            ),
            "reference_corpus_policy": (
                "Only reference split documents may be retrieved."
            ),
            "classification_metrics": classification_metrics,
            "retrieval_metrics": retrieval_metrics,
            "cases": [
                asdict(case)
                for case in case_results
            ],
        }

        if include_ragas:
            ragas_results = self._run_ragas_evaluation(
                case_results=case_results,
                sample_size=ragas_sample_size,
            )

            report["ragas"] = ragas_results

        if save_report:
            self._save_report(
                report
            )

        self._print_summary(
            report
        )

        return report

    def _validate_evaluation_split(
        self,
        split: str,
    ) -> None:
        if split not in VALID_EVALUATION_SPLITS:
            raise ValueError(
                "Invalid evaluation split. "
                f"Expected one of {VALID_EVALUATION_SPLITS}, "
                f"got '{split}'."
            )

    def _load_evaluation_records(
        self,
        split: str = DEFAULT_EVALUATION_SPLIT,
    ) -> list[dict[str, str]]:
        self._validate_evaluation_split(split)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                "Dataset manifest not found at: "
                f"{self.manifest_path}"
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError(
                    "Dataset manifest has no header."
                )

            required_columns = {
                "relative_path",
                "split",
                "category",
            }

            missing_columns = (
                required_columns
                - set(reader.fieldnames)
            )

            if missing_columns:
                raise ValueError(
                    "Dataset manifest is missing required columns: "
                    f"{sorted(missing_columns)}"
                )

            records = list(reader)

        selected_records = [
            record
            for record in records
            if record["split"].strip().lower()
            == split
        ]

        for record in selected_records:
            category = record["category"].strip()

            if category not in VALID_CATEGORIES:
                raise ValueError(
                    "Invalid gold category in manifest: "
                    f"{category}"
                )

            record["category"] = category

        return selected_records

    def _evaluate_single_case(
        self,
        record: dict[str, str],
    ) -> EvaluationCaseResult:
        relative_path = record[
            "relative_path"
        ]

        gold_category = record[
            "category"
        ]

        feedback_id = (
            record.get("feedback_id")
            or self._derive_feedback_id(
                relative_path
            )
        )

        try:
            feedback_text = (
                self._load_feedback_text(
                    relative_path
                )
            )

            rag_result = (
                self.rag_service.classify_feedback(
                    feedback_text,
                    include_retrieved_examples=True,
                )
            )

            classification = rag_result[
                "classification"
            ]

            retrieved_examples = rag_result[
                "retrieved_examples"
            ]

            retrieved_feedback_ids = [
                str(
                    example["feedback_id"]
                )
                for example in retrieved_examples
            ]

            retrieved_categories = [
                str(
                    example["category"]
                )
                for example in retrieved_examples
            ]

            gold_category_retrieved = (
                gold_category
                in retrieved_categories
            )

            retrieved_only_reference = (
                len(retrieved_examples) > 0
                and all(
                    example.get("split")
                    == "reference"
                    for example in retrieved_examples
                )
            )

            predicted_category = (
                classification.category
            )

            confidence = float(
                classification.confidence
            )

            explanation = (
                classification.explanation
            )

            flagged_keywords = list(
                classification.flagged_keywords
            )

            return EvaluationCaseResult(
                feedback_id=feedback_id,
                relative_path=relative_path,
                gold_category=gold_category,
                predicted_category=predicted_category,
                confidence=confidence,
                explanation=explanation,
                flagged_keywords=flagged_keywords,
                retrieved_feedback_ids=(
                    retrieved_feedback_ids
                ),
                retrieved_categories=(
                    retrieved_categories
                ),
                gold_category_retrieved=(
                    gold_category_retrieved
                ),
                retrieved_only_reference=(
                    retrieved_only_reference
                ),
                correct=(
                    predicted_category
                    == gold_category
                ),
                status="success",
                error=None,
            )

        except Exception as exc:
            return EvaluationCaseResult(
                feedback_id=feedback_id,
                relative_path=relative_path,
                gold_category=gold_category,
                predicted_category=None,
                confidence=None,
                explanation=None,
                flagged_keywords=[],
                retrieved_feedback_ids=[],
                retrieved_categories=[],
                gold_category_retrieved=False,
                retrieved_only_reference=False,
                correct=None,
                status="error",
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    def _load_feedback_text(
        self,
        relative_path: str,
    ) -> str:
        pdf_path = Path(
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
            document_list = [
                documents
            ]

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

        feedback_text = "\n\n".join(
            part
            for part in text_parts
            if part
        )

        if not feedback_text.strip():
            raise ValueError(
                "No usable text extracted from evaluation PDF: "
                f"{pdf_path}"
            )

        return feedback_text

    def _calculate_classification_metrics(
        self,
        cases: list[EvaluationCaseResult],
    ) -> dict[str, Any]:
        successful_cases = [
            case
            for case in cases
            if case.status == "success"
            and case.predicted_category is not None
        ]

        total = len(
            successful_cases
        )

        if total == 0:
            return {
                "evaluated_cases": 0,
                "accuracy": None,
                "macro_precision": None,
                "macro_recall": None,
                "macro_f1": None,
                "weighted_f1": None,
                "per_class": {},
                "confusion_matrix": {},
            }

        correct = sum(
            1
            for case in successful_cases
            if case.predicted_category
            == case.gold_category
        )

        accuracy = (
            correct / total
        )

        per_class: dict[
            str,
            dict[str, float | int],
        ] = {}

        for category in VALID_CATEGORIES:
            true_positive = sum(
                1
                for case in successful_cases
                if case.gold_category == category
                and case.predicted_category
                == category
            )

            false_positive = sum(
                1
                for case in successful_cases
                if case.gold_category != category
                and case.predicted_category
                == category
            )

            false_negative = sum(
                1
                for case in successful_cases
                if case.gold_category == category
                and case.predicted_category
                != category
            )

            support = sum(
                1
                for case in successful_cases
                if case.gold_category == category
            )

            precision = self._safe_divide(
                true_positive,
                true_positive + false_positive,
            )

            recall = self._safe_divide(
                true_positive,
                true_positive + false_negative,
            )

            f1 = self._safe_divide(
                2 * precision * recall,
                precision + recall,
            )

            per_class[category] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }

        macro_precision = (
            sum(
                float(
                    per_class[category][
                        "precision"
                    ]
                )
                for category in VALID_CATEGORIES
            )
            / len(VALID_CATEGORIES)
        )

        macro_recall = (
            sum(
                float(
                    per_class[category][
                        "recall"
                    ]
                )
                for category in VALID_CATEGORIES
            )
            / len(VALID_CATEGORIES)
        )

        macro_f1 = (
            sum(
                float(
                    per_class[category][
                        "f1"
                    ]
                )
                for category in VALID_CATEGORIES
            )
            / len(VALID_CATEGORIES)
        )

        weighted_f1_numerator = sum(
            float(
                per_class[category]["f1"]
            )
            * int(
                per_class[category]["support"]
            )
            for category in VALID_CATEGORIES
        )

        weighted_f1 = self._safe_divide(
            weighted_f1_numerator,
            total,
        )

        confusion_matrix: dict[
            str,
            dict[str, int],
        ] = {}

        for gold_category in VALID_CATEGORIES:
            confusion_matrix[
                gold_category
            ] = {}

            for predicted_category in VALID_CATEGORIES:
                confusion_matrix[
                    gold_category
                ][predicted_category] = sum(
                    1
                    for case in successful_cases
                    if case.gold_category
                    == gold_category
                    and case.predicted_category
                    == predicted_category
                )

        return {
            "evaluated_cases": total,
            "correct_cases": correct,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class": per_class,
            "confusion_matrix": confusion_matrix,
        }

    def _calculate_retrieval_metrics(
        self,
        cases: list[EvaluationCaseResult],
    ) -> dict[str, Any]:
        successful_cases = [
            case
            for case in cases
            if case.status == "success"
        ]

        total = len(
            successful_cases
        )

        if total == 0:
            return {
                "evaluated_cases": 0,
                "retrieval_success_rate": None,
                "reference_only_rate": None,
                "gold_category_retrieval_rate": None,
                "average_retrieved_count": None,
            }

        retrieval_success_rate = (
            sum(
                1
                for case in successful_cases
                if len(
                    case.retrieved_feedback_ids
                )
                > 0
            )
            / total
        )

        reference_only_rate = (
            sum(
                1
                for case in successful_cases
                if case.retrieved_only_reference
            )
            / total
        )

        gold_category_retrieval_rate = (
            sum(
                1
                for case in successful_cases
                if case.gold_category_retrieved
            )
            / total
        )

        average_retrieved_count = (
            sum(
                len(
                    case.retrieved_feedback_ids
                )
                for case in successful_cases
            )
            / total
        )

        return {
            "evaluated_cases": total,
            "retrieval_success_rate": (
                retrieval_success_rate
            ),
            "reference_only_rate": (
                reference_only_rate
            ),
            "gold_category_retrieval_rate": (
                gold_category_retrieval_rate
            ),
            "average_retrieved_count": (
                average_retrieved_count
            ),
        }

    def _run_ragas_evaluation(
        self,
        case_results: list[EvaluationCaseResult],
        sample_size: int | None,
    ) -> dict[str, Any]:
        try:
            from google import genai

            ragas = importlib.import_module(
                "ragas"
            )

            EvaluationDataset = (
                ragas.EvaluationDataset
            )

            llm_factory = (
                importlib.import_module(
                    "ragas.llms"
                ).llm_factory
            )

            Faithfulness = (
                importlib.import_module(
                    "ragas.metrics.collections"
                ).Faithfulness
            )

        except ImportError as exc:
            raise ImportError(
                "Ragas evaluation requires ragas and google-genai. "
                "Install them with: pip install ragas google-genai"
            ) from exc

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for Ragas evaluation."
            )

        successful_cases = [
            case
            for case in case_results
            if case.status == "success"
            and case.explanation
        ]

        if sample_size is not None:
            if sample_size <= 0:
                raise ValueError(
                    "ragas_sample_size must be greater than 0."
                )

            successful_cases = (
                successful_cases[:sample_size]
            )

        if not successful_cases:
            return {
                "evaluated_cases": 0,
                "faithfulness_average": None,
                "case_scores": [],
            }

        model_name = os.getenv(
            "RAGAS_GEMINI_MODEL",
            os.getenv(
                "GEMINI_MODEL",
                "gemini-2.5-flash",
            ),
        )

        client = genai.Client(
            api_key=api_key
        )

        evaluator_llm = llm_factory(
            model_name,
            provider="google",
            client=client,
        )

        faithfulness_metric = Faithfulness(
            llm=evaluator_llm
        )

        scores: list[dict[str, Any]] = []

        for case in successful_cases:
            contexts = self._build_ragas_context(
                case
            )

            try:
                result = faithfulness_metric.score(
                    user_input=case.explanation
                    or "",
                    response=case.explanation
                    or "",
                    retrieved_contexts=contexts,
                )

                score = self._extract_ragas_score(
                    result
                )

                scores.append(
                    {
                        "feedback_id": case.feedback_id,
                        "faithfulness": score,
                        "status": "success",
                        "error": None,
                    }
                )

            except Exception as exc:
                scores.append(
                    {
                        "feedback_id": case.feedback_id,
                        "faithfulness": None,
                        "status": "error",
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )

        valid_scores = [
            item["faithfulness"]
            for item in scores
            if item["faithfulness"] is not None
        ]

        average_score = (
            sum(valid_scores)
            / len(valid_scores)
            if valid_scores
            else None
        )

        return {
            "evaluated_cases": len(scores),
            "successful_cases": len(valid_scores),
            "faithfulness_average": average_score,
            "case_scores": scores,
        }

    def _build_ragas_context(
        self,
        case: EvaluationCaseResult,
    ) -> list[str]:
        contexts = [
            self._get_feedback_text_for_case(
                case
            )
        ]

        for feedback_id in case.retrieved_feedback_ids:
            contexts.append(
                f"Retrieved reference feedback ID: {feedback_id}"
            )

        contexts.extend(
            self._get_retrieved_feedback_texts(
                case
            )
        )

        return [
            context
            for context in contexts
            if context.strip()
        ]

    def _get_feedback_text_for_case(
        self,
        case: EvaluationCaseResult,
    ) -> str:
        return self._load_feedback_text(
            case.relative_path
        )

    def _get_retrieved_feedback_texts(
        self,
        case: EvaluationCaseResult,
    ) -> list[str]:
        feedback_text = self._load_feedback_text(
            case.relative_path
        )

        rag_result = (
            self.rag_service.classify_feedback(
                feedback_text,
                include_retrieved_examples=True,
            )
        )

        examples = rag_result.get(
            "retrieved_examples",
            [],
        )

        return [
            str(
                example.get(
                    "feedback",
                    "",
                )
            )
            for example in examples
            if example.get("feedback")
        ]

    def _extract_ragas_score(
        self,
        result: Any,
    ) -> float:
        if isinstance(
            result,
            (int, float),
        ):
            return float(
                result
            )

        value = getattr(
            result,
            "value",
            None,
        )

        if value is not None:
            return float(
                value
            )

        raise TypeError(
            "Unable to extract a numeric score from Ragas result: "
            f"{type(result).__name__}"
        )

    def _save_report(
        self,
        report: dict[str, Any],
    ) -> None:
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        split = report.get(
            "evaluation_split",
            "unknown",
        )

        json_path = (
            self.output_directory
            / f"rag_evaluation_{split}_{timestamp}.json"
        )

        csv_path = (
            self.output_directory
            / f"rag_evaluation_{split}_{timestamp}.csv"
        )

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
            )

        self._save_case_csv(
            report["cases"],
            csv_path,
        )

        print(
            f"\nEvaluation JSON saved to: {json_path}"
        )

        print(
            f"Evaluation CSV saved to: {csv_path}"
        )

    def _save_case_csv(
        self,
        cases: list[dict[str, Any]],
        csv_path: Path,
    ) -> None:
        if not cases:
            return

        fieldnames = [
            "feedback_id",
            "relative_path",
            "gold_category",
            "predicted_category",
            "confidence",
            "correct",
            "gold_category_retrieved",
            "retrieved_only_reference",
            "retrieved_feedback_ids",
            "retrieved_categories",
            "status",
            "error",
        ]

        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for case in cases:
                writer.writerow(
                    {
                        "feedback_id": case.get(
                            "feedback_id"
                        ),
                        "relative_path": case.get(
                            "relative_path"
                        ),
                        "gold_category": case.get(
                            "gold_category"
                        ),
                        "predicted_category": case.get(
                            "predicted_category"
                        ),
                        "confidence": case.get(
                            "confidence"
                        ),
                        "correct": case.get(
                            "correct"
                        ),
                        "gold_category_retrieved": case.get(
                            "gold_category_retrieved"
                        ),
                        "retrieved_only_reference": case.get(
                            "retrieved_only_reference"
                        ),
                        "retrieved_feedback_ids": ";".join(
                            case.get(
                                "retrieved_feedback_ids",
                                [],
                            )
                        ),
                        "retrieved_categories": ";".join(
                            case.get(
                                "retrieved_categories",
                                [],
                            )
                        ),
                        "status": case.get(
                            "status"
                        ),
                        "error": case.get(
                            "error"
                        ),
                    }
                )

    def _print_summary(
        self,
        report: dict[str, Any],
    ) -> None:
        metrics = report[
            "classification_metrics"
        ]

        retrieval = report[
            "retrieval_metrics"
        ]

        print(
            "\n"
            "=============================================\n"
            "RAG Evaluation Summary\n"
            "============================================="
        )

        print(
            f"Evaluation split: "
            f"{report['evaluation_split']}"
        )

        print(
            f"Evaluation cases: "
            f"{report['total_evaluation_records']}"
        )

        print(
            f"Evaluated successfully: "
            f"{metrics['evaluated_cases']}"
        )

        print(
            f"Accuracy: "
            f"{self._format_metric(metrics['accuracy'])}"
        )

        print(
            f"Macro Precision: "
            f"{self._format_metric(metrics['macro_precision'])}"
        )

        print(
            f"Macro Recall: "
            f"{self._format_metric(metrics['macro_recall'])}"
        )

        print(
            f"Macro F1: "
            f"{self._format_metric(metrics['macro_f1'])}"
        )

        print(
            f"Weighted F1: "
            f"{self._format_metric(metrics['weighted_f1'])}"
        )

        print(
            "\nRetrieval Metrics:"
        )

        print(
            f"Retrieval success rate: "
            f"{self._format_metric(retrieval['retrieval_success_rate'])}"
        )

        print(
            f"Reference-only retrieval rate: "
            f"{self._format_metric(retrieval['reference_only_rate'])}"
        )

        print(
            f"Gold-category retrieval rate: "
            f"{self._format_metric(retrieval['gold_category_retrieval_rate'])}"
        )

        print(
            f"Average retrieved examples: "
            f"{self._format_metric(retrieval['average_retrieved_count'])}"
        )

    def _derive_feedback_id(
        self,
        relative_path: str,
    ) -> str:
        digest = (
            __import__("hashlib")
            .sha256(
                relative_path.encode(
                    "utf-8"
                )
            )
            .hexdigest()
        )

        return f"FB-{digest[:12]}"

    @staticmethod
    def _safe_divide(
        numerator: float,
        denominator: float,
    ) -> float:
        if denominator == 0:
            return 0.0

        return numerator / denominator

    @staticmethod
    def _format_metric(
        value: float | None,
    ) -> str:
        if value is None:
            return "N/A"

        return f"{value:.4f}"

    @staticmethod
    def _load_environment() -> None:
        backend_directory = (
            Path(__file__).resolve().parents[1]
        )

        env_path = (
            backend_directory / ".env"
        )

        load_dotenv(
            dotenv_path=env_path
        )


def get_rag_evaluation_service(
    manifest_path: Path | str | None = None,
    output_directory: Path | str | None = None,
    rag_service: RAGService | None = None,
    document_service: Any | None = None,
) -> RAGEvaluationService:
    return RAGEvaluationService(
        manifest_path=manifest_path,
        output_directory=output_directory,
        rag_service=rag_service,
        document_service=document_service,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the AI Feedback Intelligence RAG pipeline."
        )
    )

    parser.add_argument(
        "--split",
        choices=VALID_EVALUATION_SPLITS,
        default=DEFAULT_EVALUATION_SPLIT,
        help=(
            "Dataset split to evaluate. "
            "Use development while optimizing. "
            "Use final_benchmark only for the frozen final evaluation."
        ),
    )

    parser.add_argument(
        "--ragas",
        action="store_true",
        help=(
            "Also run Ragas Faithfulness evaluation."
        ),
    )

    parser.add_argument(
        "--ragas-sample-size",
        type=int,
        default=None,
        help=(
            "Maximum number of cases for Ragas Faithfulness."
        ),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "Do not save evaluation reports."
        ),
    )

    args = parser.parse_args()

    evaluator = get_rag_evaluation_service()

    evaluator.evaluate(
        split=args.split,
        include_ragas=args.ragas,
        ragas_sample_size=args.ragas_sample_size,
        save_report=not args.no_save,
    )


if __name__ == "__main__":
    main()