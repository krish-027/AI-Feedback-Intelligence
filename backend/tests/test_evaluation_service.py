import json
import os
import time

import pytest

from backend.services.evaluation_report_service import (
    EvaluationReportInvalidError,
    EvaluationReportNotFoundError,
    EvaluationReportService,
)


def write_json(path, data):
    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )


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
def results_dir(tmp_path):
    return tmp_path


@pytest.fixture
def service(results_dir):
    return EvaluationReportService(
        results_dir=results_dir
    )


def test_get_rag_evaluation_reads_report(
    service,
    results_dir,
):
    report = make_rag_report("development")

    write_json(
        results_dir / "rag_evaluation_development_001.json",
        report,
    )

    result = service.get_rag_evaluation("development")

    assert result["evaluation_split"] == "development"
    assert result["classification_metrics"]["accuracy"] == 0.95


def test_get_latest_rag_evaluation_report(
    service,
    results_dir,
):
    old_report = make_rag_report("development")

    new_report = make_rag_report("development")
    new_report["classification_metrics"]["accuracy"] = 0.99

    old_path = (
        results_dir
        / "rag_evaluation_development_old.json"
    )

    new_path = (
        results_dir
        / "rag_evaluation_development_new.json"
    )

    write_json(old_path, old_report)
    write_json(new_path, new_report)

    old_time = time.time() - 100
    new_time = time.time()

    os.utime(
        old_path,
        (old_time, old_time),
    )

    os.utime(
        new_path,
        (new_time, new_time),
    )

    result = service.get_rag_evaluation("development")

    assert (
        result["classification_metrics"]["accuracy"]
        == 0.99
    )


def test_get_final_benchmark_rag_evaluation(
    service,
    results_dir,
):
    report = make_rag_report("final_benchmark")

    write_json(
        results_dir
        / "rag_evaluation_final_benchmark_001.json",
        report,
    )

    result = service.get_rag_evaluation(
        "final_benchmark"
    )

    assert (
        result["evaluation_split"]
        == "final_benchmark"
    )


def test_get_ablation_report(
    service,
    results_dir,
):
    report = make_ablation_report("development")

    write_json(
        results_dir
        / "rag_ablation_development.json",
        report,
    )

    result = service.get_rag_ablation(
        "development"
    )

    assert result["evaluation_split"] == "development"

    assert (
        result["rag_metrics"]["accuracy"]
        == 0.95
    )

    assert (
        result["no_rag_metrics"]["accuracy"]
        == 0.50
    )


def test_get_final_benchmark_ablation(
    service,
    results_dir,
):
    report = make_ablation_report(
        "final_benchmark"
    )

    write_json(
        results_dir
        / "rag_ablation_final_benchmark.json",
        report,
    )

    result = service.get_rag_ablation(
        "final_benchmark"
    )

    assert (
        result["evaluation_split"]
        == "final_benchmark"
    )


def test_missing_rag_report_raises(service):
    with pytest.raises(
        EvaluationReportNotFoundError
    ):
        service.get_rag_evaluation(
            "development"
        )


def test_missing_ablation_report_raises(service):
    with pytest.raises(
        EvaluationReportNotFoundError
    ):
        service.get_rag_ablation(
            "development"
        )


def test_invalid_split_raises(service):
    with pytest.raises(ValueError):
        service.get_rag_evaluation(
            "evaluation"
        )


def test_invalid_rag_report_split_is_rejected(
    service,
    results_dir,
):
    report = make_rag_report("final_benchmark")

    write_json(
        results_dir
        / "rag_evaluation_development_invalid.json",
        report,
    )

    with pytest.raises(
        EvaluationReportInvalidError
    ):
        service.get_rag_evaluation(
            "development"
        )


def test_invalid_ablation_report_split_is_rejected(
    service,
    results_dir,
):
    report = make_ablation_report(
        "final_benchmark"
    )

    write_json(
        results_dir
        / "rag_ablation_development.json",
        report,
    )

    with pytest.raises(
        EvaluationReportInvalidError
    ):
        service.get_rag_ablation(
            "development"
        )


def test_missing_required_rag_section_is_rejected(
    service,
    results_dir,
):
    report = make_rag_report("development")

    del report["cases"]

    write_json(
        results_dir
        / "rag_evaluation_development_invalid.json",
        report,
    )

    with pytest.raises(
        EvaluationReportInvalidError
    ):
        service.get_rag_evaluation(
            "development"
        )


def test_invalid_metric_above_one_is_rejected(
    service,
    results_dir,
):
    report = make_rag_report("development")

    report["classification_metrics"][
        "accuracy"
    ] = 1.5

    write_json(
        results_dir
        / "rag_evaluation_development_invalid.json",
        report,
    )

    with pytest.raises(
        EvaluationReportInvalidError
    ):
        service.get_rag_evaluation(
            "development"
        )


def test_invalid_metric_below_zero_is_rejected(
    service,
    results_dir,
):
    report = make_rag_report("development")

    report["classification_metrics"][
        "accuracy"
    ] = -0.1

    write_json(
        results_dir
        / "rag_evaluation_development_invalid.json",
        report,
    )

    with pytest.raises(
        EvaluationReportInvalidError
    ):
        service.get_rag_evaluation(
            "development"
        )


def test_invalid_report_json_is_rejected(
    service,
    results_dir,
):
    path = (
        results_dir
        / "rag_evaluation_development_invalid.json"
    )

    path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationReportInvalidError
    ):
        service.get_rag_evaluation(
            "development"
        )


def test_non_object_json_is_rejected(
    service,
    results_dir,
):
    path = (
        results_dir
        / "rag_evaluation_development_invalid.json"
    )

    path.write_text(
        json.dumps(["invalid", "report"]),
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationReportInvalidError
    ):
        service.get_rag_evaluation(
            "development"
        )


def test_summary_returns_all_available_reports(
    service,
    results_dir,
):
    write_json(
        results_dir
        / "rag_evaluation_development_001.json",
        make_rag_report("development"),
    )

    write_json(
        results_dir
        / "rag_evaluation_final_benchmark_001.json",
        make_rag_report("final_benchmark"),
    )

    write_json(
        results_dir
        / "rag_ablation_development.json",
        make_ablation_report("development"),
    )

    write_json(
        results_dir
        / "rag_ablation_final_benchmark.json",
        make_ablation_report("final_benchmark"),
    )

    result = service.get_summary()

    assert (
        result["development"]["rag_evaluation"]
        is not None
    )

    assert (
        result["development"]["rag_vs_no_rag"]
        is not None
    )

    assert (
        result["final_benchmark"]["rag_evaluation"]
        is not None
    )

    assert (
        result["final_benchmark"]["rag_vs_no_rag"]
        is not None
    )


def test_summary_handles_missing_reports(
    service,
    results_dir,
):
    result = service.get_summary()

    assert (
        result["development"]["rag_evaluation"]
        is None
    )

    assert (
        result["development"]["rag_vs_no_rag"]
        is None
    )

    assert (
        result["final_benchmark"]["rag_evaluation"]
        is None
    )

    assert (
        result["final_benchmark"]["rag_vs_no_rag"]
        is None
    )