import json
from pathlib import Path

import pytest

from backend.evaluation.experiment_service import (
    ExperimentService,
)


def create_service(
    tmp_path: Path,
) -> ExperimentService:
    return ExperimentService(
        output_directory=(
            tmp_path / "experiments"
        )
    )


def test_record_experiment(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    experiment = service.record_experiment(
        experiment_id="EXP001",
        strategy="baseline",
        split="development",
        configuration={
            "model": "gemini-2.5-flash",
            "temperature": 0.0,
        },
        metrics={
            "accuracy": 0.75,
            "macro_f1": 0.70,
        },
        decision="keep",
        notes="Initial development baseline.",
    )

    assert (
        experiment["experiment_id"]
        == "EXP001"
    )

    assert (
        experiment["split"]
        == "development"
    )

    assert (
        experiment["metrics"]["accuracy"]
        == 0.75
    )

    assert (
        experiment["decision"]
        == "keep"
    )


def test_experiment_registry_is_saved(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    service.record_experiment(
        experiment_id="EXP001",
        strategy="baseline",
        split="development",
        configuration={
            "prompt_version": "baseline",
        },
        metrics={
            "accuracy": 0.75,
        },
    )

    assert (
        service.registry_path.exists()
    )

    assert (
        service.csv_path.exists()
    )

    data = json.loads(
        service.registry_path.read_text(
            encoding="utf-8"
        )
    )

    assert len(data) == 1
    assert (
        data[0]["experiment_id"]
        == "EXP001"
    )


def test_duplicate_experiment_id_is_rejected(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    kwargs = {
        "experiment_id": "EXP001",
        "strategy": "baseline",
        "split": "development",
        "configuration": {},
        "metrics": {
            "accuracy": 0.75,
        },
    }

    service.record_experiment(
        **kwargs
    )

    with pytest.raises(
        ValueError,
        match="already exists",
    ):
        service.record_experiment(
            **kwargs
        )


def test_invalid_split_is_rejected(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    with pytest.raises(
        ValueError,
        match="Invalid experiment split",
    ):
        service.record_experiment(
            experiment_id="EXP001",
            strategy="baseline",
            split="evaluation",
            configuration={},
            metrics={
                "accuracy": 0.75,
            },
        )


def test_metric_deltas_are_calculated(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    experiment = service.record_experiment(
        experiment_id="EXP001",
        strategy="prompt_optimization",
        split="development",
        configuration={
            "prompt_version": "v2",
        },
        metrics={
            "accuracy": 0.82,
            "macro_f1": 0.78,
        },
        baseline_metrics={
            "accuracy": 0.75,
            "macro_f1": 0.70,
        },
    )

    assert (
        experiment["metric_deltas"]["accuracy"]
        == pytest.approx(0.07)
    )

    assert (
        experiment["metric_deltas"]["macro_f1"]
        == pytest.approx(0.08)
    )


def test_list_and_get_experiments(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    service.record_experiment(
        experiment_id="EXP001",
        strategy="baseline",
        split="development",
        configuration={},
        metrics={
            "accuracy": 0.75,
        },
    )

    service.record_experiment(
        experiment_id="EXP002",
        strategy="prompt_optimization",
        split="development",
        configuration={},
        metrics={
            "accuracy": 0.82,
        },
    )

    service.record_experiment(
        experiment_id="EXP003",
        strategy="final_check",
        split="final_benchmark",
        configuration={},
        metrics={
            "accuracy": 0.90,
        },
    )

    all_experiments = (
        service.list_experiments()
    )

    development_experiments = (
        service.list_experiments(
            split="development"
        )
    )

    assert len(all_experiments) == 3
    assert len(development_experiments) == 2

    experiment = service.get_experiment(
        "EXP002"
    )

    assert experiment is not None
    assert (
        experiment["metrics"]["accuracy"]
        == 0.82
    )


def test_latest_experiment(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    service.record_experiment(
        experiment_id="EXP001",
        strategy="baseline",
        split="development",
        configuration={},
        metrics={
            "accuracy": 0.75,
        },
    )

    service.record_experiment(
        experiment_id="EXP002",
        strategy="prompt_optimization",
        split="development",
        configuration={},
        metrics={
            "accuracy": 0.82,
        },
    )

    latest = (
        service.get_latest_experiment(
            split="development"
        )
    )

    assert latest is not None
    assert (
        latest["experiment_id"]
        == "EXP002"
    )


def test_best_experiment_only_considers_kept(
    tmp_path: Path,
):
    service = create_service(tmp_path)

    service.record_experiment(
        experiment_id="EXP001",
        strategy="baseline",
        split="development",
        configuration={},
        metrics={
            "accuracy": 0.75,
        },
        decision="keep",
    )

    service.record_experiment(
        experiment_id="EXP002",
        strategy="bad_prompt",
        split="development",
        configuration={},
        metrics={
            "accuracy": 0.90,
        },
        decision="revert",
    )

    service.record_experiment(
        experiment_id="EXP003",
        strategy="good_prompt",
        split="development",
        configuration={},
        metrics={
            "accuracy": 0.82,
        },
        decision="keep",
    )

    best = service.get_best_experiment(
        metric="accuracy",
        split="development",
    )

    assert best is not None
    assert (
        best["experiment_id"]
        == "EXP003"
    )