import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_SPLITS = (
    "development",
    "final_benchmark",
)

VALID_DECISIONS = (
    "keep",
    "revert",
)

DEFAULT_EXPERIMENT_DIRECTORY = Path(
    "backend/evaluation/results/experiments"
)


class ExperimentService:
    def __init__(
        self,
        output_directory: str | Path = DEFAULT_EXPERIMENT_DIRECTORY,
    ) -> None:
        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.registry_path = (
            self.output_directory
            / "experiment_registry.json"
        )

        self.csv_path = (
            self.output_directory
            / "experiment_registry.csv"
        )

    def _validate_split(
        self,
        split: str,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid experiment split '{split}'. "
                f"Expected one of: "
                f"{', '.join(VALID_SPLITS)}."
            )

    def _validate_decision(
        self,
        decision: str,
    ) -> None:
        if decision not in VALID_DECISIONS:
            raise ValueError(
                f"Invalid experiment decision '{decision}'. "
                f"Expected one of: "
                f"{', '.join(VALID_DECISIONS)}."
            )

    @staticmethod
    def _validate_non_empty(
        value: str,
        field_name: str,
    ) -> None:
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string."
            )

        if not value.strip():
            raise ValueError(
                f"{field_name} cannot be empty."
            )

    @staticmethod
    def _utc_timestamp() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat()
        )

    def _load_registry(
        self,
    ) -> list[dict[str, Any]]:
        if not self.registry_path.exists():
            return []

        with self.registry_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                "Experiment registry must contain a JSON list."
            )

        return data

    def _save_registry(
        self,
        experiments: list[dict[str, Any]],
    ) -> None:
        with self.registry_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                experiments,
                file,
                indent=2,
            )

    @staticmethod
    def _flatten_metrics(
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        flattened: dict[str, Any] = {}

        for key, value in metrics.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(
                        nested_value,
                        (dict, list),
                    ):
                        continue

                    flattened[
                        f"{key}_{nested_key}"
                    ] = nested_value
            elif isinstance(value, list):
                continue
            else:
                flattened[key] = value

        return flattened

    def record_experiment(
        self,
        experiment_id: str,
        strategy: str,
        split: str,
        configuration: dict[str, Any],
        metrics: dict[str, Any],
        baseline_metrics: dict[str, Any] | None = None,
        decision: str = "keep",
        notes: str = "",
    ) -> dict[str, Any]:
        self._validate_non_empty(
            experiment_id,
            "experiment_id",
        )

        self._validate_non_empty(
            strategy,
            "strategy",
        )

        self._validate_split(split)

        self._validate_decision(decision)

        if not isinstance(
            configuration,
            dict,
        ):
            raise TypeError(
                "configuration must be a dictionary."
            )

        if not isinstance(
            metrics,
            dict,
        ):
            raise TypeError(
                "metrics must be a dictionary."
            )

        if baseline_metrics is not None and not isinstance(
            baseline_metrics,
            dict,
        ):
            raise TypeError(
                "baseline_metrics must be a dictionary "
                "or None."
            )

        if not isinstance(notes, str):
            raise TypeError(
                "notes must be a string."
            )

        experiments = self._load_registry()

        if any(
            experiment.get("experiment_id")
            == experiment_id
            for experiment in experiments
        ):
            raise ValueError(
                f"Experiment ID '{experiment_id}' "
                f"already exists."
            )

        metric_deltas: dict[str, float] = {}

        if baseline_metrics is not None:
            for key, value in metrics.items():
                baseline_value = (
                    baseline_metrics.get(key)
                )

                if (
                    isinstance(value, (int, float))
                    and isinstance(
                        baseline_value,
                        (int, float),
                    )
                ):
                    metric_deltas[key] = (
                        float(value)
                        - float(baseline_value)
                    )

        experiment = {
            "experiment_id": experiment_id,
            "strategy": strategy,
            "split": split,
            "timestamp": self._utc_timestamp(),
            "configuration": configuration,
            "metrics": metrics,
            "baseline_metrics": (
                baseline_metrics
                if baseline_metrics is not None
                else {}
            ),
            "metric_deltas": metric_deltas,
            "decision": decision,
            "notes": notes,
        }

        experiments.append(
            experiment
        )

        self._save_registry(
            experiments
        )

        self._append_csv_row(
            experiment
        )

        return experiment

    def _append_csv_row(
        self,
        experiment: dict[str, Any],
    ) -> None:
        flattened_metrics = (
            self._flatten_metrics(
                experiment["metrics"]
            )
        )

        flattened_baseline = (
            self._flatten_metrics(
                experiment["baseline_metrics"]
            )
        )

        flattened_deltas = (
            self._flatten_metrics(
                experiment["metric_deltas"]
            )
        )

        row = {
            "experiment_id": experiment[
                "experiment_id"
            ],
            "strategy": experiment[
                "strategy"
            ],
            "split": experiment[
                "split"
            ],
            "timestamp": experiment[
                "timestamp"
            ],
            "decision": experiment[
                "decision"
            ],
            "notes": experiment[
                "notes"
            ],
        }

        for key, value in flattened_metrics.items():
            row[f"metric_{key}"] = value

        for key, value in flattened_baseline.items():
            row[f"baseline_{key}"] = value

        for key, value in flattened_deltas.items():
            row[f"delta_{key}"] = value

        file_exists = (
            self.csv_path.exists()
            and self.csv_path.stat().st_size > 0
        )

        existing_fields: list[str] = []

        if file_exists:
            with self.csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                reader = csv.reader(file)

                try:
                    existing_fields = next(
                        reader
                    )
                except StopIteration:
                    existing_fields = []

        fieldnames = list(
            dict.fromkeys(
                existing_fields
                + list(row.keys())
            )
        )

        existing_rows: list[dict[str, Any]] = []

        if file_exists:
            with self.csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                reader = csv.DictReader(
                    file
                )

                existing_rows = list(
                    reader
                )

        existing_rows.append(
            row
        )

        with self.csv_path.open(
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

            for existing_row in existing_rows:
                writer.writerow(
                    existing_row
                )

    def get_experiment(
        self,
        experiment_id: str,
    ) -> dict[str, Any] | None:
        self._validate_non_empty(
            experiment_id,
            "experiment_id",
        )

        experiments = self._load_registry()

        for experiment in experiments:
            if (
                experiment.get("experiment_id")
                == experiment_id
            ):
                return experiment

        return None

    def list_experiments(
        self,
        split: str | None = None,
    ) -> list[dict[str, Any]]:
        if split is not None:
            self._validate_split(split)

        experiments = self._load_registry()

        if split is None:
            return experiments

        return [
            experiment
            for experiment in experiments
            if experiment.get("split")
            == split
        ]

    def get_latest_experiment(
        self,
        split: str | None = None,
    ) -> dict[str, Any] | None:
        experiments = self.list_experiments(
            split=split
        )

        if not experiments:
            return None

        return experiments[-1]

    def get_best_experiment(
        self,
        metric: str = "accuracy",
        split: str = "development",
    ) -> dict[str, Any] | None:
        self._validate_non_empty(
            metric,
            "metric",
        )

        self._validate_split(split)

        experiments = self.list_experiments(
            split=split
        )

        candidates: list[
            tuple[float, dict[str, Any]]
        ] = []

        for experiment in experiments:
            if (
                experiment.get("decision")
                != "keep"
            ):
                continue

            value = experiment.get(
                "metrics",
                {},
            ).get(metric)

            if isinstance(
                value,
                (int, float),
            ):
                candidates.append(
                    (
                        float(value),
                        experiment,
                    )
                )

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: item[0],
        )[1]