import csv
import hashlib
from collections import defaultdict
from pathlib import Path


CATEGORIES = [
    "Excellent",
    "Good",
    "Need Improvements",
    "Poor",
]

DEFAULT_GROUND_TRUTH_PATH = Path(
    "Sample_Data/evaluation/ground_truth.csv"
)

DEFAULT_MANIFEST_PATH = Path(
    "Sample_Data/evaluation/dataset_manifest.csv"
)

DATASET_DIRECTORY = Path(
    "Sample_Data/customer_feedback_forms"
)

SPLIT_TARGETS = {
    "reference": 60,
    "development": 20,
    "final_benchmark": 20,
}


class DatasetManifestService:

    def __init__(
        self,
        ground_truth_path: Path | str = DEFAULT_GROUND_TRUTH_PATH,
        manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
        dataset_directory: Path | str = DATASET_DIRECTORY,
    ):
        self.ground_truth_path = Path(
            ground_truth_path
        )

        self.manifest_path = Path(
            manifest_path
        )

        self.dataset_directory = Path(
            dataset_directory
        )

    def build_manifest(self) -> list[dict[str, str]]:
        ground_truth = self._load_ground_truth()

        self._validate_ground_truth(
            ground_truth
        )

        split_assignments = (
            self._create_stratified_splits(
                ground_truth
            )
        )

        manifest = []

        for row in ground_truth:
            manifest.append(
                {
                    "feedback_id": row["feedback_id"],
                    "filename": row["filename"],
                    "relative_path": row["relative_path"],
                    "category": row["category"],
                    "split": split_assignments[
                        row["feedback_id"]
                    ],
                }
            )

        manifest.sort(
            key=lambda row: row["filename"]
        )

        return manifest

    def save_manifest(
        self,
        manifest: list[dict[str, str]],
    ) -> Path:

        self.manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "feedback_id",
            "filename",
            "relative_path",
            "category",
            "split",
        ]

        with self.manifest_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(manifest)

        return self.manifest_path

    def build_and_save(self) -> Path:
        manifest = self.build_manifest()

        path = self.save_manifest(
            manifest
        )

        self._print_summary(
            manifest
        )

        return path

    def _load_ground_truth(
        self,
    ) -> list[dict[str, str]]:

        if not self.ground_truth_path.exists():
            raise FileNotFoundError(
                "Ground-truth file not found: "
                f"{self.ground_truth_path}"
            )

        with self.ground_truth_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            if reader.fieldnames is None:
                raise ValueError(
                    "ground_truth.csv has no header."
                )

            required_fields = {
                "feedback_id",
                "filename",
                "relative_path",
                "category",
            }

            missing_fields = (
                required_fields
                - set(reader.fieldnames)
            )

            if missing_fields:
                raise ValueError(
                    "ground_truth.csv is missing "
                    f"required fields: {sorted(missing_fields)}"
                )

            return list(reader)

    def _resolve_dataset_path(
        self,
        relative_path: str,
    ) -> Path:

        path = Path(
            relative_path
        )

        if path.is_absolute():
            return path

        return self.dataset_directory / path

    def _validate_ground_truth(
        self,
        rows: list[dict[str, str]],
    ) -> None:

        if len(rows) != 100:
            raise ValueError(
                "Expected exactly 100 ground-truth rows, "
                f"but found {len(rows)}."
            )

        feedback_ids = set()
        filenames = set()
        relative_paths = set()

        for row in rows:

            feedback_id = row[
                "feedback_id"
            ].strip()

            filename = row[
                "filename"
            ].strip()

            relative_path = row[
                "relative_path"
            ].strip()

            category = row[
                "category"
            ].strip()

            if not feedback_id:
                raise ValueError(
                    "A ground-truth row has an empty feedback_id."
                )

            if not filename:
                raise ValueError(
                    "A ground-truth row has an empty filename."
                )

            if not relative_path:
                raise ValueError(
                    f"{filename}: relative_path is empty."
                )

            if category not in CATEGORIES:
                raise ValueError(
                    f"{filename}: invalid category "
                    f"'{category}'."
                )

            if feedback_id in feedback_ids:
                raise ValueError(
                    f"Duplicate feedback_id: {feedback_id}"
                )

            if filename in filenames:
                raise ValueError(
                    f"Duplicate filename: {filename}"
                )

            if relative_path in relative_paths:
                raise ValueError(
                    f"Duplicate relative_path: "
                    f"{relative_path}"
                )

            feedback_ids.add(
                feedback_id
            )

            filenames.add(
                filename
            )

            relative_paths.add(
                relative_path
            )

            pdf_path = self._resolve_dataset_path(
                relative_path
            )

            if not pdf_path.exists():
                raise FileNotFoundError(
                    "PDF referenced by ground_truth.csv "
                    f"does not exist: {pdf_path}"
                )

            if pdf_path.suffix.lower() != ".pdf":
                raise ValueError(
                    f"{pdf_path} is not a PDF file."
                )

    def _create_stratified_splits(
        self,
        rows: list[dict[str, str]],
    ) -> dict[str, str]:

        grouped_rows = defaultdict(list)

        for row in rows:
            grouped_rows[
                row["category"]
            ].append(row)

        expected_category_counts = {
            "Excellent": {
                "reference": 1,
                "development": 1,
                "final_benchmark": 1,
            },
            "Good": {
                "reference": 12,
                "development": 4,
                "final_benchmark": 4,
            },
            "Need Improvements": {
                "reference": 25,
                "development": 8,
                "final_benchmark": 8,
            },
            "Poor": {
                "reference": 22,
                "development": 7,
                "final_benchmark": 7,
            },
        }

        assignments = {}

        for category in CATEGORIES:

            category_rows = grouped_rows[
                category
            ]

            expected = (
                expected_category_counts[
                    category
                ]
            )

            expected_total = sum(
                expected.values()
            )

            if len(category_rows) != expected_total:
                raise ValueError(
                    f"Category '{category}' has "
                    f"{len(category_rows)} rows, "
                    f"but expected {expected_total}."
                )

            ordered_rows = sorted(
                category_rows,
                key=lambda row: self._stable_sort_key(
                    row["feedback_id"]
                ),
            )

            cursor = 0

            for split in [
                "reference",
                "development",
                "final_benchmark",
            ]:

                count = expected[
                    split
                ]

                selected_rows = ordered_rows[
                    cursor : cursor + count
                ]

                for row in selected_rows:
                    assignments[
                        row["feedback_id"]
                    ] = split

                cursor += count

        if len(assignments) != len(rows):
            raise ValueError(
                "Split assignment failed to cover "
                "all ground-truth rows."
            )

        self._validate_split_counts(
            rows,
            assignments,
        )

        return assignments

    @staticmethod
    def _stable_sort_key(
        value: str,
    ) -> str:

        return hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()

    def _validate_split_counts(
        self,
        rows: list[dict[str, str]],
        assignments: dict[str, str],
    ) -> None:

        split_counts = defaultdict(int)

        category_split_counts = defaultdict(
            lambda: defaultdict(int)
        )

        for row in rows:

            feedback_id = row[
                "feedback_id"
            ]

            category = row[
                "category"
            ]

            split = assignments[
                feedback_id
            ]

            split_counts[
                split
            ] += 1

            category_split_counts[
                category
            ][split] += 1

        for split, expected_count in SPLIT_TARGETS.items():

            actual_count = split_counts[
                split
            ]

            if actual_count != expected_count:
                raise ValueError(
                    f"Split '{split}' contains "
                    f"{actual_count} records; "
                    f"expected {expected_count}."
                )

        expected_category_counts = {
            "Excellent": {
                "reference": 1,
                "development": 1,
                "final_benchmark": 1,
            },
            "Good": {
                "reference": 12,
                "development": 4,
                "final_benchmark": 4,
            },
            "Need Improvements": {
                "reference": 25,
                "development": 8,
                "final_benchmark": 8,
            },
            "Poor": {
                "reference": 22,
                "development": 7,
                "final_benchmark": 7,
            },
        }

        for category, expected in (
            expected_category_counts.items()
        ):

            for split, expected_count in (
                expected.items()
            ):

                actual_count = category_split_counts[
                    category
                ][split]

                if actual_count != expected_count:
                    raise ValueError(
                        f"Category '{category}' has "
                        f"{actual_count} records in "
                        f"'{split}'; expected "
                        f"{expected_count}."
                    )

    @staticmethod
    def _print_summary(
        manifest: list[dict[str, str]],
    ) -> None:

        split_counts = defaultdict(int)

        category_split_counts = defaultdict(
            lambda: defaultdict(int)
        )

        for row in manifest:

            split = row["split"]
            category = row["category"]

            split_counts[
                split
            ] += 1

            category_split_counts[
                category
            ][split] += 1

        print(
            "=" * 60
        )

        print(
            "DATASET MANIFEST"
        )

        print(
            "=" * 60
        )

        print(
            f"Total records : {len(manifest)}"
        )

        print()

        for split in [
            "reference",
            "development",
            "final_benchmark",
        ]:

            print(
                f"{split:<18}: "
                f"{split_counts[split]}"
            )

        print()

        print(
            f"{'Category':<20}"
            f"{'Reference':<12}"
            f"{'Development':<14}"
            f"{'Final Benchmark':<16}"
        )

        print(
            "-" * 62
        )

        for category in CATEGORIES:

            print(
                f"{category:<20}"
                f"{category_split_counts[category]['reference']:<12}"
                f"{category_split_counts[category]['development']:<14}"
                f"{category_split_counts[category]['final_benchmark']:<16}"
            )


def get_dataset_manifest_service() -> DatasetManifestService:
    return DatasetManifestService()


if __name__ == "__main__":
    service = get_dataset_manifest_service()
    service.build_and_save()