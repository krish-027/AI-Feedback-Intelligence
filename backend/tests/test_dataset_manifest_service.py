import csv
from pathlib import Path

import pytest

from backend.evaluation.dataset_manifest_service import (
    CATEGORIES,
    DatasetManifestService,
    SPLIT_TARGETS,
)


def create_ground_truth_file(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "feedback_id",
                "filename",
                "relative_path",
                "category",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def create_dummy_pdfs(
    tmp_path: Path,
    rows: list[dict[str, str]],
) -> None:

    for row in rows:
        pdf_path = tmp_path / row["relative_path"]

        pdf_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf_path.write_bytes(
            b"%PDF-1.4\n"
        )


def create_valid_rows() -> list[dict[str, str]]:
    rows = []

    category_counts = {
        "Excellent": 3,
        "Good": 20,
        "Need Improvements": 41,
        "Poor": 36,
    }

    counter = 1

    for category, count in category_counts.items():

        for _ in range(count):

            filename = (
                f"Customer_Feedback_{counter:03d}.pdf"
            )

            rows.append(
                {
                    "feedback_id": (
                        f"FB-TEST{counter:03d}"
                    ),
                    "filename": filename,
                    "relative_path": filename,
                    "category": category,
                }
            )

            counter += 1

    return rows


def create_service(
    tmp_path: Path,
    rows: list[dict[str, str]],
) -> DatasetManifestService:

    ground_truth_path = (
        tmp_path / "ground_truth.csv"
    )

    manifest_path = (
        tmp_path / "dataset_manifest.csv"
    )

    create_ground_truth_file(
        ground_truth_path,
        rows,
    )

    create_dummy_pdfs(
        tmp_path,
        rows,
    )

    return DatasetManifestService(
        ground_truth_path=ground_truth_path,
        manifest_path=manifest_path,
        dataset_directory=tmp_path,
    )


def test_build_manifest_contains_100_records(
    tmp_path: Path,
) -> None:

    rows = create_valid_rows()

    service = create_service(
        tmp_path,
        rows,
    )

    manifest = service.build_manifest()

    assert len(manifest) == 100


def test_manifest_contains_expected_splits(
    tmp_path: Path,
) -> None:

    rows = create_valid_rows()

    service = create_service(
        tmp_path,
        rows,
    )

    manifest = service.build_manifest()

    splits = {
        row["split"]
        for row in manifest
    }

    assert splits == {
        "reference",
        "development",
        "final_benchmark",
    }


def test_manifest_has_60_20_20_split(
    tmp_path: Path,
) -> None:

    rows = create_valid_rows()

    service = create_service(
        tmp_path,
        rows,
    )

    manifest = service.build_manifest()

    split_counts = {}

    for split in SPLIT_TARGETS:
        split_counts[split] = sum(
            1
            for row in manifest
            if row["split"] == split
        )

    assert split_counts == {
        "reference": 60,
        "development": 20,
        "final_benchmark": 20,
    }


def test_manifest_is_stratified_by_category(
    tmp_path: Path,
) -> None:

    rows = create_valid_rows()

    service = create_service(
        tmp_path,
        rows,
    )

    manifest = service.build_manifest()

    expected = {
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

    for category in CATEGORIES:

        for split in SPLIT_TARGETS:

            actual = sum(
                1
                for row in manifest
                if row["category"] == category
                and row["split"] == split
            )

            assert actual == expected[
                category
            ][split]


def test_manifest_preserves_ground_truth_labels(
    tmp_path: Path,
) -> None:

    rows = create_valid_rows()

    service = create_service(
        tmp_path,
        rows,
    )

    manifest = service.build_manifest()

    ground_truth_by_id = {
        row["feedback_id"]: row["category"]
        for row in rows
    }

    for row in manifest:

        assert row["category"] == (
            ground_truth_by_id[
                row["feedback_id"]
            ]
        )


def test_manifest_preserves_feedback_identity(
    tmp_path: Path,
) -> None:

    rows = create_valid_rows()

    service = create_service(
        tmp_path,
        rows,
    )

    manifest = service.build_manifest()

    assert {
        row["feedback_id"]
        for row in manifest
    } == {
        row["feedback_id"]
        for row in rows
    }

    assert {
        row["filename"]
        for row in manifest
    } == {
        row["filename"]
        for row in rows
    }


def test_manifest_is_deterministic(
    tmp_path: Path,
) -> None:

    rows = create_valid_rows()

    service = create_service(
        tmp_path,
        rows,
    )

    first_manifest = (
        service.build_manifest()
    )

    second_manifest = (
        service.build_manifest()
    )

    assert first_manifest == second_manifest


def test_manifest_can_be_saved(
    tmp_path: Path,
) -> None:

    rows = create_valid_rows()

    service = create_service(
        tmp_path,
        rows,
    )

    manifest = service.build_manifest()

    output_path = service.save_manifest(
        manifest
    )

    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        saved_rows = list(reader)

    assert len(saved_rows) == 100


def test_missing_ground_truth_is_rejected(
    tmp_path: Path,
) -> None:

    service = DatasetManifestService(
        ground_truth_path=(
            tmp_path / "missing.csv"
        ),
        manifest_path=(
            tmp_path / "manifest.csv"
        ),
        dataset_directory=tmp_path,
    )

    with pytest.raises(
        FileNotFoundError
    ):

        service.build_manifest()


def test_invalid_category_is_rejected(
    tmp_path: Path,
) -> None:

    rows = create_valid_rows()

    rows[0]["category"] = "Average"

    service = create_service(
        tmp_path,
        rows,
    )

    with pytest.raises(
        ValueError,
        match="invalid category",
    ):

        service.build_manifest()


def test_duplicate_feedback_id_is_rejected(
    tmp_path: Path,
) -> None:

    rows = create_valid_rows()

    rows[1]["feedback_id"] = (
        rows[0]["feedback_id"]
    )

    service = create_service(
        tmp_path,
        rows,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate feedback_id",
    ):

        service.build_manifest()


def test_missing_pdf_is_rejected(
    tmp_path: Path,
) -> None:

    rows = create_valid_rows()

    missing_pdf = rows[0][
        "relative_path"
    ]

    service = create_service(
        tmp_path,
        rows,
    )

    missing_path = (
        tmp_path / missing_pdf
    )

    missing_path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="PDF referenced",
    ):

        service.build_manifest()