from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_INPUT_DIR = (
    PROJECT_ROOT
    / "Sample_Data"
    / "customer_feedback_forms"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "Sample_Data"
    / "evaluation"
)

ANNOTATION_FILE = (
    EVALUATION_DIR
    / "ground_truth.csv"
)

MANIFEST_FILE = (
    EVALUATION_DIR
    / "dataset_manifest.csv"
)

VALID_CATEGORIES = [
    "Excellent",
    "Good",
    "Need Improvements",
    "Poor",
]

DEFAULT_EVALUATION_RATIO = 0.20
DEFAULT_RANDOM_SEED = 42


# ============================================================
# PDF discovery
# ============================================================

def discover_pdf_files(input_dir: Path) -> list[Path]:
    """
    Recursively discover all PDF files.

    No assumptions are made about filenames or naming conventions.
    """

    if not input_dir.exists():
        raise FileNotFoundError(
            f"PDF input directory does not exist:\n{input_dir}"
        )

    pdf_files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".pdf"
    ]

    return sorted(pdf_files)


# ============================================================
# Stable document ID
# ============================================================

def generate_feedback_id(pdf_path: Path) -> str:
    """
    Generate a stable ID based on the PDF's relative path.
    """

    relative_path = pdf_path.relative_to(
        PROJECT_ROOT
    ).as_posix()

    digest = hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()[:12]

    return f"FB-{digest}"


# ============================================================
# Annotation template
# ============================================================

def create_annotation_template(
    pdf_files: list[Path],
) -> pd.DataFrame:
    """
    Create one annotation row for every discovered PDF.
    """

    rows = []

    for pdf_path in pdf_files:
        relative_path = (
            pdf_path
            .relative_to(PROJECT_ROOT)
            .as_posix()
        )

        rows.append(
            {
                "feedback_id": generate_feedback_id(
                    pdf_path
                ),
                "filename": pdf_path.name,
                "relative_path": relative_path,
                "category": "",
            }
        )

    return pd.DataFrame(rows)


def create_or_update_annotation_file(
    pdf_files: list[Path],
    annotation_file: Path,
) -> pd.DataFrame:
    """
    Create the annotation CSV.

    If the file already exists, previously entered categories
    are preserved.
    """

    new_df = create_annotation_template(pdf_files)

    if not annotation_file.exists():
        new_df.to_csv(
            annotation_file,
            index=False,
        )

        return new_df

    existing_df = pd.read_csv(
        annotation_file,
        dtype=str,
    ).fillna("")

    required_columns = {
        "feedback_id",
        "filename",
        "relative_path",
        "category",
    }

    missing_columns = (
        required_columns
        - set(existing_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Existing annotation file is missing "
            "required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    previous_labels = dict(
        zip(
            existing_df["feedback_id"],
            existing_df["category"],
        )
    )

    for index, row in new_df.iterrows():
        feedback_id = row["feedback_id"]

        if feedback_id in previous_labels:
            new_df.at[
                index,
                "category"
            ] = previous_labels[feedback_id]

    new_df.to_csv(
        annotation_file,
        index=False,
    )

    return new_df


# ============================================================
# Load and validate annotations
# ============================================================

def load_annotation_file(
    annotation_file: Path,
) -> pd.DataFrame:
    """
    Load the manually annotated CSV.
    """

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file does not exist:\n"
            f"{annotation_file}"
        )

    df = pd.read_csv(
        annotation_file,
        dtype=str,
    ).fillna("")

    required_columns = {
        "feedback_id",
        "filename",
        "relative_path",
        "category",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return df


def validate_annotations(
    df: pd.DataFrame,
) -> None:
    """
    Verify that every PDF has a valid manual label.
    """

    if df.empty:
        raise ValueError(
            "The annotation dataset is empty."
        )

    if df["feedback_id"].duplicated().any():
        duplicates = df.loc[
            df["feedback_id"].duplicated(),
            "feedback_id",
        ].tolist()

        raise ValueError(
            "Duplicate feedback IDs found: "
            + ", ".join(duplicates)
        )

    unlabeled = df[
        df["category"]
        .str.strip()
        .eq("")
    ]

    if not unlabeled.empty:
        print(
            "\nThe following PDFs are still unlabeled:"
        )

        print(
            unlabeled[
                [
                    "feedback_id",
                    "filename",
                ]
            ].to_string(index=False)
        )

        raise ValueError(
            f"\n{len(unlabeled)} PDF(s) "
            "still need a category."
        )

    invalid = sorted(
        set(df["category"])
        - set(VALID_CATEGORIES)
    )

    if invalid:
        raise ValueError(
            "Invalid categories found:\n"
            + "\n".join(
                f"- {category}"
                for category in invalid
            )
            + "\n\nValid categories are:\n"
            + "\n".join(
                f"- {category}"
                for category in VALID_CATEGORIES
            )
        )


# ============================================================
# Stratified split WITHOUT scikit-learn
# ============================================================

def create_stratified_split(
    df: pd.DataFrame,
    evaluation_ratio: float,
    random_seed: int,
) -> pd.DataFrame:
    """
    Create a deterministic stratified reference/evaluation split
    using only Python's standard library.

    Every category is split independently, so the category
    distribution is approximately preserved in both sets.
    """

    category_counts = Counter(
        df["category"]
    )

    insufficient_categories = [
        category
        for category, count
        in category_counts.items()
        if count < 2
    ]

    if insufficient_categories:
        raise ValueError(
            "Each category needs at least 2 PDFs "
            "for a stratified split.\n"
            "Insufficient categories: "
            + ", ".join(
                insufficient_categories
            )
        )

    rng = random.Random(
        random_seed
    )

    evaluation_indices = set()

    # Process each category separately.
    for category in VALID_CATEGORIES:

        category_indices = (
            df.index[
                df["category"] == category
            ]
            .tolist()
        )

        if not category_indices:
            continue

        # Deterministic shuffle.
        rng.shuffle(category_indices)

        evaluation_count = max(
            1,
            round(
                len(category_indices)
                * evaluation_ratio
            ),
        )

        # Make sure at least one document remains
        # in the reference set.
        evaluation_count = min(
            evaluation_count,
            len(category_indices) - 1,
        )

        selected = category_indices[
            :evaluation_count
        ]

        evaluation_indices.update(
            selected
        )

    result = df.copy()

    result["split"] = "reference"

    for index in evaluation_indices:
        result.at[
            index,
            "split"
        ] = "evaluation"

    return (
        result
        .sort_values("feedback_id")
        .reset_index(drop=True)
    )


# ============================================================
# Save manifest
# ============================================================

def save_manifest(
    split_df: pd.DataFrame,
    manifest_file: Path,
) -> None:
    """
    Save the final dataset manifest.
    """

    columns = [
        "feedback_id",
        "filename",
        "relative_path",
        "category",
        "split",
    ]

    split_df[columns].to_csv(
        manifest_file,
        index=False,
    )


# ============================================================
# Reporting
# ============================================================

def print_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print useful information about the final dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Total PDFs: {len(df)}"
    )

    print("\nCategory distribution:")

    category_counts = (
        df["category"]
        .value_counts()
        .reindex(
            VALID_CATEGORIES,
            fill_value=0,
        )
    )

    for category, count in (
        category_counts.items()
    ):
        print(
            f"  {category:<20} {count}"
        )

    print("\nSplit distribution:")

    split_counts = (
        df["split"]
        .value_counts()
    )

    for split, count in (
        split_counts.items()
    ):
        print(
            f"  {split:<20} {count}"
        )

    print(
        "\nCategory distribution by split:"
    )

    cross_tab = pd.crosstab(
        df["split"],
        df["category"],
    ).reindex(
        columns=VALID_CATEGORIES,
        fill_value=0,
    )

    print(
        cross_tab.to_string()
    )

    print("=" * 60)


# ============================================================
# Initial preparation
# ============================================================

def prepare_annotations() -> None:
    """
    Discover PDFs and create/update the annotation CSV.
    """

    print(
        "\nSearching for PDF files..."
    )

    pdf_files = discover_pdf_files(
        PDF_INPUT_DIR
    )

    print(
        f"Found {len(pdf_files)} PDF(s)."
    )

    if not pdf_files:
        raise ValueError(
            "No PDF files were found."
        )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = create_or_update_annotation_file(
        pdf_files,
        ANNOTATION_FILE,
    )

    labeled = (
        df["category"]
        .str.strip()
        .ne("")
        .sum()
    )

    unlabeled = (
        len(df) - labeled
    )

    print(
        "\nAnnotation file:"
    )

    print(
        ANNOTATION_FILE
    )

    print(
        "\nStatus:"
    )

    print(
        f"  Total PDFs : {len(df)}"
    )

    print(
        f"  Labeled    : {labeled}"
    )

    print(
        f"  Unlabeled  : {unlabeled}"
    )

    print(
        "\nValid categories:"
    )

    for category in VALID_CATEGORIES:
        print(
            f"  - {category}"
        )

    if unlabeled:
        print(
            "\nNext:"
        )

        print(
            "1. Open ground_truth.csv."
        )

        print(
            "2. Fill the category column."
        )

        print(
            "3. After every PDF is labeled, run:"
        )

        print(
            "   python "
            "Sample_Data\\evaluation\\"
            "prepare_evaluation_dataset.py "
            "--finalize"
        )

    else:
        print(
            "\nAll PDFs are already labeled."
        )

        print(
            "Run --finalize to create the split."
        )


# ============================================================
# Finalize dataset
# ============================================================

def finalize_dataset(
    evaluation_ratio: float,
    random_seed: int,
) -> None:
    """
    Validate manual labels and create the final split.
    """

    print(
        "\nLoading ground truth..."
    )

    df = load_annotation_file(
        ANNOTATION_FILE
    )

    print(
        "Validating labels..."
    )

    validate_annotations(df)

    print(
        "Creating stratified split..."
    )

    split_df = create_stratified_split(
        df,
        evaluation_ratio,
        random_seed,
    )

    save_manifest(
        split_df,
        MANIFEST_FILE,
    )

    print_summary(
        split_df
    )

    print(
        "\nCreated:"
    )

    print(
        f"  Ground truth : {ANNOTATION_FILE}"
    )

    print(
        f"  Manifest     : {MANIFEST_FILE}"
    )

    print(
        "\nImportant:"
    )

    print(
        "  reference  -> allowed into FAISS/RAG"
    )

    print(
        "  evaluation -> MUST NOT enter FAISS/RAG"
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Prepare and validate the "
            "PDF feedback evaluation dataset."
        )
    )

    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "Validate ground-truth labels and "
            "create the reference/evaluation split."
        ),
    )

    parser.add_argument(
        "--evaluation-ratio",
        type=float,
        default=DEFAULT_EVALUATION_RATIO,
        help=(
            "Fraction reserved for evaluation. "
            "Default: 0.20"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Random seed for reproducibility. "
            "Default: 42"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Run Stage 0.
    """

    args = parse_arguments()

    if not 0 < args.evaluation_ratio < 1:
        raise ValueError(
            "--evaluation-ratio must be between 0 and 1."
        )

    if args.finalize:
        finalize_dataset(
            evaluation_ratio=args.evaluation_ratio,
            random_seed=args.seed,
        )
    else:
        prepare_annotations()


if __name__ == "__main__":
    main()