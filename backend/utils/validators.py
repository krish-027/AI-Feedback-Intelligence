from pathlib import Path


ALLOWED_PDF_EXTENSIONS = {".pdf"}


def validate_pdf_extension(filename: str) -> bool:
    """
    Check whether the uploaded file is a PDF.
    """

    extension = Path(filename).suffix.lower()

    return extension in ALLOWED_PDF_EXTENSIONS