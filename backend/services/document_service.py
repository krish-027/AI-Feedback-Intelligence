from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, cast

import pymupdf
from dotenv import load_dotenv
from langchain_core.documents import Document


class PDFDocumentService:
    """
    PDF ingestion service with automatic OCR fallback.

    Each PDF page is independently processed.

    Normal page:
        PyMuPDF native extraction

    Image/scanned page:
        PyMuPDF OCR

    The resulting content is cleaned, explicit customer-identifying
    fields are removed, and the result is returned as a LangChain
    Document.
    """

    # ------------------------------------------------------------------
    # Default configuration
    # ------------------------------------------------------------------

    DEFAULT_OCR_LANG = "eng"
    DEFAULT_OCR_DPI = 250

    # Minimum amount of extracted text required before we consider
    # native PDF extraction usable.
    DEFAULT_MIN_NATIVE_CHARS = 50

    # Minimum proportion of alphanumeric characters in the extracted
    # text. This helps identify garbage/near-empty extraction.
    DEFAULT_MIN_ALNUM_RATIO = 0.30

    # Customer-identifying fields that should not be embedded.
    PII_FIELD_NAMES = (
        "customer name",
        "account number",
        "contact number",
    )

    def __init__(
        self,
        project_root: Path | None = None,
    ) -> None:
        """
        Initialize the document service.

        Args:
            project_root:
                Root directory of the project. If omitted, it is
                automatically inferred from this file's location.
        """

        self.project_root = (
            project_root.resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )

        # Load backend/.env.
        env_file = self.project_root / "backend" / ".env"

        if env_file.exists():
            load_dotenv(env_file)

        self.ocr_lang = os.getenv(
            "OCR_LANG",
            self.DEFAULT_OCR_LANG,
        )

        self.ocr_dpi = self._get_positive_int_env(
            "OCR_DPI",
            self.DEFAULT_OCR_DPI,
        )

        self.min_native_chars = self._get_positive_int_env(
            "OCR_MIN_NATIVE_CHARS",
            self.DEFAULT_MIN_NATIVE_CHARS,
        )

        self.min_alnum_ratio = self._get_float_env(
            "OCR_MIN_ALNUM_RATIO",
            self.DEFAULT_MIN_ALNUM_RATIO,
        )

        self.tessdata_path = self._get_tessdata_path()

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def load_pdf(
        self,
        pdf_path: str | Path,
        force_ocr: bool = False,
    ) -> Document:
        """
        Load one PDF and convert it into a LangChain Document.

        Args:
            pdf_path:
                Path to the PDF.

            force_ocr:
                If True, OCR is used for every page.

                This is primarily intended for testing the OCR path.

        Returns:
            A LangChain Document containing sanitized text and metadata.
        """

        path = Path(pdf_path).resolve()

        self._validate_pdf_path(path)

        page_results = self._extract_document_pages(
            path,
            force_ocr=force_ocr,
        )

        if not page_results:
            raise ValueError(
                f"No extractable content found in PDF: {path}"
            )

        raw_text = "\n\n".join(
            result["text"]
            for result in page_results
            if result["text"].strip()
        ).strip()

        if not raw_text:
            raise ValueError(
                f"No usable text found in PDF: {path}"
            )

        cleaned_text = self._clean_text(
            raw_text
        )

        sanitized_text = self._remove_pii(
            cleaned_text
        )

        if not sanitized_text.strip():
            raise ValueError(
                f"No usable non-PII text remains after cleaning: {path}"
            )

        page_methods = [
            result["method"]
            for result in page_results
        ]

        extraction_method = (
            self._determine_document_method(
                page_methods
            )
        )

        metadata = self._build_metadata(
            pdf_path=path,
            page_count=len(page_results),
            extraction_method=extraction_method,
            page_methods=page_methods,
            ocr_used="ocr" in page_methods,
        )

        return Document(
            page_content=sanitized_text,
            metadata=metadata,
        )

    def load_pdfs(
        self,
        pdf_paths: Iterable[str | Path],
        force_ocr: bool = False,
    ) -> list[Document]:
        """
        Load multiple PDFs.

        Args:
            pdf_paths:
                Iterable of PDF paths.

            force_ocr:
                Force OCR for every page.

        Returns:
            List of LangChain Documents.
        """

        documents: list[Document] = []

        for pdf_path in pdf_paths:

            document = self.load_pdf(
                pdf_path,
                force_ocr=force_ocr,
            )

            documents.append(
                document
            )

        return documents

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        force_ocr: bool = False,
    ) -> list[Document]:
        """
        Load all PDF files from a directory.

        This method is primarily useful for testing.

        IMPORTANT:
        Do not use this method blindly when building the final FAISS
        index. Stage 6 must select only documents marked as
        'reference' in dataset_manifest.csv.
        """

        directory_path = Path(
            directory
        ).resolve()

        if not directory_path.exists():

            raise FileNotFoundError(
                f"Directory does not exist: {directory_path}"
            )

        if not directory_path.is_dir():

            raise NotADirectoryError(
                f"Expected a directory but received: {directory_path}"
            )

        pattern = (
            "**/*.pdf"
            if recursive
            else "*.pdf"
        )

        pdf_paths = sorted(
            directory_path.glob(pattern)
        )

        if not pdf_paths:

            raise FileNotFoundError(
                f"No PDF files found in directory: {directory_path}"
            )

        return self.load_pdfs(
            pdf_paths,
            force_ocr=force_ocr,
        )

    # ==================================================================
    # PAGE EXTRACTION
    # ==================================================================

    def _extract_document_pages(
        self,
        pdf_path: Path,
        force_ocr: bool,
    ) -> list[dict[str, str]]:
        """
        Extract every page from a PDF.

        Each page independently chooses between native extraction
        and OCR.
        """

        results: list[dict[str, str]] = []

        try:

            with pymupdf.open(
                pdf_path
            ) as pdf:

                for page_number in range(
                    1,
                    pdf.page_count + 1,
                ):

                    page = pdf.load_page(
                        page_number - 1
                    )

                    native_text = cast(
                        str,
                        page.get_text(
                            "text",
                            sort=True,
                        ),
                    )

                    native_text = (
                        self._clean_text(
                            native_text
                        )
                    )

                    # --------------------------------------------------
                    # Native extraction path
                    # --------------------------------------------------

                    if (
                        not force_ocr
                        and self._is_usable_native_text(
                            native_text
                        )
                    ):

                        results.append(
                            {
                                "text": (
                                    f"[Page {page_number}]\n"
                                    f"{native_text}"
                                ),
                                "method": "native_pdf",
                            }
                        )

                        continue

                    # --------------------------------------------------
                    # OCR fallback
                    # --------------------------------------------------

                    ocr_text = self._ocr_page(
                        page
                    )

                    ocr_text = (
                        self._clean_text(
                            ocr_text
                        )
                    )

                    if not ocr_text:
                        continue

                    results.append(
                        {
                            "text": (
                                f"[Page {page_number}]\n"
                                f"{ocr_text}"
                            ),
                            "method": "ocr",
                        }
                    )

        except Exception as exc:

            if isinstance(
                exc,
                (
                    FileNotFoundError,
                    ValueError,
                    RuntimeError,
                ),
            ):
                raise

            raise RuntimeError(
                f"Failed to process PDF: {pdf_path}"
            ) from exc

        return results

    # ==================================================================
    # NATIVE TEXT DETECTION
    # ==================================================================

    def _is_usable_native_text(
        self,
        text: str,
    ) -> bool:
        """
        Determine whether native PDF extraction produced useful text.

        Returns False when:
        - text is empty
        - text is extremely short
        - extracted content contains too few alphanumeric characters
        """

        if not text.strip():
            return False

        compact_text = re.sub(
            r"\s+",
            "",
            text,
        )

        if len(compact_text) < self.min_native_chars:
            return False

        alphanumeric_count = sum(
            character.isalnum()
            for character in compact_text
        )

        if not compact_text:
            return False

        alnum_ratio = (
            alphanumeric_count
            / len(compact_text)
        )

        return (
            alnum_ratio
            >= self.min_alnum_ratio
        )

    # ==================================================================
    # OCR
    # ==================================================================

    def _ocr_page(
        self,
        page: pymupdf.Page,
    ) -> str:
        """
        OCR one PDF page using PyMuPDF's integrated OCR support.

        PyMuPDF creates an OCR TextPage using Tesseract internally
        and then allows us to extract the OCR result using the normal
        page.get_text() interface.

        The tessdata path is explicitly converted to str because the
        underlying PyMuPDF binding expects a C string path rather than
        a pathlib.Path object.
        """

        self._ensure_ocr_available()

        try:

            text_page = page.get_textpage_ocr(
                language=self.ocr_lang,
                dpi=self.ocr_dpi,
                full=True,
                tessdata=str(
                    self.tessdata_path
                ),
            )

            extracted_text = page.get_text(
                "text",
                textpage=text_page,
                sort=True,
            )

            if not isinstance(
                extracted_text,
                str,
            ):

                raise TypeError(
                    "PyMuPDF returned a non-text OCR result."
                )

            return extracted_text.strip()

        except Exception as exc:

            raise RuntimeError(
                f"PyMuPDF OCR failed on page "
                f"{page.number + 1 if page.number is not None else 'unknown'}."
            ) from exc

    def _ensure_ocr_available(
        self,
    ) -> None:
        """
        Verify that the configured Tesseract tessdata directory exists.

        This is called only when OCR is actually needed.
        """

        if self.tessdata_path is None:

            raise RuntimeError(
                "OCR is required for this PDF, but the Tesseract "
                "tessdata directory could not be located.\n\n"
                "Install Tesseract OCR and configure TESSDATA_PREFIX "
                "in backend/.env.\n\n"
                "Example:\n"
                r"TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata"
            )

        if not self.tessdata_path.exists():

            raise RuntimeError(
                "The configured Tesseract tessdata directory does "
                "not exist:\n"
                f"{self.tessdata_path}"
            )

        if not self.tessdata_path.is_dir():

            raise RuntimeError(
                "TESSDATA_PREFIX must point to a directory containing "
                "Tesseract language data."
            )

        language_file = (
            self.tessdata_path
            / f"{self.ocr_lang}.traineddata"
        )

        if not language_file.exists():

            raise RuntimeError(
                "Tesseract language data was not found:\n"
                f"{language_file}\n\n"
                f"Configured OCR language: {self.ocr_lang}"
            )

    def _get_tessdata_path(
        self,
    ) -> Path | None:
        """
        Locate the Tesseract tessdata directory.

        Priority:
        1. TESSDATA_PREFIX environment variable.
        2. Common Windows installation locations.

        Returns:
            Path to tessdata directory or None if it cannot be found.
        """

        configured_path = os.getenv(
            "TESSDATA_PREFIX"
        )

        if configured_path:

            path = Path(
                configured_path
            ).expanduser()

            if path.exists():
                return path

            # Keep the configured path even when it doesn't exist.
            # This allows _ensure_ocr_available() to provide a useful
            # error message.
            return path

        common_paths = [
            Path(
                r"C:\Program Files\Tesseract-OCR\tessdata"
            ),
            Path(
                r"C:\Program Files (x86)\Tesseract-OCR\tessdata"
            ),
        ]

        for path in common_paths:

            if path.exists():
                return path

        return None

    # ==================================================================
    # TEXT CLEANING
    # ==================================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        """
        Normalize common PDF extraction artifacts.

        Line structure is preserved because the feedback forms contain
        labels, ratings, and comments whose structure is useful.
        """

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        text = text.replace(
            "\xa0",
            " ",
        )

        cleaned_lines: list[str] = []

        for line in text.split("\n"):

            line = re.sub(
                r"[ \t]+",
                " ",
                line,
            ).strip()

            if line:
                cleaned_lines.append(
                    line
                )

        return "\n".join(
            cleaned_lines
        )

    # ==================================================================
    # PII REMOVAL
    # ==================================================================

    def _remove_pii(
        self,
        text: str,
    ) -> str:
        """
        Remove explicit customer-identifying fields.

        Currently removes:
        - Customer Name
        - Account Number
        - Contact Number

        Handles both:

            Customer Name: John Doe

        and:

            Customer Name
            John Doe
        """

        lines = text.split("\n")

        sanitized_lines: list[str] = []

        skip_next_non_empty = False

        for line in lines:

            normalized = line.strip()

            # ----------------------------------------------------------
            # Remove value following a standalone PII field.
            # ----------------------------------------------------------

            if skip_next_non_empty:

                if normalized:

                    skip_next_non_empty = False
                    continue

                continue

            # ----------------------------------------------------------
            # Check for PII field.
            # ----------------------------------------------------------

            pii_field = (
                self._match_pii_field(
                    normalized
                )
            )

            if pii_field is not None:

                # Example:
                #
                # Customer Name: John Doe
                #
                # Remove the entire line.
                if ":" in normalized:
                    continue

                # Example:
                #
                # Customer Name
                # John Doe
                #
                # Remove both lines.
                skip_next_non_empty = True

                continue

            sanitized_lines.append(
                line
            )

        sanitized_text = "\n".join(
            sanitized_lines
        )

        sanitized_text = re.sub(
            r"\n{3,}",
            "\n\n",
            sanitized_text,
        )

        return sanitized_text.strip()

    def _match_pii_field(
        self,
        line: str,
    ) -> str | None:
        """
        Detect an explicit PII field label.
        """

        for field_name in self.PII_FIELD_NAMES:

            pattern = (
                rf"^\s*"
                rf"{re.escape(field_name)}"
                rf"\s*(?::.*)?$"
            )

            if re.match(
                pattern,
                line,
                flags=re.IGNORECASE,
            ):
                return field_name

        return None

    # ==================================================================
    # METADATA
    # ==================================================================

    def _build_metadata(
        self,
        pdf_path: Path,
        page_count: int,
        extraction_method: str,
        page_methods: list[str],
        ocr_used: bool,
    ) -> dict[str, str | int | bool]:
        """
        Build metadata for the LangChain Document.

        Customer-identifying information is intentionally excluded.
        """

        try:

            relative_path = (
                pdf_path.relative_to(
                    self.project_root
                )
            )

            relative_path_string = (
                relative_path.as_posix()
            )

        except ValueError:

            relative_path_string = (
                pdf_path.as_posix()
            )

        feedback_id = (
            self._generate_feedback_id(
                relative_path_string
            )
        )

        return {
            "source": pdf_path.as_posix(),
            "filename": pdf_path.name,
            "relative_path": relative_path_string,
            "feedback_id": feedback_id,
            "page_count": page_count,
            "document_type": "customer_feedback_pdf",
            "extraction_method": extraction_method,
            "page_extraction_methods": ",".join(
                page_methods
            ),
            "ocr_used": ocr_used,
        }

    @staticmethod
    def _determine_document_method(
        page_methods: list[str],
    ) -> str:
        """
        Determine the overall extraction method.

        Possible values:
        - native_pdf
        - ocr
        - hybrid
        - unknown
        """

        method_set = set(
            page_methods
        )

        if method_set == {"native_pdf"}:
            return "native_pdf"

        if method_set == {"ocr"}:
            return "ocr"

        if (
            "native_pdf" in method_set
            and "ocr" in method_set
        ):
            return "hybrid"

        return "unknown"

    @staticmethod
    def _generate_feedback_id(
        relative_path: str,
    ) -> str:
        """
        Generate a stable feedback ID using the same strategy used
        during Stage 0.

        This allows the document to be matched to dataset_manifest.csv.
        """

        normalized_path = (
            relative_path.replace(
                "\\",
                "/",
            )
        )

        digest = hashlib.sha256(
            normalized_path.encode(
                "utf-8"
            )
        ).hexdigest()[:12]

        return f"FB-{digest}"

    # ==================================================================
    # CONFIGURATION HELPERS
    # ==================================================================

    @staticmethod
    def _get_positive_int_env(
        name: str,
        default: int,
    ) -> int:
        """
        Read a positive integer environment variable.
        """

        value = os.getenv(name)

        if value is None:
            return default

        try:

            parsed = int(value)

        except ValueError as exc:

            raise ValueError(
                f"{name} must be an integer."
            ) from exc

        if parsed <= 0:

            raise ValueError(
                f"{name} must be greater than zero."
            )

        return parsed

    @staticmethod
    def _get_float_env(
        name: str,
        default: float,
    ) -> float:
        """
        Read a floating-point environment variable.
        """

        value = os.getenv(name)

        if value is None:
            return default

        try:

            parsed = float(value)

        except ValueError as exc:

            raise ValueError(
                f"{name} must be a number."
            ) from exc

        if not 0.0 <= parsed <= 1.0:

            raise ValueError(
                f"{name} must be between 0 and 1."
            )

        return parsed

    # ==================================================================
    # VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_pdf_path(
        pdf_path: Path,
    ) -> None:
        """
        Validate the supplied PDF path.
        """

        if not pdf_path.exists():

            raise FileNotFoundError(
                f"PDF file does not exist: {pdf_path}"
            )

        if not pdf_path.is_file():

            raise ValueError(
                f"Expected a file but received: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":

            raise ValueError(
                f"Expected a PDF file but received: {pdf_path}"
            )


# ======================================================================
# Shared service instance
# ======================================================================

_document_service: PDFDocumentService | None = None


def get_document_service() -> PDFDocumentService:
    """
    Return the shared PDFDocumentService instance.

    The service is initialized once and reused.
    """

    global _document_service

    if _document_service is None:

        _document_service = (
            PDFDocumentService()
        )

    return _document_service