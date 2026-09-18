from __future__ import annotations

import hashlib
import os
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment variables from backend/.env when available.
load_dotenv()


class DocumentChunkingService:
    """
    Split LangChain Documents into retrieval-friendly chunks.

    Short documents remain as a single chunk. Longer documents are split
    recursively while maintaining a configurable overlap between chunks.
    """

    DEFAULT_CHUNK_SIZE = 900
    DEFAULT_CHUNK_OVERLAP = 150
    DEFAULT_MIN_CHUNK_LENGTH = 20

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_length: int | None = None,
    ) -> None:
        """
        Initialize the chunking service.

        Parameters
        ----------
        chunk_size:
            Maximum target size of each chunk.

        chunk_overlap:
            Number of characters shared between consecutive chunks.

        min_chunk_length:
            Minimum number of non-whitespace characters required for a chunk
            to be retained.
        """

        self.chunk_size = self._read_int_env(
            "CHUNK_SIZE",
            chunk_size,
            self.DEFAULT_CHUNK_SIZE,
        )

        self.chunk_overlap = self._read_int_env(
            "CHUNK_OVERLAP",
            chunk_overlap,
            self.DEFAULT_CHUNK_OVERLAP,
        )

        self.min_chunk_length = self._read_int_env(
            "MIN_CHUNK_LENGTH",
            min_chunk_length,
            self.DEFAULT_MIN_CHUNK_LENGTH,
        )

        self._validate_configuration()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                "",
            ],
            strip_whitespace=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, document: Document) -> list[Document]:
        """
        Split a single LangChain Document into chunks.

        Existing metadata from the original document is preserved.
        Additional chunk metadata is added to every resulting chunk.
        """

        if not isinstance(document, Document):
            raise TypeError(
                "chunk_document() expects a langchain_core.documents.Document."
            )

        text = document.page_content.strip()

        if not text:
            return []

        if len(text) < self.min_chunk_length:
            return []

        split_documents = self.text_splitter.split_documents([document])

        return self._add_chunk_metadata(
            original_document=document,
            chunks=split_documents,
        )

    def chunk_documents(
        self,
        documents: Iterable[Document],
    ) -> list[Document]:
        """
        Split multiple LangChain Documents into chunks.

        Each input document is processed independently. This is important
        because content from separate PDF pages/documents must not be mixed
        into the same chunk.
        """

        all_chunks: list[Document] = []

        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)

        return all_chunks

    def chunk_pdf_pages(
        self,
        documents: Iterable[Document],
    ) -> list[Document]:
        """
        Alias for chunk_documents().

        This method makes the service easier to understand when Stage 4
        returns one LangChain Document per PDF page.
        """

        return self.chunk_documents(documents)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _add_chunk_metadata(
        self,
        original_document: Document,
        chunks: list[Document],
    ) -> list[Document]:
        """
        Add deterministic chunk metadata to split documents.
        """

        total_chunks = len(chunks)

        result: list[Document] = []

        for index, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)

            # Preserve the original document's metadata explicitly.
            metadata.update(original_document.metadata)

            metadata["chunk_index"] = index
            metadata["chunk_count"] = total_chunks
            metadata["chunk_size"] = len(chunk.page_content)

            metadata["chunk_id"] = self._generate_chunk_id(
                original_document=original_document,
                chunk_index=index,
                chunk_text=chunk.page_content,
            )

            result.append(
                Document(
                    page_content=chunk.page_content.strip(),
                    metadata=metadata,
                )
            )

        return result

    @staticmethod
    def _generate_chunk_id(
        original_document: Document,
        chunk_index: int,
        chunk_text: str,
    ) -> str:
        """
        Generate a deterministic identifier for a chunk.

        The ID is based on the original document metadata, chunk index,
        and chunk content.
        """

        feedback_id = str(
            original_document.metadata.get(
                "feedback_id",
                original_document.metadata.get(
                    "source",
                    "unknown_document",
                ),
            )
        )

        raw_identifier = (
            f"{feedback_id}|{chunk_index}|{chunk_text}"
        )

        return hashlib.sha256(
            raw_identifier.encode("utf-8")
        ).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Validation / configuration
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        """Validate chunking parameters before creating the splitter."""

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if self.min_chunk_length < 0:
            raise ValueError(
                "min_chunk_length cannot be negative."
            )

    @staticmethod
    def _read_int_env(
        variable_name: str,
        explicit_value: int | None,
        default_value: int,
    ) -> int:
        """
        Read an integer configuration value.

        Explicit constructor arguments take priority over environment
        variables. Invalid environment values fall back to the default.
        """

        if explicit_value is not None:
            return explicit_value

        raw_value = os.getenv(variable_name)

        if raw_value is None:
            return default_value

        try:
            return int(raw_value)
        except ValueError:
            return default_value


# ----------------------------------------------------------------------
# Singleton access
# ----------------------------------------------------------------------

_chunking_service: DocumentChunkingService | None = None


def get_chunking_service() -> DocumentChunkingService:
    """
    Return the shared DocumentChunkingService instance.
    """

    global _chunking_service

    if _chunking_service is None:
        _chunking_service = DocumentChunkingService()

    return _chunking_service