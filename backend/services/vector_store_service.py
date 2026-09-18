from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from backend.services.chunking_service import get_chunking_service
from backend.services.document_service import get_document_service


class SentenceTransformerEmbeddings(Embeddings):
    """
    Small LangChain adapter around SentenceTransformer.

    This allows the existing Stage 3 embedding model to be used by
    LangChain's FAISS vector store.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """
        Load the Sentence Transformer embedding model.
        """

        self.model_name = model_name

        print(
            f"Loading embedding model: {self.model_name}"
        )

        self.model = SentenceTransformer(
            self.model_name
        )

        dimension = self.model.get_embedding_dimension()

        print(
            f"Embedding model loaded successfully "
            f"(dimension={dimension})"
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple documents."""

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for a single query."""

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()

    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""

        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(
                "The embedding model did not report an embedding dimension."
            )

        return dimension


class FAISSVectorStoreService:
    """
    Build, persist, load, and inspect the customer-feedback FAISS index.
    """

    DEFAULT_MANIFEST_PATH = Path(
        "Sample_Data/evaluation/dataset_manifest.csv"
    )

    DEFAULT_VECTOR_STORE_PATH = Path(
        "backend/vector_store/customer_feedback"
    )

    EXPECTED_REFERENCE_SPLIT = "reference"
    EXPECTED_EVALUATION_SPLIT = "evaluation"

    PII_PATTERNS = (
        re.compile(
            r"Customer Name\s*:\s*.*",
            re.IGNORECASE,
        ),
        re.compile(
            r"Account Number\s*:\s*.*",
            re.IGNORECASE,
        ),
        re.compile(
            r"Contact Number\s*:\s*.*",
            re.IGNORECASE,
        ),
    )

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        vector_store_path: str | Path | None = None,
        embedding_model_name: str = (
            SentenceTransformerEmbeddings.DEFAULT_MODEL_NAME
        ),
    ) -> None:
        """
        Initialize the FAISS vector-store service.
        """

        self.manifest_path = Path(
            manifest_path
            if manifest_path is not None
            else self.DEFAULT_MANIFEST_PATH
        )

        self.vector_store_path = Path(
            vector_store_path
            if vector_store_path is not None
            else self.DEFAULT_VECTOR_STORE_PATH
        )

        self.embedding_service = SentenceTransformerEmbeddings(
            model_name=embedding_model_name
        )

        self.document_service = get_document_service()
        self.chunking_service = get_chunking_service()

        self.vector_store: FAISS | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> FAISS:
        """
        Build the FAISS index from reference documents only.

        Evaluation documents are explicitly excluded.
        """

        print("\n=== Stage 6: Building FAISS Vector Store ===")

        manifest_rows = self._load_manifest()

        reference_rows = [
            row
            for row in manifest_rows
            if row["split"].strip().lower()
            == self.EXPECTED_REFERENCE_SPLIT
        ]

        evaluation_rows = [
            row
            for row in manifest_rows
            if row["split"].strip().lower()
            == self.EXPECTED_EVALUATION_SPLIT
        ]

        if not reference_rows:
            raise ValueError(
                "No reference documents were found in "
                "dataset_manifest.csv."
            )

        print(
            f"Manifest records: {len(manifest_rows)}"
        )

        print(
            f"Reference records: {len(reference_rows)}"
        )

        print(
            f"Evaluation records: {len(evaluation_rows)}"
        )

        if len(manifest_rows) != (
            len(reference_rows) + len(evaluation_rows)
        ):
            raise ValueError(
                "Manifest contains records with an unknown split."
            )

        documents = self._load_reference_documents(
            reference_rows
        )

        print(
            f"Reference source documents loaded: "
            f"{len(documents)}"
        )

        chunks = self.chunking_service.chunk_documents(
            documents
        )

        print(
            f"Reference chunks generated: {len(chunks)}"
        )

        if not chunks:
            raise ValueError(
                "No chunks were generated from the "
                "reference documents."
            )

        embedding_documents = self._prepare_embedding_documents(
            chunks
        )

        print(
            "Generating embeddings and building FAISS index..."
        )

        self.vector_store = FAISS.from_documents(
            documents=embedding_documents,
            embedding=self.embedding_service,
        )

        self._save_vector_store()

        print(
            "\nFAISS vector store built successfully."
        )

        print(
            f"Vectors indexed: {len(embedding_documents)}"
        )

        print(
            f"Embedding dimension: "
            f"{self.embedding_service.get_dimension()}"
        )

        print(
            f"Saved to: {self.vector_store_path}"
        )

        return self.vector_store

    def load(self) -> FAISS:
        """
        Load an existing FAISS index from disk.
        """

        index_file = (
            self.vector_store_path / "index.faiss"
        )

        metadata_file = (
            self.vector_store_path / "index.pkl"
        )

        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_file}"
            )

        if not metadata_file.exists():
            raise FileNotFoundError(
                f"FAISS metadata file not found: "
                f"{metadata_file}"
            )

        self.vector_store = FAISS.load_local(
            folder_path=str(
                self.vector_store_path
            ),
            embeddings=self.embedding_service,
            allow_dangerous_deserialization=True,
        )

        return self.vector_store

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """
        Retrieve the k most similar reference chunks.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must contain non-empty text."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        if self.vector_store is None:
            self.load()

        assert self.vector_store is not None

        return self.vector_store.similarity_search(
            query,
            k=k,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve similar reference chunks together with FAISS scores.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must contain non-empty text."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        if self.vector_store is None:
            self.load()

        assert self.vector_store is not None

        return self.vector_store.similarity_search_with_score(
            query,
            k=k,
        )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[dict[str, str]]:
        """
        Load and validate dataset_manifest.csv.
        """

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
                    "dataset_manifest.csv has no header."
                )

            required_columns = {
                "split",
                "category",
            }

            missing = (
                required_columns
                - set(reader.fieldnames)
            )

            if missing:
                raise ValueError(
                    "dataset_manifest.csv is missing "
                    f"required columns: {sorted(missing)}"
                )

            rows = list(reader)

        if not rows:
            raise ValueError(
                "dataset_manifest.csv contains no records."
            )

        return rows

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def _load_reference_documents(
        self,
        reference_rows: list[dict[str, str]],
    ) -> list[Document]:
        """
        Load PDFs belonging to the reference split.

        The manifest's relative_path is treated as the authoritative
        path to the source PDF.
        """

        documents: list[Document] = []

        for index, row in enumerate(
            reference_rows,
            start=1,
        ):
            relative_path = (
                row.get("relative_path", "")
                .strip()
            )

            if not relative_path:
                raise ValueError(
                    "A reference manifest row has an "
                    "empty relative_path."
                )

            pdf_path = Path(relative_path)

            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"Reference PDF not found: {pdf_path}"
                )

            category = row["category"].strip()

            feedback_id = (
                row.get("feedback_id", "")
                .strip()
            )

            loaded = self.document_service.load_pdf(
                pdf_path
            )

            if isinstance(loaded, Document):
                loaded_documents = [loaded]
            else:
                loaded_documents = list(loaded)

            for document in loaded_documents:
                metadata = dict(document.metadata)

                # Manifest labels are authoritative for Stage 6.
                metadata["category"] = category
                metadata["split"] = (
                    self.EXPECTED_REFERENCE_SPLIT
                )

                if feedback_id:
                    metadata["feedback_id"] = feedback_id

                metadata["manifest_relative_path"] = (
                    relative_path
                )

                documents.append(
                    Document(
                        page_content=document.page_content,
                        metadata=metadata,
                    )
                )

            if index % 10 == 0:
                print(
                    f"Loaded {index}/"
                    f"{len(reference_rows)} reference PDFs..."
                )

        return documents

    # ------------------------------------------------------------------
    # Embedding preparation
    # ------------------------------------------------------------------

    def _prepare_embedding_documents(
        self,
        chunks: list[Document],
    ) -> list[Document]:
        """
        Prepare chunks before embedding.

        PII is removed from page_content used for embeddings.

        The category and other useful metadata remain available to the
        vector store so retrieved examples can be shown to the RAG
        classifier.
        """

        prepared: list[Document] = []

        for chunk in chunks:
            cleaned_text = (
                self._remove_pii_for_embedding(
                    chunk.page_content
                )
            )

            cleaned_text = cleaned_text.strip()

            if not cleaned_text:
                continue

            metadata = dict(chunk.metadata)

            metadata["embedding_text_hash"] = (
                hashlib.sha256(
                    cleaned_text.encode("utf-8")
                ).hexdigest()[:16]
            )

            metadata["embedding_model"] = (
                self.embedding_service.model_name
            )

            metadata["embedding_dimension"] = (
                self.embedding_service.get_dimension()
            )

            prepared.append(
                Document(
                    page_content=cleaned_text,
                    metadata=metadata,
                )
            )

        return prepared

    def _remove_pii_for_embedding(
        self,
        text: str,
    ) -> str:
        """
        Remove known customer PII fields from embedding text.

        This prevents customer-specific identity information from
        becoming part of the semantic vector representation.
        """

        cleaned = text

        for pattern in self.PII_PATTERNS:
            cleaned = pattern.sub(
                "",
                cleaned,
            )

        return cleaned

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_vector_store(self) -> None:
        """
        Persist the FAISS index and its metadata locally.
        """

        if self.vector_store is None:
            raise RuntimeError(
                "Cannot save an empty vector store."
            )

        self.vector_store_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vector_store.save_local(
            str(self.vector_store_path)
        )


# ----------------------------------------------------------------------
# Singleton access
# ----------------------------------------------------------------------

_vector_store_service: (
    FAISSVectorStoreService | None
) = None


def get_vector_store_service() -> FAISSVectorStoreService:
    """
    Return the shared FAISSVectorStoreService instance.
    """

    global _vector_store_service

    if _vector_store_service is None:
        _vector_store_service = (
            FAISSVectorStoreService()
        )

    return _vector_store_service