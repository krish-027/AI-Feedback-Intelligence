from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from backend.services.chunking_service import (
    get_chunking_service,
)
from backend.services.document_service import (
    get_document_service,
)


class SentenceTransformerEmbeddings(Embeddings):

    DEFAULT_MODEL_NAME = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:

        self.model_name = model_name

        self.model = SentenceTransformer(
            self.model_name
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

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

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()

    def get_dimension(self) -> int:

        dimension = (
            self.model.get_embedding_dimension()
        )

        if dimension is None:
            raise RuntimeError(
                "Embedding model did not report "
                "an embedding dimension."
            )

        return dimension


class FAISSVectorStoreService:

    DEFAULT_MANIFEST_PATH = Path(
        "Sample_Data/evaluation/dataset_manifest.csv"
    )

    DEFAULT_VECTOR_STORE_PATH = Path(
        "backend/vector_store/customer_feedback"
    )

    REFERENCE_SPLIT = "reference"

    NON_REFERENCE_SPLITS = {
        "development",
        "final_benchmark",
    }

    VALID_SPLITS = {
        "reference",
        "development",
        "final_benchmark",
    }

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

        self.embedding_service = (
            SentenceTransformerEmbeddings(
                model_name=embedding_model_name
            )
        )

        self.document_service = (
            get_document_service()
        )

        self.chunking_service = (
            get_chunking_service()
        )

        self.vector_store: FAISS | None = None

    # ---------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------

    def _load_manifest(
        self,
    ) -> list[dict[str, str]]:

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                "Dataset manifest not found: "
                f"{self.manifest_path}"
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            if reader.fieldnames is None:
                raise ValueError(
                    "dataset_manifest.csv "
                    "has no header."
                )

            required_columns = {
                "feedback_id",
                "filename",
                "relative_path",
                "category",
                "split",
            }

            missing_columns = (
                required_columns
                - set(reader.fieldnames)
            )

            if missing_columns:
                raise ValueError(
                    "dataset_manifest.csv is missing "
                    f"required columns: "
                    f"{sorted(missing_columns)}"
                )

            rows = list(reader)

        if not rows:
            raise ValueError(
                "dataset_manifest.csv contains "
                "no records."
            )

        for row in rows:

            split = row["split"].strip()

            if split not in self.VALID_SPLITS:
                raise ValueError(
                    f"Invalid dataset split "
                    f"'{split}' for "
                    f"{row['filename']}."
                )

        return rows

    def get_records_by_split(
        self,
        split: str,
    ) -> list[dict[str, str]]:

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected one of "
                f"{sorted(self.VALID_SPLITS)}"
            )

        return [
            row
            for row in self._load_manifest()
            if row["split"].strip()
            == split
        ]

    # ---------------------------------------------------------
    # Build
    # ---------------------------------------------------------

    def build(self) -> FAISS:

        print(
            "\n=== Building FAISS Vector Store ==="
        )

        manifest_rows = (
            self._load_manifest()
        )

        reference_rows = [
            row
            for row in manifest_rows
            if row["split"].strip()
            == self.REFERENCE_SPLIT
        ]

        development_rows = [
            row
            for row in manifest_rows
            if row["split"].strip()
            == "development"
        ]

        benchmark_rows = [
            row
            for row in manifest_rows
            if row["split"].strip()
            == "final_benchmark"
        ]

        if len(reference_rows) != 60:
            raise ValueError(
                "The reference split must contain "
                f"exactly 60 records. Found "
                f"{len(reference_rows)}."
            )

        if len(development_rows) != 20:
            raise ValueError(
                "The development split must contain "
                f"exactly 20 records. Found "
                f"{len(development_rows)}."
            )

        if len(benchmark_rows) != 20:
            raise ValueError(
                "The final_benchmark split must contain "
                f"exactly 20 records. Found "
                f"{len(benchmark_rows)}."
            )

        if (
            len(reference_rows)
            + len(development_rows)
            + len(benchmark_rows)
            != 100
        ):
            raise ValueError(
                "Dataset split counts do not sum "
                "to 100."
            )

        print(
            f"Reference records      : "
            f"{len(reference_rows)}"
        )

        print(
            f"Development records    : "
            f"{len(development_rows)}"
        )

        print(
            f"Final benchmark records: "
            f"{len(benchmark_rows)}"
        )

        documents = (
            self._load_reference_documents(
                reference_rows
            )
        )

        if not documents:
            raise ValueError(
                "No reference documents were loaded."
            )

        print(
            f"Reference source documents: "
            f"{len(documents)}"
        )

        chunks = (
            self.chunking_service.chunk_documents(
                documents
            )
        )

        if not chunks:
            raise ValueError(
                "No chunks were generated from "
                "reference documents."
            )

        print(
            f"Reference chunks: "
            f"{len(chunks)}"
        )

        embedding_documents = (
            self._prepare_embedding_documents(
                chunks
            )
        )

        if not embedding_documents:
            raise ValueError(
                "No documents remained after "
                "embedding preparation."
            )

        self.vector_store = (
            FAISS.from_documents(
                documents=embedding_documents,
                embedding=self.embedding_service,
            )
        )

        self._save_vector_store()

        print(
            "\nFAISS vector store built successfully."
        )

        print(
            f"Indexed chunks: "
            f"{len(embedding_documents)}"
        )

        print(
            f"Embedding dimension: "
            f"{self.embedding_service.get_dimension()}"
        )

        print(
            f"Saved to: "
            f"{self.vector_store_path}"
        )

        return self.vector_store

    # ---------------------------------------------------------
    # Reference document loading
    # ---------------------------------------------------------

    def _load_reference_documents(
        self,
        reference_rows: list[dict[str, str]],
    ) -> list[Document]:

        documents = []

        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        for row in reference_rows:

            relative_path = Path(
                row["relative_path"]
                .strip()
            )

            pdf_path = (
                project_root
                / relative_path
            )

            if not pdf_path.exists():
                raise FileNotFoundError(
                    "Reference PDF not found: "
                    f"{pdf_path}"
                )

            loaded = (
                self.document_service.load_pdf(
                    pdf_path
                )
            )

            if not isinstance(
                loaded,
                list,
            ):
                loaded = [loaded]

            for document in loaded:

                if not isinstance(document, Document):
                    raise TypeError(
                        "Document service returned an unsupported "
                        f"type: {type(document).__name__}"
                    )

                metadata = dict(
                    document.metadata
                )

                metadata.update(
                    {
                        "feedback_id": (
                            row["feedback_id"]
                        ),
                        "filename": (
                            row["filename"]
                        ),
                        "category": (
                            row["category"]
                        ),
                        "split": (
                            row["split"]
                        ),
                        "manifest_relative_path": (
                            row["relative_path"]
                        ),
                    }
                )

                documents.append(
                    Document(
                        page_content=(
                            document.page_content
                        ),
                        metadata=metadata,
                    )
                )

        return documents

    # ---------------------------------------------------------
    # Embedding preparation
    # ---------------------------------------------------------

    def _prepare_embedding_documents(
        self,
        chunks: list[Document],
    ) -> list[Document]:

        prepared = []

        for chunk in chunks:

            cleaned_text = (
                self._remove_pii_for_embedding(
                    chunk.page_content
                )
                .strip()
            )

            if not cleaned_text:
                continue

            metadata = dict(
                chunk.metadata
            )

            if metadata.get("split") != (
                self.REFERENCE_SPLIT
            ):
                raise ValueError(
                    "Attempted to embed a non-reference "
                    f"document: "
                    f"split={metadata.get('split')}"
                )

            metadata[
                "embedding_text_hash"
            ] = hashlib.sha256(
                cleaned_text.encode("utf-8")
            ).hexdigest()[:16]

            metadata[
                "embedding_model"
            ] = self.embedding_service.model_name

            metadata[
                "embedding_dimension"
            ] = (
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

        cleaned = text

        for pattern in self.PII_PATTERNS:
            cleaned = pattern.sub(
                "",
                cleaned,
            )

        return cleaned

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def _save_vector_store(
        self,
    ) -> None:

        if self.vector_store is None:
            raise RuntimeError(
                "Cannot save an empty vector store."
            )

        self.vector_store_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vector_store.save_local(
            str(
                self.vector_store_path
            )
        )

    def load(
        self,
    ) -> FAISS:

        index_file = (
            self.vector_store_path
            / "index.faiss"
        )

        metadata_file = (
            self.vector_store_path
            / "index.pkl"
        )

        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{index_file}"
            )

        if not metadata_file.exists():
            raise FileNotFoundError(
                "FAISS metadata file not found: "
                f"{metadata_file}"
            )

        self.vector_store = (
            FAISS.load_local(
                folder_path=str(
                    self.vector_store_path
                ),
                embeddings=self.embedding_service,
                allow_dangerous_deserialization=True,
            )
        )

        return self.vector_store

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    def _get_loaded_store(
        self,
    ) -> FAISS:

        if self.vector_store is None:
            self.load()

        assert self.vector_store is not None

        return self.vector_store

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[Document]:

        if not query or not query.strip():
            raise ValueError(
                "query must contain "
                "non-empty text."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        vector_store = (
            self._get_loaded_store()
        )

        results = (
            vector_store.similarity_search(
                query,
                k=k,
            )
        )

        self._validate_reference_results(
            results
        )

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:

        if not query or not query.strip():
            raise ValueError(
                "query must contain "
                "non-empty text."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        vector_store = (
            self._get_loaded_store()
        )

        results = (
            vector_store
            .similarity_search_with_score(
                query,
                k=k,
            )
        )

        self._validate_reference_results(
            [
                document
                for document, _ in results
            ]
        )

        return results

    @staticmethod
    def _validate_reference_results(
        documents: list[Document],
    ) -> None:

        for document in documents:

            split = document.metadata.get(
                "split"
            )

            if split != "reference":
                raise ValueError(
                    "Vector store returned a "
                    "non-reference document: "
                    f"split={split}"
                )

    def get_index_document_count(
        self,
    ) -> int:

        vector_store = (
            self._get_loaded_store()
        )

        return (
            vector_store.index.ntotal
        )


_vector_store_service = None


def get_vector_store_service():
    global _vector_store_service

    if _vector_store_service is None:
        _vector_store_service = (
            FAISSVectorStoreService()
        )

    return _vector_store_service