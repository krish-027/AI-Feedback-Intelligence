import hashlib

import numpy as np
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from backend.main import app
from backend.models.classification import FeedbackClassification
from backend.services import embedding_service
from backend.services import vector_store_service
from backend.services.gemini_service import GeminiService
import backend.services.gemini_service as gemini_service_module
import backend.services.rag_chain_service as rag_chain_service_module


class OfflineEmbeddingModel:
    """
    Deterministic local replacement for SentenceTransformer
    during FastAPI/LangChain integration tests.

    The production FAISS index uses 384-dimensional embeddings,
    so the test implementation produces vectors of the same size.
    """

    DIMENSION = 384

    def __init__(self, *args, **kwargs):
        pass

    def get_embedding_dimension(self):
        return self.DIMENSION

    def encode(
        self,
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        **kwargs,
    ):
        is_single = isinstance(texts, str)

        if is_single:
            texts = [texts]

        vectors = []

        for text in texts:
            vector = np.zeros(
                self.DIMENSION,
                dtype=np.float32,
            )

            digest = hashlib.sha256(
                text.encode("utf-8")
            ).digest()

            for index, value in enumerate(digest):
                vector[index % self.DIMENSION] += (
                    value / 255.0
                )

            if normalize_embeddings:
                norm = np.linalg.norm(vector)

                if norm > 0:
                    vector = vector / norm

            vectors.append(vector)

        result = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if is_single:
            return result[0]

        return result


def offline_classification(_input):
    """
    Return deterministic structured classification for integration tests.
    """

    return FeedbackClassification(
        category="Excellent",
        confidence=0.95,
        explanation=(
            "The feedback describes a strongly positive "
            "overall experience, with helpful staff and "
            "smooth service."
        ),
        flagged_keywords=[
            "excellent",
            "helpful",
            "smooth",
        ],
    )


class OfflineGeminiLLM:
    """
    Minimal fake Gemini LLM for integration testing.

    The returned structured model is a real LangChain Runnable,
    so it can participate in the LCEL chain.
    """

    def with_structured_output(
        self,
        schema,
        method=None,
        **kwargs,
    ):
        return RunnableLambda(
            offline_classification
        )


def offline_get_gemini_service(
    api_key=None,
    model_name=None,
):
    """
    Return a GeminiService backed by the deterministic
    local test LLM instead of the real Gemini API.
    """

    return GeminiService(
        api_key="offline-test-key",
        model_name="offline-test-model",
        llm=OfflineGeminiLLM(),
    )


# ----------------------------------------------------------------------
# Replace external dependencies for this integration-test module.
# ----------------------------------------------------------------------

# Prevent SentenceTransformer from contacting Hugging Face.
embedding_service.SentenceTransformer = (
    OfflineEmbeddingModel
)

# vector_store_service has its own direct SentenceTransformer import,
# so it must be patched separately.
vector_store_service.SentenceTransformer = (
    OfflineEmbeddingModel
)

# Reset the embedding singleton so it cannot retain a real model.
embedding_service._embedding_service = None

# Reset the FAISS vector-store singleton if it already exists.
if hasattr(
    vector_store_service,
    "_vector_store_service",
):
    vector_store_service._vector_store_service = None


# GeminiService was imported directly into rag_chain_service.py,
# therefore both module references need to be patched.
gemini_service_module.get_gemini_service = (
    offline_get_gemini_service
)

rag_chain_service_module.get_gemini_service = (
    offline_get_gemini_service
)


client = TestClient(app)


def test_analysis_classify_endpoint_exists():
    response = client.post(
        "/api/analysis/classify",
        json={
            "feedback": (
                "The service was excellent. "
                "The staff were helpful and the "
                "overall experience was very smooth."
            ),
            "filename": "integration_test.txt",
        },
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert "feedback_id" in data
    assert "filename" in data
    assert "feedback" in data
    assert "classification" in data
    assert "retrieved_examples" in data
    assert "retrieved_count" in data

    classification = data["classification"]

    assert "category" in classification
    assert "confidence" in classification
    assert "explanation" in classification
    assert "flagged_keywords" in classification

    assert classification["category"] in {
        "Excellent",
        "Good",
        "Need Improvements",
        "Poor",
    }

    assert 0.0 <= classification["confidence"] <= 1.0

    assert isinstance(
        classification["explanation"],
        str,
    )

    assert isinstance(
        classification["flagged_keywords"],
        list,
    )

    assert (
        data["retrieved_count"]
        == len(data["retrieved_examples"])
    )


def test_search_endpoint_exposes_retrieval():
    response = client.post(
        "/api/search",
        json={
            "query": (
                "excellent customer service "
                "helpful staff"
            ),
            "k": 4,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "query" in data
    assert "results" in data
    assert "result_count" in data

    assert isinstance(
        data["results"],
        list,
    )

    assert (
        data["result_count"]
        == len(data["results"])
    )

    for result in data["results"]:
        assert "feedback_id" in result
        assert "category" in result
        assert "split" in result
        assert "feedback" in result
        assert "score" in result

        assert isinstance(
            result["feedback"],
            str,
        )

        assert isinstance(
            result["score"],
            (int, float),
        )


def test_evaluation_api_exposes_saved_rag_report():
    response = client.get(
        "/api/evaluation/development"
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["evaluation_split"]
        == "development"
    )

    assert "evaluation_timestamp" in data
    assert "manifest_path" in data
    assert "total_evaluation_records" in data
    assert "cases" in data

    assert isinstance(
        data["cases"],
        list,
    )


def test_evaluation_api_exposes_final_benchmark():
    response = client.get(
        "/api/evaluation/final-benchmark"
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["evaluation_split"]
        == "final_benchmark"
    )

    assert "evaluation_timestamp" in data
    assert "manifest_path" in data
    assert "total_evaluation_records" in data
    assert "cases" in data

    assert isinstance(
        data["cases"],
        list,
    )


def test_rag_vs_no_rag_api_exposes_development_report():
    response = client.get(
        "/api/evaluation/rag-vs-no-rag/development"
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["evaluation_split"]
        == "development"
    )

    assert "rag_metrics" in data
    assert "no_rag_metrics" in data
    assert "paired_comparison" in data
    assert "accuracy_difference" in data
    assert "macro_precision_difference" in data
    assert "macro_recall_difference" in data
    assert "macro_f1_difference" in data
    assert "weighted_f1_difference" in data
    assert "results" in data

    assert isinstance(
        data["results"],
        list,
    )


def test_rag_vs_no_rag_api_exposes_final_benchmark():
    response = client.get(
        "/api/evaluation/rag-vs-no-rag/final-benchmark"
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["evaluation_split"]
        == "final_benchmark"
    )

    assert "rag_metrics" in data
    assert "no_rag_metrics" in data
    assert "paired_comparison" in data
    assert "accuracy_difference" in data
    assert "macro_precision_difference" in data
    assert "macro_recall_difference" in data
    assert "macro_f1_difference" in data
    assert "weighted_f1_difference" in data
    assert "results" in data

    assert isinstance(
        data["results"],
        list,
    )


def test_health_endpoint_still_works():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_root_endpoint_still_works():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"