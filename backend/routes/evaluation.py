from fastapi import APIRouter, HTTPException

from backend.services.evaluation_report_service import (
    EvaluationReportInvalidError,
    EvaluationReportNotFoundError,
    EvaluationReportService,
)

router = APIRouter(
    prefix="/api/evaluation",
    tags=["Evaluation"],
)

evaluation_service = EvaluationReportService()


def _get_rag_report(split: str) -> dict:
    try:
        return evaluation_service.get_rag_evaluation(split)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except EvaluationReportNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except EvaluationReportInvalidError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


def _get_ablation_report(split: str) -> dict:
    try:
        return evaluation_service.get_rag_ablation(split)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except EvaluationReportNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except EvaluationReportInvalidError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get("/development")
def development_evaluation():
    """
    Return the latest saved development RAG evaluation.

    This endpoint reads the persisted report and does not rerun evaluation.
    """
    return _get_rag_report("development")


@router.get("/final-benchmark")
def final_benchmark_evaluation():
    """
    Return the latest saved final benchmark RAG evaluation.

    This endpoint reads the persisted report and does not rerun evaluation.
    """
    return _get_rag_report("final_benchmark")


@router.get("/rag-vs-no-rag/development")
def development_rag_vs_no_rag():
    """
    Return the saved development RAG-vs-No-RAG comparison.
    """
    return _get_ablation_report("development")


@router.get("/rag-vs-no-rag/final-benchmark")
def final_benchmark_rag_vs_no_rag():
    """
    Return the saved final benchmark RAG-vs-No-RAG comparison.
    """
    return _get_ablation_report("final_benchmark")


@router.get("/summary")
def evaluation_summary():
    """
    Return saved evaluation results for both development and final benchmark.
    """
    return evaluation_service.get_summary()


@router.get("/status")
def evaluation_status():
    return {
        "message": "Evaluation API is ready",
        "supported_splits": [
            "development",
            "final_benchmark",
        ],
    }