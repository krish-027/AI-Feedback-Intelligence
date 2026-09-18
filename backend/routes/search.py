from fastapi import APIRouter, HTTPException

from backend.models.search_models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from backend.services.retriever_service import get_retriever_service


router = APIRouter(
    prefix="/api/search",
    tags=["Search"],
)


@router.get("/status")
def search_status():
    return {
        "message": "Search route is ready",
    }


@router.post(
    "",
    response_model=SearchResponse,
)
def search_feedback(
    request: SearchRequest,
):
    try:
        retriever_service = get_retriever_service()

        documents_with_scores = (
            retriever_service.similarity_search_with_score(
                query=request.query,
                k=request.k,
            )
        )

        results = []

        for document, score in documents_with_scores:
            metadata = document.metadata

            results.append(
                SearchResult(
                    feedback_id=metadata["feedback_id"],
                    category=metadata["category"],
                    split=metadata["split"],
                    feedback=document.page_content.strip(),
                    score=float(score),
                )
            )

        return SearchResponse(
            query=request.query,
            results=results,
            result_count=len(results),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feedback search failed: {str(exc)}",
        ) from exc