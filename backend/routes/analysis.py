import json
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.api_models import (
    FeedbackAnalysisRequest,
    FeedbackAnalysisResponse,
    FeedbackClassificationResponse,
)
from backend.models.feedback_list_models import (
    FeedbackListItem,
    FeedbackListResponse,
)
from backend.models.individual_analysis_models import (
    IndividualFeedbackAnalysisResponse,
    RetrievedExampleResponse,
)
from backend.services.feedback_database_service import (
    get_feedback_database_service,
)
from backend.services.rag_service import get_rag_service


router = APIRouter(
    prefix="/api/analysis",
    tags=["Analysis"],
)


@router.get("/status")
def analysis_status():
    return {"message": "Analysis route is ready"}


@router.get(
    "",
    response_model=FeedbackListResponse,
)
def list_feedback(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of feedback records to return.",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of feedback records to skip.",
    ),
    db: Session = Depends(get_db),
):
    try:
        database_service = get_feedback_database_service()

        records, total = (
            database_service.get_feedback_records_paginated(
                db=db,
                limit=limit,
                offset=offset,
            )
        )

        items = [
            FeedbackListItem(
                feedback_id=record.feedback_id,
                filename=record.filename,
                category=record.category,
                confidence=record.confidence,
                created_at=record.created_at,
            )
            for record in records
        ]

        return FeedbackListResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feedback listing failed: {str(exc)}",
        ) from exc


@router.post(
    "/classify",
    response_model=FeedbackAnalysisResponse,
)
def classify_feedback(
    request: FeedbackAnalysisRequest,
    db: Session = Depends(get_db),
):
    try:
        rag_service = get_rag_service()

        result = rag_service.classify_feedback(
            feedback=request.feedback,
            include_retrieved_examples=True,
        )

        classification = result["classification"]

        database_service = get_feedback_database_service()

        feedback_id = f"API-{uuid4().hex[:12]}"

        feedback_record = database_service.create_feedback_record(
            db=db,
            feedback_id=feedback_id,
            filename=request.filename,
            feedback_text=result["feedback"],
            category=classification.category,
            confidence=classification.confidence,
            explanation=classification.explanation,
            flagged_keywords=classification.flagged_keywords,
        )

        retrieved_examples = result.get(
            "retrieved_examples",
            [],
        )

        if retrieved_examples:
            database_service.create_retrieval_records(
                db=db,
                feedback_record_id=feedback_record.id,
                retrieved_examples=retrieved_examples,
            )

        return FeedbackAnalysisResponse(
            feedback_id=feedback_id,
            filename=request.filename,
            feedback=result["feedback"],
            classification=FeedbackClassificationResponse(
                category=classification.category,
                confidence=classification.confidence,
                explanation=classification.explanation,
                flagged_keywords=classification.flagged_keywords,
            ),
            retrieved_examples=retrieved_examples,
            retrieved_count=len(retrieved_examples),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feedback classification failed: {str(exc)}",
        ) from exc


@router.get(
    "/{feedback_id}",
    response_model=IndividualFeedbackAnalysisResponse,
)
def get_feedback_analysis(
    feedback_id: str,
    db: Session = Depends(get_db),
):
    try:
        database_service = get_feedback_database_service()

        feedback_record = database_service.get_feedback_record(
            db=db,
            feedback_id=feedback_id,
        )

        if feedback_record is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Feedback record '{feedback_id}' "
                    "was not found."
                ),
            )

        retrieval_records = (
            database_service.get_retrieval_records(
                db=db,
                feedback_record_id=feedback_record.id,
            )
        )

        flagged_keywords = (
            json.loads(feedback_record.flagged_keywords)
            if feedback_record.flagged_keywords
            else []
        )

        retrieved_examples = [
            RetrievedExampleResponse(
                feedback_id=record.retrieved_feedback_id,
                category=record.retrieved_category,
                feedback=record.retrieved_feedback,
                similarity_score=record.similarity_score,
                retrieval_rank=record.retrieval_rank,
            )
            for record in retrieval_records
        ]

        return IndividualFeedbackAnalysisResponse(
            feedback_id=feedback_record.feedback_id,
            filename=feedback_record.filename,
            feedback=feedback_record.feedback_text,
            category=feedback_record.category,
            confidence=feedback_record.confidence,
            explanation=feedback_record.explanation,
            flagged_keywords=flagged_keywords,
            created_at=feedback_record.created_at,
            retrieved_examples=retrieved_examples,
            retrieved_count=len(retrieved_examples),
        )

    except HTTPException:
        raise

    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Stored feedback analysis contains "
                f"invalid data: {str(exc)}"
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Feedback analysis retrieval failed: "
                f"{str(exc)}"
            ),
        ) from exc