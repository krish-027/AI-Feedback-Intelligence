from datetime import date, datetime, time

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.analytics_models import AnalyticsResponse
from backend.services.analytics_service import get_analytics_service

router = APIRouter(
    prefix="/api/analytics",
    tags=["Analytics"],
)


@router.get("/status")
def analytics_status():
    return {"message": "Analytics route is ready"}


@router.get("", response_model=AnalyticsResponse)
def get_analytics(
    keyword_limit: int = Query(
        default=15,
        ge=1,
        le=100,
        description="Maximum number of flagged keywords to return.",
    ),
    recent_limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of recent feedback records to return.",
    ),
    category: str | None = Query(
        default=None,
        description=(
            "Optional feedback category filter. "
            "Allowed values: Excellent, Good, Need Improvements, Poor."
        ),
    ),
    start_date: date | None = Query(
        default=None,
        description="Include feedback created on or after this date.",
    ),
    end_date: date | None = Query(
        default=None,
        description="Include feedback created on or before this date.",
    ),
    db: Session = Depends(get_db),
):
    try:
        start_datetime = None
        end_datetime = None

        if start_date is not None:
            start_datetime = datetime.combine(
                start_date,
                time.min,
            )

        if end_date is not None:
            end_datetime = datetime.combine(
                end_date,
                time.max,
            )

        analytics_service = get_analytics_service()

        return analytics_service.get_dashboard_analytics(
            db=db,
            keyword_limit=keyword_limit,
            recent_limit=recent_limit,
            category=category,
            start_date=start_datetime,
            end_date=end_datetime,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analytics retrieval failed: {str(exc)}",
        ) from exc