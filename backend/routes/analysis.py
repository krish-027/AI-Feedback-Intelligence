from fastapi import APIRouter


router = APIRouter(
    prefix="/api/analysis",
    tags=["Analysis"]
)


@router.get("/status")
def analysis_status():
    return {
        "message": "Analysis route is ready"
    }