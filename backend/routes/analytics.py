from fastapi import APIRouter


router = APIRouter(
    prefix="/api/analytics",
    tags=["Analytics"]
)


@router.get("/status")
def analytics_status():
    return {
        "message": "Analytics route is ready"
    }