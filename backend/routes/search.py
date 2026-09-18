from fastapi import APIRouter


router = APIRouter(
    prefix="/api/search",
    tags=["Search"]
)


@router.get("/status")
def search_status():
    return {
        "message": "Search route is ready"
    }