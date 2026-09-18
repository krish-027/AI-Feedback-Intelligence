from fastapi import APIRouter


router = APIRouter(
    prefix="/api/upload",
    tags=["Upload"]
)


@router.get("/status")
def upload_status():
    return {
        "message": "Upload route is ready"
    }