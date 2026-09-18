from fastapi import APIRouter


router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"]
)


@router.get("/status")
def chat_status():
    return {
        "message": "Chat route is ready"
    }