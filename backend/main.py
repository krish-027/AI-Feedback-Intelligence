from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings

from backend.routes.upload import router as upload_router
from backend.routes.analysis import router as analysis_router
from backend.routes.search import router as search_router
from backend.routes.analytics import router as analytics_router
from backend.routes.chat import router as chat_router


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-powered customer feedback intelligence system "
        "for analyzing PDF feedback forms."
    ),
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Routers
# --------------------------------------------------

app.include_router(upload_router)
app.include_router(analysis_router)
app.include_router(search_router)
app.include_router(analytics_router)
app.include_router(chat_router)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "AI Customer Feedback Intelligence API",
        "status": "running",
        "version": settings.APP_VERSION,
    }


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
    }