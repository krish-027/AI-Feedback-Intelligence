import os

from dotenv import load_dotenv


# Load variables from .env
load_dotenv()


class Settings:
    APP_NAME: str = os.getenv(
        "APP_NAME",
        "AI Customer Feedback Intelligence"
    )

    APP_VERSION: str = os.getenv(
        "APP_VERSION",
        "1.0.0"
    )

    GEMINI_API_KEY: str = os.getenv(
        "GEMINI_API_KEY",
        ""
    )

    GEMINI_MODEL: str = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash"
    )

    UPLOAD_DIR: str = os.getenv(
        "UPLOAD_DIR",
        "data/feedback_pdfs"
    )

    EXTRACTED_DIR: str = os.getenv(
        "EXTRACTED_DIR",
        "data/extracted"
    )

    RESULTS_DIR: str = os.getenv(
        "RESULTS_DIR",
        "data/results"
    )

    VECTOR_STORE_DIR: str = os.getenv(
        "VECTOR_STORE_DIR",
        "vector_store"
    )


settings = Settings()