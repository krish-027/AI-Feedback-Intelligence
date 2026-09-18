from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.upload_models import UploadResponse
from backend.services.document_service import get_document_service
from backend.services.feedback_database_service import (
    get_feedback_database_service,
)
from backend.services.rag_service import get_rag_service


router = APIRouter(
    prefix="/api/upload",
    tags=["Upload"],
)


@router.get("/status")
def upload_status():
    return {"message": "Upload route is ready"}


@router.post(
    "/pdf",
    response_model=UploadResponse,
)
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    temporary_path: Path | None = None

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty.",
            )

        with NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temporary_file:
            temporary_file.write(file_bytes)
            temporary_path = Path(
                temporary_file.name
            )

        document_service = get_document_service()

        documents = document_service.load_pdf(
            temporary_path
        )

        if not isinstance(documents, list):
            documents = [documents]

        page_contents = []
        for document in documents:
            if isinstance(document, tuple):
                page_content = document[0]
            else:
                page_content = document.page_content

            if page_content:
                page_contents.append(page_content.strip())

        extracted_text = "\n\n".join(page_contents).strip()

        if not extracted_text:
            raise ValueError(
                "No text could be extracted from the PDF."
            )

        rag_service = get_rag_service()

        result = rag_service.classify_feedback(
            feedback=extracted_text,
            include_retrieved_examples=True,
        )

        classification = result["classification"]

        database_service = get_feedback_database_service()

        feedback_id = f"PDF-{uuid4().hex[:12]}"

        feedback_record = (
            database_service.create_feedback_record(
                db=db,
                feedback_id=feedback_id,
                filename=file.filename,
                feedback_text=result["feedback"],
                category=classification.category,
                confidence=classification.confidence,
                explanation=classification.explanation,
                flagged_keywords=(
                    classification.flagged_keywords
                ),
            )
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

        return UploadResponse(
            feedback_id=feedback_id,
            filename=file.filename,
            status="success",
            message=(
                "PDF uploaded, analyzed, and "
                "stored successfully."
            ),
            extracted_text=extracted_text,
            category=classification.category,
            confidence=classification.confidence,
            explanation=classification.explanation,
            flagged_keywords=(
                classification.flagged_keywords
            ),
            retrieved_count=len(
                retrieved_examples
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"PDF processing and analysis failed: "
                f"{str(exc)}"
            ),
        ) from exc

    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass