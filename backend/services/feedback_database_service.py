import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.database_models import FeedbackRecord, RetrievalRecord


class FeedbackDatabaseService:

    def create_feedback_record(
        self,
        db: Session,
        feedback_id: str,
        filename: str,
        feedback_text: str,
        category: str,
        confidence: float,
        explanation: str,
        flagged_keywords: list[str],
    ):
        record = FeedbackRecord(
            feedback_id=feedback_id,
            filename=filename,
            feedback_text=feedback_text,
            category=category,
            confidence=confidence,
            explanation=explanation,
            flagged_keywords=json.dumps(flagged_keywords),
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return record

    def create_retrieval_records(
        self,
        db: Session,
        feedback_record_id: int,
        retrieved_examples: list[dict],
    ):
        records = []

        for rank, example in enumerate(retrieved_examples, start=1):
            record = RetrievalRecord(
                feedback_record_id=feedback_record_id,
                retrieved_feedback_id=example["feedback_id"],
                retrieved_category=example["category"],
                similarity_score=float(example.get("score", 0.0)),
                retrieved_feedback=example["feedback"],
                retrieval_rank=rank,
            )

            db.add(record)
            records.append(record)

        db.commit()

        for record in records:
            db.refresh(record)

        return records

    def get_feedback_record(
        self,
        db: Session,
        feedback_id: str,
    ):
        statement = select(FeedbackRecord).where(
            FeedbackRecord.feedback_id == feedback_id
        )

        return db.execute(statement).scalar_one_or_none()

    def get_retrieval_records(
        self,
        db: Session,
        feedback_record_id: int,
    ):
        statement = (
            select(RetrievalRecord)
            .where(
                RetrievalRecord.feedback_record_id == feedback_record_id
            )
            .order_by(RetrievalRecord.retrieval_rank)
        )

        return list(db.execute(statement).scalars().all())

    def get_all_feedback_records(
        self,
        db: Session,
    ):
        statement = select(FeedbackRecord).order_by(
            FeedbackRecord.created_at.desc()
        )

        return list(db.execute(statement).scalars().all())

    def get_feedback_records_paginated(
        self,
        db: Session,
        limit: int,
        offset: int,
    ):
        if limit < 1:
            raise ValueError("limit must be at least 1.")

        if offset < 0:
            raise ValueError("offset cannot be negative.")

        records = (
            db.query(FeedbackRecord)
            .order_by(FeedbackRecord.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        total = db.query(FeedbackRecord).count()

        return records, total


def get_feedback_database_service():
    return FeedbackDatabaseService()