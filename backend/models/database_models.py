from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class FeedbackRecord(Base):
    __tablename__ = "feedback_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    feedback_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    feedback_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    explanation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    flagged_keywords: Mapped[str] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    retrieval_records: Mapped[list["RetrievalRecord"]] = relationship(
        back_populates="feedback_record",
        cascade="all, delete-orphan",
    )


class RetrievalRecord(Base):
    __tablename__ = "retrieval_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    feedback_record_id: Mapped[int] = mapped_column(
        ForeignKey("feedback_records.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    retrieved_feedback_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    retrieved_category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    similarity_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    retrieved_feedback: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    retrieval_rank: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    feedback_record: Mapped["FeedbackRecord"] = relationship(
        back_populates="retrieval_records",
    )