from backend.database import Base, engine  # pyright: ignore[reportAttributeAccessIssue]
from backend.models.database_models import FeedbackRecord


def initialize_database():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    initialize_database()
    print("Database tables initialized successfully.")