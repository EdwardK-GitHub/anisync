from sqlalchemy import text

from app.db import Base, engine
from app import models  # noqa: F401


def main() -> None:
    """
    Initialize the database schema.

    Important:
    - pgvector extension must exist before creating vector(384) columns.
    - This script is safe to run multiple times.
    """
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")


if __name__ == "__main__":
    main()
