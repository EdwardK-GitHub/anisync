import datetime as dt

from sqlalchemy import select

from app.db import SessionLocal
from app.models import User
from app.security import hash_password


DEMO_USERS = [
    ("host@example.com", "Host Edward", "AniSyncDemo123!"),
    ("kai@example.com", "Kai", "AniSyncDemo123!"),
    ("mina@example.com", "Mina", "AniSyncDemo123!"),
    ("theo@example.com", "Theo", "AniSyncDemo123!"),
]


def main() -> None:
    """
    Seed demo users only.

    We intentionally do not seed a room, so the local test can verify:
    - host creates room
    - other users join by code
    - each user submits text
    - host computes
    - everyone votes
    """
    db = SessionLocal()
    try:
        for email, display_name, password in DEMO_USERS:
            existing = db.scalar(select(User).where(User.email == email))
            if existing:
                print(f"User already exists: {email}")
                continue

            db.add(
                User(
                    email=email,
                    display_name=display_name,
                    password_hash=hash_password(password),
                )
            )
            print(f"Created demo user: {email}")

        db.commit()
        print("Demo users seeded.")
        print("Password for all demo users: AniSyncDemo123!")
    finally:
        db.close()


if __name__ == "__main__":
    main()
