#!/usr/bin/env python3
"""
Seed local fake users for end-to-end testing.
"""
from sqlalchemy import select

from app.db import SessionLocal
from app.models import User
from app.security import hash_password

TEST_USERS = [
    ("alice@example.com", "Alice", "Passw0rd!alice"),
    ("bob@example.com", "Bob", "Passw0rd!bob"),
    ("cara@example.com", "Cara", "Passw0rd!cara"),
    ("dan@example.com", "Dan", "Passw0rd!dan"),
]


def main():
    db = SessionLocal()
    try:
        for email, display_name, password in TEST_USERS:
            existing = db.scalar(select(User).where(User.email == email))
            if existing is None:
                user = User(
                    email=email,
                    display_name=display_name,
                    password_hash=hash_password(password),
                )
                db.add(user)
        db.commit()
        print("Seeded test users.")
    finally:
        db.close()


if __name__ == "__main__":
    main()