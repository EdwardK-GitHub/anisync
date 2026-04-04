from fastapi import Request
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import User

password_hasher = PasswordHash.recommended()
SESSION_USER_ID_KEY = "user_id"


def hash_password(password: str) -> str:
    """Hash a plaintext password using a modern recommended algorithm."""
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a stored hash."""
    return password_hasher.verify(password, password_hash)


def login_user(request: Request, user: User) -> None:
    """Store the logged-in user ID in the signed session."""
    request.session[SESSION_USER_ID_KEY] = user.id


def logout_user(request: Request) -> None:
    """Clear the session."""
    request.session.clear()


def get_current_user(request: Request, db: Session) -> User | None:
    """Return the current logged-in user or None."""
    user_id = request.session.get(SESSION_USER_ID_KEY)
    if not user_id:
        return None
    stmt = select(User).where(User.id == user_id)
    return db.scalar(stmt)