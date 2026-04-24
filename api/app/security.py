from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from app.models import User


password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """
    Hash a password using Argon2.

    Never store plain-text passwords.
    """
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a plain-text password against the stored Argon2 hash.
    """
    try:
        return password_hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def get_current_user(request: Request, db: Session) -> User:
    """
    Read the current logged-in user from the signed session cookie.
    """
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    user = db.get(User, int(user_id))
    if not user:
        request.session.clear()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    return user
