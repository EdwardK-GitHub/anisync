from pathlib import Path
import secrets
import string

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from app.db import get_db
from app.embeddings import embed_text
from app.models import Room, RoomMember, RoomQuerySubmission, User
from app.recommendations import compute_recommendations_for_room
from app.security import get_current_user, hash_password, login_user, logout_user, verify_password
from app.settings import settings

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AniSync")
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    same_site="lax",
    https_only=settings.environment == "production",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def render(
    request: Request,
    template_name: str,
    db: Session,
    **context,
) -> HTMLResponse:
    """
    Central template helper so every page gets current_user automatically.
    """
    context["current_user"] = get_current_user(request, db)
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context=context,
    )


def redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url=url, status_code=303)


def require_user(request: Request, db: Session) -> User:
    user = get_current_user(request, db)
    if user is None:
        raise RuntimeError("Unauthenticated access")
    return user


def generate_room_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user is None:
        return redirect("/login")
    return redirect("/dashboard")


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user:
        return redirect("/dashboard")
    return render(request, "register.html", db, error=None)


@app.post("/register", response_class=HTMLResponse)
def register_user(
    request: Request,
    email: str = Form(...),
    display_name: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    if db.scalar(select(User).where(User.email == email)):
        return render(request, "register.html", db, error="That email is already registered.")

    user = User(
        email=email.strip().lower(),
        display_name=display_name.strip(),
        password_hash=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    login_user(request, user)
    return redirect("/dashboard")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user:
        return redirect("/dashboard")
    return render(request, "login.html", db, error=None)


@app.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.scalar(select(User).where(User.email == email.strip().lower()))
    if user is None or not verify_password(password, user.password_hash):
        return render(request, "login.html", db, error="Invalid email or password.")

    login_user(request, user)
    return redirect("/dashboard")


@app.post("/logout")
def logout_route(request: Request):
    logout_user(request)
    return redirect("/login")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user is None:
        return redirect("/login")

    room_ids = [member.room_id for member in db.scalars(select(RoomMember).where(RoomMember.user_id == user.id)).all()]
    rooms = []
    if room_ids:
        rooms = list(db.scalars(select(Room).where(Room.id.in_(room_ids)).order_by(Room.created_at.desc())).all())

    return render(request, "dashboard.html", db, rooms=rooms, error=None)


@app.get("/rooms/new", response_class=HTMLResponse)
def create_room_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user is None:
        return redirect("/login")
    return render(request, "create_room.html", db, error=None)


@app.post("/rooms/new", response_class=HTMLResponse)
def create_room(
    request: Request,
    title: str = Form(...),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if user is None:
        return redirect("/login")

    title = title.strip()
    if not title:
        return render(request, "create_room.html", db, error="Room title is required.")

    code = generate_room_code()
    while db.scalar(select(Room).where(Room.code == code)) is not None:
        code = generate_room_code()

    room = Room(title=title, code=code, host_user_id=user.id, status="open")
    db.add(room)
    db.commit()
    db.refresh(room)

    membership = RoomMember(room_id=room.id, user_id=user.id)
    db.add(membership)
    db.commit()

    return redirect(f"/rooms/{room.code}")


@app.post("/rooms/join")
def join_room(
    request: Request,
    code: str = Form(...),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if user is None:
        return redirect("/login")

    room = db.scalar(select(Room).where(Room.code == code.strip().upper()))
    if room is None:
        return render(request, "dashboard.html", db, rooms=[], error="Room code not found.")

    membership = db.scalar(
        select(RoomMember).where(RoomMember.room_id == room.id, RoomMember.user_id == user.id)
    )
    if membership is None:
        db.add(RoomMember(room_id=room.id, user_id=user.id))
        db.commit()

    return redirect(f"/rooms/{room.code}")


@app.get("/rooms/{code}", response_class=HTMLResponse)
def room_page(
    request: Request,
    code: str,
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if user is None:
        return redirect("/login")

    room = db.scalar(select(Room).where(Room.code == code.upper()))
    if room is None:
        return redirect("/dashboard")

    # Auto-join if the user visits directly and isn't already a member.
    membership = db.scalar(
        select(RoomMember).where(RoomMember.room_id == room.id, RoomMember.user_id == user.id)
    )
    if membership is None:
        db.add(RoomMember(room_id=room.id, user_id=user.id))
        db.commit()

    members = list(
        db.scalars(
            select(RoomMember)
            .where(RoomMember.room_id == room.id)
        ).all()
    )
    member_users = []
    for member in members:
        member_user = db.scalar(select(User).where(User.id == member.user_id))
        if member_user:
            member_users.append(member_user)

    submission = db.scalar(
        select(RoomQuerySubmission).where(
            RoomQuerySubmission.room_id == room.id,
            RoomQuerySubmission.user_id == user.id,
        )
    )

    all_submissions = list(
        db.scalars(select(RoomQuerySubmission).where(RoomQuerySubmission.room_id == room.id)).all()
    )
    submitted_user_ids = {row.user_id for row in all_submissions}

    can_compute = room.host_user_id == user.id and len(submitted_user_ids) >= 2 and room.status != "computing"

    return render(
        request,
        "room.html",
        db,
        room=room,
        members=member_users,
        submission=submission,
        submitted_user_ids=submitted_user_ids,
        can_compute=can_compute,
        error=None,
    )


@app.post("/rooms/{code}/submit", response_class=HTMLResponse)
def submit_room_preference(
    request: Request,
    code: str,
    query_text: str = Form(...),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if user is None:
        return redirect("/login")

    room = db.scalar(select(Room).where(Room.code == code.upper()))
    if room is None:
        return redirect("/dashboard")

    query_text = query_text.strip()
    if not query_text:
        return render(
            request,
            "room.html",
            db,
            room=room,
            members=[],
            submission=None,
            submitted_user_ids=set(),
            can_compute=False,
            error="Please enter a preference before submitting.",
        )

    query_embedding = embed_text(query_text)

    submission = db.scalar(
        select(RoomQuerySubmission).where(
            RoomQuerySubmission.room_id == room.id,
            RoomQuerySubmission.user_id == user.id,
        )
    )

    if submission is None:
        submission = RoomQuerySubmission(
            room_id=room.id,
            user_id=user.id,
            query_text=query_text,
            query_embedding=query_embedding,
        )
        db.add(submission)
    else:
        submission.query_text = query_text
        submission.query_embedding = query_embedding

    db.commit()
    return redirect(f"/rooms/{room.code}")


@app.post("/rooms/{code}/compute")
def compute_room_results(
    request: Request,
    code: str,
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if user is None:
        return redirect("/login")

    room = db.scalar(select(Room).where(Room.code == code.upper()))
    if room is None:
        return redirect("/dashboard")

    if room.host_user_id != user.id:
        return redirect(f"/rooms/{room.code}")

    room.status = "computing"
    db.commit()

    try:
        result = compute_recommendations_for_room(db, room)
        room.results_json = result
        room.status = "results_ready"
        db.commit()
        return redirect(f"/rooms/{room.code}/results")
    except Exception as exc:
        room.status = "compute_failed"
        room.results_json = {"error": str(exc)}
        db.commit()
        return redirect(f"/rooms/{room.code}")


@app.get("/rooms/{code}/results", response_class=HTMLResponse)
def results_page(
    request: Request,
    code: str,
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if user is None:
        return redirect("/login")

    room = db.scalar(select(Room).where(Room.code == code.upper()))
    if room is None:
        return redirect("/dashboard")

    membership = db.scalar(
        select(RoomMember).where(RoomMember.room_id == room.id, RoomMember.user_id == user.id)
    )
    if membership is None:
        return redirect("/dashboard")

    if room.status != "results_ready" or not room.results_json:
        return redirect(f"/rooms/{room.code}")

    return render(
        request,
        "results.html",
        db,
        room=room,
        results=room.results_json.get("final_results", []),
        participant_names=room.results_json.get("participants_used", []),
        iteration_info=room.results_json.get("iterations", []),
    )