from pathlib import Path

from sqlalchemy import select
from supabase import create_client

from app.config import get_settings
from app.db import SessionLocal
from app.models import CatalogItem


def content_type_for(path: Path) -> str:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if path.suffix.lower() == ".png":
        return "image/png"
    if path.suffix.lower() == ".webp":
        return "image/webp"
    return "application/octet-stream"


def upload_file(storage, bucket: str, local_path: Path, object_path: str) -> str:
    """
    Upload one local media file to Supabase Storage and return its public URL.
    """
    with local_path.open("rb") as file:
        storage.from_(bucket).upload(
            path=object_path,
            file=file,
            file_options={
                "content-type": content_type_for(local_path),
                "upsert": "true",
            },
        )

    return storage.from_(bucket).get_public_url(object_path)


def main() -> None:
    """
    Deployment helper.

    After restoring the preprocessed local database into Supabase:
    1. Upload local media files to Supabase Storage.
    2. Update image paths in catalog_items to public Supabase URLs.

    This does not re-run dataset preprocessing.
    """
    settings = get_settings()

    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env.")

    media_root = settings.resolved_media_root
    bucket = settings.supabase_storage_bucket

    supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
    db = SessionLocal()

    try:
        items = list(db.scalars(select(CatalogItem).order_by(CatalogItem.id)))
        print(f"Syncing media for {len(items)} catalog items...")

        for index, item in enumerate(items, start=1):
            item_id = item.source_item_id

            poster_file = media_root / "posters" / f"{item_id}.jpg"
            thumb_file = media_root / "thumbnails" / f"{item_id}.jpg"

            if poster_file.exists():
                item.image_local_path = upload_file(
                    supabase.storage,
                    bucket,
                    poster_file,
                    f"posters/{item_id}.jpg",
                )

            if thumb_file.exists():
                item.thumbnail_local_path = upload_file(
                    supabase.storage,
                    bucket,
                    thumb_file,
                    f"thumbnails/{item_id}.jpg",
                )

            if index % 250 == 0:
                db.commit()
                print(f"Synced {index}/{len(items)} items...")

        db.commit()
        print("Supabase media sync complete.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
