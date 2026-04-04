#!/usr/bin/env python3
"""
Load the processed anime catalog JSONL into PostgreSQL.

This script upserts records by source_item_id so it can be re-run safely.
"""
import argparse
import json
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert

from app.db import SessionLocal
from app.models import CatalogItem


def chunked(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True, help="Path to processed catalog JSONL")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    processed_path = Path(args.processed)
    if not processed_path.exists():
        raise FileNotFoundError(processed_path)

    with processed_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    db = SessionLocal()
    try:
        for batch in chunked(rows, args.batch_size):
            stmt = insert(CatalogItem).values(batch)
            update_columns = {
                "title": stmt.excluded.title,
                "search_text": stmt.excluded.search_text,
                "text_blob": stmt.excluded.text_blob,
                "year": stmt.excluded.year,
                "season": stmt.excluded.season,
                "media_type": stmt.excluded.media_type,
                "status": stmt.excluded.status,
                "episodes": stmt.excluded.episodes,
                "score": stmt.excluded.score,
                "top_tags": stmt.excluded.top_tags,
                "metadata_json": stmt.excluded.metadata_json,
                "embedding": stmt.excluded.embedding,
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=[CatalogItem.source_item_id],
                set_=update_columns,
            )
            db.execute(stmt)

        db.commit()
        print(f"Loaded {len(rows)} catalog records.")
    finally:
        db.close()


if __name__ == "__main__":
    main()