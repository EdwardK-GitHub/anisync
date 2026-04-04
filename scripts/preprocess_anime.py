#!/usr/bin/env python3
"""
Offline preprocessing for the anime-offline-database JSONL file.

What this script does:
1. Reads the manually-downloaded raw JSONL dataset.
2. Skips the metadata line.
3. Filters anime records according to the design document.
4. Builds:
   - source_item_id
   - search_text
   - text_blob
   - metadata_json
   - score
   - top_tags
5. Embeds text_blob in batches using the selected sentence-transformers model.
6. Writes a processed JSONL file with normalized embeddings.

This script is intentionally offline.
"""
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from sentence_transformers import SentenceTransformer

ALLOWED_TYPES = {"TV", "MOVIE", "OVA", "ONA", "SPECIAL"}
ALLOWED_STATUSES = {"FINISHED", "ONGOING"}


def stable_sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def choose_score(score_obj: dict | None) -> float | None:
    if not score_obj:
        return None
    for key in ("arithmeticMean", "arithmeticGeometricMean", "median"):
        value = score_obj.get(key)
        if value is not None:
            return float(value)
    return None


def build_source_item_id(record: dict) -> str:
    sources = record.get("sources") or []
    if sources:
        return stable_sha1("|".join(sorted(sources)))
    return stable_sha1(record["title"])


def build_search_text(record: dict) -> str:
    synonyms = record.get("synonyms") or []
    values = [record["title"], *synonyms]
    return " | ".join(values)


def build_text_blob(record: dict) -> str:
    anime_season = record.get("animeSeason") or {}
    duration = record.get("duration") or {}
    score = choose_score(record.get("score"))
    synonyms = record.get("synonyms") or []
    studios = record.get("studios") or []
    tags = record.get("tags") or []

    parts = []

    def add(label: str, value):
        if value is None:
            return
        if value == "":
            return
        if isinstance(value, list) and not value:
            return
        parts.append(f"{label}: {value}")

    add("title", record.get("title"))
    add("type", record.get("type"))
    add("status", record.get("status"))
    add("year", anime_season.get("year"))
    add("season", anime_season.get("season"))
    add("episodes", record.get("episodes"))
    add("duration_value", duration.get("value"))
    add("duration_unit", duration.get("unit"))
    add("score", score)
    add("synonyms", ", ".join(synonyms))
    add("studios", ", ".join(studios))
    add("tags", ", ".join(tags))

    return " | ".join(parts)


def retain_record(record: dict, current_year: int) -> bool:
    anime_season = record.get("animeSeason") or {}
    year = anime_season.get("year")
    media_type = record.get("type")
    status = record.get("status")

    if year is None:
        return False
    if not (1960 <= int(year) <= current_year + 1):
        return False
    if media_type not in ALLOWED_TYPES:
        return False
    if status not in ALLOWED_STATUSES:
        return False
    return True


def iter_raw_anime(raw_jsonl_path: Path):
    with raw_jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            obj = json.loads(line)

            # The official JSONL file uses the first line for metadata.
            # We treat any line without "title" as non-anime metadata and skip it.
            if line_number == 0 and "title" not in obj:
                continue
            if "title" not in obj:
                continue

            yield obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Path to manually-downloaded anime-offline-database.jsonl")
    parser.add_argument("--out", required=True, help="Output processed JSONL path")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformers model name",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    current_year = date.today().year
    rows = []

    for record in iter_raw_anime(raw_path):
        if not retain_record(record, current_year):
            continue

        anime_season = record.get("animeSeason") or {}
        duration = record.get("duration") or {}

        row = {
            "source_item_id": build_source_item_id(record),
            "title": record.get("title"),
            "search_text": build_search_text(record),
            "text_blob": build_text_blob(record),
            "year": anime_season.get("year"),
            "season": anime_season.get("season"),
            "media_type": record.get("type"),
            "status": record.get("status"),
            "episodes": record.get("episodes"),
            "score": choose_score(record.get("score")),
            "top_tags": (record.get("tags") or [])[:3],
            "metadata_json": {
                "year": anime_season.get("year"),
                "season": anime_season.get("season"),
                "episodes": record.get("episodes"),
                "duration_seconds": duration.get("value"),
                "synonyms": record.get("synonyms") or [],
                "studios": record.get("studios") or [],
                "sources": record.get("sources") or [],
                "relatedAnime": record.get("relatedAnime") or [],
                "tags": record.get("tags") or [],
            },
        }
        rows.append(row)

    print(f"Retained {len(rows)} anime rows after filtering.")

    model = SentenceTransformer(args.model)

    texts = [row["text_blob"] for row in rows]
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    with out_path.open("w", encoding="utf-8") as f:
        for row, embedding in zip(rows, embeddings, strict=True):
            row["embedding"] = [float(x) for x in embedding]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote processed catalog to: {out_path}")


if __name__ == "__main__":
    main()