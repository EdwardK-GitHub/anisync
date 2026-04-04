from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.kmeans import choose_best_kmeans_result
from app.models import CatalogItem, Room, RoomMember, RoomQuerySubmission


def _embedding_array(items: list[CatalogItem]) -> np.ndarray:
    """
    Convert item.embedding lists into a NumPy matrix of shape (n_items, 384).
    """
    return np.asarray([np.asarray(item.embedding, dtype=np.float32) for item in items], dtype=np.float32)


def top_100_for_query(db: Session, query_embedding: list[float]) -> list[CatalogItem]:
    """
    Exact top-100 semantic retrieval from PostgreSQL using pgvector cosine distance.

    Lower cosine distance is better.
    """
    stmt = (
        select(CatalogItem)
        .order_by(CatalogItem.embedding.cosine_distance(query_embedding))
        .limit(100)
    )
    return list(db.scalars(stmt).all())


def merge_unique_items(item_lists: list[list[CatalogItem]]) -> list[CatalogItem]:
    """
    Merge multiple retrieval lists while preserving first-seen order.
    """
    seen: set[int] = set()
    merged: list[CatalogItem] = []

    for item_list in item_lists:
        for item in item_list:
            if item.id not in seen:
                seen.add(item.id)
                merged.append(item)

    return merged


def _cluster_member_indices(labels: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Turn a label vector into one index array per cluster.
    """
    return [np.where(labels == cluster_idx)[0] for cluster_idx in range(k)]


def _cluster_scores(
    cluster_points: np.ndarray,
    query_vectors: np.ndarray,
) -> tuple[list[float], float, float, tuple[float, ...]]:
    """
    Score one cluster against all users.

    Returns:
    - per_user_scores
    - cluster_score (average over users)
    - coherence (average squared distance to centroid)
    - centroid_lexicographic_key (deterministic tie-break helper)
    """
    per_user_scores: list[float] = []

    for q in query_vectors:
        sims = cluster_points @ q
        per_user_scores.append(float(sims.mean()))

    cluster_score = float(sum(per_user_scores) / len(per_user_scores))

    centroid = cluster_points.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm != 0:
        centroid = centroid / centroid_norm

    coherence = float(((cluster_points - centroid) ** 2).sum(axis=1).mean())
    lexicographic_key = tuple(np.round(centroid, 8).tolist())

    return per_user_scores, cluster_score, coherence, lexicographic_key


def _choose_best_cluster(
    member_indices: list[np.ndarray],
    points: np.ndarray,
    query_vectors: np.ndarray,
) -> dict[str, Any]:
    """
    Choose the next cluster using the design document tie-break rules.
    """
    cluster_rows: list[dict[str, Any]] = []

    for idxs in member_indices:
        cluster_points = points[idxs]
        per_user_scores, cluster_score, coherence, lexicographic_key = _cluster_scores(
            cluster_points, query_vectors
        )

        cluster_rows.append(
            {
                "indices": idxs,
                "cluster_score": cluster_score,
                "min_user_score": float(min(per_user_scores)),
                "coherence": coherence,
                "size": int(len(idxs)),
                "lexicographic_key": lexicographic_key,
                "per_user_scores": per_user_scores,
            }
        )

    # Tie-break order:
    # 1. higher cluster_score
    # 2. higher min_user_score
    # 3. lower coherence
    # 4. larger size
    # 5. smaller centroid lexicographic key
    cluster_rows.sort(
        key=lambda row: (
            -round(row["cluster_score"], 6),
            -row["min_user_score"],
            row["coherence"],
            -row["size"],
            row["lexicographic_key"],
        )
    )
    return cluster_rows[0]


def compute_recommendations_for_room(db: Session, room: Room) -> dict[str, Any]:
    """
    Full room recommendation pipeline.

    This function follows the design document exactly:
    - collect submitted users
    - embed queries already stored in DB
    - top-100 retrieval per user
    - merge + dedupe
    - iterative manual K-means
    - select cluster by average query-to-cluster similarity
    - stop by safeguards
    - final ranking by average similarity to original query vectors
    """
    member_stmt = (
        select(RoomMember)
        .where(RoomMember.room_id == room.id)
        .options(selectinload(RoomMember.user))
    )
    members = list(db.scalars(member_stmt).all())
    member_user_ids = {member.user_id for member in members}

    submission_stmt = (
        select(RoomQuerySubmission)
        .where(RoomQuerySubmission.room_id == room.id)
        .options(selectinload(RoomQuerySubmission.user))
    )
    submissions = [s for s in db.scalars(submission_stmt).all() if s.user_id in member_user_ids]

    if len(submissions) < 2:
        raise ValueError("At least 2 submitted users are required.")

    participant_names = [submission.user.display_name for submission in submissions]
    query_vectors = np.asarray(
        [np.asarray(submission.query_embedding, dtype=np.float32) for submission in submissions],
        dtype=np.float32,
    )

    retrieval_lists: list[list[CatalogItem]] = []
    retrieval_counts: dict[str, int] = {}

    for submission in submissions:
        results = top_100_for_query(db, submission.query_embedding)
        retrieval_lists.append(results)
        retrieval_counts[submission.user.display_name] = len(results)

    current_items = merge_unique_items(retrieval_lists)
    current_points = _embedding_array(current_items)

    iterations: list[dict[str, Any]] = []
    iteration = 0
    weak_shrink_count = 0
    tiny_gain_count = 0
    previous_selected_score: float | None = None

    while True:
        n = len(current_items)

        if n <= 10:
            break

        if iteration >= 8:
            break

        chosen_k, kmeans_result = choose_best_kmeans_result(current_points, query_vectors)
        member_indices = _cluster_member_indices(kmeans_result.labels, chosen_k)

        best_cluster = _choose_best_cluster(member_indices, current_points, query_vectors)
        selected_size = int(len(best_cluster["indices"]))
        selected_score = float(best_cluster["cluster_score"])

        iterations.append(
            {
                "iteration": iteration,
                "input_count": n,
                "chosen_k": chosen_k,
                "silhouette": kmeans_result.silhouette,
                "selected_cluster_size": selected_size,
                "selected_cluster_score": selected_score,
                "selected_cluster_min_user_score": best_cluster["min_user_score"],
                "selected_cluster_coherence": best_cluster["coherence"],
            }
        )

        # Stopping safeguards.
        if selected_size >= 0.95 * n:
            weak_shrink_count += 1
        else:
            weak_shrink_count = 0

        if previous_selected_score is not None and (selected_score - previous_selected_score) < 0.005:
            tiny_gain_count += 1
        else:
            tiny_gain_count = 0

        previous_selected_score = selected_score

        if weak_shrink_count >= 2 or tiny_gain_count >= 2:
            break

        current_items = [current_items[idx] for idx in best_cluster["indices"]]
        current_points = current_points[best_cluster["indices"]]
        iteration += 1

    final_rows: list[dict[str, Any]] = []

    for item in current_items:
        anime_vec = np.asarray(item.embedding, dtype=np.float32)
        sims = [float(np.dot(q, anime_vec)) for q in query_vectors]
        avg_score = float(sum(sims) / len(sims))
        min_score = float(min(sims))
        std_score = float(np.std(np.asarray(sims, dtype=np.float32)))

        final_rows.append(
            {
                "catalog_item_id": item.id,
                "title": item.title,
                "media_type": item.media_type,
                "year": item.year,
                "status": item.status,
                "episodes": item.episodes,
                "top_tags": item.top_tags[:3] if item.top_tags else [],
                "catalog_score": item.score,
                "evaluation_score": avg_score,
                "worst_user_score": min_score,
                "std_score": std_score,
            }
        )

    final_rows.sort(
        key=lambda row: (
            -row["evaluation_score"],
            -row["worst_user_score"],
            row["std_score"],
            -(row["catalog_score"] or -999.0),
            row["title"].lower(),
        )
    )

    final_rows = final_rows[:10]

    return {
        "participants_used": participant_names,
        "participant_count": len(participant_names),
        "retrieval_counts": retrieval_counts,
        "iterations": iterations,
        "final_results": final_rows,
    }