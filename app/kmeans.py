from dataclasses import dataclass

import numpy as np
from sklearn.metrics import silhouette_score


@dataclass
class KMeansResult:
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    silhouette: float | None = None


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize rows to unit length, avoiding divide-by-zero issues.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def average_similarity_to_queries(points: np.ndarray, query_vectors: np.ndarray) -> np.ndarray:
    """
    For each point, compute its average dot-product similarity to all query vectors.

    Shape:
    - points: (n_points, d)
    - query_vectors: (n_queries, d)

    Returns:
    - scores: (n_points,)
    """
    sims = points @ query_vectors.T
    return sims.mean(axis=1)


def deterministic_farthest_init(
    points: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Deterministic farthest-point-style initialization.

    Rule:
    1. first centroid = point with highest average similarity to all queries
    2. each next centroid = point farthest from the already chosen centroid set
    """
    n = len(points)
    avg_scores = average_similarity_to_queries(points, query_vectors)
    first_idx = int(np.argmax(avg_scores))
    chosen = [first_idx]

    while len(chosen) < k:
        chosen_points = points[np.array(chosen)]
        distances = ((points[:, None, :] - chosen_points[None, :, :]) ** 2).sum(axis=2)
        min_distances = distances.min(axis=1)

        # Never re-choose an already selected point.
        min_distances[np.array(chosen)] = -1.0
        next_idx = int(np.argmax(min_distances))
        chosen.append(next_idx)

    return points[np.array(chosen)].copy()


def run_manual_kmeans(
    points: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    n_init: int = 10,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> KMeansResult:
    """
    Manual K-means implementation.

    Inputs:
    - points are normalized anime embeddings
    - query_vectors are used only for deterministic first-centroid seeding

    Returns:
    - best result over n_init restarts by lowest inertia
    """
    best_result: KMeansResult | None = None
    n = len(points)

    if k < 2:
        raise ValueError("k must be at least 2")

    if n < k:
        raise ValueError("Cannot cluster fewer points than k")

    for _restart in range(n_init):
        centroids = deterministic_farthest_init(points, query_vectors, k)
        labels = np.zeros(n, dtype=np.int32)

        for _ in range(max_iter):
            distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)

            new_centroids = centroids.copy()
            for cluster_idx in range(k):
                member_mask = new_labels == cluster_idx

                if member_mask.sum() == 0:
                    # Empty-cluster repair:
                    # reset centroid to point with largest current error.
                    current_errors = distances[np.arange(n), new_labels]
                    worst_point_idx = int(np.argmax(current_errors))
                    new_centroids[cluster_idx] = points[worst_point_idx]
                else:
                    cluster_points = points[member_mask]
                    centroid = cluster_points.mean(axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm == 0.0:
                        new_centroids[cluster_idx] = centroid
                    else:
                        new_centroids[cluster_idx] = centroid / norm

            movement = np.linalg.norm(new_centroids - centroids, axis=1).max()
            centroids = new_centroids
            labels = new_labels

            if movement < tol:
                break

        final_distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        inertia = float(final_distances[np.arange(n), labels].sum())

        result = KMeansResult(labels=labels.copy(), centroids=centroids.copy(), inertia=inertia)

        if best_result is None or result.inertia < best_result.inertia:
            best_result = result

    if best_result is None:
        raise RuntimeError("K-means failed to produce a result")

    return best_result


def candidate_k_values(n_points: int) -> list[int]:
    """
    Build the candidate K list from the design document:
    K_max(n) = min(8, floor(n / 5))
    """
    k_max = min(8, n_points // 5)
    if k_max < 2:
        return []
    return list(range(2, k_max + 1))


def minimum_cluster_size(n_points: int) -> int:
    """
    Valid cluster size threshold from the design document:
    max(3, ceil(0.05 * n_points))
    """
    return max(3, int(np.ceil(0.05 * n_points)))


def choose_best_kmeans_result(
    points: np.ndarray,
    query_vectors: np.ndarray,
) -> tuple[int, KMeansResult]:
    """
    Try all candidate K values, score valid ones with silhouette, and apply fallback.

    This function returns:
    - chosen_k
    - chosen_kmeans_result
    """
    ks = candidate_k_values(len(points))

    valid_results: list[tuple[int, KMeansResult]] = []
    min_size = minimum_cluster_size(len(points))

    for k in ks:
        result = run_manual_kmeans(points, query_vectors, k=k)

        counts = np.bincount(result.labels, minlength=k)
        if counts.min() < min_size:
            continue

        # silhouette_score requires at least 2 labels and fewer labels than samples.
        sil = float(silhouette_score(points, result.labels, metric="euclidean"))
        result.silhouette = sil
        valid_results.append((k, result))

    if valid_results:
        valid_results.sort(key=lambda item: item[1].silhouette, reverse=True)
        best_k, best_result = valid_results[0]
        if best_result.silhouette is not None and best_result.silhouette >= 0.05:
            return best_k, best_result

    # Fallback rule from the design document.
    fallback_k = 2 if len(points) < 20 else 3
    fallback_k = min(fallback_k, max(2, len(points) - 1))
    fallback_result = run_manual_kmeans(points, query_vectors, k=fallback_k)
    return fallback_k, fallback_result