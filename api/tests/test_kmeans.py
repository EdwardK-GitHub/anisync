import numpy as np

from app.ml.kmeans import choose_k_and_cluster, manual_kmeans, silhouette_score


def test_manual_kmeans_finds_two_simple_clusters():
    left = np.array([[0.0, 1.0], [0.1, 0.99], [-0.1, 0.98]], dtype=np.float32)
    right = np.array([[1.0, 0.0], [0.99, 0.1], [0.98, -0.1]], dtype=np.float32)
    x = np.vstack([left, right])

    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    result = manual_kmeans(x, 2, random_seed=1)

    assert result.k == 2
    assert set(result.assignments.tolist()) == {0, 1}
    assert result.objective >= 0


def test_silhouette_score_range():
    x = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        dtype=np.float32,
    )
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    assignments = np.array([0, 0, 1, 1])

    score = silhouette_score(x, assignments)

    assert -1.0 <= score <= 1.0


def test_choose_k_and_cluster_small_pool_falls_back_to_two():
    x = np.eye(4, dtype=np.float32)
    result = choose_k_and_cluster(x)

    assert result.k == 2
