import numpy as np

from video_rag.search import rank_vectors


def test_rank_vectors_returns_nearest_neighbor():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    order, scores = rank_vectors(query, embeddings, top_k=2)
    assert int(order[0]) == 1
    assert scores[0] > scores[1]
