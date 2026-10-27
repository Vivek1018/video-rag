from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from video_rag.config import Settings


@dataclass(frozen=True)
class SearchHit:
    score: float
    source: str
    start: float | None
    end: float | None
    text: str
    summary: str

    def format_line(self) -> str:
        span = ""
        if self.start is not None and self.end is not None:
            span = f" [{self.start:.2f} - {self.end:.2f}]"
        preview = (self.summary or self.text).replace("\n", " ")
        if len(preview) > 140:
            preview = preview[:137] + "..."
        return f"{self.score:.3f} | {self.source}{span} {preview}"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def rank_vectors(
    query_vector: np.ndarray, embeddings: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    query = _l2_normalize(query_vector.reshape(1, -1))[0]
    matrix = _l2_normalize(embeddings)
    scores = matrix @ query
    order = np.argsort(-scores)[:top_k]
    return order, scores[order]


def search(query: str, settings: Settings, top_k: int | None = None) -> list[SearchHit]:
    from FlagEmbedding import BGEM3FlagModel

    top_k = top_k or settings.top_k
    emb_path = settings.embeddings_dir / "segment_embeddings.npy"
    meta_path = settings.embeddings_dir / "segment_metadata.json"
    if not emb_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            "Index not found. Run: video-rag index"
        )

    embeddings = np.load(emb_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    model = BGEM3FlagModel(settings.bge_model, use_fp16=False)
    query_vec = np.asarray(model.encode([query])["dense_vecs"][0], dtype=np.float32)
    order, scores = rank_vectors(query_vec, embeddings, top_k)

    hits: list[SearchHit] = []
    for index, score in zip(order, scores):
        item = metadata[int(index)]
        hits.append(
            SearchHit(
                score=float(score),
                source=str(item.get("source", "")),
                start=item.get("start"),
                end=item.get("end"),
                text=str(item.get("text", "")),
                summary=str(item.get("summary", "")),
            )
        )
    return hits
