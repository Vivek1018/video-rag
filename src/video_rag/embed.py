from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from video_rag.config import Settings
from video_rag.segments import write_json
from video_rag.summarize import load_summaries


def _encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    outputs = model.encode(texts, batch_size=batch_size, max_length=8192)
    return np.asarray(outputs["dense_vecs"], dtype=np.float32)


def build_index(settings: Settings) -> tuple[Path, Path]:
    from FlagEmbedding import BGEM3FlagModel

    rows = load_summaries(settings)
    documents = [row.get("summary") or row["text"] for row in rows]
    print(f"Loading {settings.bge_model}...")
    model = BGEM3FlagModel(settings.bge_model, use_fp16=False)
    print(f"Embedding {len(documents)} segments...")
    embeddings = _encode(model, documents, settings.embed_batch_size)
    del model
    gc.collect()

    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    emb_path = settings.embeddings_dir / "segment_embeddings.npy"
    meta_path = settings.embeddings_dir / "segment_metadata.json"
    np.save(emb_path, embeddings)
    write_json(meta_path, rows)
    print(f"Saved embeddings -> {emb_path}  shape={embeddings.shape}")
    print(f"Saved metadata   -> {meta_path}")
    return emb_path, meta_path
