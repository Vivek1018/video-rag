from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_batch_size: int
    default_language: str
    qwen_backend: str
    ollama_base_url: str
    qwen_model: str
    hf_token: str | None
    bge_model: str
    embed_batch_size: int
    media_dir: Path
    transcripts_dir: Path
    summaries_dir: Path
    embeddings_dir: Path
    top_k: int

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        root = _repo_root()
        if env_file is None:
            env_file = root / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)

        return cls(
            repo_root=root,
            whisper_model=os.getenv("WHISPER_MODEL", "medium"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            whisper_batch_size=int(os.getenv("WHISPER_BATCH_SIZE", "4")),
            default_language=os.getenv("DEFAULT_LANGUAGE", "en"),
            qwen_backend=os.getenv("QWEN_BACKEND", "ollama"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            qwen_model=os.getenv("QWEN_MODEL", "qwen2.5:3b"),
            hf_token=os.getenv("HF_TOKEN") or None,
            bge_model=os.getenv("BGE_MODEL", "BAAI/bge-m3"),
            embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "16")),
            media_dir=_resolve_path(os.getenv("MEDIA_DIR", "data/media"), root),
            transcripts_dir=_resolve_path(
                os.getenv("TRANSCRIPTS_DIR", "data/transcripts"), root
            ),
            summaries_dir=_resolve_path(
                os.getenv("SUMMARIES_DIR", "data/summaries"), root
            ),
            embeddings_dir=_resolve_path(
                os.getenv("EMBEDDINGS_DIR", "data/embeddings"), root
            ),
            top_k=int(os.getenv("TOP_K", "5")),
        )
