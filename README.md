# Video RAG — Retrieval-Augmented Generation for Video Content

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Natural-language search over video (and audio) archives. The pipeline is local:

**WhisperX** transcription → **Qwen2.5-3B** segment summaries → **BGE-M3** embeddings → cosine retrieval of the matching time range.

Built as a personal project while studying multimodal embeddings for an M.Sc. in Data Analytics at Stiftung Universität Hildesheim.

This repository does **not** ship personal media. Add your own files under `data/media/`.

## Pipeline

1. **Transcribe** — WhisperX with word-level alignment. Language is taken from the filename (`talk-Deutsch.mp4` → `de`), then file tags, then a default.
2. **Summarize** — Qwen2.5-3B (Ollama `qwen2.5:3b` by default, or Hugging Face `Qwen/Qwen2.5-3B-Instruct`) writes a short summary per timestamped segment.
3. **Embed** — BGE-M3 dense vectors (1024-d) for those summaries, stored as NumPy + JSON metadata.
4. **Search** — a text query is embedded with the same model and ranked by cosine similarity. Hits include source file, start/end time, summary, and transcript snippet.

## Quickstart

```bash
git clone https://github.com/Vivek1018/video-rag.git
cd video-rag

uv venv
uv pip install -e ".[all]"
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Put `.mp4` / `.mp3` / `.wav` files in `data/media/`, then:

```bash
# WhisperX needs FFmpeg on PATH
video-rag transcribe
video-rag summarize          # requires Ollama: ollama pull qwen2.5:3b
video-rag index
video-rag search -q "when do they discuss the cost reduction" --top-k 5
```

One-shot ingest:

```bash
video-rag ingest
video-rag search -q "opening remarks"
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `medium` | WhisperX model size |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `QWEN_BACKEND` | `ollama` | `ollama` or `transformers` |
| `QWEN_MODEL` | `qwen2.5:3b` | Ollama tag or HF model id |
| `BGE_MODEL` | `BAAI/bge-m3` | Embedding model |
| `TOP_K` | `5` | Default hits per query |

## Layout

```text
src/video_rag/     # transcribe, summarize, embed, search, CLI
data/media/        # your videos/audio (not committed)
data/transcripts/  # generated
data/summaries/    # generated
data/embeddings/   # generated
```

## Author

**Vivek Kumar Verma** — [LinkedIn](https://www.linkedin.com/in/vivek-kumar-verma-0206765b/) · Vivek.verma1018@gmail.com
