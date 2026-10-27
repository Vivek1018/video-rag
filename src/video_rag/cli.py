from __future__ import annotations

import argparse
import sys
from pathlib import Path

from video_rag.config import Settings


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_env(args.env_file)


def _cmd_transcribe(args: argparse.Namespace) -> int:
    from video_rag.transcribe import transcribe_media

    written = transcribe_media(_settings(args), skip_existing=not args.force)
    print(f"Transcripts: {len(written)}")
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    from video_rag.summarize import summarize_segments

    summarize_segments(_settings(args))
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    from video_rag.embed import build_index

    build_index(_settings(args))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from video_rag.search import search

    settings = _settings(args)
    hits = search(args.query, settings, top_k=args.top_k)
    print(f'Query: "{args.query}"')
    for hit in hits:
        print(hit.format_line())
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from video_rag.embed import build_index
    from video_rag.summarize import summarize_segments
    from video_rag.transcribe import transcribe_media

    settings = _settings(args)
    transcribe_media(settings, skip_existing=not args.force)
    summarize_segments(settings)
    build_index(settings)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video-segment retrieval (WhisperX → Qwen2.5-3B → BGE-M3)."
    )
    parser.add_argument("--env-file", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    transcribe_p = sub.add_parser("transcribe", help="Transcribe media with WhisperX.")
    transcribe_p.add_argument("--force", action="store_true")
    transcribe_p.set_defaults(func=_cmd_transcribe)

    summarize_p = sub.add_parser("summarize", help="Summarize segments with Qwen2.5-3B.")
    summarize_p.set_defaults(func=_cmd_summarize)

    index_p = sub.add_parser("index", help="Embed summaries with BGE-M3.")
    index_p.set_defaults(func=_cmd_index)

    search_p = sub.add_parser("search", help="Retrieve video segments for a text query.")
    search_p.add_argument("--query", "-q", required=True)
    search_p.add_argument("--top-k", type=int, default=None)
    search_p.set_defaults(func=_cmd_search)

    ingest_p = sub.add_parser("ingest", help="Transcribe, summarize, and index.")
    ingest_p.add_argument("--force", action="store_true")
    ingest_p.set_defaults(func=_cmd_ingest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
