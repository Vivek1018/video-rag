from __future__ import annotations

import gc
from pathlib import Path

from video_rag.config import Settings
from video_rag.language import resolve_language

MEDIA_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".webm"}


def discover_media(media_dir: Path) -> list[Path]:
    if not media_dir.is_dir():
        raise FileNotFoundError(
            f"Media directory not found: {media_dir}. Add files under data/media/."
        )
    paths = sorted(
        p
        for p in media_dir.iterdir()
        if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No media files in {media_dir}")
    return paths


def save_transcript(output_path: Path, result: dict) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for segment in result["segments"]:
            handle.write(
                f"[{segment['start']:.2f} - {segment['end']:.2f}]: {segment['text']}\n"
            )
    return output_path


def transcribe_media(settings: Settings, skip_existing: bool = True) -> list[Path]:
    import whisperx

    paths = discover_media(settings.media_dir)
    settings.transcripts_dir.mkdir(parents=True, exist_ok=True)

    model = whisperx.load_model(
        settings.whisper_model,
        settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    align_cache: dict[str, tuple] = {}
    written: list[Path] = []

    def get_align_model(language_code: str):
        if language_code not in align_cache:
            align_cache[language_code] = whisperx.load_align_model(
                language_code=language_code, device=settings.whisper_device
            )
        return align_cache[language_code]

    try:
        for path in paths:
            output_path = settings.transcripts_dir / f"{path.stem}.txt"
            if skip_existing and output_path.is_file():
                print(f"Skipping (exists): {path.name}")
                written.append(output_path)
                continue

            lang, source = resolve_language(
                path.name, str(path), default=settings.default_language
            )
            print(f"Processing: {path.name} [lang={lang} via {source}]...")
            try:
                audio = whisperx.load_audio(str(path))
                result = model.transcribe(
                    audio, batch_size=settings.whisper_batch_size, language=lang
                )
                align_model, metadata = get_align_model(result["language"])
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio,
                    settings.whisper_device,
                    return_char_alignments=False,
                )
                save_transcript(output_path, result)
                written.append(output_path)
                print(f"  Saved: {output_path}")
            except Exception as exc:
                print(f"  ERROR on {path.name}: {exc}")
            finally:
                gc.collect()
    finally:
        del model
        align_cache.clear()
        gc.collect()

    return written
