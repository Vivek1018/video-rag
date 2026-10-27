"""Resolve transcription language from filename tags, then file metadata."""

from __future__ import annotations

import re
from pathlib import Path

LANG_ALIASES = {
    "deutsch": "de",
    "german": "de",
    "de": "de",
    "english": "en",
    "eng": "en",
    "en": "en",
    "espanol": "es",
    "español": "es",
    "spanish": "es",
    "es": "es",
    "portuguese": "pt",
    "portugues": "pt",
    "português": "pt",
    "brazilian": "pt",
    "brasil": "pt",
    "pt": "pt",
    "hindi": "hi",
    "hi": "hi",
    "french": "fr",
    "francais": "fr",
    "français": "fr",
    "fr": "fr",
}


def language_from_filename(filename: str) -> str | None:
    """Parse language codes from filename tokens (e.g. talk-Deutsch.mp4)."""
    stem = Path(filename).stem.lower()
    tokens = re.split(r"[-_\s]+", stem)
    tokens = [re.sub(r"[^a-zà-ü]+", "", token) for token in tokens]

    found: list[str] = []
    for token in tokens:
        if token in LANG_ALIASES:
            code = LANG_ALIASES[token]
            if code not in found:
                found.append(code)
    return found[0] if found else None


def language_from_metadata(file_path: str) -> str | None:
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return None

    try:
        audio_meta = MutagenFile(file_path, easy=True)
        if audio_meta is None:
            return None
        for key in ["language", "tlan", "\xa9lan"]:
            if key in audio_meta and audio_meta[key]:
                lang_val = str(audio_meta[key][0]).strip().lower()
                if lang_val in LANG_ALIASES:
                    return LANG_ALIASES[lang_val]
                if len(lang_val) >= 2:
                    return lang_val[:2]
    except Exception:
        return None
    return None


def resolve_language(
    filename: str, file_path: str, default: str = "en"
) -> tuple[str, str]:
    lang = language_from_filename(filename)
    source = "filename"
    if not lang:
        lang = language_from_metadata(file_path)
        source = "metadata"
    if not lang:
        lang = default
        source = "default"
    return lang, source
