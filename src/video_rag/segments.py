from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SEGMENT_RE = re.compile(
    r"^\[\s*(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\s*\]:\s*(?P<text>.+)$"
)


def parse_transcript_file(file_path: Path) -> list[dict[str, Any]]:
    source = file_path.name
    rows: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = SEGMENT_RE.match(line)
        if not match:
            rows.append(
                {"source": source, "start": None, "end": None, "text": line}
            )
            continue
        text = match.group("text").strip()
        if not text:
            continue
        rows.append(
            {
                "source": source,
                "start": float(match.group("start")),
                "end": float(match.group("end")),
                "text": text,
            }
        )
    return rows


def load_all_transcripts(transcript_dir: Path) -> list[dict[str, Any]]:
    files = sorted(p for p in transcript_dir.glob("*.txt") if p.is_file())
    rows: list[dict[str, Any]] = []
    for file_path in files:
        rows.extend(parse_transcript_file(file_path))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
