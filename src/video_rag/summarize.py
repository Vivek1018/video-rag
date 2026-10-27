from __future__ import annotations

import json
from pathlib import Path

import requests

from video_rag.config import Settings
from video_rag.segments import load_all_transcripts, write_json

SYSTEM_PROMPT = (
    "You summarize a timestamped transcript segment from a video or audio file. "
    "Write 1-2 sentences in the same language as the segment. "
    "Keep names, numbers, and the main claim. Do not add facts that are not present."
)


def _summarize_ollama(text: str, settings: Settings) -> str:
    url = settings.ollama_base_url.rstrip("/") + "/api/generate"
    prompt = f"{SYSTEM_PROMPT}\n\nSegment:\n{text}\n\nSummary:"
    response = requests.post(
        url,
        json={"model": settings.qwen_model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return str(response.json().get("response", "")).strip()


def _summarize_transformers(text: str, settings: Settings, pipe: object) -> str:
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nSegment:\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    outputs = pipe(prompt, max_new_tokens=128, do_sample=False)
    generated = outputs[0]["generated_text"]
    if isinstance(generated, str) and "assistant" in generated:
        return generated.split("assistant")[-1].strip()
    return str(generated).strip()


def summarize_segments(settings: Settings) -> Path:
    rows = load_all_transcripts(settings.transcripts_dir)
    if not rows:
        raise FileNotFoundError(
            f"No transcripts in {settings.transcripts_dir}. Run: video-rag transcribe"
        )

    pipe = None
    if settings.qwen_backend == "transformers":
        from transformers import pipeline

        pipe = pipeline(
            "text-generation",
            model=settings.qwen_model,
            token=settings.hf_token,
        )

    summarized: list[dict] = []
    for index, row in enumerate(rows, start=1):
        text = row["text"]
        print(f"Summarizing {index}/{len(rows)}: {row['source']} [{row['start']}-{row['end']}]")
        try:
            if settings.qwen_backend == "transformers":
                summary = _summarize_transformers(text, settings, pipe)
            else:
                summary = _summarize_ollama(text, settings)
        except Exception as exc:
            print(f"  WARN: summary failed ({exc}); using transcript text.")
            summary = text
        summarized.append({**row, "summary": summary or text})

    settings.summaries_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.summaries_dir / "segment_summaries.json"
    write_json(output_path, summarized)
    print(f"Saved {len(summarized)} summaries -> {output_path}")
    return output_path


def load_summaries(settings: Settings) -> list[dict]:
    path = settings.summaries_dir / "segment_summaries.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Summaries not found at {path}. Run: video-rag summarize"
        )
    return json.loads(path.read_text(encoding="utf-8"))
