from __future__ import annotations
from pathlib import Path
from typing import List

def _fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    hh, rem = divmod(ms, 3600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def text_to_srt_lines(
    text: str,
    total_duration_sec: float,
    words_per_line: int = 7,
    min_line_sec: float = 1.2
) -> List[str]:
    """
    Split text into subtitle lines, distributing time by word share
    but clamping to a minimum duration per cue.
    """
    words = text.split()
    if not words:
        return []

    blocks = [
        " ".join(words[i:i + words_per_line])
        for i in range(0, len(words), words_per_line)
    ]
    total_words = len(words)
    lines, cursor = [], 0.0

    for idx, blk in enumerate(blocks, start=1):
        w = len(blk.split())
        dur = max(min_line_sec, total_duration_sec * (w / max(1, total_words)))
        start = cursor
        end = min(total_duration_sec, start + dur) if idx < len(blocks) else total_duration_sec
        if end <= start:
            end = min(total_duration_sec, start + min_line_sec)
        cursor = end
        lines += [f"{idx}", f"{_fmt_ts(start)} --> {_fmt_ts(end)}", blk, ""]

    return lines

def write_srt(path: Path, text: str, total_duration_sec: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(text_to_srt_lines(text, total_duration_sec)),
        encoding="utf-8"
    )
    return path
