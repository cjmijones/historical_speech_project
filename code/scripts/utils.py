# code/scripts/utils.py
from __future__ import annotations
import re
from typing import List

def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "item"

def normalize_whitespace(text: str) -> str:
    """
    Normalize line endings, collapse runs of spaces,
    preserve paragraph breaks (double newlines).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Protect double newlines temporarily
    text = re.sub(r"\n{2,}", "<<<PARA>>>", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n", " ", text)
    # Restore paragraph breaks as double newlines
    text = text.replace("<<<PARA>>>", "\n\n")
    # Collapse >2 paragraph breaks to exactly two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# Simple sentence-boundary regex: split on . ! ? followed by space and uppercase/quote/digit
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

def split_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    parts = _SENT_BOUNDARY.split(text)
    # Merge short fragments to avoid ultra-short sentences
    merged: List[str] = []
    buf = ""
    for p in parts:
        if not buf:
            buf = p
        elif len(buf.split()) < 6:
            buf += " " + p
        else:
            merged.append(buf.strip())
            buf = p
    if buf:
        merged.append(buf.strip())
    return merged

def estimate_duration_sec(words: int, wpm: int = 160) -> float:
    return max(1.0, 60.0 * words / max(80, wpm))
