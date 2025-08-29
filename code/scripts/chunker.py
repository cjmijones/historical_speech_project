# code/scripts/chunker.py
from __future__ import annotations
from typing import List, Tuple
from .utils import split_sentences, estimate_duration_sec

def chunk_by_target_seconds(
    body: str,
    target_seconds: int = 60,
    wpm: int = 160,
    max_seconds: int = 68
) -> List[Tuple[str, int, int, int, int, float]]:
    """
    Chunk text by sentence boundaries to ~target_seconds per chunk (by WPM estimate).
    Returns list of tuples:
      (chunk_text, start_char, end_char, start_word, end_word, est_sec)
    """
    sents = split_sentences(body)

    # Build character offsets (rough, but stable with our normalization)
    offsets = []
    pos = 0
    for s in sents:
        start = pos
        pos += len(s)
        offsets.append((s, start, pos))
        pos += 1  # simulate a space/newline join

    chunks = []
    cur_text: List[str] = []
    cur_start_char = 0 if not offsets else offsets[0][1]
    cur_words = 0
    cur_start_word = 0
    word_cursor = 0

    def flush(end_char):
        nonlocal cur_text, cur_start_char, cur_words, cur_start_word, word_cursor
        if not cur_text:
            return
        text = " ".join(cur_text).strip()
        est = estimate_duration_sec(cur_words, wpm=wpm)
        chunks.append((text, cur_start_char, end_char, cur_start_word, cur_start_word + cur_words, est))
        cur_text = []
        cur_words = 0
        cur_start_char = end_char
        cur_start_word = word_cursor

    for sent, s0, s1 in offsets:
        w = len(sent.split())
        projected = estimate_duration_sec(cur_words + w, wpm=wpm)

        # If adding this sentence would push us well beyond target and we already have content, flush first
        if cur_text and (projected > target_seconds and projected > 0.6 * max_seconds):
            flush(s0)

        cur_text.append(sent)
        cur_words += w
        word_cursor += w

        # If we've reached target_seconds, flush at sentence boundary
        if estimate_duration_sec(cur_words, wpm=wpm) >= target_seconds and cur_text:
            flush(s1)

    # Final flush
    if offsets:
        flush(offsets[-1][2])

    return chunks
