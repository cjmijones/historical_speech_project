# create_inserts_from_timestamps.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import re
import itertools

# -----------------------
# Normalization helpers
# -----------------------
_WORD_CHARS = re.compile(r"[A-Za-z0-9']+")
SPACE_RE = re.compile(r"\s+")

def _norm_space(s: str) -> str:
    return SPACE_RE.sub(" ", s).strip()

def _norm_token(s: str) -> str:
    """
    Normalize a single token:
      - lowercased
      - keep letters/digits/apostrophes only
      - strip leading/trailing non-word chars (handles '-constructed', 'york.', 'union,')
    """
    # find the first alnum/apostrophe run
    m = _WORD_CHARS.findall(s.lower())
    return m[0] if m else ""

def _phrase_to_tokens(phrase: str) -> List[str]:
    return [_norm_token(t) for t in phrase.split() if _norm_token(t)]

def _word_items_from_json(word_json: List[Dict[str, Any]]) -> List[Tuple[str, float, float, int]]:
    """
    Convert word-level timestamp JSON items to a list of normalized tokens with (start, end, idx).
    Each input item has keys: word, start, end, position, matched (ignored).
    """
    out = []
    for item in word_json:
        tok = _norm_token(item["word"])
        if tok:
            out.append((tok, float(item["start"]), float(item["end"]), int(item.get("position", len(out)))))
    return out

# -----------------------
# Phrase matching
# -----------------------
def _find_first_span(tokens: List[Tuple[str, float, float, int]], phrase_tokens: List[str]) -> Tuple[float, float] | None:
    """
    Slide over normalized tokens to find the first contiguous window equal to phrase_tokens.
    Returns (t_start, t_end) or None if not found.
    """
    if not phrase_tokens:
        return None
    n = len(phrase_tokens)
    toks_only = [t[0] for t in tokens]
    for i in range(0, len(tokens) - n + 1):
        window = toks_only[i:i+n]
        if window == phrase_tokens:
            t_start = tokens[i][1]
            t_end   = tokens[i+n-1][2]
            return (t_start, t_end)
    return None

# -----------------------
# Asset + layout helpers
# -----------------------
def _slugify_for_asset(phrase: str) -> str:
    # simple slug: keep letters/digits, replace spaces with underscores
    base = re.sub(r"[^A-Za-z0-9]+", " ", phrase).strip().lower()
    slug = "_".join(base.split())
    return f"assets_cache\\pexels_{slug}.jpg"

_POSITIONS = [
    "top_left", "top_right", "mid_left", "mid_right"
]

def _next_pos(i: int) -> str:
    return _POSITIONS[i % len(_POSITIONS)]

# -----------------------
# Public builder
# -----------------------
def build_inserts_for_chunk(
    wordlevel_json: List[Dict[str, Any]],
    phrases: List[str],
    *,
    fps: int = 24,
    min_duration: float = 3.5,
    pad_pre: float = 0.5,
    pad_post: float = 0.0,
    avoid_overlaps: bool = False,
    gap_after: float = 1.0,
    min_start_time: float = 1.0,                 # earliest allowed overlay start
    max_end_time: float | None = None,           # latest allowed overlay end (optional)
    default_entry: str = "fade_in",
    default_exit: str = "fade_out",
) -> Dict[str, Any]:
    """
    Build video inserts with proper scheduling:
      - Do NOT shift inserts to min_start_time.
      - Only select inserts whose natural start (after pad_pre) >= current threshold.
      - Threshold starts at `min_start_time` and advances by `gap_after` after each selection.
      - Additionally, regardless of `min_duration`, NO overlay extends past the end timestamp
        of the final word. Only the final selected overlay may be shortened (or dropped if it
        would be zero-length) to meet this constraint.
    """
    tokens = _word_items_from_json(wordlevel_json)

    # Edge case: no tokens
    if not tokens:
        return {"fps": fps, "inserts": []}

    # Hard ceiling: end time of the final word in the JSON
    final_word_end = max(t[2] for t in tokens)

    # Build all proposals with their NATURAL timings (only padding & min_duration applied)
    raw_proposals = []
    for i, phrase in enumerate(phrases):
        ptoks = _phrase_to_tokens(phrase)
        if not ptoks:
            continue
        span = _find_first_span(tokens, ptoks)
        if not span:
            continue

        t0, t1 = span  # natural phrase window from timestamps
        t_start = max(0.0, t0 - pad_pre)
        t_end   = max(t1 + pad_post, t_start + min_duration)

        # Keep proposals as-is here. We will clamp ONLY the final selected insert later.
        # However, if an explicit max_end_time was provided by caller, honor it now.
        if max_end_time is not None and t_end > max_end_time:
            continue

        raw_proposals.append({
            "t_start": t_start,
            "t_end": t_end,
            "type": "image_overlay",
            "asset": _slugify_for_asset(phrase),
            "layout": {
                "pos": _next_pos(i),
                "scale": 0.6,
                "margin": [24, 24],
            },
            "anim": {
                "entry": {"type": default_entry, "duration": 0.75},
                "exit":  {"type": default_exit, "duration": 0.75},
            },
            "reason": f"keyword: {phrase}",
        })

    # Sort by natural start time (stable)
    raw_proposals.sort(key=lambda x: (x["t_start"], x["t_end"]))

    # Greedy selection against a moving threshold:
    selected: List[Dict[str, Any]] = []
    threshold = float(min_start_time)

    for ins in raw_proposals:
        if ins["t_start"] >= threshold:
            selected.append(ins)
            # Even if overlaps are "allowed", we still advance the threshold to enforce spacing preference
            next_threshold = ins["t_end"] + gap_after
            threshold = next_threshold

    # Enforce: nothing shows past the final word end; only the final selected insert may be shortened.
    if selected:
        last = selected[-1]
        if last["t_end"] > final_word_end:
            # Clamp the final insert to end exactly at the final word end
            clamped_end = final_word_end
            # If clamping would result in non-positive duration, drop the final insert
            if clamped_end <= last["t_start"] + 1e-6:
                selected.pop()
            else:
                last["t_end"] = clamped_end

    return {"fps": fps, "inserts": selected}


# -----------------------
# CLI usage (optional)
# -----------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Create inserts JSON from word timestamps and keyphrases.")
    ap.add_argument("--timestamps_json", type=str, required=True, help="Path to word-level timestamps JSON file (array).")
    ap.add_argument("--phrases_json", type=str, required=False, help="Path to a JSON list of phrases. If omitted, reads from stdin.")
    ap.add_argument("--out", type=str, required=True, help="Output path for inserts JSON.")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--min_duration", type=float, default=3.5)
    ap.add_argument("--pad_pre", type=float, default=0.0)
    ap.add_argument("--pad_post", type=float, default=0.0)
    args = ap.parse_args()

    word_ts = json.loads(Path(args.timestamps_json).read_text(encoding="utf-8"))
    if args.phrases_json:
        phrases = json.loads(Path(args.phrases_json).read_text(encoding="utf-8"))
    else:
        # read newline-delimited phrases from stdin
        import sys
        phrases = [ln.strip() for ln in sys.stdin if ln.strip()]

    result = build_inserts_for_chunk(
        word_ts, phrases,
        fps=args.fps,
        min_duration=args.min_duration,
        pad_pre=args.pad_pre,
        pad_post=args.pad_post,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote inserts to {args.out}")
