# ollama_commentary.py  (you can paste this whole cell into a Jupyter notebook)

from __future__ import annotations
import json, shutil, subprocess, textwrap, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# --- Configuration ---
DEFAULT_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama HTTP API

# --- Dataclass for clarity (optional) ---
@dataclass
class Chunk:
    parent_id: str
    chunk_id: str
    index: int
    text: str
    approx_word_count: int
    est_duration_sec: float | None = None
    start_char: int | None = None
    end_char: int | None = None
    start_word: int | None = None
    end_word: int | None = None
    meta: Dict[str, Any] | None = None

# ---------- Core prompting ----------
def _build_prompt(text: str, target_words: int) -> str:
    # Keep the instructions crisp and clamp output length with ±10% tolerance
    return textwrap.dedent(f"""
    You are a clear, neutral explainer for a modern audience.

    Task: Write a plain-language explanation and commentary that briefly
    explains the meaning and purpose of the passage for a modern audience.
    Paraphrase instead of quoting except for very short phrases if necessary.
    Avoid flowery language and keep it concrete and accessible.

    Length: about {target_words} words (±10%). No title, no bullet points—
    just a concise paragraph (two if needed).

    PASSAGE:
    \"\"\"{text.strip()}\"\"\"
    """).strip()

def _soft_trim_to_target_words(s: str, target: int, pct_tolerance: float = 0.10) -> str:
    """If the model runs long, softly trim to within tolerance (end at sentence boundary when possible)."""
    words = s.split()
    upper = int(round(target * (1 + pct_tolerance)))
    if len(words) <= upper:
        return s.strip()

    # Try to cut near the upper bound, then backtrack to last sentence end.
    cut = upper
    truncated = " ".join(words[:cut])
    # Prefer ending at punctuation.
    for mark in [". ", "? ", "! "]:
        idx = truncated.rfind(mark)
        if idx != -1 and idx > len(truncated) * 0.6:  # only if reasonably far in
            return truncated[: idx + 1].strip()
    return truncated.strip()

# ---------- Ollama call helpers ----------
def _ollama_via_http(model: str, prompt: str, timeout: int = 180, options: Optional[dict] = None) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

def _ollama_via_subprocess(model: str, prompt: str, timeout: int = 180) -> str:
    """
    Fallback to `ollama run` when HTTP API isn't available.
    Note: This doesn’t support fine control; it’s a pragmatic backup.
    """
    if not shutil.which("ollama"):
        raise RuntimeError("Ollama not found on PATH and HTTP API unavailable.")
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ollama run failed: {proc.stderr.decode('utf-8', 'ignore')}")
    return proc.stdout.decode("utf-8", "ignore").strip()

def generate_commentary(
    text: str,
    approx_word_count: int,
    *,
    ratio: float = 0.75,         # 70–80% sweet spot; default 75%
    model: str = DEFAULT_MODEL,
    options: Optional[dict] = None,
) -> str:
    target_words = max(30, int(round(approx_word_count * ratio)))
    prompt = _build_prompt(text, target_words)

    try:
        reply = _ollama_via_http(model, prompt, options=options)
    except Exception:
        # As a convenience, retry once quickly (helps when ollama is initializing)
        time.sleep(1.0)
        try:
            reply = _ollama_via_http(model, prompt, options=options)
        except Exception:
            # Fallback to subprocess
            reply = _ollama_via_subprocess(model, prompt)

    # Soft-trim if the model runs long
    reply = _soft_trim_to_target_words(reply, target_words)
    return reply.strip()

# ---------- JSON pipeline ----------
def add_commentary_to_chunk_dict(
    chunk: Dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    ratio: float = 0.75,
    options: Optional[dict] = None,
    add_word_count_field: bool = True,
) -> Dict[str, Any]:
    text = chunk.get("text", "")
    approx_wc = int(chunk.get("approx_word_count", max(1, len(text.split()))))

    commentary = generate_commentary(text, approx_wc, ratio=ratio, model=model, options=options)

    out = dict(chunk)
    out["commentary"] = commentary
    if add_word_count_field:
        out["commentary_word_count"] = len(commentary.split())
    return out

def process_json_file(
    in_path: str | Path,
    out_path: str | Path | None = None,
    *,
    model: str = DEFAULT_MODEL,
    ratio: float = 0.75,
    options: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Reads a single JSON object (like your example), adds 'commentary', returns it,
    and optionally writes to out_path.
    """
    in_path = Path(in_path)
    with in_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    updated = add_commentary_to_chunk_dict(obj, model=model, ratio=ratio, options=options)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(updated, f, ensure_ascii=False, indent=2)

    return updated

def process_many_json_files(
    inputs: List[str | Path],
    out_dir: str | Path | None = None,
    *,
    model: str = DEFAULT_MODEL,
    ratio: float = 0.75,
    options: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """
    Batch version: each input is a single JSON object (one chunk per file).
    If out_dir is given, writes each updated JSON next to the input filename.
    """
    results = []
    for p in inputs:
        p = Path(p)
        out_path = None
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_path = out_dir / p.name
        results.append(process_json_file(p, out_path, model=model, ratio=ratio, options=options))
    return results
