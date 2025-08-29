# code/scripts/split_text.py
"""
Backwards-compatible wrapper that delegates to ingest_folder().
This reads raw JSON from data/raw_texts and writes processed JSON
(master + chunks + manifest) into data/processed_texts.
"""
from __future__ import annotations
from pathlib import Path
from .ingest import ingest_folder

def split_raw_texts(
    input_dir: str = "data/raw_texts",
    output_dir: str = "data/processed_texts",
    target_seconds: int = 60,
    wpm: int = 160
) -> None:
    ingest_folder(Path(input_dir), Path(output_dir), target_seconds=target_seconds, wpm=wpm)

if __name__ == "__main__":
    split_raw_texts()
