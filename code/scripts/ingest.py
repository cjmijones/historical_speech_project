# code/scripts/ingest.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List, Dict

from pydantic import BaseModel
from .models import RawDocument, ProcessedDocument, ProcessedChunk
from .utils import normalize_whitespace, slugify
from .chunker import chunk_by_target_seconds

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: Any) -> None:
    """
    Save either a plain dict/list or a Pydantic BaseModel.
    For BaseModel we use model_dump_json() to ensure JSON-safe types (HttpUrl, datetime, etc.).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, BaseModel):
        path.write_text(obj.model_dump_json(indent=2), encoding="utf-8")
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_raw(raw: RawDocument) -> ProcessedDocument:
    clean_body = normalize_whitespace(raw.body)
    word_count = len(clean_body.split())
    char_count = len(clean_body)
    est_sec = round(word_count * 60.0 / 160.0, 2)  # default 160 wpm estimate

    norm = {
        "id": raw.id,
        "title": raw.title.strip(),
        "author": raw.author.strip(),
        "orator": (raw.orator or raw.author).strip(),
        "date": raw.date.strip(),
        "language": raw.language.strip(),
        "public_domain": raw.public_domain,
        "source_title": raw.source_title.strip(),
        "source_url": str(raw.source_url) if raw.source_url else None,
        "source_archive_url": str(raw.source_archive_url) if raw.source_archive_url else None,
        "license": raw.license.strip(),
        "tags": raw.tags,
        "topics": raw.topics,
        "translator": raw.translator,
        "editor": raw.editor,
        "location": raw.location,
        "edition": raw.edition,
        "notes": raw.notes,
    }

    return ProcessedDocument(
        **norm,
        normalized={"body": clean_body},
        word_count=word_count,
        char_count=char_count,
        est_read_time_sec=est_sec,
    )

def chunk_document(
    doc: ProcessedDocument,
    target_seconds: int = 60,
    wpm: int = 160
) -> List[ProcessedChunk]:
    body = doc.normalized["body"]
    tuples = chunk_by_target_seconds(body, target_seconds=target_seconds, wpm=wpm)
    chunks: List[ProcessedChunk] = []
    for idx, (text, c0, c1, w0, w1, est_sec) in enumerate(tuples):
        chunk_id = slugify(f"{doc.id}-part-{idx+1:02d}")
        chunks.append(ProcessedChunk(
            parent_id=doc.id,
            chunk_id=chunk_id,
            index=idx,
            text=text,
            approx_word_count=w1 - w0,
            est_duration_sec=round(est_sec, 2),
            start_char=c0,
            end_char=c1,
            start_word=w0,
            end_word=w1,
            meta={
                "title": doc.title,
                "author": doc.author,
                "orator": doc.orator,
                "date": doc.date,
                "language": doc.language,
                "source_title": doc.source_title,
                "source_url": str(doc.source_url) if doc.source_url else None,
                "license": doc.license
            }
        ))
    return chunks

def ingest_folder(
    input_dir: Path,
    output_dir: Path,
    target_seconds: int = 60,
    wpm: int = 160
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    raws = list(input_dir.glob("*.json"))
    if not raws:
        print(f"[ingest] No JSON files in {input_dir}")
        return

    for p in raws:
        data = load_json(p)
        try:
            raw = RawDocument(**data)
        except Exception as e:
            print(f"[ingest] Validation failed for {p.name}: {e}")
            continue

        processed = normalize_raw(raw)

        # Write normalized master  ⟵ CHANGED: pass the model, not dict
        master_out = output_dir / f"{processed.id}.json"
        save_json(master_out, processed)  # was processed.model_dump()

        # Chunk + write chunks and manifest
        chunks = chunk_document(processed, target_seconds=target_seconds, wpm=wpm)
        chunk_dir = output_dir / f"{processed.id}_chunks"
        manifest = []
        for ch in chunks:
            cpath = chunk_dir / f"{ch.chunk_id}.json"
            save_json(cpath, ch)  # ⟵ CHANGED: pass the model, not dict
            manifest.append({
                "chunk_id": ch.chunk_id,
                "index": ch.index,
                "approx_word_count": ch.approx_word_count,
                "est_duration_sec": ch.est_duration_sec,
                "file": cpath.name
            })

        save_json(chunk_dir / "manifest.json", {
            "parent_id": processed.id,
            "count": len(chunks),
            "target_seconds": target_seconds,
            "wpm": wpm,
            "chunks": manifest
        })

        print(f"[ingest] {p.name} → {master_out.name} + {len(chunks)} chunks")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Validate, normalize, and chunk raw JSON texts.")
    ap.add_argument("--input", default="data/raw_texts", help="Folder of raw .json files")
    ap.add_argument("--output", default="data/processed_texts", help="Folder to write processed JSON + chunks")
    ap.add_argument("--target-seconds", type=int, default=60, help="Target duration per chunk")
    ap.add_argument("--wpm", type=int, default=160, help="Words per minute (TTS pacing)")
    args = ap.parse_args()

    ingest_folder(Path(args.input), Path(args.output), target_seconds=args.target_seconds, wpm=args.wpm)
