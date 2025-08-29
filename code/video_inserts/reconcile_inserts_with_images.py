# reconcile_inserts_with_images.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json, re, logging, argparse

from video_inserts.image_cache_manager import ensure_images_for_phrases

# Matches "keyword: ..." (case-insensitive) and removes the prefix
KEYWORD_PREFIX_RE = re.compile(r"^\s*keyword\s*:\s*", re.IGNORECASE)


# -----------------------------
# Logging
# -----------------------------
def _mk_logger(name="reconcile_inserts", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


# -----------------------------
# Helpers
# -----------------------------
def _extract_phrases_from_inserts_doc(doc: Dict[str, Any], logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Prefer doc['keywords'] if available; otherwise parse phrases from inserts[*]['reason'].
    Accepts formats like 'keyword: The People' (case-insensitive).
    Returns a de-duplicated list (case-insensitive) preserving first-seen order.
    """
    phrases: List[str] = []

    # 1) Top-level keywords (highest priority)
    if isinstance(doc.get("keywords"), list) and doc["keywords"]:
        phrases = [str(p).strip() for p in doc["keywords"] if str(p).strip()]
        if logger:
            logger.info(f"Found {len(phrases)} phrase(s) from doc['keywords'].")

    # 2) Else, collect from inserts[*]['reason']
    if not phrases:
        for ins in doc.get("inserts", []):
            reason = ins.get("reason", "")
            if not reason:
                continue
            ph = KEYWORD_PREFIX_RE.sub("", reason).strip()
            if ph:
                phrases.append(ph)
        if logger:
            logger.info(f"Found {len(phrases)} phrase(s) from inserts[*]['reason'].")

    # 3) De-duplicate (case-insensitive) while preserving order
    seen = set()
    uniq: List[str] = []
    for p in phrases:
        pl = p.lower()
        if pl not in seen:
            uniq.append(p)
            seen.add(pl)

    if logger:
        logger.info(f"{len(uniq)} unique phrase(s) after de-duplication.")
    return uniq


def _normalize_sep(p: str) -> str:
    # Ensure Windows/Unix friendly stored path
    return p.replace("\\", "/")  # store forward slashes in JSON for portability


# -----------------------------
# Public API
# -----------------------------
def reconcile_inserts_json(
    in_path: Path | str,
    out_path: Path | str | None = None,
    *,
    overlay_dir: Path | str = "assets/overlay_images",
    preferred_source: str = "replicate",
    replicate_api_key: str | None = None,
    replicate_model: str = "black-forest-labs/flux-schnell",
    replicate_inputs: Dict[str, Any] | None = None,
    log_level: int = logging.INFO,
) -> Dict[str, Any]:
    """
    Reads an inserts JSON, ensures images exist for all phrases (via image_cache_manager),
    updates each insert['asset'] to the resolved image path, and writes the updated JSON.

    Returns: the updated JSON dict.
    """
    logger = _mk_logger(level=log_level)

    in_path = Path(in_path)
    out_path = Path(out_path) if out_path else in_path
    overlay_dir = Path(overlay_dir)

    logger.info(f"Loading inserts JSON: {in_path}")
    doc = json.loads(in_path.read_text(encoding="utf-8"))

    phrases = _extract_phrases_from_inserts_doc(doc, logger=logger)
    if not phrases:
        raise ValueError("No phrases found in JSON (keywords or reason fields).")

    logger.info(f"Resolving images for {len(phrases)} phrase(s) into: {overlay_dir}")
    phrase_to_path = ensure_images_for_phrases(
        phrases,
        overlay_dir=overlay_dir,
        preferred_source=preferred_source,
        replicate_api_key=replicate_api_key,
        replicate_model=replicate_model,
        replicate_inputs=replicate_inputs or {"width": 1024},
        logger=logger,
        progress=True,
    )

    # Update each insert's asset path based on its phrase
    updates = 0
    for ins in doc.get("inserts", []):
        reason = ins.get("reason", "")
        phrase = KEYWORD_PREFIX_RE.sub("", reason).strip() if reason else ""
        if not phrase:
            continue
        new_path = phrase_to_path.get(phrase)
        if new_path:
            old_asset = ins.get("asset")
            ins["asset"] = _normalize_sep(new_path)
            updates += 1
            logger.info(f"Updated asset for '{phrase}': {old_asset} -> {ins['asset']}")
        else:
            logger.warning(f"No resolved image path for phrase: {phrase!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    logger.info(f"Saved updated inserts JSON: {out_path} (assets updated: {updates})")

    return doc


# -----------------------------
# CLI
# -----------------------------
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconcile inserts JSON with cached/generated overlay images.")
    p.add_argument("--in", dest="in_path", required=True, help="Path to the input inserts JSON.")
    p.add_argument("--out", dest="out_path", default=None, help="Optional output path. Defaults to overwrite input.")
    p.add_argument("--overlay-dir", dest="overlay_dir", default="assets/overlay_images", help="Directory to store overlay images and _image_map.json.")
    p.add_argument("--preferred-source", dest="preferred_source", default="replicate", choices=("replicate", "wikiimages"), help="Primary source (fallback will be the other).")
    p.add_argument("--replicate-api-key", dest="replicate_api_key", default=None, help="Replicate API key.")
    p.add_argument("--replicate-model", dest="replicate_model", default="black-forest-labs/flux-schnell", help="Replicate model identifier or version hash.")
    p.add_argument("--width", type=int, default=1024, help="Image width for Replicate (if applicable).")
    p.add_argument("--height", type=int, default=None, help="Image height for Replicate (optional).")
    p.add_argument("--log-level", dest="log_level", default="INFO", choices=("DEBUG","INFO","WARNING","ERROR","CRITICAL"), help="Logging verbosity.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    replicate_inputs: Dict[str, Any] = {"width": args.width}
    if args.height:
        replicate_inputs["height"] = args.height

    reconcile_inserts_json(
        args.in_path,
        out_path=args.out_path,
        overlay_dir=args.overlay_dir,
        preferred_source=args.preferred_source,
        replicate_api_key=args.replicate_api_key,
        replicate_model=args.replicate_model,
        replicate_inputs=replicate_inputs,
        log_level=level,
    )
