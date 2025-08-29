# image_cache_manager.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Tuple, Any
from pathlib import Path
import json, time, re, os, mimetypes, hashlib, logging
import requests

try:
    from PIL import Image
    import io
except Exception:
    Image = None

# Optional import; only needed if you use the replicate source
try:
    import replicate  # pip install replicate
except Exception:
    replicate = None

# -----------------------------
# Logging
# -----------------------------
def _mk_logger(name="image_cache_manager", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(level)
    return logger

# -----------------------------
# Paths & basic config (now passed in)
# -----------------------------
DEFAULT_EXT = ".jpg"

def _now_ts() -> float:
    return time.time()

def _map_path(overlay_dir: Path) -> Path:
    return overlay_dir / "_image_map.json"

def _load_map(overlay_dir: Path, logger: Optional[logging.Logger] = None) -> Dict:
    path = _map_path(overlay_dir)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if logger: logger.debug(f"Loaded map: {path}")
            return data
        except Exception as e:
            if logger: logger.warning(f"Failed to read map {path}: {e}")
    return {"version": 1, "updated": _now_ts(), "map": {}, "aliases": {}}

def _save_map(overlay_dir: Path, m: Dict, logger: Optional[logging.Logger] = None) -> None:
    m["version"] = int(m.get("version", 1))
    m["updated"] = _now_ts()
    overlay_dir.mkdir(parents=True, exist_ok=True)
    path = _map_path(overlay_dir)
    path.write_text(json.dumps(m, indent=2), encoding="utf-8")
    if logger: logger.debug(f"Saved map: {path}")

# -----------------------------
# Anchor normalization
# -----------------------------
_ARTICLES = {"the","a","an","this","that","these","those","its","their","his","her"}
_WORD_RE  = re.compile(r"[A-Za-z][A-Za-z'-]*")

def _tokens(s: str) -> List[str]:
    return _WORD_RE.findall(s)

def _is_article(tok: str) -> bool:
    return tok.lower() in _ARTICLES

def _anchor_from_phrase(phrase: str) -> str:
    """
    Examples:
      "a well constructed Union" -> "union"
      "the Union"                -> "union"
      "violence of faction"      -> "violence of faction"
      "State of New York"        -> "new york"
    """
    raw = _tokens(phrase)
    if not raw:
        return ""
    if _is_article(raw[0]):
        raw = raw[1:] or raw
    low = [t.lower() for t in raw]
    if "of" in low and 0 < low.index("of") < len(low) - 1:
        return " ".join(low)
    if len(raw) >= 2 and all(t[0].isupper() for t in raw[-2:]):
        return " ".join(t.lower() for t in raw[-2:])
    return raw[-1].lower()

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

# -----------------------------
# Errors
# -----------------------------
class ImageSourceError(RuntimeError): pass

# -----------------------------
# Wikimedia (fallback) fetcher
# -----------------------------
def _detect_ext_from_headers(headers: Dict[str, str]) -> str:
    ctype = headers.get("Content-Type", "")
    ext = mimetypes.guess_extension(ctype.split(";")[0].strip()) or DEFAULT_EXT
    return ".jpg" if ext in (".jpe", ".jpeg", ".jpg") else ext

def _http_get_bytes(url: str, timeout: float = 25.0) -> Tuple[bytes, str]:
    r = requests.get(url, timeout=timeout, stream=True)
    if r.status_code != 200:
        raise ImageSourceError(f"HTTP {r.status_code} for {url}")
    return r.content, _detect_ext_from_headers(r.headers)

def _convert_image_bytes(data: bytes, target_fmt: str) -> Tuple[bytes, str]:
    """
    Convert raw image bytes to target_fmt ('jpg'/'png'). Returns (bytes, ext).
    Requires Pillow; if unavailable, returns original data/extension.
    """
    if Image is None:
        return data, None  # no conversion possible

    target_fmt = target_fmt.lower()
    if target_fmt not in {"jpg", "jpeg", "png"}:
        return data, None
    ext = ".jpg" if target_fmt in {"jpg", "jpeg"} else ".png"

    im = Image.open(io.BytesIO(data)).convert("RGB" if ext == ".jpg" else "RGBA")
    out = io.BytesIO()
    save_kwargs = {"quality": 95} if ext == ".jpg" else {}
    im.save(out, format="JPEG" if ext == ".jpg" else "PNG", **save_kwargs)
    return out.getvalue(), ext


def fetch_from_wikiimages(query: str, logger: Optional[logging.Logger] = None) -> Tuple[bytes, str]:
    if logger: logger.info(f"[wikiimages] Searching for: {query!r}")
    srch = requests.get(
        "https://en.wikipedia.org/w/rest.php/v1/search/page",
        params={"q": query, "limit": 1},
        timeout=20
    )
    if srch.status_code != 200:
        raise ImageSourceError(f"wiki search failed {srch.status_code}")
    js = srch.json()
    pages = js.get("pages") or []
    if not pages:
        raise ImageSourceError("no wiki hits")

    page_id = pages[0].get("id")
    media = requests.get(
        f"https://en.wikipedia.org/w/rest.php/v1/page/{page_id}/media",
        timeout=25
    )
    if media.status_code != 200:
        raise ImageSourceError(f"wiki media fetch failed {media.status_code}")
    mjs = media.json()

    candidates = []
    for item in mjs.get("items", []):
        if item.get("type") == "image":
            src = None
            if "original" in item:
                src = item["original"].get("source")
            elif "srcset" in item and item["srcset"]:
                src = item["srcset"][-1].get("src")
            elif "src" in item:
                src = item.get("src")
            if src:
                candidates.append(src)

    if not candidates:
        raise ImageSourceError("no wiki images on page")

    url = candidates[0]
    if logger: logger.info(f"[wikiimages] Downloading: {url}")
    return _http_get_bytes(url)

# -----------------------------
# Replicate SDK fetcher (robust to strings / file-like)
# -----------------------------
def fetch_from_replicate_sdk(
    prompt: str,
    *,
    api_key: str,
    model: str = "black-forest-labs/flux-schnell",
    logger: Optional[logging.Logger] = None,
    **model_inputs
) -> Tuple[bytes, str]:
    """
    Uses Replicate's official Python SDK.
    Accepts both file-like outputs and plain URL strings.
    Returns: (bytes, extension)
    """
    if replicate is None:
        raise ImageSourceError("replicate SDK not installed. pip install replicate")

    client = replicate.Client(api_token=api_key)
    full_prompt = model_inputs.pop("prompt_override", None) or \
        f"In a 1780s style illustration output an image that represents the concept of {prompt}"

    # Get output format from model_inputs or default to "jpg"
    output_format = model_inputs.pop("output_format", "jpg")
    
    if logger: logger.info(f"[replicate] Running model={model} prompt={full_prompt!r}")
    output = client.run(model, input={"prompt": full_prompt, "output_format": output_format, **model_inputs})

    if not output:
        raise ImageSourceError("replicate returned no outputs")

    first = output[0]

    # normalize outputs into (data, ext_detected)
    if hasattr(first, "read"):  # SDK file-like
        url = None
        if hasattr(first, "url"):
            try:
                url = first.url()
            except Exception:
                url = None
        data = first.read()
        ext_detected = (os.path.splitext(url)[1].lower() if url else None)
    elif isinstance(first, str):  # plain URL
        if logger: logger.info(f"[replicate] Downloading output URL")
        data, ext_detected = _http_get_bytes(first)
    elif isinstance(first, (bytes, bytearray)):  # raw bytes
        data, ext_detected = (bytes(first), None)
    else:
        raise ImageSourceError(f"replicate output not recognized: {type(first)}")

    # If caller requested a specific format (jpg/png), convert if needed
    requested = output_format.lower() if isinstance(output_format, str) else None
    if requested in {"jpg", "jpeg", "png"}:
        # If detected extension doesn't match request, or is None, convert
        if ext_detected not in {f".{requested}", ".jpeg"}:
            conv_data, conv_ext = _convert_image_bytes(data, requested)
            if conv_ext:  # conversion succeeded (Pillow available)
                return conv_data, conv_ext
            # else: fall through and return original bytes/ext

    # Fallback: return original; pick a sensible ext
    if ext_detected in {".jpg", ".jpeg", ".png", ".webp"}:
        return data, ext_detected
    # default if unknown
    return data, ".webp"


    # Unexpected format
    raise ImageSourceError(f"replicate output not recognized: {type(first)}")

# -----------------------------
# Source wrapper (Replicate default, Wikimedia fallback)
# -----------------------------
def fetch_image_with_fallback(
    phrase: str,
    *,
    preferred: str = "replicate",
    replicate_api_key: Optional[str] = None,
    replicate_model: str = "black-forest-labs/flux-schnell",
    replicate_inputs: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bytes, str, str]:
    """
    Returns (bytes, ext, source_used)
    Tries preferred source, then falls back to the other.
    """
    replicate_inputs = replicate_inputs or {}
    errors = []
    if logger: logger.debug(f"fetch_image_with_fallback preferred={preferred}")

    if preferred == "replicate":
        try:
            if not replicate_api_key:
                raise ImageSourceError("replicate_api_key is required when preferred='replicate'")
            data, ext = fetch_from_replicate_sdk(
                phrase, api_key=replicate_api_key, model=replicate_model,
                logger=logger, **replicate_inputs
            )
            if logger: logger.info("[replicate] Succeeded")
            return data, ext, "replicate"
        except Exception as e:
            errors.append(f"replicate: {e}")
            if logger: logger.warning(f"[replicate] Failed, falling back to wikiimages: {e}")
            try:
                data, ext = fetch_from_wikiimages(phrase, logger=logger)
                if logger: logger.info("[wikiimages] Succeeded (fallback)")
                return data, ext, "wikiimages"
            except Exception as e2:
                errors.append(f"wikiimages: {e2}")
                raise ImageSourceError("; ".join(errors))

    elif preferred == "wikiimages":
        try:
            data, ext = fetch_from_wikiimages(phrase, logger=logger)
            if logger: logger.info("[wikiimages] Succeeded")
            return data, ext, "wikiimages"
        except Exception as e:
            errors.append(f"wikiimages: {e}")
            if logger: logger.warning(f"[wikiimages] Failed, falling back to replicate: {e}")
            try:
                if not replicate_api_key:
                    raise ImageSourceError("replicate_api_key required for fallback to replicate")
                data, ext = fetch_from_replicate_sdk(
                    phrase, api_key=replicate_api_key, model=replicate_model,
                    logger=logger, **replicate_inputs
                )
                if logger: logger.info("[replicate] Succeeded (fallback)")
                return data, ext, "replicate"
            except Exception as e2:
                errors.append(f"replicate: {e2}")
                raise ImageSourceError("; ".join(errors))
    else:
        raise ValueError("preferred must be 'replicate' or 'wikiimages'")

# -----------------------------
# Public API
# -----------------------------
@dataclass
class FetchResult:
    image_path: Path
    anchor_key: str
    from_cache: bool
    source_used: Optional[str]  # None if from cache

def get_or_create_image_for_phrase(
    phrase: str,
    *,
    overlay_dir: Path | str,                       # NEW: configurable
    preferred_source: str = "replicate",
    replicate_api_key: Optional[str] = None,       # pass explicitly
    replicate_model: str = "black-forest-labs/flux-schnell",
    replicate_inputs: Optional[Dict[str, Any]] = None,
    anchor_override: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> FetchResult:
    """
    Resolve an image for the phrase via cache → (replicate → wikiimages) with a stable anchor key.
    """
    overlay_dir = Path(overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    logger = logger or _mk_logger()

    mapping = _load_map(overlay_dir, logger=logger)

    anchor = (anchor_override or _anchor_from_phrase(phrase)).strip()
    if not anchor:
        raise ValueError(f"Could not derive anchor from phrase: {phrase!r}")

    # Alias fast‑path
    alias_key = mapping.get("aliases", {}).get(phrase.lower())
    if alias_key:
        mapped = mapping["map"].get(alias_key)
        if mapped and Path(mapped).exists():
            if logger: logger.info(f"[cache] HIT alias='{phrase.lower()}' -> anchor='{alias_key}' -> {mapped}")
            return FetchResult(Path(mapped), alias_key, True, None)

    # Direct anchor fast‑path
    mapped = mapping["map"].get(anchor)
    if mapped and Path(mapped).exists():
        mapping["aliases"][phrase.lower()] = anchor
        _save_map(overlay_dir, mapping, logger=logger)
        if logger: logger.info(f"[cache] HIT anchor='{anchor}' -> {mapped}")
        return FetchResult(Path(mapped), anchor, True, None)

    # Need to fetch/generate
    if logger: logger.info(f"[cache] MISS for anchor='{anchor}' (phrase: {phrase!r})")
    data, ext, used = fetch_image_with_fallback(
        phrase,
        preferred=preferred_source,
        replicate_api_key=replicate_api_key,
        replicate_model=replicate_model,
        replicate_inputs=replicate_inputs or {},
        logger=logger,
    )

    # Save under anchor‑based filename
    ext = ext if ext.startswith(".") else f".{ext}"
    fname = f"{_slug(anchor)}{ext}"
    out_path = overlay_dir / fname
    if out_path.exists():
        h = hashlib.sha1((phrase + str(_now_ts())).encode("utf-8")).hexdigest()[:6]
        out_path = overlay_dir / f"{_slug(anchor)}_{h}{ext}"
    out_path.write_bytes(data)
    if logger: logger.info(f"[save] Wrote image -> {out_path}")

    # Update map & alias
    mapping["map"][anchor] = str(out_path)
    mapping["aliases"][phrase.lower()] = anchor
    _save_map(overlay_dir, mapping, logger=logger)

    return FetchResult(out_path, anchor, False, used)

def ensure_images_for_phrases(
    phrases: List[str],
    *,
    overlay_dir: Path | str,                       # NEW: configurable
    preferred_source: str = "replicate",
    replicate_api_key: Optional[str] = None,
    replicate_model: str = "black-forest-labs/flux-schnell",
    replicate_inputs: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    progress: bool = True,
) -> Dict[str, str]:
    """
    Bulk resolve: returns {phrase -> image_path}; reuses cache where possible.
    """
    overlay_dir = Path(overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    logger = logger or _mk_logger()
    results: Dict[str, str] = {}
    n = len(phrases)
    if progress and logger:
        logger.info(f"Ensuring images for {n} phrase(s) into {overlay_dir}")

    for i, p in enumerate(phrases, 1):
        if progress and logger:
            logger.info(f"[{i}/{n}] Resolving: {p!r}")
        fr = get_or_create_image_for_phrase(
            p,
            overlay_dir=overlay_dir,
            preferred_source=preferred_source,
            replicate_api_key=replicate_api_key,
            replicate_model=replicate_model,
            replicate_inputs=replicate_inputs,
            logger=logger,
        )
        results[p] = str(fr.image_path)
        if progress and logger:
            logger.info(f"[{i}/{n}] ✓ {p!r} -> {fr.image_path} (source={fr.source_used or 'cache'})")

    if progress and logger:
        logger.info("All phrases resolved.")
    return results
