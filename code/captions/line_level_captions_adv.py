#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
from typing import List, Dict, Any, Tuple

# ----------------------------- spaCy (optional) -----------------------------
def _ensure_sentencizer(nlp):
    """Ensure sentence boundaries exist even if parser/senter isn't present."""
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

try:
    import spacy  # type: ignore
    _NLP = _ensure_sentencizer(spacy.load("en_core_web_sm"))
except Exception:
    _NLP = None  # graceful fallback without spaCy

_EOS_LINE_RE = re.compile(r"[.?!][\"')\]]*\s*$")  # end-of-sentence at the end of a (sub)string
_ENTITY_LEADERS = {"new", "los", "las", "san", "santa", "st", "st.", "saint", "fort"}


# ----------------------------- I/O helper -----------------------------
def output_subtitles_to_json(subtitles: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subtitles, f, indent=4, ensure_ascii=False)


# ----------------------------- Line splitting -----------------------------
def split_text_into_lines(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert word-level ASR tokens into line-level caption entries.

    Each input item: {"word": str, "start": float, "end": float, ...}
    Output entries:
      {
        "word": "<display text>",
        "start": float,
        "end": float,
        "textcontents": [ original word dicts ... ]
      }
    """
    MaxChars = 30        # max characters in a line (approx)
    MaxDuration = 2.5    # seconds
    MaxGap = 1.5         # seconds (split on long silences)

    subtitles: List[Dict[str, Any]] = []
    line: List[Dict[str, Any]] = []
    line_duration = 0.0

    for idx, word_data in enumerate(data):
        line.append(word_data)
        line_duration += word_data["end"] - word_data["start"]

        temp_text = " ".join(item["word"] for item in line)  # for char count only
        new_line_chars = len(temp_text)

        duration_exceeded = line_duration > MaxDuration
        chars_exceeded = new_line_chars > MaxChars

        if idx > 0:
            gap = word_data["start"] - data[idx - 1]["end"]
            maxgap_exceeded = gap > MaxGap
        else:
            maxgap_exceeded = False

        if duration_exceeded or chars_exceeded or maxgap_exceeded:
            if line:
                subtitles.append({
                    "word": " ".join(item["word"] for item in line),
                    "start": line[0]["start"],
                    "end": line[-1]["end"],
                    "textcontents": line[:],  # copy
                })
                line = []
                line_duration = 0.0

    if line:
        subtitles.append({
            "word": " ".join(item["word"] for item in line),
            "start": line[0]["start"],
            "end": line[-1]["end"],
            "textcontents": line[:],
        })

    return subtitles


# ----------------------------- Capitalization helpers -----------------------------
def _split_ws_core(word: str) -> Tuple[str, str, str]:
    """
    Split a token into (leading_whitespace, core, trailing_whitespace).
    Core may include punctuation (e.g., 'York.').
    """
    m = re.match(r"^(\s*)(\S+?)(\s*)$", word)
    if not m:
        # Pure whitespace or empty (unlikely for ASR tokens); treat as leading ws.
        return (word, "", "")
    return m.group(1), m.group(2), m.group(3)


def _smart_word_cap(core: str) -> str:
    """
    Title-case a core token while preserving:
      - ALLCAPS acronyms (USA, NATO)
      - Mixed-case brands (iPhone, eBay)
      - Inner hyphen/apostrophe (Jean-Luc, O'Neill)
    """
    if not core:
        return core

    alpha = re.sub(r"[^A-Za-z]", "", core)
    if alpha and (alpha.isupper() or (not alpha.islower() and not alpha.isupper())):
        return core  # keep ALLCAPS or mixed-case as-is

    m = re.match(r"^([\w'\-]+)$", core)
    if not m:
        # If there are surrounding punct chars, title-case only the word-ish middle part
        m2 = re.match(r"^(\W*)([\w'\-]+)(\W*)$", core)
        if not m2:
            return core
        pre, mid, post = m2.groups()
        parts = re.split(r"([\'-])", mid)
        cap_parts = [p.capitalize() if p.isalpha() else p for p in parts]
        return f"{pre}{''.join(cap_parts)}{post}"

    # core is purely word-ish
    parts = re.split(r"([\'-])", core)
    cap_parts = [p.capitalize() if p.isalpha() else p for p in parts]
    return "".join(cap_parts)


def _token_ends_sentence(core: str) -> bool:
    """Return True if core looks like it ends with sentence punctuation."""
    return bool(re.search(r"[.?!][\"')\]]*$", core))


def _build_join_and_spans_from_cores(cores: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Join cores with single spaces and return char spans [start,end) for each original core.
    This is used to map spaCy tokens back to word indices.
    """
    spans: List[Tuple[int, int]] = []
    cur = 0
    out_parts: List[str] = []
    for i, c in enumerate(cores):
        start = cur
        out_parts.append(c)
        cur += len(c)
        end = cur
        spans.append((start, end))
        if i < len(cores) - 1:
            out_parts.append(" ")
            cur += 1
    return "".join(out_parts), spans


def _capitalize_line(words: List[str], force_sentence_start_cap: bool) -> Tuple[str, List[str], bool]:
    """
    Capitalize a line with two phases:
      1) Intra-line sentence starts: capitalize first alphabetic core after . ? !
      2) NER/PROPN pass (spaCy if available), plus a small leader-word heuristic (New/Los/San...)

    Returns: (display_text, updated_token_strings, ends_sentence)
    """
    if not words:
        return "", [], False

    # Parse tokens into whitespace + core pieces to preserve spacing.
    pres, cores, posts = [], [], []
    for w in words:
        pre, core, post = _split_ws_core(w)
        pres.append(pre)
        cores.append(core)
        posts.append(post)

    # -------- Phase 1: intra-line sentence starts --------
    at_sentence_start = force_sentence_start_cap
    phase1 = []
    for i, core in enumerate(cores):
        new_core = core
        # Only capitalize the first *alphabetic* core at a sentence start
        if at_sentence_start and re.search(r"[A-Za-z]", core):
            if core[:1].islower():
                new_core = core[:1].upper() + core[1:]
            at_sentence_start = False
        # After placing this word, if it ends a sentence, the next alpha core starts a sentence
        if _token_ends_sentence(new_core):
            at_sentence_start = True
        phase1.append(new_core)

    # -------- Phase 2: NER/PROPN + small leader heuristic --------
    is_ent_or_propn = [False] * len(phase1)
    if _NLP is not None:
        joined, spans = _build_join_and_spans_from_cores(phase1)
        doc = _NLP(joined)
        # Mark indices whose char spans overlap any PROPN or any token inside an entity
        for idx, (start, end) in enumerate(spans):
            toks = [t for t in doc if not (t.idx >= end or (t.idx + len(t)) <= start)]
            if any(t.pos_ == "PROPN" or t.ent_iob_ != "O" for t in toks):
                is_ent_or_propn[idx] = True

    phase2 = phase1[:]
    for i, core in enumerate(phase1):
        if is_ent_or_propn[i]:
            # Title-case this core unless it's ALLCAPS
            if not (core.isupper() and core.isalpha()):
                phase2[i] = _smart_word_cap(core)

    # Leader heuristic: if a PROPN/entity follows a known leader (new/los/las/san/...), capitalize leader.
    for i in range(1, len(phase2)):
        if is_ent_or_propn[i]:
            prev = phase2[i - 1]
            if prev and prev.isalpha() and prev.lower() in _ENTITY_LEADERS:
                phase2[i - 1] = _smart_word_cap(prev)

    # Reassemble tokens (preserve original leading whitespace per token); no extra joining spaces
    updated_tokens = [f"{pre}{core}{post}" for pre, core, post in zip(pres, phase2, posts)]
    display_text = "".join(updated_tokens).lstrip()  # trim any leading ws for display aesthetics
    ends_sentence = bool(_EOS_LINE_RE.search(display_text))
    return display_text, updated_tokens, ends_sentence


def apply_capitalization_to_subtitles(subtitles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Post-process line-level subtitles sequentially, carrying sentence-boundary context across lines.

    Rules:
      - Only capitalize the first word in a line if the previous line ended with . ! ?
      - Always title-case proper nouns (PROPN) and named entities when spaCy is available.
      - Apply the same capitalization to each token in `textcontents[i]["word"]`.
    """
    results: List[Dict[str, Any]] = []
    prev_ended_sentence = True  # treat start of transcript as sentence start

    for item in subtitles:
        token_strs = [w["word"] for w in item.get("textcontents", [])]
        display, updated_tokens, ends_sentence = _capitalize_line(
            token_strs, force_sentence_start_cap=prev_ended_sentence
        )

        # Write back to textcontents
        new_textcontents = []
        for wdict, neww in zip(item.get("textcontents", []), updated_tokens):
            wnew = dict(wdict)
            wnew["word"] = neww
            new_textcontents.append(wnew)

        new_item = dict(item)
        new_item["word"] = display
        new_item["textcontents"] = new_textcontents

        results.append(new_item)
        prev_ended_sentence = ends_sentence

    return results

def split_lines_with_capitalization(wordlevel_json="./temp/word_timestamps.json", out_path="./temp/word_line_timestamps.json"):

    with open(wordlevel_json, "r", encoding="utf-8") as f:
        wordlevel_info = json.load(f)

    linelevel_subtitles = split_text_into_lines(wordlevel_info)

    linelevel_subtitles = apply_capitalization_to_subtitles(linelevel_subtitles)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Write output
    output_subtitles_to_json(linelevel_subtitles, out_path)


# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render captions from word-level info JSON.")
    parser.add_argument(
        "--wordlevel_json",
        type=str,
        default="./temp/word_timestamps.json",
        help="Path to the word-level info JSON file (default: temp/word_timestamps.json)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./temp/word_line_timestamps.json",
        help="Path to output line-level JSON file (default: temp/word_line_timestamps.json)",
    )
    args = parser.parse_args()

    # Load word-level input
    with open(args.wordlevel_json, "r", encoding="utf-8") as f:
        wordlevel_info = json.load(f)

    # 1) Split into lines
    linelevel_subtitles = split_text_into_lines(wordlevel_info)

    # 2) Apply capitalization across lines and tokens
    linelevel_subtitles = apply_capitalization_to_subtitles(linelevel_subtitles)

    # Optional: preview
    # for line in linelevel_subtitles:
    #     print(json.dumps(line, indent=4, ensure_ascii=False))

    # Ensure output dir exists
    out_dir = os.path.dirname(args.out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Write output
    output_subtitles_to_json(linelevel_subtitles, args.out_path)
