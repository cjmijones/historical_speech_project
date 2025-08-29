# keyphrase_extractor.py
"""
Keyword/Phrase extractor with spaCy (preferred) and a regex fallback.
- Prefers noun phrases and salient POS patterns (Adj|Noun combos)
- Scores phrases by frequency, early-position bonus, and length bonus
- Returns top_k unique phrases preserving their first-occurrence order

Usage:
    python keyphrase_extractor.py "Your text here"
or import:
    from keyphrase_extractor import extract_keyphrases
"""

from __future__ import annotations
from typing import List, Tuple, Iterable
import re
import sys
from collections import Counter, defaultdict

# ---------------------------
# Stopwords (compact English set)
# ---------------------------
_STOPWORDS = {
    "a","an","and","are","as","at","be","been","but","by","for","from","had","has",
    "have","he","her","hers","him","his","i","in","into","is","it","its","itself",
    "me","my","myself","nor","of","on","or","our","ours","ourselves","she","so",
    "that","the","their","theirs","them","themselves","there","these","they","this",
    "to","too","was","were","what","when","where","which","who","whom","why","will",
    "with","you","your","yours","yourself","yourselves","we","us","do","does","did",
    "not","no","than","then","if","shall","may","might","can","could","would","should",
    "none"
}

# Add near the other constants
_ARTICLES = {"the","a","an","this","that","these","those","its","their","his","her"}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

def _leading_article(str_tokens: list[str]) -> tuple[list[str], bool]:
    """Remove a single leading article if present; return (tokens_wo_article, had_article?)."""
    if str_tokens and str_tokens[0].lower() in _ARTICLES:
        return str_tokens[1:], True
    return str_tokens, False

def _core_token_set(phrase: str) -> tuple[frozenset[str], bool]:
    """
    Tokens lowercased; ignore one leading article for 'core' comparison.
    Returns (core_token_set, had_leading_article).
    """
    toks = _WORD_RE.findall(phrase)
    toks_wo_art, had_art = _leading_article(toks)
    return frozenset(t.lower() for t in toks_wo_art if t), had_art

def _prune_contained_variants(phrases: list[tuple[str, float, int]]) -> list[tuple[str, float, int]]:
    """
    Given [(phrase, score, pos)], drop any phrase whose *core* token set
    is a subset of a longer kept phrase's *core* token set.
    Preference order when cores tie/overlap:
      - more tokens first (longer phrase)
      - has leading article beats no-article (keeps 'The People' over 'People')
      - higher score
      - earlier position
    """
    with_meta = []
    for p, sc, pos in phrases:
        core_set, has_art = _core_token_set(p)
        with_meta.append((p, sc, pos, core_set, has_art, len(core_set)))

    # sort so we try to keep better reps first
    with_meta.sort(key=lambda x: (-x[5], not x[4], -x[1], x[2]))
    kept: list[tuple[str, float, int, frozenset[str], bool, int]] = []

    for cand in with_meta:
        _, _, _, core_i, _, _ = cand
        # if this core is subset of any kept core, drop it
        if any(core_i and core_i.issubset(core_k) for *_, core_k, __, ___ in kept):
            continue
        kept.append(cand)

    # restore by appearance for stable reading
    kept.sort(key=lambda x: x[2])
    return [(p, sc, pos) for (p, sc, pos, *_rest) in kept]


# ---------------------------
# Try spaCy (with model) else fallback
# ---------------------------
def _load_spacy():
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            # Model missing; allow user to still run (fallback path)
            return None
    except Exception:
        return None

_NLP = _load_spacy()

# ---------------------------
# Text cleaning helpers
# ---------------------------
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[^\w]+|[^\w]+$")

def _clean_token(tok: str) -> str:
    tok = _PUNCT_EDGE_RE.sub("", tok)
    return tok

def _normalize_space(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip()

# ---------------------------
# Candidate generation (spaCy)
# ---------------------------
def _spacy_candidates(text: str) -> Iterable[Tuple[str, int]]:
    """
    Yield (phrase, start_char_index) using spaCy noun_chunks and POS patterns.
    """
    # We re-load inside to avoid None doc if _NLP became None
    import spacy  # safe; already available if this is called
    nlp = _NLP or spacy.blank("en")  # blank if model missing (no noun_chunks)
    doc = nlp(text)

    # If we have a pipeline with parser/pos, we can do noun_chunks & patterns
    has_parser = "parser" in nlp.pipe_names or "senter" in nlp.pipe_names
    has_tagger = "tagger" in nlp.pipe_names or "morphologizer" in nlp.pipe_names

    yielded = set()

    if has_parser and hasattr(doc, "noun_chunks"):
        for nc in doc.noun_chunks:
            phrase = _normalize_space(nc.text)
            phrase = _clean_token(phrase)
            if phrase and phrase.lower() not in _STOPWORDS and not phrase.isdigit():
                key = phrase.lower()
                if key not in yielded:
                    yielded.add(key)
                    yield (phrase, nc.start_char)

    if has_tagger:
        # POS pattern: (Adj|NOUN)+ NOUN  (e.g., "well constructed Union", "popular governments")
        # Build spans from POS tags
        i = 0
        while i < len(doc):
            tok = doc[i]
            if tok.pos_ in {"ADJ", "NOUN", "PROPN"}:
                j = i
                seen_noun = tok.pos_ in {"NOUN", "PROPN"}
                while j + 1 < len(doc) and doc[j + 1].pos_ in {"ADJ", "NOUN", "PROPN"}:
                    j += 1
                    if doc[j].pos_ in {"NOUN", "PROPN"}:
                        seen_noun = True
                if seen_noun:
                    span = doc[i:j+1]
                    phrase = _normalize_space(span.text)
                    phrase = _clean_token(phrase)
                    if phrase and phrase.lower() not in _STOPWORDS and not phrase.isdigit():
                        key = phrase.lower()
                        if key not in yielded:
                            yielded.add(key)
                            yield (phrase, span.start_char)
                i = j + 1
            else:
                i += 1

    # Fallback inside spaCy path: if nothing yielded, do regex n-grams
    if not yielded:
        yield from _regex_ngram_candidates(text)

# ---------------------------
# Candidate generation (regex fallback)
# ---------------------------
def _regex_ngram_candidates(text: str, n_max: int = 4) -> Iterable[Tuple[str, int]]:
    """
    Very simple fallback: tokenize by non-letters, drop stopwords,
    yield 1–4 gram candidates with character offsets.
    """
    tokens = []
    # Track original positions for start_char index
    for m in re.finditer(r"[A-Za-z][A-Za-z'-]*", text):
        tok = m.group(0)
        lower = tok.lower()
        if lower not in _STOPWORDS:
            tokens.append((tok, m.start()))

    # unigrams to 4-grams
    for n in range(1, n_max + 1):
        for i in range(0, len(tokens) - n + 1):
            words, start = zip(*tokens[i:i+n])
            phrase = " ".join(words)
            phrase = _normalize_space(phrase)
            phrase = _clean_token(phrase)
            if phrase and phrase.lower() not in _STOPWORDS and not phrase.isdigit():
                yield (phrase, start[0])

# ---------------------------
# Scoring
# ---------------------------
def _score_candidates(cands: Iterable[Tuple[str, int]], text_len: int) -> List[Tuple[str, float, int]]:
    """
    Score by:
      - frequency
      - early position bonus (earlier start => higher score)
      - length bonus (more words up to 4)
    """
    freq = Counter()
    first_pos = {}
    length_bonus = defaultdict(float)

    for phrase, pos in cands:
        key = phrase.lower()
        freq[key] += 1
        if key not in first_pos:
            first_pos[key] = pos
        n_words = max(1, len(phrase.split()))
        length_bonus[key] = max(length_bonus[key], min(1.0, (n_words - 1) / 3.0))  # 0..1

    scored = []
    for key, count in freq.items():
        pos = first_pos[key]
        pos_bonus = 1.0 + (1.0 - (pos / max(1, text_len))) * 0.6  # up to +0.6 if very early
        len_bonus = 1.0 + length_bonus[key] * 0.5                 # up to +0.5 for longer phrases
        score = count * pos_bonus * len_bonus
        scored.append((key, score, pos))

    # Sort by score desc, then by earliest position asc
    scored.sort(key=lambda x: (-x[1], x[2]))
    return scored

# ---------------------------
# Public API
# ---------------------------
def extract_keyphrases(text: str, top_k: int = 5) -> List[str]:
    if not text or not text.strip():
        return []
    text = _normalize_space(text)

    # Get candidates (spaCy or regex)
    if _NLP is not None:
        candidates = list(_spacy_candidates(text))     # [(phrase, pos)]
    else:
        candidates = list(_regex_ngram_candidates(text))

    # Map first original casing & position
    first_surface: dict[str, tuple[str, int]] = {}
    for surf, pos in candidates:
        k = surf.lower()
        if k not in first_surface:
            first_surface[k] = (surf, pos)

    # Score
    scored = _score_candidates(candidates, len(text))  # [(lower_key, score, pos)]

    # Unique by first appearance
    seen = set()
    ordered = sorted(scored, key=lambda x: x[2])
    uniq_lowers: List[str] = []
    for key, _, _pos in ordered:
        if key not in seen:
            seen.add(key)
            uniq_lowers.append(key)

    # Keep a wider pool by score, then filter by that pool
    final_by_score = [k for k, *_ in scored]
    selected_pool = set(final_by_score[: top_k * 4])  # buffer
    prelim = [k for k in uniq_lowers if k in selected_pool][: max(top_k * 3, top_k)]

    # Recover original surface forms & their scores/positions
    prelim_with_meta: List[tuple[str, float, int]] = []
    score_map = {k: s for k, s, _ in scored}
    pos_map   = {k: p for k, _, p in scored}
    for k in prelim:
        surf, first_pos = first_surface.get(k, (k, pos_map.get(k, 10**9)))
        prelim_with_meta.append((surf, score_map.get(k, 0.0), first_pos))

    # NEW: prune contained phrases (People vs The People, etc.)
    pruned = _prune_contained_variants(prelim_with_meta)

    # Take top_k by score but present in appearance order
    pruned.sort(key=lambda x: (-x[1], x[2]))
    top = pruned[: top_k]
    top.sort(key=lambda x: x[2])

    # Pretty-case: keep existing caps; otherwise Title-case multiword tokens
    pretty = []
    for surf, _, _ in top:
        if any(ch.isupper() for ch in surf):
            pretty.append(surf)
        else:
            words = [w if w.isupper() else (w.capitalize() if len(w) > 2 else w) for w in surf.split()]
            pretty.append(" ".join(words))
    return pretty

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        # sample: Federalist No. 10 opening
        input_text = (
            "To the People of the State of New York: "
            "AMONG the numerous advantages promised by a well constructed Union, "
            "none deserves to be more accurately developed than its tendency to break "
            "and control the violence of faction. The friend of popular governments "
            "never finds himself so much alarmed for their character and fate, as when "
            "he contemplates their propensity to this dangerous vice. He will not fail, "
            "therefore, to set a due value on any plan which, without violating the principles "
            "to which he is attached, provides a proper cure for it."
        )

    phrases = extract_keyphrases(input_text, top_k=15)
    for i, p in enumerate(phrases, 1):
        print(f"{i}. {p}")
