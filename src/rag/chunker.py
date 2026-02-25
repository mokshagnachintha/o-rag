"""
chunker.py — Load .txt and .pdf files, split into overlapping chunks,
and compute per-chunk TF-IDF vectors (stored for hybrid retrieval).
No heavy NLP libraries — pure Python + PyMuPDF.
"""
import re
import math
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# ---- optional PDF support (PyMuPDF) ----
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

CHUNK_SIZE   = 350   # tokens (approx words) per chunk
CHUNK_OVERLAP = 60   # overlapping tokens between consecutive chunks

# Minimal English stopwords (keeps index small)
_STOP = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can of in on at to for "
    "from by with about as into through during including before after "
    "above below between each other than and or but not this that "
    "these those i me my we our you your he she it its they them their "
    "what which who whom when where why how all both each few more most "
    "other some such no nor only same so than too very just".split()
)


# ------------------------------------------------------------------ #
#  Text extraction                                                     #
# ------------------------------------------------------------------ #

def _extract_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_pdf(path: str) -> str:
    if not PDF_SUPPORT:
        raise RuntimeError(
            "PyMuPDF not installed. Install it with: pip install pymupdf"
        )
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def extract_text(path: str) -> str:
    """Return plain text from .txt or .pdf file."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    return _extract_txt(path)


# ------------------------------------------------------------------ #
#  Tokenisation                                                        #
# ------------------------------------------------------------------ #

_RE_WORD = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    raw = _RE_WORD.findall(text.lower())
    return [t for t in raw if t not in _STOP and len(t) > 1]


# ------------------------------------------------------------------ #
#  Chunking                                                            #
# ------------------------------------------------------------------ #

def _split_sentences(text: str) -> List[str]:
    """Naive sentence splitter — avoids pulling in NLTK."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str) -> List[str]:
    """
    Split text into overlapping chunks of ~CHUNK_SIZE words.
    Returns list of raw (un-tokenised) chunk strings.
    """
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ------------------------------------------------------------------ #
#  TF-IDF helpers                                                      #
# ------------------------------------------------------------------ #

def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {term: cnt / total for term, cnt in counts.items()}


def compute_tfidf_vecs(
    all_token_lists: List[List[str]],
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Compute TF-IDF for each chunk.
    Returns (list_of_tfidf_dicts, idf_dict).
    """
    N = len(all_token_lists)
    # Document frequency
    df: Dict[str, int] = {}
    for toks in all_token_lists:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    idf: Dict[str, float] = {
        t: math.log((N + 1) / (cnt + 1)) + 1.0
        for t, cnt in df.items()
    }

    vecs = []
    for toks in all_token_lists:
        tf = _compute_tf(toks)
        vecs.append({t: tf[t] * idf[t] for t in tf})
    return vecs, idf


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def process_document(path: str) -> List[dict]:
    """
    Full pipeline: extract → chunk → tokenise → TF-IDF.

    Returns list of chunk dicts:
        {chunk_idx, text, tokens, tfidf_vec}
    """
    raw_text   = extract_text(path)
    raw_chunks = chunk_text(raw_text)
    token_lists = [tokenise(c) for c in raw_chunks]
    tfidf_vecs, _ = compute_tfidf_vecs(token_lists)

    result = []
    for idx, (text, tokens, vec) in enumerate(
        zip(raw_chunks, token_lists, tfidf_vecs)
    ):
        result.append(
            {
                "chunk_idx": idx,
                "text": text,
                "tokens": tokens,
                "tfidf_vec": vec,
            }
        )
    return result
