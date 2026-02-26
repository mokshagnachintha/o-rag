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

# ---- optional PDF support (PyMuPDF or pypdf fallback) ----
try:
    import fitz  # PyMuPDF (desktop)
    PDF_SUPPORT  = True
    _PDF_BACKEND = "pymupdf"
except ImportError:
    try:
        import pypdf as _pypdf  # pure-Python fallback (Android)
        PDF_SUPPORT  = True
        _PDF_BACKEND = "pypdf"
    except ImportError:
        PDF_SUPPORT  = False
        _PDF_BACKEND = None


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
            "No PDF library available. Install pymupdf or pypdf."
        )
    if _PDF_BACKEND == "pymupdf":
        doc = fitz.open(path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)
    else:
        # pypdf fallback
        reader = _pypdf.PdfReader(path)
        return "\n".join(
            (p.extract_text() or "") for p in reader.pages
        )


def resolve_uri(path: str) -> str:
    """
    On Android, plyer.filechooser returns a content:// URI instead of a
    real file path.  Copy the content into app-private storage and return
    the real path so Python's open() / pypdf can read it.
    On all other platforms (or if path is already a file path) returns path
    unchanged.
    """
    import os
    if not path.startswith("content://"):
        return path
    try:
        from jnius import autoclass  # type: ignore
        PythonActivity   = autoclass("org.kivy.android.PythonActivity")
        Uri              = autoclass("android.net.Uri")
        ctx = PythonActivity.mActivity
        uri = Uri.parse(path)
        # Get the display name from the content resolver
        name = "attachment"
        cursor = ctx.getContentResolver().query(uri, None, None, None, None)
        if cursor:
            try:
                if cursor.moveToFirst():
                    idx = cursor.getColumnIndex("_display_name")
                    if idx >= 0:
                        name = cursor.getString(idx)
            finally:
                cursor.close()
        # Copy bytes to private storage.
        # Use getFd() + os.dup() so Python owns its own fd while the
        # PFD is closed normally — avoids IllegalStateException from
        # calling close() after detachFd().
        dest_dir = os.path.join(
            os.environ.get("ANDROID_PRIVATE", "/tmp"), "attachments"
        )
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name)
        pfd = ctx.getContentResolver().openFileDescriptor(uri, "r")
        if pfd is None:
            raise RuntimeError(f"openFileDescriptor returned None for URI: {path}")
        try:
            fd_dup = os.dup(pfd.getFd())     # duplicate — Python owns this fd
            with os.fdopen(fd_dup, "rb") as src_f:
                data = src_f.read()
        finally:
            pfd.close()                      # safe: pfd still owns the original fd
        with open(dest, "wb") as out_f:
            out_f.write(data)
        return dest
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Could not read file from device: {e}") from e


def extract_text(path: str) -> str:
    """Return plain text from .txt or .pdf file (resolves content:// URIs)."""
    path = resolve_uri(path)
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
