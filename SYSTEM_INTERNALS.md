# System Internals — How the RAG App Works Under the Hood

This document explains at the **operating system and binary level** exactly what happens
from the moment you run the app to the moment you get an answer. Every step is traced
through the actual source code.

---

## Table of Contents

1. [High-Level Picture](#1-high-level-picture)
2. [Phase 1 — Model Startup (System Level)](#2-phase-1--model-startup-system-level)
3. [Phase 2 — Document Ingestion (Text Processing Pipeline)](#3-phase-2--document-ingestion-text-processing-pipeline)
4. [Phase 3 — Query Time (Retrieval + Inference)](#4-phase-3--query-time-retrieval--inference)
5. [The GGUF File Format](#5-the-gguf-file-format)
6. [Tokenisation Deep Dive](#6-tokenisation-deep-dive)
7. [TF-IDF Math — How Chunks Are Scored](#7-tf-idf-math--how-chunks-are-scored)
8. [BM25 Math — The Other Half of Retrieval](#8-bm25-math--the-other-half-of-retrieval)
9. [The Prompt — What the Model Actually Receives](#9-the-prompt--what-the-model-actually-receives)
10. [How the LLM Generates Text (Autoregression)](#10-how-the-llm-generates-text-autoregression)
11. [Memory Layout During Inference](#11-memory-layout-during-inference)
12. [Threading Model](#12-threading-model)
13. [SQLite Storage Layout](#13-sqlite-storage-layout)
14. [End-to-End Trace — One Question, Every Step](#14-end-to-end-trace--one-question-every-step)

---

## 1. High-Level Picture

```
┌──────────────────────────────────────────────────────────────────┐
│  User types question                                             │
│         │                                                        │
│  ┌──────▼────────┐   top-4 chunks   ┌─────────────────────┐     │
│  │ HybridRetriever│ ──────────────► │  build_rag_prompt() │     │
│  │ (RAM, ~21 rows)│                 │  Gemma instruct fmt │     │
│  └───────────────┘                 └──────────┬──────────┘     │
│         ▲                                      │                │
│         │ reload()                   HTTP POST │ /completion    │
│         │                                      ▼                │
│  ┌──────┴───────┐              ┌───────────────────────────┐    │
│  │  ragapp.db   │              │  llama-server.exe          │    │
│  │  (SQLite)    │              │  port 8082 (subprocess)    │    │
│  │  chunks tbl  │              │  Gemma 1B Q5_K_M GGUF     │    │
│  └──────────────┘              │  ~812 MB mmap'd in RAM    │    │
│         ▲                      └───────────────────────────┘    │
│         │ INSERT                         │                       │
│  ┌──────┴───────┐                       │ content (text)        │
│  │  chunker.py  │                       ▼                       │
│  │  PDF→chunks  │              ┌──────────────────┐             │
│  └──────┬───────┘              │  Answer printed  │             │
│         │                      └──────────────────┘             │
│  PyMuPDF (fitz)                                                  │
│  reads PDF bytes                                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1 — Model Startup (System Level)

### Step 1 — Python calls `llm.load(model_path, n_ctx=4096, n_threads=4)`

Source: `src/rag/llm.py`, class `LlamaCppModel.load()`

```python
def load(self, model_path, n_ctx=2048, n_threads=4, n_gpu_layers=0):
    with self._lock:                  # acquire threading.Lock
        self._unload_internal()       # kill any previously running server

        # Try backend 1: llama-cpp-python
        try:
            Llama = _get_llama()      # import llama_cpp.Llama
            ...                       # FAILS on Python 3.13 Windows — no wheel
        except RuntimeError:
            pass

        # Try backend 2: Ollama desktop
        if _ollama_reachable():       # HTTP GET localhost:11434 — FAILS, not installed
            ...

        # Backend 3: llama-server.exe subprocess
        _extract_zip_if_needed()      # unzip llamacpp_bin.zip if first run
        _start_llama_server(model_path, n_ctx, n_threads)
```

### Step 2 — ZIP extraction (first run only)

`_extract_zip_if_needed()` checks if `llamacpp_bin/llama-server.exe` already exists.
If not, it opens `llamacpp_bin.zip` (29.8 MB, llama.cpp release b8123 Windows CPU build)
and extracts all files into `llamacpp_bin/`.

```
llamacpp_bin.zip
  └── llama-server.exe    ← the HTTP inference server binary
  └── ggml.dll
  └── llama.dll
  └── ...other runtime DLLs
```

### Step 3 — `_start_llama_server()` launches a child process

```python
cmd = [
    "llamacpp_bin/llama-server.exe",
    "--model",    "Gemma-3-1B-...Q5_K_M.gguf",
    "--ctx-size", "4096",
    "--threads",  "4",
    "--port",     "8082",
    "--host",     "127.0.0.1",
]
_LLAMASERVER_PROC = subprocess.Popen(
    cmd,
    stdout=subprocess.DEVNULL,   # suppress server logs
    stderr=subprocess.DEVNULL,
    creationflags=CREATE_NO_WINDOW,   # invisible on Windows taskbar
)
```

**What `llama-server.exe` does at OS level:**

1. Parses CLI flags
2. Calls `llama_model_load()` which **memory-maps the GGUF file** into the process's virtual address space using `mmap()` / `CreateFileMapping()` on Windows
3. Allocates the **KV cache** in RAM: `n_ctx × n_layers × 2 × head_dim × sizeof(float16)` bytes
   - For Gemma 1B, n_layers=18, n_ctx=4096: ~72 MB for the KV cache
4. Allocates the **compute buffer** for matrix multiplications (~50–100 MB)
5. Starts an **HTTP server** on `127.0.0.1:8082` using a built-in HTTP library
6. Exposes endpoints: `/health`, `/completion`, `/tokenize`, `/embedding`

### Step 4 — Python polls `/health` until the model is ready

```python
def _wait_for_server(port, timeout=180):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True      # server ready
        except Exception:
            pass
        time.sleep(0.5)              # poll every 500ms
```

The server returns `{"status":"ok"}` once the model is fully loaded into RAM.
On a 4-core CPU this typically takes **3–8 seconds** for the 812 MB model.

### Model memory layout at this point

```
llama-server.exe process (Windows):
┌─────────────────────────────────────────────────┐
│ Virtual Address Space                           │
│                                                 │
│  0x...000  PE image (exe + DLLs) ~30 MB         │
│  0x...000  GGUF mmap region      ~812 MB        │
│            ├── token embeddings  ~33 MB         │
│            ├── 18× transformer   ~780 MB        │
│            │   layers (weights)                 │
│            └── output head       ~1 MB          │
│  0x...000  KV cache (FP16)        ~72 MB        │
│  0x...000  compute buffer         ~80 MB        │
│  0x...000  HTTP server heap       ~10 MB        │
│                                                 │
│  Total RAM used: ~975 MB                        │
└─────────────────────────────────────────────────┘
```

The **GGUF weights are memory-mapped** (not copied into RAM all at once). The OS pages in
weight data on demand as each matrix multiply is executed. On first inference all pages are
faulted in; subsequent inferences are faster.

---

## 3. Phase 2 — Document Ingestion (Text Processing Pipeline)

When you run `:add file.pdf` or call `ingest_document()`, a background thread runs this
full pipeline:

### Step 1 — PDF binary → raw Unicode text

Source: `src/rag/chunker.py`, `_extract_pdf()`

```python
def _extract_pdf(path: str) -> str:
    doc = fitz.open(path)           # PyMuPDF opens the PDF binary
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))   # renders page to plain text
    doc.close()
    return "\n".join(pages)
```

**What PyMuPDF does:** Opens the PDF binary, parses the cross-reference table, decompresses
page content streams (usually Flate/zlib compressed), runs the PDF rendering engine on each
page's content stream to extract character positions and Unicode codepoints, sorts them into
reading order, and returns plain text.

For a 15-page paper like "Attention Is All You Need" this produces ~25,000 words of raw text.

### Step 2 — Raw text → overlapping word chunks

Source: `src/rag/chunker.py`, `chunk_text()`

```python
CHUNK_SIZE    = 350    # words per chunk
CHUNK_OVERLAP = 60     # words shared between adjacent chunks

def chunk_text(text: str) -> List[str]:
    words = text.split()           # split on whitespace
    chunks = []
    start = 0
    while start < len(words):
        end   = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP   # advance by 290 words
    return chunks
```

**Why overlapping?** A sentence that spans the boundary between two 350-word windows would
be split in half. The 60-word overlap ensures every sentence appears complete in at least one
chunk. For 25,000 words this produces roughly **85 chunks** (25000 ÷ 290 ≈ 86).

Actual result for the Attention paper: **21 chunks** (the PDF has ~7,300 words of real content
after stripping headers/footers/references).

```
Chunk 0: words[0..349]
Chunk 1: words[290..639]    ← 60 words overlap with chunk 0
Chunk 2: words[580..929]    ← 60 words overlap with chunk 1
...
```

### Step 3 — Each chunk → token list (stopword-filtered)

Source: `src/rag/chunker.py`, `tokenise()`

```python
_RE_WORD = re.compile(r"[a-z0-9]+")

def tokenise(text: str) -> List[str]:
    raw = _RE_WORD.findall(text.lower())              # extract alphanum words
    return [t for t in raw if t not in _STOP          # remove ~70 stopwords
                            and len(t) > 1]           # drop single chars
```

Example:
```
Input:  "The Transformer architecture uses self-attention mechanisms."
Output: ["transformer", "architecture", "uses", "attention", "mechanisms"]
       (removed: "the", "self" not in stop list but "-" stripped)
```

This token list is stored in the `chunks.tokens` column as JSON and is used by BM25 at query time.

### Step 4 — All chunks together → TF-IDF vectors

Source: `src/rag/chunker.py`, `compute_tfidf_vecs()`

This is computed **across all chunks at once** (corpus-level) so IDF values are meaningful.

```python
def compute_tfidf_vecs(all_token_lists):
    N = len(all_token_lists)          # total number of chunks

    # Document frequency: how many chunks contain each term
    df = {}
    for toks in all_token_lists:
        for t in set(toks):            # set() — count each term once per chunk
            df[t] = df.get(t, 0) + 1

    # IDF: log((N+1)/(df+1)) + 1  (smoothed to avoid div-by-zero)
    idf = {
        t: math.log((N + 1) / (cnt + 1)) + 1.0
        for t, cnt in df.items()
    }

    # TF-IDF vector per chunk: {term: (term_count/chunk_len) * idf}
    vecs = []
    for toks in all_token_lists:
        tf = Counter(toks)
        total = len(toks)
        tfidf = {t: (tf[t] / total) * idf[t] for t in tf}
        vecs.append(tfidf)

    return vecs, idf
```

A TF-IDF vector for a chunk is a **sparse dictionary** like:
```python
{
    "transformer":   0.0842,
    "attention":     0.0731,
    "encoder":       0.0654,
    "decoder":       0.0601,
    "softmax":       0.0420,
    ...
}
```
Terms that appear in many chunks (like "model") get low IDF, so their TF-IDF score is low.
Rare domain-specific terms (like "multihead") get high IDF, high score — exactly what you want.

### Step 5 — Persist to SQLite

Source: `src/rag/db.py`, `insert_chunks()`

```python
def insert_chunks(doc_id, chunks):
    rows = [
        (
            doc_id,
            c["chunk_idx"],
            c["text"],
            json.dumps(c["tokens"]),     # list → JSON string
            pickle.dumps(c["tfidf_vec"]) # dict → binary blob via pickle
        )
        for c in chunks
    ]
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO chunks(doc_id,chunk_idx,text,tokens,tfidf_vec) VALUES(?,?,?,?,?)",
            rows,
        )
```

The TF-IDF dict is stored as a pickled binary blob in the `tfidf_vec` BLOB column.
Pickle is used (rather than JSON) because it is ~3x faster to deserialise floats from pickle
than from JSON, and the retriever loads all chunks into RAM on every `reload()` call.

---

## 4. Phase 3 — Query Time (Retrieval + Inference)

### Step 1 — Retriever loads all chunks into RAM

Source: `src/rag/retriever.py`, `HybridRetriever.reload()`

```python
def reload(self):
    from .db import load_all_chunks
    self._chunks = load_all_chunks()       # List[dict], one per chunk
    total = sum(len(c["tokens"]) for c in self._chunks)
    self._avg_dl = total / len(self._chunks)   # average chunk length (for BM25)
```

`load_all_chunks()` does a single `SELECT * FROM chunks` and unpickles
the `tfidf_vec` blob back into a Python dict for every row.

For 21 chunks this is near-instant (~5ms). For 10,000 chunks it would take ~200ms;
production systems use a vector database — this app keeps it simple with SQLite.

### Step 2 — Query is tokenised (same pipeline as ingestion)

```python
q_tokens = tokenise("What is the main contribution of the Transformer architecture?")
# Result: ["main", "contribution", "transformer", "architecture"]
```

### Step 3 — BM25 scoring (all 21 chunks, O(n·q) time)

For each query token, IDF is computed over the loaded corpus:

```
N  = 21 chunks
df("transformer") = 14 chunks contain it
IDF("transformer") = log((21 - 14 + 0.5) / (14 + 0.5) + 1) = log(1.45) = 0.372
```

For each chunk, BM25 score is:

```
score(chunk, query) = Σ_t  IDF(t) × tf(t,d)×(k1+1) / (tf(t,d) + k1×(1 - b + b×|d|/avgdl))

Where:
  k1 = 1.5   (TF saturation — diminishing returns for repeated terms)
  b  = 0.75  (length normalisation — penalise overly long chunks)
```

A chunk that mentions "transformer" 5 times gets a higher score than one that mentions it
once, but not 5× higher — k1 saturates the benefit of repetition.

### Step 4 — TF-IDF cosine scoring (all 21 chunks, O(n·|vocab|) time)

The query is represented as a sparse TF vector:
```python
q_vec = {"main": 0.25, "contribution": 0.25, "transformer": 0.25, "architecture": 0.25}
```

For each chunk, cosine similarity is computed as a sparse dot product:
```
cos(q, chunk_i) = (q · chunk_i) / (|q| × |chunk_i|)

dot product = sum of (q[term] × chunk_i[term]) for every term that appears in BOTH
```

Because both vectors are sparse this is very fast — only shared terms matter.

### Step 5 — Hybrid combination and top-k selection

Both score lists are **min-max normalised to [0, 1]** independently, then combined:

```python
final_score[i] = 0.5 × bm25_normalised[i] + 0.5 × cosine_normalised[i]
```

Results are sorted descending, top 4 chunks selected:

```
Rank 1: chunk_12  score=0.91  "The Transformer uses multi-head attention..."
Rank 2: chunk_08  score=0.84  "attention mechanism allows the model to..."
Rank 3: chunk_15  score=0.77  "We describe the Transformer architecture..."
Rank 4: chunk_03  score=0.72  "encoder-decoder structure with stacked..."
```

### Step 6 — Prompt construction

Source: `src/rag/llm.py`, `build_rag_prompt()`

The 4 chunk texts are concatenated with `---` separators and wrapped in Gemma's
instruction-tuning format:

```
<start_of_turn>user
You are a helpful assistant. Answer ONLY based on the provided context.
If the answer is not in the context, say "I don't know."

Context:
The Transformer uses multi-head attention in both encoder and decoder...

---

attention mechanism allows the model to jointly attend to information
from different representation subspaces at different positions...

---

We describe the Transformer architecture and explain the encoder stack...

---

encoder-decoder structure with stacked self-attention and feed-forward layers...

Question: What is the main contribution of the Transformer architecture?<end_of_turn>
<start_of_turn>model
```

The `<start_of_turn>model` at the end is the **generation cue** — the model continues
from this point, generating the answer token by token.

### Step 7 — HTTP POST to llama-server

Source: `src/rag/llm.py`, `_gen_via_server()`

```python
payload = json.dumps({
    "prompt":       "<start_of_turn>user\n...full prompt...<start_of_turn>model\n",
    "n_predict":    512,          # max tokens to generate
    "temperature":  0.7,          # sampling randomness
    "top_p":        0.9,          # nucleus sampling threshold
    "stream":       False,        # get full response at once
    "cache_prompt": False,
}).encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8082/completion",
    data    = payload,
    headers = {"Content-Type": "application/json"},
    method  = "POST",
)

with urllib.request.urlopen(req) as resp:
    body = json.loads(resp.read())
answer = body["content"]        # the generated text
```

**Why stdlib `urllib` instead of `requests`?** Zero extra dependency — works on Android,
Windows, any Python 3.x without installing anything.

---

## 5. The GGUF File Format

The `Gemma-3-1B-...-Q5_K_M.gguf` file is a binary container defined by the llama.cpp project.

```
Bytes 0-3:   Magic "GGUF"
Bytes 4-7:   Version (3)
Bytes 8-15:  n_tensors (count of weight matrices)
Bytes 16-23: n_kv (metadata key-value count)
...
Metadata section:
  key: "general.architecture"  value: "gemma3"
  key: "gemma3.context_length" value: 8192
  key: "gemma3.embedding_length" value: 1152
  key: "gemma3.block_count"    value: 18
  key: "tokenizer.ggml.model"  value: "llama"  (SentencePiece BPE)
  ... (dozens more keys)
...
Tensor info section (one entry per weight matrix):
  name, shape, quantization_type, offset_into_data_section
...
Data section (aligned to 32 bytes):
  Raw quantized weight bytes for every tensor
```

**Q5_K_M quantization:** Each weight is stored as 5.68 bits on average using a
"K-quant" block scheme. 256 weights are grouped into a block. Each block stores:
- A float16 scale factor (shared across 256 weights)
- A float16 min value
- 256 × 5-bit quantized values (packed)

At inference time, llama.cpp dequantizes blocks on-the-fly during matrix multiplications,
converting them back to float32 for the arithmetic.

---

## 6. Tokenisation Deep Dive

There are **two completely separate tokenisation steps** in this system:

### A. RAG tokenisation (chunker.py) — for text search

```python
_RE_WORD = re.compile(r"[a-z0-9]+")   # alphanumeric only

def tokenise(text):
    raw = _RE_WORD.findall(text.lower())
    return [t for t in raw if t not in _STOP and len(t) > 1]
```

- Input: arbitrary text
- Output: list of lowercase ASCII words, no punctuation, no stopwords
- Purpose: build BM25/TF-IDF index for retrieval
- ~70 English stopwords removed (a, the, is, are, was, ...)
- Example: `"multi-head attention"` → `["multi", "head", "attention"]`

### B. LLM tokenisation (inside llama-server) — for the neural network

This happens **inside the C++ binary**, not in Python. Gemma uses a
**SentencePiece BPE (Byte-Pair Encoding)** tokenizer with a 256,000 token vocabulary.

```
Text:  "The Transformer architecture uses self-attention"
Tokens: ["▁The", "▁Transform", "er", "▁architect", "ure", "▁uses", "▁self", "-", "attention"]
IDs:    [651, 24882, 643, 79946, 1280, 3327, 1102, 117, 17038]
```

The `▁` (underscore) marks the start of a word (space-aware tokenisation).
Common words like "The" are single tokens. Rare words are split into subwords.
The full prompt (context + question) becomes ~2,000–2,500 token IDs passed to the transformer.

---

## 7. TF-IDF Math — How Chunks Are Scored

**Term Frequency (TF):** How often a term appears in a chunk, normalised by chunk length.

$$TF(t, d) = \frac{\text{count}(t, d)}{|d|}$$

If chunk $d$ has 200 tokens and "attention" appears 8 times: $TF = 8/200 = 0.04$

**Inverse Document Frequency (IDF):** How rare the term is across all chunks. Rare = more informative.

$$IDF(t) = \log\left(\frac{N+1}{df(t)+1}\right) + 1$$

If "attention" appears in 14 of 21 chunks: $IDF = \log(22/15) + 1 = 0.38 + 1 = 1.38$

If "quantum" appears in 0 of 21 chunks: $IDF = \log(22/1) + 1 = 3.09 + 1 = 4.09$  ← much higher weight

**TF-IDF score:**
$$\text{tfidf}(t, d) = TF(t, d) \times IDF(t) = 0.04 \times 1.38 = 0.055$$

**Cosine similarity between query vector $q$ and chunk vector $d$:**

$$\text{cos\_sim}(q, d) = \frac{\sum_{t \in q \cap d} q[t] \times d[t]}{\sqrt{\sum q[t]^2} \times \sqrt{\sum d[t]^2}}$$

Only terms in **both** vectors contribute — hence the fast sparse dot product.

---

## 8. BM25 Math — The Other Half of Retrieval

BM25 (Okapi BM25) is the gold-standard probabilistic retrieval function used by
Elasticsearch, Lucene, and every serious search engine.

$$\text{BM25}(q, d) = \sum_{t \in q} IDF_{BM25}(t) \times \frac{TF(t,d) \times (k_1 + 1)}{TF(t,d) + k_1 \times \left(1 - b + b \times \frac{|d|}{avgdl}\right)}$$

**Parameters in this app:**
- $k_1 = 1.5$ — TF saturation constant. If a word appears many times, score increases but saturates. $k_1=0$ means TF doesn't matter; $k_1=\infty$ means raw TF count.
- $b = 0.75$ — Length normalisation. $b=0$ ignores length; $b=1$ fully normalises. $0.75$ is standard.

**Why combine BM25 + cosine?**

| Scenario | BM25 wins | Cosine wins |
|---|---|---|
| Exact keyword match | ✓ | |
| Synonym/paraphrase | | ✓ |
| Query term repeated in chunk | ✓ (saturated) | (double-counted) |
| Short precise chunks | ✓ | |
| Semantic similarity | | ✓ |

The hybrid at $\alpha = 0.5$ gets the best of both.

---

## 9. The Prompt — What the Model Actually Receives

Gemma 3 is an **instruction-tuned** model trained with turn-based conversation.
It expects a specific format:

```
<start_of_turn>user
{system instructions + context + question}
<end_of_turn>
<start_of_turn>model
```

The model was trained to generate text that completes this turn — i.e., it will
produce a helpful response to the question, staying within the bounds set by "Answer ONLY
based on the context."

**Token budget breakdown for a typical RAG query:**

```
<start_of_turn>user\n          →    4 tokens
System instruction             →   38 tokens
Context (4 chunks × ~350 words) →  ~1,800 tokens
Question (typical)             →   15 tokens
<end_of_turn>\n<start_of_turn>model\n → 6 tokens
─────────────────────────────────────────────────
Total prompt tokens:           ~1,863 tokens
Max answer tokens:             512 tokens
Total context window needed:   ~2,375 tokens → requires n_ctx ≥ 2500
(We use n_ctx=4096 to be safe)
```

If the prompt exceeds n_ctx, llama-server returns HTTP 400 with:
```json
{"error": {"message": "request (2136 tokens) exceeds the available context size (2048)"}}
```

---

## 10. How the LLM Generates Text (Autoregression)

Inside `llama-server.exe`, each `POST /completion` request triggers this loop:

### 1. Prompt encoding (prefill)

The full prompt string is split into tokens by the SentencePiece tokenizer.
These token IDs are fed through the **Gemma 1B transformer** in one forward pass
(the "prefill" or "context encoding" phase):

```
For each transformer layer (18 layers):
  1. Layer norm on input
  2. Self-attention:
     Q = input × W_Q    (query projection, 1152 → 8 heads × 256 dim)
     K = input × W_K    (key projection)
     V = input × W_V    (value projection)
     scores = softmax(Q × K^T / √256)   ← scaled dot product attention
     output = scores × V
  3. Add & norm (residual connection)
  4. Feed-forward network:
     x = GELU(input × W1) × W2     (1152 → 4608 → 1152)
  5. Add & norm
```

The output of the last layer for the **final token position** is a vector of 1,152 floats.

### 2. Vocabulary projection

This 1,152-dimensional vector is multiplied by the output embedding matrix
(dimensions 1,152 × 256,000) to produce **logits** — one raw score per vocabulary token.

### 3. Sampling

```
logits → divide by temperature (0.7) → softmax → probability distribution

Temperature 0.7 makes the distribution sharper (more confident) than temperature 1.0.

top_p = 0.9: sort tokens by probability descending, sum until cumulative ≥ 0.9,
             discard the rest, renormalise, sample from this nucleus.
```

One token is sampled. It is appended to the sequence.

### 4. The KV cache — why generation is fast

The K and V matrices from the prefill are **cached in RAM** (the KV cache, ~72 MB).
For each new token generated, only the **new token's row** in the attention matrices
needs to be computed, then attended against all cached K/V pairs.

Without the KV cache, generating 100 tokens would require 100 full forward passes
over the entire context. With the cache, each new token requires only a single
lightweight forward pass over the 18 layers for that one new token position.

Typical generation speed: **3–8 tokens per second** on a 4-core CPU for this 1B model.

---

## 11. Memory Layout During Inference

```
Process: llama-server.exe
Virtual memory at inference time:

  ┌──────────────────────────────────────────────────────────────┐
  │ GGUF weights (mmap)                        ~812 MB           │
  │   token_embd.weight          [256000 × 1152]  FP16/Q5_K    │
  │   blk.0.attn_q.weight        [1152 × 1152]    Q5_K_M       │
  │   blk.0.attn_k.weight        [1152 × 256]     Q5_K_M       │
  │   blk.0.attn_v.weight        [1152 × 256]     Q5_K_M       │
  │   blk.0.attn_output.weight   [1152 × 1152]    Q5_K_M       │
  │   blk.0.ffn_gate.weight      [1152 × 4608]    Q5_K_M       │
  │   blk.0.ffn_up.weight        [1152 × 4608]    Q5_K_M       │
  │   blk.0.ffn_down.weight      [4608 × 1152]    Q5_K_M       │
  │   ...× 18 layers...                                         │
  │   output.weight              [256000 × 1152]  Q5_K_M       │
  ├──────────────────────────────────────────────────────────────│
  │ KV cache (dynamic, FP16)                   ~72 MB           │
  │   18 layers × 2 (K+V) × 4096 tokens × 256 dim × 2 bytes   │
  ├──────────────────────────────────────────────────────────────│
  │ Compute scratch buffer                      ~80 MB           │
  │   Intermediate activations, dequant buffer, matmul temp     │
  ├──────────────────────────────────────────────────────────────│
  │ HTTP server + misc                         ~10 MB            │
  └──────────────────────────────────────────────────────────────┘
                                          Total: ~975 MB
```

---

## 12. Threading Model

```
Main thread (Python CLI / Kivy UI)
    │
    ├── threading.Lock (_LLAMASERVER_LOCK)
    │       guards subprocess start/stop
    │
    ├── threading.Thread: ingest_document()
    │       PDF → chunks → DB → retriever.reload()
    │       daemon=True (dies if main exits)
    │
    ├── threading.Thread: load_model()
    │       blocks until _wait_for_server() returns
    │       daemon=True
    │
    ├── threading.Thread: auto_download_default()
    │       HuggingFace download with progress callback
    │       daemon=True
    │
    └── threading.Thread: ask() / query()
            retriever.query() → build_rag_prompt() → HTTP POST
            calls stream_cb on Kivy main thread via Clock.schedule_once()
            daemon=True
```

The `generate()` call inside `ask()` is **blocking HTTP** — it waits for the full
response from llama-server before calling `on_done`. In the Android UI, streaming mode
is used instead, where `stream_cb` is called with each token and `Clock.schedule_once()`
ensures Kivy UI updates happen on the main thread.

---

## 13. SQLite Storage Layout

Database file: `~/ragapp.db` (Windows: `C:\Users\<name>\ragapp.db`)

```sql
CREATE TABLE documents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,              -- "1706.03762v7 (1).pdf"
    path       TEXT NOT NULL UNIQUE,       -- full file path
    added_at   TEXT DEFAULT (datetime('now')),
    num_chunks INTEGER DEFAULT 0
);

CREATE TABLE chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_idx  INTEGER NOT NULL,           -- 0, 1, 2, ...
    text       TEXT NOT NULL,              -- raw chunk text (~350 words)
    tokens     TEXT,                       -- JSON: ["word1","word2",...]
    tfidf_vec  BLOB                        -- pickle: {"word1":0.042, ...}
);

CREATE INDEX idx_chunks_doc ON chunks(doc_id);
```

**Storage sizes for the Attention paper (21 chunks):**

| Column | Type | Size |
|---|---|---|
| text (raw) | TEXT | ~1.5 KB per chunk × 21 = ~32 KB |
| tokens (json) | TEXT | ~400 bytes per chunk × 21 = ~8 KB |
| tfidf_vec (pickle) | BLOB | ~2 KB per chunk × 21 = ~42 KB |
| Total DB file | | ~100 KB |

`ON DELETE CASCADE` means deleting a document automatically deletes all its chunks.
`PRAGMA journal_mode=WAL` means writes don't lock readers — safe for concurrent
background ingestion while the UI reads from the retriever.

---

## 14. End-to-End Trace — One Question, Every Step

**Question:** `"What is the main contribution of the Transformer architecture?"`

```
t=0ms    User hits Enter in CLI

t=1ms    cli.py: raw = input("You> ")  →  calls self._ask(raw)

t=2ms    _ask() → retriever.query("What is the main contribution...", top_k=4)
         tokenise() → ["main", "contribution", "transformer", "architecture"]
         _bm25_scores(q_tokens):
           for each of 21 chunks, compute BM25 score in pure Python
           21 × 4 token lookups ≈ 84 dict operations → ~0.5ms
         _cosine_scores(q_tokens):
           sparse dot product of query TF vec against 21 precomputed tfidf_vecs
           ~0.3ms
         normalise + combine + sort → top 4 chunks
         Total retrieval: ~2ms

t=4ms    build_rag_prompt(top_4_chunks, question)
         String concatenation → ~9,000 char prompt string
         ~0.2ms

t=4ms    _gen_via_server(prompt, max_tokens=512, ...)
         json.dumps(payload) → ~9,100 bytes
         urllib.request.Request (builds HTTP request object, no I/O yet)
         urllib.request.urlopen(req) → TCP connect to 127.0.0.1:8082
         TCP connect on loopback: ~0.1ms

t=4ms    HTTP request bytes sent to llama-server over loopback socket:
         POST /completion HTTP/1.1\r\n
         Host: 127.0.0.1:8082\r\n
         Content-Type: application/json\r\n
         Content-Length: 9100\r\n
         \r\n
         {"prompt": "<start_of_turn>user\n...", "n_predict": 512, ...}

t=4ms    llama-server receives request, parses JSON
         SentencePiece tokenizes prompt → ~2,136 tokens

t=4ms    PREFILL phase begins:
         2,136 tokens fed through all 18 transformer layers
         Each layer: Q/K/V projections + attention + FFN
         On 4 CPU cores, BLAS-optimised matrix multiply (OpenBLAS or similar)
         ~300ms–800ms depending on CPU

t=600ms  GENERATION phase begins (autoregressive):
         Sample token 1 → e.g. "The" (token 651)
         Sample token 2 → e.g. "main" (token 3476)
         Sample token 3 → e.g. "contribution" (token 19461)
         ...
         Each token: ~120ms–300ms on CPU
         Target answer ≈ 60 tokens

t=10s    ~60 tokens generated:
         "The main contribution of the Transformer is its use of attention
          mechanisms to compute representations without using sequence-aligned
          recurrent networks or convolutions entirely..."
         Generation stops when model emits <end_of_turn> token or hits n_predict limit

t=10s    llama-server sends HTTP 200 response:
         {"content": "The main contribution...", "tokens_predicted": 62, ...}

t=10s    Python receives response, json.loads(), extracts body["content"]

t=10s    cli.py prints answer to terminal
         Total time: ~10 seconds  (0.6s prefill + ~9s generation at 7 tok/s)
```

---

## Summary Table

| Stage | Where | Time | What happens |
|---|---|---|---|
| Model load | `llm.py` → OS | 3–8s | mmap 812MB GGUF, alloc KV cache, start HTTP server |
| PDF extraction | `chunker.py` → PyMuPDF | 0.5–2s | Decompress PDF streams, extract Unicode text |
| Chunking | `chunker.py` | <10ms | Split 7K words into 21 overlapping 350-word windows |
| Tokenisation | `chunker.py` | <5ms | Regex + stopword filter → word lists |
| TF-IDF build | `chunker.py` | <10ms | Per-corpus IDF, per-chunk TF×IDF sparse vectors |
| DB insert | `db.py` → SQLite | <20ms | Pickle vectors, JSON tokens, bulk INSERT |
| Retriever reload | `retriever.py` | <5ms | SELECT all chunks, unpickle vectors into RAM |
| BM25 scoring | `retriever.py` | ~0.5ms | 21 chunks × 4 query tokens, pure Python |
| Cosine scoring | `retriever.py` | ~0.3ms | 21 sparse dot products |
| Prompt build | `llm.py` | <1ms | String concat, Gemma turn format |
| HTTP POST | `llm.py` → socket | <1ms | Loopback TCP, 9KB payload |
| Prefill (LLM) | `llama-server.exe` | 0.3–0.8s | 2136 tokens × 18 layers forward pass |
| Generation (LLM) | `llama-server.exe` | 8–15s | 60 tokens × 18 layers, KV cache accelerated |
| Total query | end-to-end | ~10–16s | (on a 4-core laptop CPU) |
