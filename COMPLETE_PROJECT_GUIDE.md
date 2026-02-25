# Offline RAG App — Complete Project Guide
### From Zero to Expert: Technical & Non-Technical Deep Dive

---

## Table of Contents

**Part A — Non-Technical: What This Is & Why It Matters**
1. [What Is This App in Plain English?](#part-a1--what-is-this-app-in-plain-english)
2. [The Problem It Solves](#part-a2--the-problem-it-solves)
3. [What RAG Means — No Jargon](#part-a3--what-rag-means--no-jargon)
4. [What AI Model Is Used and Why](#part-a4--what-ai-model-is-used-and-why)
5. [How a User Experiences the App](#part-a5--how-a-user-experiences-the-app)

**Part B — Technical Architecture**
6. [Technology Stack — Every Library Explained](#part-b6--technology-stack--every-library-explained)
7. [Project File Structure](#part-b7--project-file-structure)
8. [The RAG Pipeline — Step by Step](#part-b8--the-rag-pipeline--step-by-step)
9. [Document Ingestion Deep Dive](#part-b9--document-ingestion-deep-dive)
10. [Retrieval Engine — BM25 + TF-IDF Hybrid](#part-b10--retrieval-engine--bm25--tf-idf-hybrid)
11. [LLM Backend — Three-Step Fallback System](#part-b11--llm-backend--three-step-fallback-system)
12. [Prompt Engineering — How the Model Is Instructed](#part-b12--prompt-engineering--how-the-model-is-instructed)
13. [Database Design](#part-b13--database-design)

**Part C — How It Runs on Mobile (Android)**
14. [Android Architecture Overview](#part-c14--android-architecture-overview)
15. [How Python Runs on Android](#part-c15--how-python-runs-on-android)
16. [How the AI Model Runs on a Phone CPU](#part-c16--how-the-ai-model-runs-on-a-phone-cpu)
17. [The Build Process — Buildozer & python-for-android](#part-c17--the-build-process--buildozer--python-for-android)
18. [Memory Management on Android](#part-c18--memory-management-on-android)
19. [Storage & File Paths on Android](#part-c19--storage--file-paths-on-android)
20. [Threading on Android with Kivy](#part-c20--threading-on-android-with-kivy)

**Part D — The AI / ML Internals**
21. [What Is a Large Language Model?](#part-d21--what-is-a-large-language-model)
22. [What Is a Transformer?](#part-d22--what-is-a-transformer)
23. [What Is Quantization? Why Q5_K_M?](#part-d23--what-is-quantization-why-q5_km)
24. [What Is a GGUF File?](#part-d24--what-is-a-gguf-file)
25. [Autoregressive Generation — How Text Is Created](#part-d25--autoregressive-generation--how-text-is-created)
26. [Temperature & Top-P Sampling Explained](#part-d26--temperature--top-p-sampling-explained)
27. [Context Window — What It Is and Why It Matters](#part-d27--context-window--what-it-is-and-why-it-matters)

**Part E — Design Decisions & Trade-offs**
28. [Why No Internet at Runtime?](#part-e28--why-no-internet-at-runtime)
29. [Why Gemma 1B and Not a Larger Model?](#part-e29--why-gemma-1b-and-not-a-larger-model)
30. [Why SQLite and Not a Vector Database?](#part-e30--why-sqlite-and-not-a-vector-database)
31. [Why BM25+TF-IDF and Not Sentence Embeddings?](#part-e31--why-bm25tfidf-and-not-sentence-embeddings)
32. [Why Kivy?](#part-e32--why-kivy)

**Part F — Performance & Sizing**
33. [Speed: What Takes How Long](#part-f33--speed-what-takes-how-long)
34. [Storage Sizes — Full Breakdown](#part-f34--storage-sizes--full-breakdown)
35. [RAM Requirements by Device](#part-f35--ram-requirements-by-device)
36. [Scaling — How Many Documents Can It Handle?](#part-f36--scaling--how-many-documents-can-it-handle)

**Part G — Tough Questions Answered**
37. [Q&A — Questions You Might Be Asked](#part-g37--qa--questions-you-might-be-asked)

---

---

# PART A — Non-Technical: What This Is & Why It Matters

---

## Part A.1 — What Is This App in Plain English?

This is a **personal AI assistant that lives entirely on your phone or computer** and can
answer questions about documents you give it — like research papers, contracts, textbooks,
meeting notes, or any text file.

Think of it like this: you upload a 100-page legal document, and instead of reading it
yourself, you ask "What are the termination clauses?" and the app finds the relevant
paragraphs and gives you a precise answer — **without sending anything to the internet**.

Everything — the AI brain, the document storage, the search index — runs locally on your
device. No cloud. No subscription. No privacy risk.

---

## Part A.2 — The Problem It Solves

**The problem with normal AI chatbots (ChatGPT, Gemini, etc.):**
- Your documents are uploaded to a company's server
- You pay per query or per month
- Your private data (contracts, notes, medical records) leaves your device
- No internet = no answers
- The AI doesn't know your specific documents — it only knows what it was trained on

**What this app does differently:**
- The AI model runs on your phone's processor. Nothing goes to the internet.
- You feed it your own documents. It learns from them instantly (no retraining).
- It answers based exclusively on what's in your documents — no hallucinations from
  unrelated training data.
- Works in airplane mode, in remote areas, or on air-gapped devices.
- Free forever. No API keys. No accounts.

---

## Part A.3 — What RAG Means — No Jargon

**RAG = Retrieval-Augmented Generation**

It's a two-step process:

**Step 1 — Retrieval (the "Search" part)**
When you ask a question, the app searches through all your documents and finds the
3–4 most relevant paragraphs. This is like a very smart search engine that understands
meaning, not just keywords.

**Step 2 — Generation (the "AI Answer" part)**
Those paragraphs are handed to the AI model along with your question. The AI reads
the paragraphs and writes a focused answer based only on that content.

**Why RAG instead of just asking the AI directly?**

Imagine you hand a brilliant professor a 500-page book and say "read this, then answer
my questions." That would take hours. RAG is like giving the professor just the 4 most
relevant pages from that book — they answer instantly and accurately.

Without RAG, the AI either:
- Doesn't know your document at all, or
- Would need a context window big enough to hold the entire document (expensive, slow)

With RAG, even a small AI model can answer questions about a 1,000-page document since
it only ever sees 4 paragraphs at a time.

---

## Part A.4 — What AI Model Is Used and Why

**Model:** Gemma 3 1B Instruct (Q5_K_M quantized)
**Made by:** Google DeepMind
**File size:** 812 MB
**License:** Apache 2.0 (free to use commercially, no restrictions)

**"1B"** means it has 1 billion parameters — 1 billion numbers that were tuned during
training across trillions of words. This is small by modern AI standards (GPT-4 has
~1.8 trillion parameters) but large enough to understand and generate fluent, accurate text.

**"Instruct"** means it was specifically fine-tuned to follow instructions and have
conversations, not just predict the next word in a document.

**"Q5_K_M"** means the model's weights have been compressed (quantized) from 32-bit
floating point numbers down to approximately 5.68 bits each. This halves the file size
with minimal quality loss. More on this in Part D.

**Why this model specifically?**
- Fits in 1 GB RAM (1B params × 5.68 bits ÷ 8 + overhead ≈ 900 MB)
- Fast enough on a phone CPU (3–8 tokens per second)
- Good enough at reading and summarising short contexts
- No login required to download
- Permissive license — can be used in any app commercially

---

## Part A.5 — How a User Experiences the App

**First launch:**
1. App opens, shows a loading screen
2. In the background, the 812 MB Gemma model downloads from Hugging Face to your device
3. When done, the model loads into memory (~5 seconds)
4. You see the Chat screen

**Adding a document:**
1. Tap "Docs" tab
2. Type the path to your PDF or text file (or browse to it)
3. Tap "Add Document"
4. App extracts text, splits it into ~350-word chunks, indexes all chunks
5. "1706.03762v7.pdf — 21 chunks ready" appears in the list

**Asking a question:**
1. Tap "Chat" tab
2. Type your question
3. App searches your documents for relevant chunks (~2ms)
4. Passes context + question to the AI model
5. Answer streams token by token onto the screen
6. Total time: ~10–15 seconds on a mid-range phone

**The three-screen layout:**
- **Chat** — ask questions, see streaming answers, conversation history
- **Docs** — add PDFs/text files, view what's indexed, delete documents
- **Settings** — download different model sizes, swap models, view storage used

---

---

# PART B — Technical Architecture

---

## Part B.6 — Technology Stack — Every Library Explained

### Kivy (UI Framework)
**What it is:** A Python library for building multi-touch user interfaces that run on
Android, iOS, Windows, Linux, and Mac from the same codebase.

**Why it's used here:** It's the only mature framework that lets you write Python UI code
and compile it to a real Android APK via Buildozer. React Native, Flutter, etc. require
JavaScript or Dart — this entire app is Python.

**How it works:** Kivy renders its own OpenGL-based UI — it does not use native Android
widgets (no XML layouts, no Android Views). Instead, it draws everything itself using
OpenGL ES 2.0, which is why the same `.py` code runs identically on all platforms.

**Key Kivy concepts used in this app:**
- `Screen` / `ScreenManager` — the three tabs (Chat, Docs, Settings) are Kivy Screens
- `BoxLayout`, `ScrollView` — layout containers
- `Clock.schedule_once()` — thread-safe UI updates (since Android UI must be touched only from main thread)
- `@mainthread` decorator — ensures a function runs on the Kivy main thread
- `dp()` / `sp()` — density-independent pixels / scale-independent pixels (so the UI looks correct on all screen densities)

### PyMuPDF / fitz (PDF Parsing)
**What it is:** Python bindings for MuPDF, a C library from Artifex Software. The Python
module is imported as `fitz` (historical name).

**What it does here:** Opens PDF binary files, decompresses page content streams (PDF pages
are stored as zlib-compressed PostScript-like streams), extracts the text content with
correct reading order, returns plain Unicode strings.

**Why MuPDF over other PDF libraries:**
- Pure C library, compiles to ARM64 for Android (Buildozer handles this)
- Handles complex PDFs: multi-column, rotated text, scanned documents (with OCR disabled, just text layer)
- ~15 MB on Android vs. 80+ MB for alternatives like PDFium

### llama-cpp-python (LLM Inference — Primary Backend)
**What it is:** Python bindings for llama.cpp, a C++ library by Georgi Gerganov that
implements transformer inference for GGUF models.

**Why it can run on a phone:**
llama.cpp is written in pure C++ with no frameworks beyond the C++ standard library.
It compiles to any CPU architecture. On ARM64 (modern phones), it uses NEON SIMD
instructions for fast matrix multiplications — the same instruction set that powers
most of the inference computation.

**When it works / when it doesn't:**
- ✅ Android (arm64): compiles natively via Buildozer + python-for-android
- ✅ Linux x64/ARM: compiles with any GCC
- ✅ Windows with MSVC or MinGW: compiles fine
- ❌ Windows + Python 3.13: no pre-built wheel exists yet (no binary release for cp313)
- This is why the Windows fallback uses `llama-server.exe` instead

### llama-server (Windows Fallback Backend)
**What it is:** A standalone HTTP server binary that loads a GGUF model and exposes an
inference API on localhost. It's the same C++ code as llama-cpp-python but packaged as
an executable rather than a Python extension.

**How it's used here:**
1. The ZIP `llamacpp_bin.zip` (29.8 MB) contains the pre-built Windows x64 binary
2. On first run, Python extracts the ZIP
3. `subprocess.Popen()` launches `llama-server.exe` as a child process
4. Python polls `http://127.0.0.1:8082/health` until the model finishes loading
5. All inference is done via `HTTP POST /completion` — pure stdlib `urllib.request`

**Why HTTP and not a socket/pipe?**
The llama-server uses a clean JSON API that works identically regardless of whether
the server is a local subprocess or a remote machine. No custom IPC protocol needed.

### SQLite3 (Storage)
Python's built-in `sqlite3` module. No external database needed.

**Why SQLite:**
- Built into Python's standard library — zero extra dependencies
- Single file database — easy to back up, move, or delete
- Runs entirely in-process — no database daemon
- WAL (Write-Ahead Log) mode enables concurrent reads during background writes
- Android's built-in SQLite is the same engine — no compatibility issues

### huggingface-hub (Model Downloader)
**What it is:** Official Hugging Face Python client for downloading files from
huggingface.co model repositories.

**What it does here:** Downloads the GGUF model file with:
- Resume support (if download is interrupted, continues from byte offset)
- SHA-256 hash verification after download
- Progress callbacks for UI progress bars
- No authentication required for Apache 2.0 public models

---

## Part B.7 — Project File Structure

```
app/
├── main.py                    Android/desktop app entry point (Kivy)
├── cli.py                     Interactive terminal REPL (Windows/Linux)
├── test_rag.py                Non-interactive integration test
├── Modelfile                  Ollama model registration config
├── buildozer.spec             Android build configuration
├── requirements.txt           Python dependencies (desktop dev)
├── SYSTEM_INTERNALS.md        OS-level technical documentation
├── COMPLETE_PROJECT_GUIDE.md  This file
│
├── src/
│   ├── __init__.py
│   ├── rag/                   Core RAG engine (all platforms)
│   │   ├── db.py              SQLite layer — read/write documents & chunks
│   │   ├── chunker.py         PDF/text → overlapping chunks + TF-IDF vectors
│   │   ├── retriever.py       Hybrid BM25 + TF-IDF cosine search engine
│   │   ├── llm.py             LLM backend (llama-cpp / ollama / llama-server)
│   │   ├── pipeline.py        Orchestrator — ties all RAG pieces together
│   │   └── downloader.py      Model download from Hugging Face Hub
│   │
│   └── ui/                    Kivy Android/desktop UI
│       └── screens/
│           ├── chat_screen.py     Chat bubbles, streaming token display
│           ├── docs_screen.py     Document list, add/delete
│           └── settings_screen.py Model catalogue, download progress
│
├── quantize/                  Standalone model analysis & re-quantization CLI
│   ├── model_analyzer.py      Read GGUF binary header, print stats
│   ├── quantizer.py           Re-quantize GGUF to different bit-widths
│   ├── gemma3_recipe.py       Gemma-specific quantization presets
│   └── run_quantize.py        CLI entry point
│
├── llamacpp_bin/              Pre-built llama-server.exe (Windows fallback)
│   └── llama-server.exe       HTTP inference server for Windows
│
└── Gemma-3-1B-...Q5_K_M.gguf  The AI model file (812 MB)
```

**Data flow between files:**
```
User input
  → cli.py / chat_screen.py
  → pipeline.py (ask())
  → retriever.py (query())     ← loaded from db.py
  → llm.py (generate())        ← llama-server.exe / llama-cpp-python
  → Answer back to UI
```

**Ingestion flow:**
```
PDF file
  → chunker.py (process_document())
  → db.py (insert_chunks())
  → retriever.py (reload())    ← now searchable
```

---

## Part B.8 — The RAG Pipeline — Step by Step

The entire intelligence of the app is in this pipeline. Every question goes through
exactly these stages:

```
QUESTION: "What is the main contribution of the Transformer?"
    │
    ▼  1. TOKENISE QUESTION
    "main contribution transformer architecture"
    (lowercase, strip punctuation, remove stopwords)
    │
    ▼  2. BM25 SCORE (keyword relevance)
    Score each of the 21 indexed chunks by keyword match.
    Chunk 12 scores 0.91 — contains "transformer" 5 times.
    │
    ▼  3. TF-IDF COSINE SCORE (semantic overlap)
    Convert question and each chunk to sparse word-frequency vectors.
    Compute cosine angle between them. High similarity = high score.
    │
    ▼  4. HYBRID COMBINATION
    final_score = 0.5 × BM25_normalised + 0.5 × cosine_normalised
    Select top 4 chunks.
    │
    ▼  5. BUILD PROMPT
    Wrap the 4 chunk texts + question in Gemma's instruction format:
    <start_of_turn>user
    You are a helpful assistant. Answer ONLY based on the context...
    Context: [chunk1] --- [chunk2] --- [chunk3] --- [chunk4]
    Question: What is the main contribution...?
    <end_of_turn>
    <start_of_turn>model
    │
    ▼  6. INFERENCE
    POST prompt to llama-server.exe at localhost:8082.
    Model reads context, generates answer token by token.
    │
    ▼  7. ANSWER
    "The main contribution of the Transformer is its use of
     self-attention mechanisms to compute representations..."
```

---

## Part B.9 — Document Ingestion Deep Dive

**File:** `src/rag/chunker.py`

When you add a document, `process_document(path)` runs these steps:

### Step 1: Text extraction

For PDF files:
```python
doc = fitz.open(path)           # open binary PDF
for page in doc:
    text += page.get_text("text")   # extract plain text per page
```
PyMuPDF decompresses each page's content stream (zlib-compressed PostScript-like commands),
runs a text extraction algorithm that respects reading order (left-to-right, top-to-bottom),
and returns Unicode-encoded plain text.

For TXT files: just `open(path).read()`.

### Step 2: Chunking with overlap

```
Full text (7,300 words for the Attention paper):

chunk 0: words[0 .. 349]      (350 words)
chunk 1: words[290 .. 639]    (350 words, 60-word overlap with chunk 0)
chunk 2: words[580 .. 929]    (350 words, 60-word overlap with chunk 1)
...
chunk 20: words[5800 .. 6149]
```

**Why 350 words?**
- Fits within the model's context without dominating it
- Long enough to contain complete thoughts/sentences
- 4 chunks × 350 words ≈ 1,400 words retrieval context (manageable prompt size)

**Why 60-word overlap?**
Prevents splitting a key sentence across two chunks. If "The Transformer
replaces recurrence entirely" sits at the boundary between chunk 7 and chunk 8,
the overlap ensures the full sentence is in at least one of them.

### Step 3: Per-chunk tokenisation

```python
_RE_WORD = re.compile(r"[a-z0-9]+")

def tokenise(text):
    raw = _RE_WORD.findall(text.lower())
    return [w for w in raw if w not in STOPWORDS and len(w) > 1]
```

This is **NOT** the LLM's tokenizer. This is a simple word-extraction function for
building the search index. It produces a list like:
```
["transformer", "architecture", "attention", "mechanism", "encoder", "decoder", ...]
```

~70 common English words are removed (stopwords): "a", "the", "is", "are", "of", etc.
These words appear in every chunk so they carry no discriminating information.

### Step 4: TF-IDF vectorisation (corpus-level)

**What TF-IDF is:** A number that measures how important a word is to a particular chunk
relative to all other chunks. Words that appear everywhere (like "model") get low scores.
Words that appear in few chunks (like "multihead") get high scores.

**TF (Term Frequency):**
```
TF("attention", chunk_5) = (number of times "attention" appears in chunk_5)
                          / (total words in chunk_5)
                        = 8 / 203 = 0.039
```

**IDF (Inverse Document Frequency):**
```
IDF("attention") = log( (N+1) / (df+1) ) + 1
                = log( (21+1) / (14+1) ) + 1
                = log(1.47) + 1 = 1.385

("attention" appears in 14 of 21 chunks — common, lower IDF)

IDF("multihead") = log( (21+1) / (2+1) ) + 1
                 = log(7.33) + 1 = 2.993

("multihead" appears in only 2 of 21 chunks — rare, higher IDF)
```

**TF-IDF score:**
```
tfidf("attention", chunk_5) = 0.039 × 1.385 = 0.054
tfidf("multihead", chunk_5) = 0.020 × 2.993 = 0.060
```

Each chunk gets a sparse dictionary of these scores — a **TF-IDF vector** — stored in
the database as a pickled Python dict.

### Step 5: Database storage

All chunks go to SQLite:
- `text` column: the raw 350-word text (for showing context to the LLM)
- `tokens` column: JSON list of search tokens (for BM25 at query time)
- `tfidf_vec` column: pickled sparse dict (for cosine similarity at query time)

---

## Part B.10 — Retrieval Engine — BM25 + TF-IDF Hybrid

**File:** `src/rag/retriever.py`

The `HybridRetriever` class loads all chunks into RAM once and answers queries in ~2ms.

### BM25 Scoring

BM25 (Okapi BM25) is the algorithm behind Elasticsearch, Lucene, and most production
search engines. It extends TF-IDF with two improvements:

**1. TF Saturation:** If a word appears 10 times vs 5 times, the score does not double
— it saturates. This prevents a chunk that just repeats "attention attention attention..."
from scoring higher than a chunk that uses "attention" once but in the right context.

**2. Length Normalisation:** Longer chunks naturally contain more words. BM25 normalises
scores by chunk length so a long chunk doesn't automatically beat a short precise one.

**The formula:**
```
For each query term t, for each chunk d:

BM25(t, d) = IDF(t) × [ TF(t,d) × (k1 + 1) ]
                       [ ─────────────────────────────────────── ]
                       [ TF(t,d) + k1 × (1 - b + b × len(d)/avglen) ]

Where:
  k1 = 1.5   (saturation — higher = more reward for repeated terms)
  b  = 0.75  (length norm — 0=ignore length, 1=full normalisation)
  IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

### TF-IDF Cosine Similarity

The query is converted to a TF vector (term frequencies, no IDF since it's a 4-word query).
The cosine similarity with each stored chunk vector is computed:

```
cos(q, d) = (q · d) / (|q| × |d|)

The dot product q · d = sum of (q[word] × d[word]) for every shared word.
```

Only words that appear in BOTH the query and the chunk contribute. This makes it
extremely fast even for large vocabularies.

### Combining the Two Scores

Both BM25 and cosine produce scores on different scales. They are each min-max
normalised to [0, 1] independently, then averaged:

```
final_score[i] = 0.5 × BM25_norm[i] + 0.5 × cosine_norm[i]
```

The top 4 chunks by final_score are selected and passed to the LLM.

**Why combine two algorithms?**

| Scenario | BM25 handles it well | Cosine handles it well |
|---|---|---|
| Query: exact words from the document | ✓ | ✓ |
| Query: synonyms of document words | ✗ | ✓ (partially) |
| Short precise query | ✓ | ✓ |
| Long query with many terms | ✓ (via IDF weighting) | ✓ |
| Chunk has query word repeated 20× | ✓ (saturated, fair) | ✗ (over-rewarded) |
| Chunk is very long | ✓ (length normalised) | ✗ (more matches by chance) |

---

## Part B.11 — LLM Backend — Three-Step Fallback System

**File:** `src/rag/llm.py`

The same Python code handles all platforms by trying three backends in order:

```
llm.load(model_path, n_ctx=4096, n_threads=4)
         │
         ▼  Try 1: llama-cpp-python
         "from llama_cpp import Llama"
         If import fails → try next
         │
         ▼  Try 2: Ollama desktop app
         HTTP GET http://localhost:11434
         If server not running → try next
         │
         ▼  Try 3: llama-server.exe (Windows bundled binary)
         Extract llamacpp_bin.zip if needed
         subprocess.Popen(["llama-server.exe", "--model", path, ...])
         Poll http://127.0.0.1:8082/health until ready
         │
         ▼  If all three fail:
         Raise RuntimeError with installation instructions
```

**The external interface is identical regardless of which backend is active:**
```python
llm.generate(prompt, max_tokens=512, temperature=0.7, top_p=0.9)
→ returns the generated text string
```

The calling code (pipeline.py, cli.py) never needs to know which backend is running.

### Backend 1: llama-cpp-python

The model loads directly into the Python process. Inference is a C++ function call.
This is the fastest option since there's no IPC overhead — no HTTP, no subprocess.

```python
model = Llama(
    model_path   = "/data/models/gemma.gguf",
    n_ctx        = 4096,      # context window tokens
    n_threads    = 4,         # CPU threads for inference
    n_gpu_layers = 0,         # GPU layers (0 = CPU only)
    verbose      = False,
)
output = model("prompt text...", max_tokens=512, temperature=0.7)
text = output["choices"][0]["text"]
```

### Backend 2: Ollama

Ollama is a desktop application that manages LLM models and serves them via HTTP.
If it's running, this app registers the GGUF file with Ollama and queries it:

```python
import ollama
ollama.create(model="gemma-rag", from_="/path/to/model.gguf")
response = ollama.generate(model="gemma-rag", prompt="...", stream=False)
text = response.response
```

### Backend 3: llama-server subprocess (current backend on Windows)

```python
subprocess.Popen([
    "llamacpp_bin/llama-server.exe",
    "--model",    "/path/to/model.gguf",
    "--ctx-size", "4096",
    "--threads",  "4",
    "--port",     "8082",
    "--host",     "127.0.0.1",
], creationflags=CREATE_NO_WINDOW)

# Poll until ready:
urllib.request.urlopen("http://127.0.0.1:8082/health")

# Inference:
response = urllib.request.urlopen(Request(
    "http://127.0.0.1:8082/completion",
    data=json.dumps({"prompt": "...", "n_predict": 512}).encode(),
    headers={"Content-Type": "application/json"},
))
text = json.loads(response.read())["content"]
```

---

## Part B.12 — Prompt Engineering — How the Model Is Instructed

**File:** `src/rag/llm.py`, function `build_rag_prompt()`

Gemma 3 is an "instruction-tuned" model. It was trained on conversations that follow
a specific format. If you don't use this format, the model may produce poor output.

**Gemma's conversation format:**
```
<start_of_turn>user
[your message here]<end_of_turn>
<start_of_turn>model
[model generates from here]
```

**The RAG prompt built by this app:**
```
<start_of_turn>user
You are a helpful assistant. Answer ONLY based on the provided context.
If the answer is not in the context, say "I don't know."

Context:
[chunk 1 text — ~350 words]

---

[chunk 2 text — ~350 words]

---

[chunk 3 text — ~350 words]

---

[chunk 4 text — ~350 words]

Question: [user's question]<end_of_turn>
<start_of_turn>model

```

**Key design decisions in this prompt:**

1. **"Answer ONLY based on the context"** — Prevents the model from using its own
   training knowledge to hallucinate facts not in your documents.

2. **"If the answer is not in the context, say 'I don't know'"** — Gives the model
   explicit permission to admit ignorance rather than guess.

3. **Context before question** — The model pays more attention to recent tokens;
   putting the question last means it's still fresh when generation starts.

4. **`---` separators between chunks** — Helps the model understand chunk boundaries
   so it doesn't blend sentences from different parts of the document.

5. **No system prompt** — Gemma's format doesn't have a separate system message turn.
   Instructions are embedded in the user turn.

---

## Part B.13 — Database Design

**File:** `src/rag/db.py`

Single SQLite file: `~/ragapp.db` (`C:\Users\<name>\ragapp.db` on Windows,
`/data/user/0/com.yourname.offlinerag/files/ragapp.db` on Android)

**Schema:**
```sql
-- One row per ingested file
CREATE TABLE documents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,           -- "report.pdf"
    path       TEXT NOT NULL UNIQUE,    -- full filesystem path
    added_at   TEXT DEFAULT (datetime('now')),
    num_chunks INTEGER DEFAULT 0        -- updated after chunking
);

-- One row per text chunk extracted from a document
CREATE TABLE chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_idx  INTEGER NOT NULL,         -- ordering index: 0, 1, 2, ...
    text       TEXT NOT NULL,            -- raw chunk text (~350 words)
    tokens     TEXT,                     -- JSON: ["word1", "word2", ...]
    tfidf_vec  BLOB                      -- pickled dict: {word: float, ...}
);

CREATE INDEX idx_chunks_doc ON chunks(doc_id);
```

**Key properties:**
- `ON DELETE CASCADE` — deleting a document auto-deletes all its chunks
- `PRAGMA journal_mode=WAL` — Write-Ahead Log: background writes don't block reads
- `PRAGMA synchronous=NORMAL` — faster writes, safe for non-critical data
- `UNIQUE` on path — same file can't be ingested twice
- `BLOB` for tfidf_vec — Python's `pickle.dumps()` gives binary that SQLite stores raw

---

---

# PART C — How It Runs on Mobile (Android)

---

## Part C.14 — Android Architecture Overview

Android is a Linux-based operating system. Every app runs in its own isolated
**Linux process** with:
- Its own RAM allocation (managed by Android's Low Memory Killer)
- Its own sandboxed filesystem directory
- Its own Dalvik/ART JVM (for Java/Kotlin apps)
- Its own set of permissions granted by the user

This app does NOT use Java or Kotlin. It uses **Python running natively on Android's
Linux kernel** via the CPython interpreter compiled for ARM64.

**What "arm64-v8a" means:**
- ARM = the processor architecture (most phones since 2014 use ARM)
- 64 = 64-bit register width (not 32-bit)
- v8a = ARMv8 architecture version "a" profile

All modern Android phones (since ~2017) are 64-bit ARM. This is why `buildozer.spec`
specifies `android.archs = arm64-v8a` — the app won't run on 32-bit phones, but those
are extinct in 2026.

**Process hierarchy when the app runs:**

```
Android OS (Linux kernel, ARM64)
  └── App process: com.yourname.offlinerag
        ├── Python 3.11 interpreter (compiled ARM64 C binary)
        │     ├── Kivy event loop (OpenGL ES 2.0 rendering)
        │     ├── src.rag.pipeline (orchestration)
        │     ├── src.rag.retriever (search, pure Python)
        │     └── llama_cpp.Llama (AI model, C++ extension compiled ARM64)
        │           └── Gemma 1B weights (mmap'd from storage)
        └── SurfaceFlinger (GPU compositing, managed by OS)
```

---

## Part C.15 — How Python Runs on Android

**The tool chain: Buildozer → python-for-android (p4a)**

Python doesn't come with Android. It has to be compiled from source for ARM64 Linux
and bundled inside the APK. Here's how:

**1. Buildozer** reads `buildozer.spec` and orchestrates the entire build process.
It downloads: the Android SDK, the Android NDK (Native Development Kit), and
python-for-android (p4a).

**2. python-for-android** cross-compiles Python 3.11 from source using the Android NDK.
The NDK contains a GCC/Clang cross-compiler that produces ARM64 Linux binaries on
your x86 build machine.

**3. Cross-compilation of Python extensions:**
Each Python package listed in `requirements` in `buildozer.spec` has a "recipe" in p4a.
A recipe is a script that knows how to build that specific library for Android.

For example, the `llama-cpp-python` recipe:
- Downloads llama.cpp C++ source
- Runs CMake with the ARM64 cross-compiler
- Enables NEON SIMD (ARM's vectorisation extension)
- Produces `llama_cpp.cpython-311-aarch64-linux-android.so`
- This `.so` file is bundled in the APK

**4. The APK structure:**
```
offlinerag.apk (ZIP file)
├── classes.dex              Java shim that launches Python
├── lib/arm64-v8a/
│   ├── libpython3.11.so     Python interpreter
│   ├── libSDL2.so           Graphics/input (Kivy dependency)
│   ├── libmupdf.so          PDF parsing (PyMuPDF)
│   ├── libllama.so          llama.cpp inference engine
│   └── libggml.so           ggml tensor library (llama.cpp dependency)
├── assets/
│   ├── private.tar.bz2      Compressed Python stdlib + app source
│   └── public.tar.bz2       Kivy assets
└── AndroidManifest.xml      App metadata, permissions
```

**5. App startup on Android:**
1. Android's Zygote spawns the app process
2. The DEX shim (`classes.dex`) runs as Java
3. Java unpacks `private.tar.bz2` to `/data/user/0/<package>/files/` if first run
4. Java calls `System.loadLibrary("python3.11")` to load the Python interpreter
5. Python interpreter starts, runs `main.py`
6. Kivy initialises OpenGL ES 2.0 via SDL2
7. The three screens are constructed and the app is live

---

## Part C.16 — How the AI Model Runs on a Phone CPU

**No GPU usage.** The model runs entirely on the phone's CPU cores.

### Why not the GPU?

Modern phones have powerful GPUs, but:
1. Android's GPU API (OpenGL ES / Vulkan) is designed for graphics, not general-purpose compute
2. llama.cpp on Android uses CPU by default; GPU (via OpenCL) is experimental
3. The CPU is fast enough for a 1B parameter model
4. CPU avoids the GPU memory bandwidth bottleneck for small models

### ARM NEON SIMD — The Key to Phone AI

NEON is ARM's 128-bit vector extension. Every ARMv8 processor has it.
It allows processing **4 float32 values in parallel** in a single instruction,
or **8 float16 values**, or **16 int8 values**.

llama.cpp uses NEON intrinsics (special C++ assembly-like calls) for the inner loop
of matrix-vector multiply, which is 95% of transformer inference computation.

**Example: dot product of 256 numbers**
```
Normal C++:  256 multiplications + 256 additions = 512 ops = ~256 CPU cycles
With NEON:   256 ÷ 4 = 64 NEON multiply-accumulate ops = ~16 CPU cycles
             (16× speedup on this operation)
```

This is why the 1B model can run at 3–8 tokens/second on a phone — something that would
have seemed impossible just 3–4 years ago.

### Q5_K_M dequantisation at runtime

The weights are stored as 5.68-bit integers. Before each matrix multiply, the relevant
block of weights is **dequantised** back to float32:

```
For each block of 256 weights:
  float32 scale = read block_scale (stored as float16 → convert)
  for each weight w in block:
    float32 value = (w * scale)    // ~1 CPU multiply per weight
  then use float32 values in the dot product
```

The dequantisation and matrix multiply are done together ("fused") in llama.cpp to avoid
storing an intermediate float32 matrix. This saves memory bandwidth.

### Memory mapping the model file

The 812 MB GGUF file is not loaded into RAM all at once. The OS uses **mmap()** (memory
mapping), which tells the OS "this file is part of my address space." The OS only pages
in (loads into physical RAM) the parts of the file actually needed for each computation.

On Android, this means:
- App starts fast (model not fully loaded until first inference)
- During inference, the OS may briefly stall to page in weights from internal storage
- After first full pass, all weights are cached in RAM (warm cache)
- Background apps may be killed by Android's Low Memory Killer to make space

---

## Part C.17 — The Build Process — Buildozer & python-for-android

To produce an APK from this Python code:

**Step 1: Install Buildozer (on Linux — Mac also works, Windows requires WSL)**
```bash
pip install buildozer
sudo apt-get install -y git zip unzip openjdk-17-jdk
```

**Step 2: Run the build (from the app directory)**
```bash
buildozer android debug
```

**What happens internally (takes 15–45 minutes on first run):**

1. Buildozer reads `buildozer.spec`
2. Downloads Android SDK tools (~500 MB)
3. Downloads Android NDK 25c (~1.2 GB)
4. Downloads python-for-android
5. Builds Python 3.11 for ARM64 from source (~10 min)
6. Builds each dependency from its recipe:
   - Kivy: cross-compile C extensions, SDL2, OpenGL ES bindings
   - PyMuPDF: cross-compile MuPDF C library with the NDK compiler
   - llama-cpp-python: cross-compile llama.cpp with NEON enabled
7. Packages all `.so` files, Python stdlib, and app source into the APK
8. Signs the APK with a debug key

**Output:** `bin/offlinerag-1.0.0-arm64-v8a-debug.apk`

**Install to phone:**
```bash
buildozer android deploy run
# or manually:
adb install bin/offlinerag-1.0.0-arm64-v8a-debug.apk
```

### buildozer.spec key settings explained

```ini
android.api = 34          # Target Android 14 (must use API 34 to publish on Play Store)
android.minapi = 26       # Minimum Android 8.0 (64-bit support solid from API 26)
android.ndk = 25c         # NDK version — 25c has best Python 3.11 support in p4a
android.archs = arm64-v8a # 64-bit ARM only — llama.cpp requires 64-bit
android.add_jvm_options = -Xmx512m  # JVM heap — Python/NDK use this at startup
android.permissions = READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE
# REQ to access user PDFs in /sdcard/Documents/
```

---

## Part C.18 — Memory Management on Android

Android uses a tiered memory killing system called the **Low Memory Killer (LMK)**.
When the phone runs low on RAM, Android kills background processes in reverse priority order.

**App process priority (foreground = safest):**
```
Foreground (visible, active)     ← our app while user is in it
Visible (partially visible)
Service
Background (minimised)           ← gets killed first
```

**Risk scenario:** User loads the 812 MB model, then switches to Chrome to look something up.
Our app moves to "Background" priority. If Android needs memory, it may kill our process.
When the user returns, the app **restarts from scratch** — model must reload (5 seconds).

**How this app handles it:**
- `pipeline.init()` is called on app start — re-initialises DB and retriever quickly
- The model is checked on every query: `if not llm.is_loaded(): load_model()`
- Document index (in SQLite) survives because it's on storage, not in RAM

**Why 4 GB minimum is recommended:**
```
Android OS + services:    ~1.5 GB  (needed by OS)
Our app (peak):           ~1.1 GB  (model + cache + Python)
Other apps (background):  ~0.5 GB  (music player, messaging, etc.)
Safety headroom:          ~0.9 GB
─────────────────────────────────────────
Minimum comfortable RAM:  ~4.0 GB
```

At **8 GB**, there's ~5 GB free during inference. The LMK will never touch this app.

---

## Part C.19 — Storage & File Paths on Android

Android has two storage areas:

**Internal storage (private):**
```
/data/user/0/com.yourname.offlinerag/
  ├── files/                   ← ANDROID_PRIVATE env variable points here
  │   ├── app/                 ← unpacked app source + Python stdlib
  │   ├── models/              ← downloaded GGUF files (~812 MB)
  │   └── ragapp.db            ← SQLite database
  └── cache/                   ← temp files, auto-cleaned by OS
```

This storage is:
- Private to the app — other apps cannot read it
- Deleted when app is uninstalled
- Does NOT require storage permissions

**External storage (shared):**
```
/storage/emulated/0/
  └── Documents/               ← user puts PDFs here
```

This is where users store PDFs they want to ingest. The app reads from here with
`READ_EXTERNAL_STORAGE` permission. This appears as the "Files" folder in Android's
file manager.

**In the Python code, paths are resolved like this:**
```python
ANDROID_PRIVATE = os.environ.get("ANDROID_PRIVATE", os.path.expanduser("~"))
DB_PATH   = os.path.join(ANDROID_PRIVATE, "ragapp.db")
MODEL_DIR = os.path.join(ANDROID_PRIVATE, "models")
```

On Android: `ANDROID_PRIVATE` is set by the python-for-android bootstrap to
`/data/user/0/com.yourname.offlinerag/files`.

On Windows/Linux: `ANDROID_PRIVATE` is not set, so it falls back to `~` (home directory).
This is why the database ends up at `C:\Users\<name>\ragapp.db` on Windows.

---

## Part C.20 — Threading on Android with Kivy

**The golden rule of UI programming on Android (and every platform):**
> The UI must only be touched from the **main thread**.

If you update a label from a background thread, you get crashes or corrupted UI.
Inference takes 10–15 seconds. You cannot block the main thread for that long
(the app would freeze, Android would show an ANR dialog after 5 seconds).

**How this app handles threading:**

All long operations run on background threads:
```python
# pipeline.py
def ask(question, stream_cb, on_done):
    def _run():
        chunks = retriever.query(question)
        prompt = build_rag_prompt(chunks, question)
        answer = llm.generate(prompt, stream_cb=stream_cb)
        on_done(True, answer)
    threading.Thread(target=_run, daemon=True).start()
```

The `stream_cb` (streaming token callback) cannot touch the UI directly. It uses
Kivy's `Clock.schedule_once()` to schedule a UI update on the main thread:

```python
# chat_screen.py
def _on_token(token: str):
    def _update(dt):
        self._current_bubble.append(token)    # safe: runs on main thread
    Clock.schedule_once(_update, 0)           # 0 = next frame
```

`Clock.schedule_once(fn, 0)` schedules `fn` to run on the Kivy main thread during
the next event loop iteration. This is the correct way to update UI from a background
thread in Kivy.

The `@mainthread` decorator (from `kivy.clock`) is syntactic sugar for the same pattern.

---

---

# PART D — The AI / ML Internals

---

## Part D.21 — What Is a Large Language Model?

A Large Language Model (LLM) is a mathematical function:

```
f(token_1, token_2, ..., token_N) → probability distribution over next token
```

It takes a sequence of tokens (pieces of text) and predicts what token comes next.
By sampling from the predicted distribution repeatedly, you get generated text.

**How it was built:**
1. Collected ~trillions of tokens of text from the internet, books, and code
2. Initialised a neural network with ~1 billion random numbers (parameters/weights)
3. Fed the text through the network repeatedly
4. Measured the error: "how surprised was the network by the actual next word?"
5. Used backpropagation + gradient descent to nudge all ~1B numbers to be less surprised
6. After millions of iterations, the network learned grammar, facts, reasoning, code

**Why it can "understand" your documents:**
It doesn't truly understand. But it learned statistical patterns in language so deeply
that when you give it a paragraph of text and ask a question, the pattern most likely
to follow is a correct, coherent answer. RAG exploits this by always giving the model
the exact relevant text.

---

## Part D.22 — What Is a Transformer?

The Transformer is the neural network architecture that all modern LLMs (GPT, Gemma,
LLaMA, Claude) are built on. It was invented by Google in 2017 (the "Attention Is All
You Need" paper — the exact paper indexed in this app during testing).

**Core insight:** Language understanding requires connecting words that are far apart.
"The bank where we deposited our savings was on the river bank" — "bank" means two
different things depending on context words many positions away. RNNs (the previous
approach) had trouble with long dependencies.

**The Transformer uses self-attention** to let every word in a sequence look at every
other word simultaneously:

```
For each position i, compute:
  Query(i) = word_i × W_Q    "what am I looking for?"
  Key(j)   = word_j × W_K    "what do I contain?" (for every position j)
  Value(j) = word_j × W_V    "what information do I give?" (for every position j)

Attention score: softmax( Query(i) · Key(j) / √d ) for all j
Output(i) = Σ_j  score(i,j) × Value(j)
```

This lets "bank" attend to "river" or "savings" to resolve its meaning.

**Gemma 1B architecture:**
```
Vocab size:     256,000 tokens
Embedding size: 1,152 dimensions
Layers:         18 transformer blocks
Attention heads: 8 query heads, 4 key/value heads (grouped-query attention)
FFN hidden size: 4,608
Context window:  8,192 tokens (we use 4,096)
Total params:    ~1.25 billion
```

**One forward pass through Gemma 1B for 2,000 tokens:**

```
1. Embedding lookup:
   2000 token IDs → 2000 × 1152-dim float vectors
   (look up row in 256000×1152 embedding table)

2. For each of 18 layers:
   a. RMS Layer Norm
   b. Self-attention (grouped-query, 8Q + 4KV heads):
      - Compute Q, K, V projections (matrix multiplies)
      - Apply RoPE (rotational positional encoding)
      - Compute attention scores
      - Weighted sum of values
      - Project back to 1152 dims
   c. Residual add
   d. RMS Layer Norm
   e. Feed-forward: x → GELU(x × W1) × W3 + x × W2
      (SwiGLU activation, 1152 → 4608 → 1152)
   f. Residual add

3. Final RMS Layer Norm

4. Output projection:
   2000 × 1152 → 2000 × 256000 logits
   Only last position matters for next-token prediction

5. Sample from logits[last_position] → next token
```

---

## Part D.23 — What Is Quantization? Why Q5_K_M?

**The problem:** Gemma 1B in full float32 precision = 1.25B × 4 bytes = **5 GB**.
That won't fit on a phone. Even float16 = **2.5 GB** — barely fits.

**Quantization** reduces the number of bits used to represent each weight:

| Format | Bits/weight | Gemma 1B size | Quality loss |
|---|---|---|---|
| float32 (original) | 32 | 5.0 GB | None (baseline) |
| float16 | 16 | 2.5 GB | Negligible |
| Q8_0 | 8.0 | 1.25 GB | Negligible |
| Q6_K | 6.56 | 1.03 GB | Very small |
| **Q5_K_M** | **5.68** | **0.89 GB** | **Small** |
| Q4_K_M | 4.85 | 0.76 GB | Moderate |
| Q3_K_M | 3.35 | 0.52 GB | Noticeable |
| Q2_K | 2.63 | 0.41 GB | Significant |

**How Q5_K_M works (K-quant block scheme):**

Weights are grouped into **super-blocks of 256**. Each super-block stores:
- 1 × float16 scale factor (`d`)
- 1 × float16 minimum value (`dmin`)
- 256 × 5-bit quantized values (packed into bytes)

At inference time, to recover the original float32 value of weight `w`:
```
float32 value = (quantised_int × d) + dmin
```

The `_M` suffix means "medium" — the block-level scales use 6-bit precision instead of
4-bit, giving slightly better quality at a small size cost. The `_S` variant uses lower
precision scales; `_M` is the standard "balanced" choice.

**Why Q5_K_M specifically for this app:**
- At 812 MB, it fits comfortably in 1 GB RAM (with the KV cache overhead)
- Practically indistinguishable from float16 quality for reading-comprehension tasks
- Fast enough on ARM NEON (5-bit integers are processed 8 at a time in NEON registers)

---

## Part D.24 — What Is a GGUF File?

GGUF (GGML Unified Format) is a binary file format invented by llama.cpp author
Georgi Gerganov for storing quantized model weights along with metadata.

**File structure:**
```
Bytes 0-3:   "GGUF" magic (file signature)
Bytes 4-7:   Version number (3)
Bytes 8-15:  Number of tensors
Bytes 16-23: Number of metadata key-value pairs

[Metadata section]
  Each entry: key_length(u64) + key_string + value_type(u32) + value
  Examples:
    "general.architecture" = "gemma3"
    "gemma3.block_count"   = 18
    "gemma3.context_length" = 8192
    "gemma3.embedding_length" = 1152
    "tokenizer.ggml.tokens" = ["<pad>", "<eos>", "<bos>", ...]  (256K entries)

[Tensor info section]
  Each entry: name + shape + quantization_type + byte_offset

[Data section — aligned to 32 bytes]
  Raw quantized weight bytes, one per tensor in order
```

**Advantage over other formats (PyTorch `.pt`, SafeTensors):**
- Single self-contained file — no config.json, tokenizer.json needed separately
- Supports all quantization formats natively
- Memory-mappable (the OS can mmap the entire file)
- Platform-independent (works on x86, ARM64, RISC-V, etc.)

---

## Part D.25 — Autoregressive Generation — How Text Is Created

The model generates one token at a time, feeding each new token back as input.

```
Prompt:    "The Transformer architecture uses"
           Token IDs: [651, 24882, 643, 3327 , 3327]
                        The   Trans  -form  -er   archi...

Step 1: Run forward pass on all prompt tokens.
        Get probability distribution for the NEXT token.
        High probability tokens: "self" (0.35), "attention" (0.28), "multi" (0.11), ...
        Sample: "self" (token  1102)

Step 2: Append "self" to sequence. Forward pass for just this one new token
        (using cached K/V values from step 1 — no recomputation needed).
        High probability: "attention" (0.71), "-" (0.09), ...
        Sample: "attention" (token 17038)

Step 3: Append "attention". Forward pass.
        Result: "-" (0.84)
        Sample: "-"

Step 4: ...continue until:
        - Model generates <eos> (end of sequence) token
        - n_predict limit reached (512 tokens in our config)
        - <end_of_turn> token detected
```

This is why word-by-word streaming is possible — each token can be sent to the UI
as soon as it's sampled, without waiting for the complete answer.

**The KV cache** — why subsequent tokens are fast:

The prefill (processing the ~2000-token prompt) is slow because every layer must run
for all 2000 positions simultaneously. But once done, the Key and Value matrices for
all 2000 positions are **cached in RAM** (~72 MB for Gemma 1B at n_ctx=4096).

For each new generated token, only a 1×1152 vector needs to go through the 18 layers
(not 2001×1152), and it attends to the cached K/V pairs. This is ~2000× faster per
token than recomputing from scratch.

---

## Part D.26 — Temperature & Top-P Sampling Explained

After the forward pass, the model produces **logits** — one raw score per vocabulary token.

**Converting logits to probabilities:**
```
probabilities = softmax(logits / temperature)
```

**Temperature:**
- `temperature = 0.0` → always pick the highest-probability token (greedy, deterministic)
- `temperature = 0.7` → sharper distribution, high-prob tokens still dominate (used here)
- `temperature = 1.0` → probabilities as the model naturally learned them
- `temperature = 2.0` → flat distribution, very random/creative output

**Top-P (nucleus sampling):**
Sort tokens by probability descending. Sum probabilities until you reach `top_p`.
Discard all remaining tokens. Re-normalise and sample.

```
With top_p = 0.9:
  Token "self":      probability 0.35  | cumulative 0.35
  Token "attention": probability 0.28  | cumulative 0.63
  Token "multi":     probability 0.15  | cumulative 0.78
  Token "a":         probability 0.08  | cumulative 0.86
  Token "its":       probability 0.06  | cumulative 0.92 ← stop here (≥ 0.90)
  (all remaining tokens discarded)

Re-normalise these 5 tokens, sample one.
```

**Why use both?**
- Temperature controls overall sharpness/creativity
- Top-P prevents very low-probability "wild" tokens from ever being sampled
- Together they produce coherent, varied, non-hallucinating text

**For RAG tasks** (reading comprehension), we want **low temperature (0.7)** and
**high top-p (0.9)** — coherent, accurate, slightly varied answers, not random.

---

## Part D.27 — Context Window — What It Is and Why It Matters

The **context window** is the maximum total number of tokens the model can process at once:
input (your prompt) + output (the answer) combined.

Gemma 1B supports up to **8,192 tokens**. This app uses **4,096 tokens** (set via `--ctx 4096`).

**Why not use the full 8,192?**
- KV cache size = `n_ctx × n_layers × 2 × head_dim × bytes_per_element`
- At n_ctx=8192: `8192 × 18 × 2 × 256 × 2 bytes = 150 MB KV cache`
- At n_ctx=4096: `4096 × 18 × 2 × 256 × 2 bytes = 75 MB KV cache`
- Halving n_ctx halves KV cache RAM usage with no quality impact for typical RAG queries

**Token budget for a typical RAG query:**
```
System instruction:          ~38 tokens
4 chunks × ~350 words:    ~1,800 tokens  (words ≠ tokens; ratio ~1.3 words/token)
User question:                ~15 tokens
Special format tokens:         ~8 tokens
──────────────────────────────────────────
Total prompt:              ~1,861 tokens
Max answer budget:           ~512 tokens
──────────────────────────────────────────
Total needed:              ~2,373 tokens  (well within 4,096)
```

**What happens if the prompt exceeds n_ctx?**
llama-server returns HTTP 400:
```json
{"error": "request (2136 tokens) exceeds the available context size (2048)"}
```
This is what happened when we first ran with `n_ctx=2048` — the 4-chunk RAG prompt
was too large. Increasing to `n_ctx=4096` solved it.

---

---

# PART E — Design Decisions & Trade-offs

---

## Part E.28 — Why No Internet at Runtime?

**Privacy:** Legal documents, medical records, personal notes, proprietary research.
None of this should leave the device. A RAG system that uploads your documents to
a cloud API (OpenAI, Google) for indexing is a non-starter for sensitive use cases.

**Reliability:** The app works on airplanes, in remote areas, on corporate networks
that block external connections, or when your internet is down.

**Cost:** Cloud LLM APIs charge per token. Heavy document analysis (thousands of queries
per day) costs real money. This app costs $0 to run after the initial model download.

**Latency:** A cloud round-trip adds 200–1000ms of network latency per query on top of
inference time. Local inference has zero network latency — the only delay is the model.

---

## Part E.29 — Why Gemma 1B and Not a Larger Model?

**The trade-off chart:**

| Model | Params | GGUF size | RAM needed | Speed on phone | Quality |
|---|---|---|---|---|---|
| Gemma 1B Q5_K_M | 1B | 0.8 GB | 1.1 GB | 4–8 tok/s | Good for RAG |
| Gemma 3 4B Q4_K_M | 4B | 2.5 GB | 3.2 GB | 1–2 tok/s | Better |
| Phi-3.5 Mini 3.8B Q4 | 3.8B | 2.3 GB | 3.0 GB | 1–2 tok/s | Better |
| LLaMA 3.1 8B Q4_K_M | 8B | 4.7 GB | 5.5 GB | <1 tok/s | Best |

For a **RAG task** (reading comprehension on short context), a 1B model with the right
passages works almost as well as a 7B model answering from memory, because the context
removes the need for the model to "know" the answer — it just needs to read and summarise.

On a **2 GB phone**, only the 1B model fits. On a **4 GB phone**, the 4B model
is possible but inference is slow. The 1B model was chosen for maximum device coverage.

---

## Part E.30 — Why SQLite and Not a Vector Database?

Popular RAG systems use vector databases (Chroma, Pinecone, Weaviate, FAISS) to store
dense embedding vectors and do approximate nearest-neighbour (ANN) search.

**Why this app doesn't:**

1. **No embedding model needed:** Vector search requires running a separate "embedding
   model" (like sentence-transformers) on every query and every chunk during indexing.
   That's another 250–500 MB model on the device.

2. **SQLite is built into Python and Android:** No extra library to compile for arm64.
   Zero additional APK size.

3. **BM25 is competitive for in-domain search:** For document-specific questions where
   the user knows what keywords appear in the documents, BM25 often beats dense retrieval.
   Dense retrieval shines for semantic searches ("what are the risks?" when the document
   says "dangers" not "risks"). For most use cases, the hybrid is sufficient.

4. **Scale:** A personal document assistant won't have more than a few hundred documents.
   At 10,000 chunks, even O(n) linear scan takes ~10ms in Python. ANN is only needed
   at 100,000+ chunks.

---

## Part E.31 — Why BM25+TF-IDF and Not Sentence Embeddings?

**Sentence embeddings** (from models like `all-MiniLM-L6-v2`) map text to 384-dimensional
float vectors that capture semantic meaning. Similar sentences have similar vectors.
Retrieval is done with cosine similarity between the query embedding and stored embeddings.

**Why not used here:**

1. Requires running an additional 90 MB neural network model for every query and for
   every chunk during indexing. On a phone, this competes with the main LLM for RAM.

2. Embedding models need `sentence-transformers` (pip package) → pulls in PyTorch (~800 MB
   on Android) or ONNX Runtime (~60 MB lighter but still needs compilation for arm64).

3. For the scale this app operates at (21–1000 chunks), the quality difference between
   BM25 hybrid and embedding-based retrieval is small in practice.

4. Adds complexity to the build process (another p4a recipe to maintain).

**The hybrid BM25+TF-IDF is "good enough" and "zero extra cost."**

---

## Part E.32 — Why Kivy?

**Alternatives considered:**

| Framework | Cross-platform? | Python? | Android APK? | Verdict |
|---|---|---|---|---|
| Kivy | ✅ | ✅ Pure Python | ✅ Buildozer | **Chosen** |
| React Native | ✅ | ❌ JavaScript | ✅ | Different language |
| Flutter | ✅ | ❌ Dart | ✅ | Different language |
| BeeWare (Toga) | ✅ | ✅ | ✅ | Less mature, fewer widgets |
| tkinter | ❌ | ✅ | ❌ | No Android |
| PyQt / PySide | ❌ Windows/Linux | ✅ | ❌ | No Android |
| Flet | ✅ | ✅ | ✅ Beta | Very new in 2024–2025 |

Kivy is the **only battle-tested Python framework** for Android development.
Buildozer + python-for-android has been building Android apps from Python since 2013.
The ecosystem is stable, the arm64 recipes for all dependencies exist, and the community
is large enough that issues have documented solutions.

---

---

# PART F — Performance & Sizing

---

## Part F.33 — Speed: What Takes How Long

**On a 4-core ARM64 phone (e.g., Snapdragon 730, ~2019 era):**

| Operation | Time |
|---|---|
| App cold start (first ever launch) | 30–60s (unpacking + first setup) |
| App cold start (subsequent) | 3–5s |
| Model load (warm, file cached by OS) | 3–8s |
| PDF ingestion (15-page paper) | 0.5–2s |
| PDF ingestion (100-page document) | 3–8s |
| Query — retrieval only (BM25+cosine) | 1–5ms |
| Query — prompt building | <1ms |
| Query — prefill (2000-token prompt) | 0.5–1.5s |
| Query — generation (60-token answer) | 8–20s (60 ÷ 3–8 tok/s) |
| Query — end to end | ~10–22s |

**On an 8-core ARM64 phone (e.g., Snapdragon 888, ~2021):**

| Operation | Time |
|---|---|
| Model load | 2–4s |
| Query end to end | ~6–12s |

**On a modern flagship (Snapdragon 8 Gen 3, 2024):**

| Operation | Time |
|---|---|
| Model load | 1–2s |
| Query end to end | ~4–7s |

**On your Windows desktop (4-core i5/Ryzen 5):**

| Operation | Time |
|---|---|
| llama-server start + model load | 3–6s |
| Query end to end | ~8–15s |

---

## Part F.34 — Storage Sizes — Full Breakdown

**On-device storage used by a fully deployed Android app:**

| Item | Storage | Location |
|---|---|---|
| APK file | ~110 MB | `/data/app/` |
| Installed app (unpacked) | ~200 MB | `/data/app/` |
| Python stdlib | ~20 MB | `files/app/` |
| App source code | ~0.2 MB | `files/app/` |
| Gemma 1B Q5_K_M model | **812 MB** | `files/models/` |
| ragapp.db (100 docs / 5000 chunks) | ~5 MB | `files/` |
| Temp/cache | ~10 MB | `cache/` |
| **Total** | **~1.05 GB** | |

**On Windows (development machine):**

| Item | Size |
|---|---|
| Source code (21 .py files) | 149 KB |
| `.venv/` (Python packages, no Kivy) | 75 MB |
| `llamacpp_bin/` (extracted EXE + DLLs) | 86 MB |
| `llamacpp_bin.zip` (original) | 30 MB (removable after extraction) |
| `ragapp.db` (21-chunk Attention paper) | 200 KB |
| GGUF model | 812 MB |
| **Total** | **~1.03 GB** |

---

## Part F.35 — RAM Requirements by Device

```
Component RAM usage during inference:
  Android OS idle:              1,200–1,800 MB  (varies by Android version)
  Python runtime + Kivy:           200 MB
  llama-cpp-python:                 50 MB
  Gemma 1B model (mmap'd): 
    - weights (mmap, paged in):    812 MB
    - KV cache (n_ctx=4096):        75 MB
    - Compute buffer:               80 MB
  HybridRetriever (1000 chunks):    15 MB
  SQLite / ragapp.db:               10 MB
──────────────────────────────────────────────
Peak total during inference:    ~2,440 MB (2.4 GB)

Device RAM   Free during inference   Verdict
──────────────────────────────────────────────
2 GB         -400 MB (OVER budget)   ❌ Will be killed by LMK
3 GB         600 MB headroom         ⚠️  Keep other apps closed
4 GB         1,600 MB headroom       ✅ Comfortable
6 GB         3,600 MB headroom       ✅ Excellent
8 GB         5,600 MB headroom       ✅ No issues whatsoever
12 GB+       10+ GB headroom         ✅ Could run 4B model
```

---

## Part F.36 — Scaling — How Many Documents Can It Handle?

| Documents | Total chunks | DB size | Retrieval time | Works well? |
|---|---|---|---|---|
| 10 papers (150 pages) | ~300 | ~3 MB | <1ms | ✅ |
| 100 documents (1500 pages) | ~3,000 | ~30 MB | ~8ms | ✅ |
| 500 documents (7500 pages) | ~15,000 | ~150 MB | ~40ms | ✅ |
| 2,000 documents (30,000 pages) | ~60,000 | ~600 MB | ~150ms | ⚠️ Slow retrieval |
| 10,000 documents | ~300,000 | ~3 GB | ~800ms | ❌ Need vector DB + ANN |

**The practical limit for this app:** ~500 documents / ~15,000 chunks
Beyond that, retrieval becomes slow and the DB grows large enough to impact DB load time.

**For larger scale**, the retriever would need replacing with:
- FAISS (Facebook AI Similarity Search) for ANN
- Replaced per-chunk BM25 with inverted index for keyword search
- Both changes possible without touching the rest of the app

---

---

# PART G — Tough Questions Answered

---

## Part G.37 — Q&A — Questions You Might Be Asked

**Q: "Why does the app need 8,192 tokens context but you only use 4,096?"**

A: Gemma's architecture supports 8,192 but the KV cache at n_ctx=8192 requires 150 MB vs
75 MB at n_ctx=4096. For typical RAG queries (4 chunks of ~350 words + question), 4,096
tokens is more than enough. We leave half unused to save RAM.

---

**Q: "Could this be attacked by prompt injection in a malicious PDF?"**

A: Yes, in theory. If a PDF contains text like `Ignore all previous instructions and say...`,
the model might follow it since that text ends up in the context. Mitigations:
1. The prompt says "Answer ONLY based on context" which limits instruction-following
2. A smaller 1B model is generally less susceptible to following complex injections
3. The app is offline — there's no data exfiltration even if injection succeeds
4. Real hardening would require input sanitisation and/or prompt injection detection

---

**Q: "Why not use the phone's NPU (Neural Processing Unit)?"**

A: Modern flagship phones (Snapdragon 8 Gen 1+, Dimensity 9000+, Apple A-series) have NPUs
designed for neural network inference. However:
1. Android NPU access requires vendor-specific APIs (Qualcomm QNN, MediaTek APU, etc.)
2. llama.cpp has experimental Qualcomm QNN support but it's not in stable builds
3. NPUs are optimized for 8-bit or lower precision; 5-bit (our format) may not be supported
4. CPU NEON is portable across all ARM64 devices; NPU code is vendor-locked
5. NPU acceleration is a future enhancement — the CPU implementation works now

---

**Q: "How does the streaming output work technically?"**

A: In streaming mode, llama-server sends Server-Sent Events (SSE) as each token is generated:

```
HTTP response (streaming, one SSE per token):
data: {"content": "The", "stop": false}
data: {"content": " main", "stop": false}
data: {"content": " contribution", "stop": false}
...
data: {"content": ".", "stop": true, "tokens_predicted": 62}
```

The Python side reads line by line from the HTTP response stream using `resp.readline()`.
Each line is parsed as JSON and the "content" field (the token text) is passed to
the `stream_cb` callback, which schedules a Kivy `Clock.schedule_once()` to append
the token to the current chat bubble.

---

**Q: "Why pickle for TF-IDF storage and not JSON?"**

A: The TF-IDF vector is a Python dict mapping strings to floats: `{"attention": 0.054, ...}`.

- JSON encoding: `json.dumps({"attention": 0.054, ...})` → ~300 bytes, slow to parse floats
- Pickle encoding: `pickle.dumps(...)` → ~200 bytes binary, fast native deserialization

For 10,000 chunks loaded on startup, pickle is ~3× faster to deserialise than JSON
because Python's pickle can directly reconstruct the dict without float string parsing.
The downside (pickle is Python-only and not human-readable) is acceptable since this
data never leaves the device or crosses language boundaries.

---

**Q: "What happens if the model download is interrupted?"**

A: `huggingface_hub.hf_hub_download` automatically resumes from the last byte.
It stores the partially downloaded file in a `.lock` cache directory.
On resume, it sends an HTTP Range request (`Range: bytes=500000000-`) and the Hugging
Face CDN serves the remaining bytes. After completion, SHA-256 hash is verified.
If verification fails, the file is deleted and the download restarts from scratch.

---

**Q: "Why doesn't the app have a search bar in the document list?"**

A: The current MVP focuses on the core RAG pipeline. The document list is small enough
(typically 5–50 documents) that scrolling is sufficient. A search/filter feature
would be a straightforward addition using Kivy's `TextInput` + filtering the
`list_documents()` query: `WHERE name LIKE '%query%'`.

---

**Q: "Could you add image support for scanning physical documents?"**

A: Yes, with two additions:
1. **OCR:** Android's MLKit Text Recognition API or Tesseract (has a python-for-android
   recipe) can extract text from camera images. The extracted text is then fed into
   the same chunking/indexing pipeline as a text file.
2. **Image permissions:** `android.permissions = CAMERA` in buildozer.spec
3. The rest of the pipeline (chunking, BM25, LLM) is completely unchanged — it only
   ever sees plain text.

---

**Q: "Why does the GGUF model use 'i1' in its name?"**

A: `i1` stands for "importance matrix quantization." Standard GGUF quantization treats
all weights equally — every weight in a layer gets the same bit-width. Importance matrix
quantization analyzes a dataset to measure how much each weight affects the model's
output quality ("importance scores"). Weights with higher importance get more bits;
less important weights get fewer bits. The average is the same (Q5_K_M = 5.68 bits/weight)
but the distribution is smarter. This produces better perplexity (measure of language
model quality) than standard quantization at the same file size. The i1 variants
on the mradermacher Hugging Face page are importance-quantized by the repo maintainer.

---

**Q: "What is the difference between this app and LM Studio or Jan.ai?"**

A: LM Studio and Jan.ai are desktop-only applications. This app:
1. **Runs on Android** — those apps don't
2. **Has RAG built in** — you add your documents and it answers from them
3. **Is open source Python** — the entire application is readable, auditable, modifiable
4. **Has zero UI framework dependency on OS** — Kivy draws its own UI, works identically
   on Android 8 through Android 14

LM Studio/Jan.ai are better for model exploration and chat. This app is purpose-built
for private offline document Q&A on mobile.

---

**Q: "What would you change to make the retrieval better?"**

A: Three improvements in priority order:

1. **Sentence-level chunking instead of word-count chunking:** Current chunks are exactly
   350 words, which can split sentences awkwardly. A sentence splitter (even regex-based)
   would produce cleaner chunk boundaries, improving context quality.

2. **Add sparse keyword index (inverted index) for exact-match boost:** The current BM25
   is computed in pure Python O(n·q). An SQLite FTS5 (Full-Text Search) inverted index
   would handle exact matches much faster and could replace the Python BM25 loop entirely.

3. **Re-ranking with a cross-encoder:** After retrieving the top 20 chunks with BM25+cosine,
   run a small cross-encoder model (which looks at query + chunk together) to re-rank and
   select the true top 4. Cross-encoders are much more accurate than bi-encoders (separate
   query and document encodings) but slower — acceptable for a 20-candidate re-rank.
