# O-RAG — Offline RAG Android App

A fully offline AI assistant for Android that can answer questions from your own PDF and TXT documents. Everything runs on your device — no internet required after setup, no data sent to any server.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [How It Works — Simple Explanation](#how-it-works--simple-explanation)
3. [Architecture Overview](#architecture-overview)
4. [Project Structure](#project-structure)
5. [AI Models Used](#ai-models-used)
6. [RAG Pipeline — In Detail](#rag-pipeline--in-detail)
7. [Retrieval — How the App Finds Relevant Text](#retrieval--how-the-app-finds-relevant-text)
8. [Chat History Compression](#chat-history-compression)
9. [Android Service](#android-service)
10. [Database](#database)
11. [UI Screens](#ui-screens)
12. [Building the APK](#building-the-apk)
13. [Two Build Flavors](#two-build-flavors)
14. [Device Storage Breakdown](#device-storage-breakdown)
15. [Permissions](#permissions)
16. [Performance Decisions](#performance-decisions)
17. [Known Limitations](#known-limitations)
18. [Dependencies](#dependencies)

---

## What It Does

- Chat freely with an AI (like a local ChatGPT) — no internet needed
- Attach a PDF or TXT file and ask questions about it — the AI reads your document and answers from it
- All processing happens on your phone — your documents never leave your device
- Works completely offline after the first model setup

---

## How It Works — Simple Explanation

Think of it in three steps:

**1. You upload a document (e.g. a PDF)**
The app breaks your document into small chunks of ~80 words each. It stores these chunks in a local database on your phone.

**2. You ask a question**
The app searches through all the chunks and finds the 2 most relevant ones — like a search engine over your document.

**3. The AI answers**
The 2 best chunks are handed to the AI along with your question. The AI reads them and generates an answer — it stays strictly within what's in your document.

If you have no document uploaded, the app just chats with you normally (like a regular AI assistant).

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│               Android App (Kivy UI)          │
│                                              │
│  chat_screen.py  ←→  pipeline.py            │
│       │                   │                  │
│       │            ┌──────┴──────┐           │
│       │            │             │           │
│       │        retriever       llm.py        │
│       │        (BM25+TF-IDF    (Qwen via     │
│       │        +Semantic)      llama-server) │
│       │            │             │           │
│       │          db.py        nomic embed    │
│       │        (SQLite)       (port 8083)    │
│       │                          │           │
│  ┌────┴────────────────────────┐ │           │
│  │     Android Foreground      │─┘           │
│  │  Service (service/main.py)  │             │
│  │  Qwen server  port 8082     │             │
│  └─────────────────────────────┘             │
└─────────────────────────────────────────────┘
```

The app has two separate processes running at the same time:
- **Main process** — the Kivy UI (what the user sees)
- **Service process** — a background Android service that keeps the Qwen AI server alive even when the app is minimised

---

## Project Structure

```
.
├── main.py                    # App entry point — starts UI + service
├── buildozer.spec             # Android build config (permissions, SDK version, etc.)
├── requirements.txt           # Python dependencies
│
├── rag/                       # Core RAG logic
│   ├── pipeline.py            # Orchestrates everything: ingest → retrieve → generate
│   ├── chunker.py             # Splits documents into ~80-word chunks
│   ├── retriever.py           # Hybrid search: BM25 + TF-IDF + Semantic
│   ├── db.py                  # SQLite database for documents and chunks
│   ├── llm.py                 # LLM backend (Qwen via llama-server)
│   └── downloader.py          # Downloads/extracts models on first launch
│
├── ui/
│   └── screens/
│       └── chat_screen.py     # The entire UI — chat + document upload
│
├── service/
│   └── main.py                # Android background service — keeps llama-server alive
│
├── assets/
│   ├── app_icon.png
│   └── app_splash.png
│
└── .github/
    └── workflows/
        ├── build_apk.yml          # Bundled build — models inside APK (~1.3 GB)
        └── build_apk_slim.yml     # Slim build — models download on first launch (~120 MB)
```

---

## AI Models Used

### 1. Qwen 2.5 1.5B Instruct Q4_K_M — Generation Model
- **Job**: Answer questions, chat, generate responses
- **Size**: ~1.1 GB on disk
- **Port**: 8082
- **Quantization**: Q4_K_M — 4-bit quantization, good quality for a 1.5B parameter model
- **Context window**: 768 tokens (enough for a RAG prompt + answer)
- **Max reply**: 320 tokens (~240 words)
- **Source**: [Qwen/Qwen2.5-1.5B-Instruct-GGUF on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)

### 2. Nomic Embed Text v1.5 Q4_K_M — Embedding Model
- **Job**: Convert text into numbers (vectors) so the app can measure similarity between your question and document chunks
- **Size**: ~80 MB on disk
- **Port**: 8083
- **Context window**: 128 tokens
- **Lazy start**: Only starts when you upload your first PDF — saves ~300 MB RAM if you only use direct chat
- **Source**: [nomic-ai/nomic-embed-text-v1.5-GGUF on HuggingFace](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF)

Both models run via **llama-server** — a pre-compiled ARM64 binary bundled inside the APK. It serves both models over localhost HTTP so the Python app can call them like a web API.

---

## RAG Pipeline — In Detail

RAG stands for **Retrieval Augmented Generation**. Instead of asking the AI to remember everything, you give it the relevant text at question time.

### Step 1 — Document Ingestion (when you tap + and pick a file)

```
PDF / TXT file
     │
     ▼
chunker.py — splits into ~80-word chunks with 15-word overlap
     │
     ▼
db.py — saves each chunk to SQLite (text + BM25 tokens + TF-IDF vector)
     │
     ▼
retriever.reload() — loads all chunks into RAM, starts background embedding
```

- **Chunk size**: 80 words — chosen to fit inside Nomic's 128-token context window
- **Overlap**: 15 words — so sentences that cross chunk boundaries don't lose meaning
- **Duplicate protection**: If you re-upload the same file, old chunks are deleted before adding new ones (no duplicates)
- **PDF streaming**: Large PDFs are copied in 1 MB pieces to avoid RAM spikes

### Step 2 — Query (when you send a message)

```
Your question
     │
     ▼
retriever.query() — scores every chunk against your question
     │
     ▼
Top 2 chunks selected
     │
     ▼
build_rag_prompt() — formats: [system] + [context chunks] + [question]
     │
     ▼
llm.generate() — Qwen reads the prompt and streams the answer back
```

### Step 3 — Answer Streaming

Tokens come back from Qwen one at a time. Instead of updating the screen 30 times per second, the app batches tokens for 80ms and then updates — smooth display without draining the CPU.

---

## Retrieval — How the App Finds Relevant Text

The app uses **three methods at once** and combines their scores:

### BM25 (30% weight)
Classic keyword search used by search engines. Rewards chunks that contain your exact query words. Works immediately — no setup needed.

### TF-IDF Cosine (20% weight)
Measures how unusual the matching words are. If your document is about cats and you mention "cat" — it's not very informative. But if you mention "hyperthyroidism" — that's very specific and scored higher.

### Semantic Embeddings (50% weight)
Converts both your question and each chunk into a list of ~768 numbers (a vector). Chunks whose meaning is closest to your question score highest — even if they don't share the same exact words. Uses the Nomic embedding model.

**Combined score formula:**
```
score = 0.30 × BM25 + 0.20 × TF-IDF + 0.50 × Semantic
```

If Nomic is not yet running (embeddings not ready), falls back to:
```
score = 0.50 × BM25 + 0.50 × TF-IDF
```

**Embedding cap**: Only the first 30 chunks get semantic embeddings computed. For larger documents, chunks 31+ use BM25+TF-IDF only. This keeps startup fast — 30 HTTP calls to Nomic instead of potentially hundreds.

---

## Chat History Compression

When you chat without a document (direct mode), the app keeps context of the conversation. But sending the entire history to Qwen every turn wastes context window space.

**Solution — rolling compression:**
- Last **3 turns** are kept word-for-word (verbatim)
- Turns older than that are compressed to their **first sentence only** (no LLM call needed — just a string split)
- The compressed summary is injected into the system message: *"Earlier in our conversation: [summary]"*

This means Qwen always has recent context without the oldest messages eating up all the 768 tokens.

---

## Android Service

The app runs a **foreground service** (`service/main.py`) in a separate Android process.

**Why?** Android can kill background apps to save RAM. If the Qwen server (a subprocess) lived inside the main app process, it would die every time the user switches apps. The service keeps it alive.

**How it works:**
1. Service starts when the app opens
2. It waits until the Qwen model file is on disk (extracted by the main app)
3. Starts `llama-server` as a subprocess on port 8082 with Qwen
4. Checks every 10 seconds — if the server crashes, restarts it automatically
5. Nomic server (port 8083) is **not** supervised by the service — it starts lazily only when needed

**Important**: The service is declared as `:foreground` in `buildozer.spec`. This means p4a (the Android build tool) automatically promotes it and shows a persistent notification — your Python code does **not** need to call `startForeground()` itself.

---

## Database

Uses **SQLite** stored at `$ANDROID_PRIVATE/ragapp.db`.

### Tables

**documents**
```sql
id         INTEGER PRIMARY KEY
name       TEXT            -- filename shown in UI
path       TEXT UNIQUE     -- full path on device
added_at   TEXT            -- timestamp
num_chunks INTEGER         -- how many chunks were created
```

**chunks**
```sql
id         INTEGER PRIMARY KEY
doc_id     INTEGER REFERENCES documents(id) ON DELETE CASCADE
chunk_idx  INTEGER         -- order within document
text       TEXT            -- the actual chunk text
tokens     TEXT            -- JSON list of lowercase words (for BM25)
tfidf_vec  BLOB            -- pickled {term: score} dict (for TF-IDF cosine)
```

### Integrity fixes
- `PRAGMA foreign_keys=ON` — enabled on every connection so `ON DELETE CASCADE` actually fires when a document is deleted
- WAL journal mode — faster concurrent reads/writes
- Re-upload protection — old chunks are deleted before new ones are inserted

---

## UI Screens

The app has a single screen (`chat_screen.py`) with everything in one place:

### Chat area
- Dark theme (#1a1a1a background, ChatGPT-style)
- User messages appear as bubbles on the right
- AI messages stream in on the left with an "AI" avatar
- Typing indicator (animated dots) while the AI is thinking

### Input bar (bottom)
- **[+] button** — opens the Android file picker to select a PDF or TXT
- **Text input pill** — type your message
- **[↑] send button** — dims when the model is not ready yet

### Attachment preview card
- Shows above the input bar when you pick a file
- Displays filename, file type, and size
- [×] button to remove the attachment before sending

### Document status card
- Appears inline in the chat while the file is being processed
- Shows a progress bar during ingestion
- Updates to a success message when done

### Welcome / loading message
- Shows on first launch: "Preparing model…"
- Updates to "Ready! Ask me anything." once Qwen is loaded
- Shows a download progress bar if the model needs to be downloaded first

---

## Building the APK

The APK is built automatically using **GitHub Actions** — you don't need Android Studio or a local Android SDK.

### Prerequisites

1. Fork or push this repo to GitHub
2. Go to **Actions** tab — builds start automatically on every push

### Build process (what happens automatically)

1. Ubuntu 22.04 runner starts
2. Python 3.11 + Java 17 installed
3. Buildozer + python-for-android installed
4. llama.cpp cloned from GitHub
5. APK built with Kivy + all Python dependencies
6. llama-server compiled from source for ARM64 Android
7. Models downloaded from HuggingFace (bundled build only)
8. Models + binary injected into the APK zip file
9. APK signed with a debug key
10. APK uploaded as a GitHub Actions artifact

**Build time**: ~45-60 minutes (first build), ~20-30 minutes (cached)

### Downloading the APK

Go to **Actions** → click the latest successful run → scroll to **Artifacts** at the bottom → download `O-RAG-debug-apk` or `O-RAG-slim-apk`.

---

## Two Build Flavors

### 1. Bundled APK — `build_apk.yml`
- **Triggers**: Automatically on every `git push` to main/master
- **Models**: Qwen (~1.1 GB) + Nomic (~80 MB) packed inside the APK
- **APK size**: ~1.3 GB
- **Artifact**: `O-RAG-debug-apk`
- **First launch**: Extracts models to device storage, then ready
- **Works offline**: Yes, from the very first launch
- **Total device storage used**: ~2.5 GB (APK copy + extracted models)

### 2. Slim APK — `build_apk_slim.yml`
- **Triggers**: Manual only (Actions → Run workflow button)
- **Models**: None bundled — downloaded on first launch via HuggingFace
- **APK size**: ~120 MB
- **Artifact**: `O-RAG-slim-apk`
- **First launch**: Downloads ~1.2 GB over WiFi (one time only, then cached)
- **Works offline**: Yes, after first-launch download completes
- **Total device storage used**: ~1.32 GB (no extra APK copy of models)
- **Savings**: ~1.18 GB vs bundled build

**Which to choose?**
- Use **Bundled** if you want to install and use immediately with no internet
- Use **Slim** if your device has limited storage (saves ~1.18 GB permanently)

---

## Device Storage Breakdown

### Bundled APK
| Item | Size |
|------|------|
| APK file in /data/app/ | ~1.3 GB |
| Extracted Qwen model | ~1.1 GB |
| Extracted Nomic model | ~80 MB |
| SQLite DB + app data | ~10 MB |
| **Total** | **~2.5 GB** |

### Slim APK
| Item | Size |
|------|------|
| APK file in /data/app/ | ~120 MB |
| Downloaded Qwen model | ~1.1 GB |
| Downloaded Nomic model | ~80 MB |
| SQLite DB + app data | ~10 MB |
| **Total** | **~1.32 GB** |

---

## Permissions

| Permission | Why it's needed |
|------------|----------------|
| `READ_EXTERNAL_STORAGE` | Read files from older Android versions |
| `WRITE_EXTERNAL_STORAGE` | Write files on older Android versions |
| `MANAGE_EXTERNAL_STORAGE` | Access all files (needed for file picker) |
| `READ_MEDIA_IMAGES` / `READ_MEDIA_VIDEO` | Media picker access on Android 13+ |
| `INTERNET` | Model download on first launch (slim build), HuggingFace |
| `FOREGROUND_SERVICE` | Keep the AI server running in background |
| `FOREGROUND_SERVICE_SPECIAL_USE` | Required for foreground services on Android 14 |
| `POST_NOTIFICATIONS` | Show the foreground service notification on Android 13+ |

---

## Performance Decisions

Every setting below was chosen carefully for a 1.5B model on a mobile device:

| Setting | Value | Reason |
|---------|-------|--------|
| Context window (`n_ctx`) | 768 tokens | RAG prompt ≈ 410 tokens + 320 reply = 730 max — fits with headroom |
| Max reply tokens | 320 | ~240 words — enough for a detailed answer |
| Chunk size | 80 words | Fits Nomic's 128-token context window (80 words ≈ 96 tokens) |
| Chunk overlap | 15 words | Prevents meaning loss at chunk boundaries |
| Top-K retrieval | 2 chunks | 2 × 800 chars ≈ 200 tokens of context — fits budget |
| Embedding cap | 30 chunks | Avoids 100s of serial HTTP calls on large docs |
| Embedding text cap | 300 chars | ≈ 100 tokens — matches Nomic ctx=128 |
| Token batch flush | 80ms | Smooth streaming UI without per-token screen redraws |
| PDF copy chunk | 1 MB | Streams large PDFs to disk without loading entire file into RAM |
| Nomic start | Lazy | Only starts when a PDF is uploaded — saves ~300 MB RAM for chat-only users |
| History verbatim | 3 turns | Keeps recent context accurate |
| History compression | First sentence | No LLM call needed — just a string split |
| JVM heap | 768 MB | `-Xmx768m` — prevents Gradle OOM during APK assembly |

---

## Known Limitations

- **History is not saved** — restarting the app clears the conversation (by design, saves storage)
- **RAG has no memory across questions** — each question is independent; the AI doesn't remember what you asked before in RAG mode
- **Large documents** — chunks 31+ don't get semantic embeddings (BM25+TF-IDF only). For a very large PDF this is fine for most questions.
- **PDF quality** — scanned PDFs (images of text) are not supported. The document must have selectable text.
- **Single document at a time** — you can upload multiple files and they all go into the same index, but there's no per-document filtering
- **Cold start** — the first time llama-server loads Qwen takes ~10-20 seconds depending on device speed. After that it stays loaded in the background service.

---

## Dependencies

### Python packages (installed by p4a during build)
| Package | Purpose |
|---------|---------|
| `kivy==2.3.0` | UI framework for Android |
| `pypdf` | Extract text from PDF files |
| `huggingface-hub` | Download models from HuggingFace |
| `requests` / `certifi` / `urllib3` | HTTP for HuggingFace downloads |
| `sqlite3` | Built-in Python — document/chunk database |
| `plyer` | Android file picker integration |
| `tqdm` | Progress bars during download |
| `filelock` / `packaging` / `fsspec` | HuggingFace hub dependencies |

### System components
| Component | Purpose |
|-----------|---------|
| `llama-server` (ARM64 binary) | Serves both AI models over localhost HTTP |
| Android NDK 25b | Compiles llama-server for ARM64 |
| Android SDK 34 | Build tools, APK signing |
| Python-for-Android (p4a) | Cross-compiles Python + packages for Android |
| Buildozer | Orchestrates the entire Android build |

### Build environment
| Tool | Version |
|------|---------|
| Python | 3.11 |
| Java | 17 (Temurin) |
| Android API target | 34 (Android 14) |
| Android min API | 26 (Android 8.0) |
| Android ABI | arm64-v8a only |
| Cython | 0.29.37 (pinned for pyjnius compatibility) |

---

## Quick Reference — File by File

| File | What it does |
|------|-------------|
| `main.py` | Starts the app, registers crash handler, calls `init()` after 0.3s |
| `rag/pipeline.py` | Central hub — connects UI to chunker, retriever, LLM, and downloader |
| `rag/chunker.py` | Reads PDF/TXT, splits into 80-word overlapping chunks |
| `rag/retriever.py` | Holds all chunks in RAM, scores them against queries |
| `rag/db.py` | SQLite CRUD — insert/delete documents and chunks |
| `rag/llm.py` | Talks to llama-server HTTP API, builds prompts, streams tokens |
| `rag/downloader.py` | Extracts models from APK assets or downloads from HuggingFace |
| `service/main.py` | Android background service — launches and supervises Qwen server |
| `ui/screens/chat_screen.py` | Entire UI — chat bubbles, file picker, progress bars, token streaming |
| `buildozer.spec` | Android build config — permissions, SDK/NDK versions, service declaration |
| `.github/workflows/build_apk.yml` | CI: bundled APK with models inside |
| `.github/workflows/build_apk_slim.yml` | CI: slim APK, models download on first launch |
