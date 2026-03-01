# O-RAG (Offline Retrieval-Augmented Generation) Architecture Overview

This document explains the "Why" and "How" of the O-RAG Android application, breaking down the framework choices and the complete user journey from tapping the app icon to receiving an AI-generated answer.

---

## 1. Why we chose this framework (The "Why")

Building a heavy LLM application that runs entirely offline on an Android phone is extremely difficult because mobile phones have strict limits on battery, RAM, and storage. Here is why we made our specific architectural choices:

### Why Kivy & Python-for-Android (p4a)?
1.  **Fast Prototyping:** AI logic and text chunking are mathematically easiest to write in Python. Kivy lets us write both the UI and the AI logic in the exact same language.
2.  **p4a (Python-for-Android):** Mobile phones don't understand Python. `p4a` brings a portable Python interpreter and wraps it inside a native Android APK shell so the phone can run it perfectly.
3.  **Buildozer:** This automates compiling complex libraries, converting pure Python into an Android container, and requesting OS permissions (Storage, Internet) seamlessly.

### Why `llama.cpp` over PyTorch or TensorFlow Lite?
1.  **Memory Constraints:** `llama.cpp` is written in raw C++. It is the absolute fast, most RAM-efficient way to run gigabyte-sized LLMs like Qwen (specifically quantized `.gguf` formats) on mobile CPUs.
2.  **Pure CPU performance:** It is specifically heavily optimized for ARM64 mobile processors. 

### Why SQLite instead of Pinecone or ChromaDB?
Mobile users cannot install massive standalone Vector Database engines, and they don't want an app reliant on cloud APIs. SQLite is a lightweight, invisible database that lives natively inside a single file on the phone, preventing bloat.

### Why the custom "Fake PyMuPDF" (pypdf fallback)?
The original PyMuPDF library relies on massive C++ dependencies that crash Android Compilers (`p4a`). Falling back to a lightweight, pure-Python library (`pypdf`) keeps the Android compilation smooth and dramatically reduces the final `.apk` file size.

---

## 2. From Start to Chat (The "How" of opening the App)

When a user taps the O-RAG app icon, a complex choreographed sequence fires off:

### Step 1: `main.py` (The Front Door & The UI)
1.  **UI Render:** Kivy immediately draws the `ChatScreen` so the user isn't staring at a blank screen.
2.  **The Background Engine:** It fires up `service/main.py`. This is an Android Foreground Service. It tells the Android OS: *"Never kill this background process."* This ensures the heavy AI models stay loaded in RAM even if the user minimizes the app to text a friend.
3.  **The Orchestrator:** Finally, it calls `pipeline.init()`.

### Step 2: `pipeline.py` (The Manager)
1.  **Database Check:** It tells `db.py` to ensure the SQLite file (`ragapp.db`) exists and is ready.
2.  **Retrieval Boot:** It tells `retriever.py` to read the database and memorize the search index.
3.  **Model Boot:** It checks with `downloader.py`. If the 1GB model files are missing, it rips them out of the APK or downloads them from the internet. Finally, it tells `llm.py` to load those models into the phone's RAM.

*If the user just types "Hello!", they chat directly with the LLM (no RAG involved) via `llm.py`'s 8082 generation server. The app remembers the **last 3 messages back-and-forth** to maintain basic conversational history without overflowing the context window.*

---

## 3. The RAG Process (Uploading a PDF)

When the user uploads a PDF, the AI needs to ingest it.

### Step 1: Ingestion (`chunker.py`)
1.  **Extract:** It reads the raw text out of the 50-page PDF.
2.  **Chunk:** It slices the giant text down into bite-sized chunks of **80 words** each, with a 15-word overlap so sentences aren't randomly cut in half.
3.  **Math:** It drops useless grammar words ("the", "is") and creates a TF-IDF mathematical vector that identifies the most important, unique keywords in that specific chunk.

### Step 2: Storage (`db.py`)
1.  The raw 80-word chunk (needed later for human-readable context) AND the math vector are permanently saved inside the hidden Android `ragapp.db` SQLite hard drive file. This data lives forever and does not disappear when the app is restarted (until the user clicks "Delete Documents").

---

## 4. Querying the RAG (Getting an Answer)

When a user asks: *"What did the fox jump over?"* 

### Step 1: The Search (`retriever.py`)
The search engine fires up. It compares the user's question against every mathematical vector in the database across all uploaded PDFs. It grabs the best **Top 2 Chunks** (the 2 paragraphs most genetically similar to the user's question).

### Step 2: Building the Prompt (`llm.py`)
The phone has strictly limited RAM. To prevent crashes, the context window is hard-capped at **768 tokens (~600 words)** total. 
The system builds a giant prompt string:
1.  **The Context:** The 2 best 80-word chunks retrieved from the database.
2.  **The Question:** The user's prompt.
3.  **The Rules:** A strict hidden system prompt that commands the AI: *"Answer ONLY based on the provided context. If the answer is not in the context, say 'I don't know.' DO NOT just repeat the question."*

### Step 3: The Generation (`llm.py`)
This massive glued string is sent to the Qwen engine running on localhost Port 8082. The C++ engine processes it, strips out any internal `<think>`-tags if it babbles, and streams the final, clean answer back to the UI one word at a time.
