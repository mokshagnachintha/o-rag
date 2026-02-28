"""
pipeline.py — Orchestrates the full RAG pipeline:
    ingest document → retrieve context → generate answer
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Optional

from .db       import init_db, insert_document, update_doc_chunk_count
from .chunker  import process_document
from .db       import insert_chunks
from .retriever import HybridRetriever
from .llm      import llm, build_rag_prompt, build_direct_prompt, list_available_models
from .downloader import auto_download_default, model_dest_path, QWEN_MODEL, NOMIC_MODEL


# Module-level retriever (shared across the whole app)
retriever = HybridRetriever(alpha=0.5)


# ------------------------------------------------------------------ #
#  Initialisation                                                      #
# ------------------------------------------------------------------ #

# Registered callbacks for auto-download progress UI
_auto_dl_progress_cb: Optional[Callable[[float, str], None]] = None
_auto_dl_done_cb:     Optional[Callable[[bool, str], None]]  = None


def register_auto_download_callbacks(
    on_progress: Optional[Callable[[float, str], None]],
    on_done:     Optional[Callable[[bool, str], None]],
) -> None:
    """
    Call from UI to receive auto-download / model-load progress events.
    If the model is already loaded by the time this is called, on_done
    fires immediately so the UI never gets stuck in the loading state.
    """
    global _auto_dl_progress_cb, _auto_dl_done_cb
    _auto_dl_progress_cb = on_progress
    _auto_dl_done_cb     = on_done

    # Race guard 1: LLM already loaded in this process
    if on_done and llm.is_loaded():
        on_done(True, "Models ready: Qwen + Nomic")
        return

    # Race guard 2: foreground service already has llama-server running —
    # skip extraction/download and connect immediately.
    import urllib.request
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8082/health", timeout=1
        ) as r:
            if r.status == 200:
                llm._backend = "llama_server"  # mark as connected
                if on_done:
                    on_done(True, "Models ready: Qwen + Nomic (service)")
                return
    except Exception:
        pass


def init() -> None:
    """Call once at app start: set up DB, retriever, then auto-download default model."""
    init_db()
    retriever.reload()
    _start_auto_download()


def _start_auto_download() -> None:
    """Ensure Qwen + Nomic are on disk, then load Qwen and start Nomic server."""
    def _progress(frac: float, text: str):
        if _auto_dl_progress_cb:
            _auto_dl_progress_cb(frac, text)

    def _done(success: bool, _msg: str):
        """Called when BOTH models are ready on disk."""
        qwen_path  = model_dest_path(QWEN_MODEL["filename"])
        nomic_path = model_dest_path(NOMIC_MODEL["filename"])

        if not success:
            if _auto_dl_done_cb:
                _auto_dl_done_cb(False, _msg)
            return

        # ── 1. Start the Nomic embedding server (port 8083) ─────────── #
        # Done first while RAM is still clean; it only needs 512 context
        # and ~200 MB RAM for the 80 MB Q4 model.
        from .llm import start_nomic_server
        import os
        if os.path.isfile(nomic_path):
            threading.Thread(
                target=start_nomic_server,
                args=(nomic_path,),
                daemon=True,
            ).start()

        # ── 2. Load Qwen for generation (always use Qwen, not Nomic) ── #
        if not llm.is_loaded():
            load_model(
                qwen_path,
                on_progress=_progress,
                on_done=lambda ok, msg: (
                    _auto_dl_done_cb(ok, msg) if _auto_dl_done_cb else None
                ),
            )
        elif _auto_dl_done_cb:
            _auto_dl_done_cb(True, "Models ready: Qwen + Nomic")

    auto_download_default(on_progress=_progress, on_done=_done)


# ------------------------------------------------------------------ #
#  Document ingestion                                                  #
# ------------------------------------------------------------------ #

def ingest_document(
    file_path: str,
    on_done: Optional[Callable[[bool, str], None]] = None,
) -> None:
    """
    Ingest a .txt or .pdf file in a background thread.
    on_done(success: bool, message: str) is called on completion.
    """
    def _run():
        try:
            name = Path(file_path).name
            doc_id = insert_document(name, file_path)
            chunks = process_document(file_path)
            insert_chunks(doc_id, chunks)
            update_doc_chunk_count(doc_id, len(chunks))
            retriever.reload()
            if on_done:
                on_done(True, f"Ingested '{name}' — {len(chunks)} chunks")
        except Exception as e:
            if on_done:
                on_done(False, f"Error: {e}")

    threading.Thread(target=_run, daemon=True).start()


# ------------------------------------------------------------------ #
#  Model management                                                    #
# ------------------------------------------------------------------ #

def load_model(
    model_path: str,
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_done: Optional[Callable[[bool, str], None]] = None,
) -> None:
    """Load a GGUF model in a background thread, with live progress callbacks."""
    def _run():
        try:
            llm.load(model_path, on_progress=on_progress)
            if on_done:
                on_done(True, f"Model loaded: {Path(model_path).name}")
        except Exception as e:
            if on_done:
                on_done(False, f"Failed to load model: {e}")

    threading.Thread(target=_run, daemon=True).start()


def get_available_models() -> list[str]:
    return list_available_models()


def is_model_loaded() -> bool:
    return llm.is_loaded()


# ------------------------------------------------------------------ #
#  Query                                                               #
# ------------------------------------------------------------------ #

def chat_direct(
    question: str,
    history: list | None = None,
    stream_cb: Optional[Callable[[str], None]] = None,
    on_done: Optional[Callable[[bool, str], None]] = None,
) -> None:
    """
    Chat directly with the LLM — no document retrieval.
    history: list of (user_text, assistant_text) tuples for multi-turn context.
    """
    def _run():
        try:
            if not llm.is_loaded():
                if on_done:
                    on_done(False, "No LLM model loaded. Please load a GGUF model first.")
                return

            prompt = build_direct_prompt(question, history)
            answer = llm.generate(prompt, stream_cb=stream_cb)
            answer = answer.strip()
            if on_done:
                on_done(True, answer)

        except Exception as e:
            if on_done:
                on_done(False, f"Error during inference: {e}")

    threading.Thread(target=_run, daemon=True).start()


def ask(
    question: str,
    top_k: int = 4,
    stream_cb: Optional[Callable[[str], None]] = None,
    on_done: Optional[Callable[[bool, str], None]] = None,
) -> None:
    """
    Run a RAG query in a background thread.

    Args:
        question : user question
        top_k    : number of chunks to retrieve
        stream_cb: called with each new LLM token (for streaming UI)
        on_done  : called with (success, full_answer_or_error)
    """
    def _run():
        try:
            if retriever.is_empty():
                if on_done:
                    on_done(False, "No documents ingested yet.")
                return

            if not llm.is_loaded():
                if on_done:
                    on_done(False, "No LLM model loaded. Please load a GGUF model first.")
                return

            results = retriever.query(question, top_k=top_k)
            if not results:
                if on_done:
                    on_done(False, "No relevant context found.")
                return

            context_chunks = [text for text, _ in results]
            prompt = build_rag_prompt(context_chunks, question)

            answer = llm.generate(prompt, stream_cb=stream_cb)
            answer = answer.strip()

            if on_done:
                on_done(True, answer)

        except Exception as e:
            if on_done:
                on_done(False, f"Error during inference: {e}")

    threading.Thread(target=_run, daemon=True).start()
