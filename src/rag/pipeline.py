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
from .llm      import llm, build_rag_prompt, list_available_models
from .downloader import auto_download_default, model_dest_path, DEFAULT_MODEL


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
    """Call from UI to receive auto-download progress events."""
    global _auto_dl_progress_cb, _auto_dl_done_cb
    _auto_dl_progress_cb = on_progress
    _auto_dl_done_cb     = on_done


def init() -> None:
    """Call once at app start: set up DB, retriever, then auto-download default model."""
    init_db()
    retriever.reload()
    _start_auto_download()


def _start_auto_download() -> None:
    """Download Gemma 3 4B Q4 in the background; auto-load when done."""
    def _progress(frac: float, text: str):
        if _auto_dl_progress_cb:
            _auto_dl_progress_cb(frac, text)

    def _done(success: bool, path: str):
        if success:
            # Auto-load the model right after download finishes
            if not llm.is_loaded():
                load_model(
                    path,
                    on_done=lambda ok, msg: (
                        _auto_dl_done_cb(ok, msg) if _auto_dl_done_cb else None
                    ),
                )
            elif _auto_dl_done_cb:
                _auto_dl_done_cb(True, f"Model ready: {Path(path).name}")
        else:
            if _auto_dl_done_cb:
                _auto_dl_done_cb(False, path)  # path contains error msg here

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
    on_done: Optional[Callable[[bool, str], None]] = None,
) -> None:
    """Load a GGUF model in a background thread."""
    def _run():
        try:
            llm.load(model_path)
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
