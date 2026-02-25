"""
downloader.py — Download Gemma GGUF models from Hugging Face Hub.

Uses `huggingface_hub.hf_hub_download` which:
  • Resumes interrupted downloads automatically
  • Verifies SHA-256 integrity after download
  • Reports byte-level download progress via tqdm callback

Default auto-download (runs on first app launch, no login required):
  mradermacher/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking-i1-GGUF
  →  Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking.i1-Q4_K_M.gguf (~806 MB)

Full catalogue (user can pick any in Settings — all Apache 2.0, no login needed):
  i1-Q2_K    ~690 MB  (smallest)
  i1-Q4_K_M  ~806 MB  ← AUTO (recommended)
  i1-Q5_K_M  ~851 MB
  i1-Q6_K    ~1.0 GB  (highest quality)
"""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Callable, Optional


# ------------------------------------------------------------------ #
#  Catalogue of available Gemma GGUF models                           #
# ------------------------------------------------------------------ #

# The model that is bundled in the APK and auto-used on first launch (Apache 2.0, no login needed)
DEFAULT_MODEL: dict = {
    "label":    "Gemma 3 1B i1-Q5_K_M uncensored (bundled, ~851 MB)",
    "repo_id":  "mradermacher/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking-i1-GGUF",
    "filename": "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking.i1-Q5_K_M.gguf",
    "size_mb":  851,
}

GEMMA_MODELS: list[dict] = [
    DEFAULT_MODEL,
    {
        "label":    "Gemma 3 1B i1-Q2_K (~690 MB)",
        "repo_id":  "mradermacher/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking-i1-GGUF",
        "filename": "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking.i1-Q2_K.gguf",
        "size_mb":  690,
    },
    {
        "label":    "Gemma 3 1B i1-Q5_K_M (~851 MB)",
        "repo_id":  "mradermacher/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking-i1-GGUF",
        "filename": "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking.i1-Q5_K_M.gguf",
        "size_mb":  851,
    },
    {
        "label":    "Gemma 3 1B i1-Q6_K (~1.0 GB)",
        "repo_id":  "mradermacher/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking-i1-GGUF",
        "filename": "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking.i1-Q6_K.gguf",
        "size_mb":  1010,
    },
]


# ------------------------------------------------------------------ #
#  Destination directory (same as llm.py models dir)                  #
# ------------------------------------------------------------------ #

# App root: src/rag/downloader.py → ../../..
_APP_ROOT_DL = Path(__file__).resolve().parent.parent.parent


def _models_dir() -> str:
    base = os.environ.get("ANDROID_PRIVATE", os.path.expanduser("~"))
    d = os.path.join(base, "models")
    os.makedirs(d, exist_ok=True)
    return d


def model_dest_path(filename: str) -> str:
    return os.path.join(_models_dir(), filename)


def is_downloaded(filename: str) -> bool:
    return os.path.isfile(model_dest_path(filename)) or _bundled_model_path(filename) is not None


def _bundled_model_path(filename: str) -> Optional[str]:
    """
    Return the path to the GGUF if it was bundled inside the APK or
    sits in the project root (desktop).  Returns None if not found.

    On Android, python-for-android extracts all app files to the
    directory pointed to by ANDROID_APP_PATH (p4a >= 2023.09) or to
    $ANDROID_PRIVATE/app/ on older builds.
    """
    candidates = [
        # Desktop / development: model sitting next to main.py
        str(_APP_ROOT_DL / filename),
        # Android: p4a extracts app files to ANDROID_APP_PATH
        os.path.join(os.environ.get("ANDROID_APP_PATH", ""), filename),
        # Android alternative layout (older p4a)
        os.path.join(os.environ.get("ANDROID_PRIVATE", ""), "app", filename),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


# ------------------------------------------------------------------ #
#  Download logic                                                      #
# ------------------------------------------------------------------ #

def _get_hf_hub():
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed.\n"
            "Install it with: pip install huggingface-hub"
        )


def _expected_bytes(repo_id: str, filename: str) -> int:
    """Return the file size in bytes from the HF Hub metadata (no download)."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        url  = hf_hub_url(repo_id=repo_id, filename=filename)
        meta = get_hf_file_metadata(url)
        return meta.size or 0
    except Exception:
        return 0


def download_model(
    repo_id:     str,
    filename:    str,
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_done:     Optional[Callable[[bool, str], None]]  = None,
) -> None:
    """
    Download a GGUF file from Hugging Face to the local models/ folder.
    Runs in a background thread.

    Progress is reported by polling the partial file size every 0.5 s,
    so it works with any version of huggingface_hub.

    on_progress(fraction 0-1, status_text) — called ~2×/sec during download
    on_done(success, dest_path_or_error)   — called on completion
    """
    def _run():
        dest = model_dest_path(filename)

        if os.path.isfile(dest):
            if on_progress:
                on_progress(1.0, "Already downloaded.")
            if on_done:
                on_done(True, dest)
            return

        hf_hub_download = _get_hf_hub()

        if on_progress:
            on_progress(0.0, "Connecting to Hugging Face...")

        # Fetch expected file size before download starts
        total_bytes = _expected_bytes(repo_id, filename)

        # --- progress poller (runs in its own thread) ---
        _stop_poll = threading.Event()

        def _poller():
            # huggingface_hub writes to a .incomplete temp file first
            inc_path = dest + ".incomplete"
            while not _stop_poll.wait(0.5):
                check = inc_path if os.path.isfile(inc_path) else dest
                if os.path.isfile(check):
                    done = os.path.getsize(check)
                    if total_bytes:
                        frac = min(done / total_bytes, 0.99)
                        mb_d = done        / 1_048_576
                        mb_t = total_bytes / 1_048_576
                        if on_progress:
                            on_progress(frac, f"{mb_d:.0f} / {mb_t:.0f} MB")
                    else:
                        mb_d = done / 1_048_576
                        if on_progress:
                            on_progress(0.0, f"{mb_d:.0f} MB downloaded...")

        poll_thread = threading.Thread(target=_poller, daemon=True)
        poll_thread.start()

        try:
            # Build kwargs carefully — older HF versions don't have some args
            kwargs: dict = {
                "repo_id":  repo_id,
                "filename": filename,
                "local_dir": _models_dir(),
            }
            # local_dir_use_symlinks added in ~0.17; silently skip if absent
            try:
                import inspect
                from huggingface_hub import hf_hub_download as _hfd
                if "local_dir_use_symlinks" in inspect.signature(_hfd).parameters:
                    kwargs["local_dir_use_symlinks"] = False
            except Exception:
                pass

            cached = hf_hub_download(**kwargs)

            _stop_poll.set()
            poll_thread.join(timeout=1)

            if os.path.abspath(cached) != os.path.abspath(dest):
                shutil.copy2(cached, dest)

            if on_progress:
                on_progress(1.0, "Download complete.")
            if on_done:
                on_done(True, dest)

        except Exception as e:
            _stop_poll.set()
            if on_done:
                on_done(False, f"Download failed: {e}")

    threading.Thread(target=_run, daemon=True).start()


def auto_download_default(
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_done:     Optional[Callable[[bool, str], None]]  = None,
) -> None:
    """
    Locate or download the default model.  Priority order:
      1. Already present in models/ dir (from a previous run)
      2. Bundled inside the APK (present in app directory after install)
      3. Download from Hugging Face Hub
    """
    filename = DEFAULT_MODEL["filename"]
    dest     = model_dest_path(filename)

    # 1. Already in models/ dir
    if os.path.isfile(dest):
        if on_progress:
            on_progress(1.0, "Model ready.")
        if on_done:
            on_done(True, dest)
        return

    # 2. Bundled with the APK — use it in place, no copy needed
    bundled = _bundled_model_path(filename)
    if bundled and bundled != dest:
        if on_progress:
            on_progress(1.0, "Using bundled model.")
        if on_done:
            on_done(True, bundled)
        return

    # 3. Fall back to downloading from Hugging Face
    download_model(
        repo_id     = DEFAULT_MODEL["repo_id"],
        filename    = DEFAULT_MODEL["filename"],
        on_progress = on_progress,
        on_done     = on_done,
    )
