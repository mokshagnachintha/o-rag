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
#  Catalogue of available Mobile RAG GGUF models                     #
# ------------------------------------------------------------------ #

# The primary Generation model
QWEN_MODEL: dict = {
    "label":    "Qwen 2.5 1.5B Instruct Q4_K_M (~1.1 GB)",
    "repo_id":  "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "size_mb":  1120,
}

# The primary Embedding model
NOMIC_MODEL: dict = {
    "label":    "Nomic Embed Text v1.5 Q4_K_M (~80 MB)",
    "repo_id":  "nomic-ai/nomic-embed-text-v1.5-GGUF",
    "filename": "nomic-embed-text-v1.5.Q4_K_M.gguf",
    "size_mb":  80,
}

MOBILE_MODELS: list[dict] = [QWEN_MODEL, NOMIC_MODEL]


# ------------------------------------------------------------------ #
#  Destination directory (same as llm.py models dir)                  #
# ------------------------------------------------------------------ #

# App root: rag/downloader.py → ../..
_APP_ROOT_DL = Path(__file__).resolve().parent.parent


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


def _extract_model_from_apk(
    asset_name:  str,
    dest_path:   str,
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_done:     Optional[Callable[[bool, str], None]]  = None,
) -> None:
    """
    Internal helper: extract any APK asset entry to an arbitrary dest_path.
    Runs in a background thread.
    """
    def _run():
        try:
            from android import mActivity  # type: ignore
        except ImportError:
            if on_done:
                on_done(False, "Not on Android — skipping asset extraction.")
            return

        try:
            import zipfile as _zf

            apk_path = str(mActivity.getPackageCodePath())
            entry  = f"assets/{asset_name}"
            label  = os.path.basename(asset_name)

            with _zf.ZipFile(apk_path, "r") as zf:
                info  = zf.getinfo(entry)
                total = info.file_size
                print(f"[downloader] Extracting {entry}  ({total//1_048_576} MB) → {dest_path}")

                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                copied = 0
                if on_progress:
                    on_progress(0.0, f"Extracting {label}…")

                with zf.open(info) as zin, open(dest_path, "wb") as f:
                    while True:
                        chunk = zin.read(512 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        copied += len(chunk)
                        if on_progress and total > 0:
                            frac = min(copied / total, 0.99)
                            mb_d = copied // 1_048_576
                            mb_t = total  // 1_048_576
                            on_progress(frac, f"Extracting {label}… {mb_d} / {mb_t} MB")

            if on_progress:
                on_progress(1.0, f"Extracted {label}.")
            if on_done:
                on_done(True, dest_path)

        except Exception as e:
            if on_done:
                on_done(False, str(e))

    threading.Thread(target=_run, daemon=True).start()


def extract_from_apk_asset(
    asset_name:  str = "models/model.gguf",
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_done:     Optional[Callable[[bool, str], None]]  = None,
) -> None:
    """Backward-compatible wrapper: extracts Qwen model from APK assets."""
    _extract_model_from_apk(
        asset_name  = asset_name,
        dest_path   = model_dest_path(QWEN_MODEL["filename"]),
        on_progress = on_progress,
        on_done     = on_done,
    )



def auto_download_default(
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_done:     Optional[Callable[[bool, str], None]]  = None,
) -> None:
    """
    Ensure both the Qwen (Generation) and Nomic (Embedding) models are ready.
    Logic priority:
      1. Already present in models/ dir
      2. Bundled inside the APK (Android) -> Extract Qwen
      3. Bundled in project dev folder
      4. Download from HuggingFace
    """
    qwen_dest  = model_dest_path(QWEN_MODEL["filename"])
    nomic_dest = model_dest_path(NOMIC_MODEL["filename"])

    def _prepare_nomic():
        # Step 2: Ensure Nomic embedding model is present
        if os.path.isfile(nomic_dest) and os.path.getsize(nomic_dest) > 10 * 1024 * 1024:
            if on_progress: on_progress(1.0, "All models ready.")
            if on_done: on_done(True, "All models ready.")
            return

        # On Android, try extracting from APK first
        if os.environ.get("ANDROID_PRIVATE"):
            def _after_nomic_extract(ok, path_or_err):
                if ok:
                    if on_progress: on_progress(1.0, "All models ready.")
                    if on_done: on_done(True, "All models ready.")
                else:
                    # Fallback: download from HuggingFace
                    download_model(
                        repo_id     = NOMIC_MODEL["repo_id"],
                        filename    = NOMIC_MODEL["filename"],
                        on_progress = on_progress,
                        on_done     = on_done,
                    )
            _extract_model_from_apk(
                asset_name   = "models/nomic.gguf",
                dest_path    = nomic_dest,
                on_progress  = on_progress,
                on_done      = _after_nomic_extract,
            )
            return

        # Desktop / no APK: download directly from HF
        download_model(
            repo_id     = NOMIC_MODEL["repo_id"],
            filename    = NOMIC_MODEL["filename"],
            on_progress = on_progress,
            on_done     = on_done
        )

    def _prepare_qwen():
        # Step 1: Ensure Qwen generation model is present
        
        # 1. Already on disk
        if os.path.isfile(qwen_dest) and os.path.getsize(qwen_dest) > 50 * 1024 * 1024:
            _prepare_nomic()
            return

        # 2. Extract from APK asset (Android only)
        if os.environ.get("ANDROID_PRIVATE"):
            def _after_extract(ok, path_or_err):
                if ok:
                    _prepare_nomic()
                else:
                    # Fallback to HF download for Qwen
                    download_model(
                        repo_id     = QWEN_MODEL["repo_id"],
                        filename    = QWEN_MODEL["filename"],
                        on_progress = on_progress,
                        on_done     = lambda ok, msg: _prepare_nomic() if ok else on_done(False, msg),
                    )

            extract_from_apk_asset("models/model.gguf", on_progress, _after_extract)
            return

        # 3. Bundled with the project (desktop dev)
        bundled = _bundled_model_path(QWEN_MODEL["filename"])
        if bundled and bundled != qwen_dest:
            _prepare_nomic()
            return

        # 4. Download from Hugging Face
        download_model(
            repo_id     = QWEN_MODEL["repo_id"],
            filename    = QWEN_MODEL["filename"],
            on_progress = on_progress,
            on_done     = lambda ok, msg: _prepare_nomic() if ok else on_done(False, msg),
        )

    # Start the chain
    _prepare_qwen()
