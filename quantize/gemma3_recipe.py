"""
gemma3_recipe.py — End-to-end recipe for producing the i1-Q5_K_M variant of:
  mradermacher/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking-i1-GGUF

What this recipe does
─────────────────────
  Step 1 — Download source Q8_0 (~1.1 GB)
            Q8_0 is near-lossless (8-bit), the best base for re-quantizing.
            It comes from the same mradermacher repo (Apache 2.0, no login).

  Step 2 — Download imatrix calibration file (1.5 MB)
            mradermacher includes a pre-computed importance matrix with every
            i1-GGUF repo.  This tells the quantizer which weights matter most,
            so it can assign extra bits there and fewer bits elsewhere —
            yielding the "i1" (imatrix) quality advantage.

  Step 3 — Re-quantize Q8_0 → i1-Q5_K_M
            QType Q5_K_M:  5.68 bits/weight, ~851 MB, high quality.
            With the imatrix the quality beats a plain static Q5_K_M.

            Requires the `llama-quantize` (or `llama.cpp quantize`) binary
            on your PATH or pointed to via --quantize-bin.
            It is included with most llama-cpp-python pip wheels as
            `llama_cpp/quantize` on Linux/macOS or
            `llama_cpp\\quantize.exe` on Windows.

Usage
─────
    # Full recipe (download + quantize):
    python -m quantize.run_quantize recipe --target q5_k_m

    # Only download the source files (no quantization):
    python -m quantize.run_quantize recipe --download-only

    # Only quantize (source already downloaded):
    python -m quantize.run_quantize recipe --quantize-only \\
        --quantize-bin path/to/llama-quantize

    # Use a custom llama-quantize binary:
    python -m quantize.run_quantize recipe \\
        --target q5_k_m --quantize-bin C:/llama.cpp/build/bin/llama-quantize.exe
"""
from __future__ import annotations

import os
import sys
import shutil
import threading
import subprocess
from pathlib import Path
from typing import Callable, Optional


# ------------------------------------------------------------------ #
#  Recipe constants                                                    #
# ------------------------------------------------------------------ #

REPO_ID = "mradermacher/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking-i1-GGUF"

# The best source for re-quantization inside this repo (near-lossless)
SOURCE_FILE  = "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking.Q8_0.gguf"
SOURCE_MB    = 1100  # approx

# Importance matrix — calibration data supplied by mradermacher
IMATRIX_FILE = "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking.imatrix"
IMATRIX_MB   = 2     # approx

# Primary output target described by the recipe
TARGET_FILE  = "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking.i1-Q5_K_M.gguf"
TARGET_MB    = 851   # approx

# All re-quantizable targets available from this recipe
RECIPE_TARGETS: dict[str, dict] = {
    "q5_k_m": {
        "quant_type":   "Q5_K_M",
        "output_file":  "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking.i1-Q5_K_M.gguf",
        "size_mb":      851,
        "description":  "High quality, ~851 MB — best accessible quality/size ratio",
    },
    "q5_k_s": {
        "quant_type":   "Q5_K_S",
        "output_file":  "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking.i1-Q5_K_S.gguf",
        "size_mb":      836,
        "description":  "Slightly smaller than Q5_K_M, ~836 MB",
    },
    "q6_k": {
        "quant_type":   "Q6_K",
        "output_file":  "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking.i1-Q6_K.gguf",
        "size_mb":      1000,
        "description":  "Very high quality, ~1.0 GB",
    },
    "q4_k_m": {
        "quant_type":   "Q4_K_M",
        "output_file":  "Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking.i1-Q4_K_M.gguf",
        "size_mb":      806,
        "description":  "Recommended default, ~806 MB",
    },
}


# ------------------------------------------------------------------ #
#  Models directory (shared with main app)                             #
# ------------------------------------------------------------------ #

def _models_dir() -> Path:
    base = os.environ.get("ANDROID_PRIVATE", os.path.expanduser("~"))
    d = Path(base) / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_path(filename: str) -> Path:
    return _models_dir() / filename


# ------------------------------------------------------------------ #
#  Step 1 + 2 — Download source + imatrix                             #
# ------------------------------------------------------------------ #

def _download_file(
    repo_id:     str,
    filename:    str,
    label:       str,
    size_mb:     int,
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_done:     Optional[Callable[[bool, str], None]]  = None,
) -> None:
    """Download a single file from HF Hub with progress polling."""
    dest = _model_path(filename)

    if dest.is_file():
        if on_progress:
            on_progress(1.0, f"Already present: {filename}")
        if on_done:
            on_done(True, str(dest))
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        if on_done:
            on_done(False, "huggingface_hub not installed — run: pip install huggingface-hub")
        return

    if on_progress:
        on_progress(0.0, f"Connecting for {label}...")

    total_bytes = size_mb * 1_048_576
    stop_poll   = threading.Event()

    def _poller():
        inc = str(dest) + ".incomplete"
        while not stop_poll.wait(0.5):
            check = inc if os.path.isfile(inc) else str(dest)
            if os.path.isfile(check):
                done = os.path.getsize(check)
                if total_bytes:
                    frac = min(done / total_bytes, 0.99)
                    if on_progress:
                        on_progress(frac, f"{label}: {done/1_048_576:.0f}/{size_mb} MB")
                else:
                    if on_progress:
                        on_progress(0.0, f"{label}: {done/1_048_576:.0f} MB...")

    poll = threading.Thread(target=_poller, daemon=True)
    poll.start()

    try:
        models_dir = str(_models_dir())
        kwargs: dict = {
            "repo_id":   repo_id,
            "filename":  filename,
            "local_dir": models_dir,
        }
        try:
            import inspect
            if "local_dir_use_symlinks" in inspect.signature(hf_hub_download).parameters:
                kwargs["local_dir_use_symlinks"] = False
        except Exception:
            pass

        cached = hf_hub_download(**kwargs)
        stop_poll.set()
        poll.join(1)

        final = str(dest)
        if os.path.abspath(cached) != os.path.abspath(final):
            shutil.copy2(cached, final)

        if on_progress:
            on_progress(1.0, f"{label}: complete.")
        if on_done:
            on_done(True, final)

    except Exception as e:
        stop_poll.set()
        if on_done:
            on_done(False, f"Download failed ({label}): {e}")


def download_source(
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_done:     Optional[Callable[[bool, str], None]]  = None,
) -> None:
    """Download the Q8_0 source GGUF (Step 1) — background thread."""
    threading.Thread(
        target=_download_file,
        args=(REPO_ID, SOURCE_FILE, "Q8_0 source", SOURCE_MB, on_progress, on_done),
        daemon=True,
    ).start()


def download_imatrix(
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_done:     Optional[Callable[[bool, str], None]]  = None,
) -> None:
    """Download the importance matrix calibration file (Step 2) — background thread."""
    threading.Thread(
        target=_download_file,
        args=(REPO_ID, IMATRIX_FILE, "imatrix", IMATRIX_MB, on_progress, on_done),
        daemon=True,
    ).start()


# ------------------------------------------------------------------ #
#  Step 3 — Re-quantize Q8_0 → target quant using imatrix             #
# ------------------------------------------------------------------ #

def _find_quantize_bin() -> str | None:
    """
    Try to locate the llama-quantize binary.
    Checks:
      1. PATH
      2. Inside the llama_cpp Python package (pip-installed wheel)
      3. Common build output locations
    """
    # 1. On PATH
    found = shutil.which("llama-quantize") or shutil.which("quantize")
    if found:
        return found

    # 2. llama_cpp package wheel (llama-cpp-python pip install includes the binary)
    try:
        import llama_cpp
        pkg_dir = Path(llama_cpp.__file__).parent
        for name in ("llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"):
            candidate = pkg_dir / name
            if candidate.is_file():
                return str(candidate)
    except ImportError:
        pass

    # 3. Common llama.cpp build paths
    common = [
        Path("llama.cpp") / "build" / "bin" / "llama-quantize",
        Path("llama.cpp") / "build" / "bin" / "llama-quantize.exe",
        Path("llama.cpp") / "quantize",
        Path("llama.cpp") / "quantize.exe",
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
    ]
    for c in common:
        if c.is_file():
            return str(c)

    return None


def quantize_to_target(
    target:       str       = "q5_k_m",
    quantize_bin: Optional[str] = None,
    n_threads:    int       = 4,
    on_progress:  Optional[Callable[[str], None]] = None,
    on_done:      Optional[Callable[[bool, str], None]] = None,
) -> None:
    """
    Re-quantize Q8_0 source → target quant type using the imatrix.
    Runs in a background thread.

    Parameters
    ----------
    target       : key from RECIPE_TARGETS (e.g. "q5_k_m")
    quantize_bin : path to llama-quantize binary (auto-detected if None)
    n_threads    : CPU threads for quantization
    on_progress  : callback(status_message: str)
    on_done      : callback(success: bool, message: str)
    """
    if target not in RECIPE_TARGETS:
        msg = f"Unknown target '{target}'. Valid: {list(RECIPE_TARGETS)}"
        if on_done:
            on_done(False, msg)
        return

    meta = RECIPE_TARGETS[target]

    def _run():
        src_path   = _model_path(SOURCE_FILE)
        imat_path  = _model_path(IMATRIX_FILE)
        dst_path   = _model_path(meta["output_file"])
        quant_type = meta["quant_type"]

        # Guard: source must exist
        if not src_path.is_file():
            if on_done:
                on_done(False, f"Source not found: {src_path}\nRun with --download-only first.")
            return

        # Guard: output already exists
        if dst_path.is_file():
            size_mb = dst_path.stat().st_size / 1_048_576
            if on_done:
                on_done(True, f"Output already exists: {dst_path.name}  ({size_mb:.0f} MB)")
            return

        # Locate binary
        bin_path = quantize_bin or _find_quantize_bin()
        if not bin_path:
            # Fallback — try llama-cpp-python Python API (no imatrix support, static quant)
            _quantize_via_python_api(
                src=str(src_path),
                dst=str(dst_path),
                quant_type=quant_type,
                n_threads=n_threads,
                on_progress=on_progress,
                on_done=on_done,
            )
            return

        # Build command
        cmd = [bin_path, str(src_path), str(dst_path), quant_type]
        if imat_path.is_file():
            cmd += ["--imatrix", str(imat_path)]
            if on_progress:
                on_progress(f"Using imatrix: {imat_path.name}")
        else:
            if on_progress:
                on_progress("WARNING: imatrix file not found — producing static quant (no i1 quality).")

        cmd += ["--threads", str(n_threads)]

        if on_progress:
            on_progress(f"Running: {' '.join(str(c) for c in cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                stripped = line.rstrip()
                if stripped and on_progress:
                    on_progress(stripped)

            proc.wait()

            if proc.returncode == 0 and dst_path.is_file():
                size_mb = dst_path.stat().st_size / 1_048_576
                if on_done:
                    on_done(True, f"Done: {dst_path.name}  ({size_mb:.0f} MB)")
            else:
                if on_done:
                    on_done(False, f"llama-quantize exited with code {proc.returncode}")

        except FileNotFoundError:
            if on_done:
                on_done(False, f"Binary not found: {bin_path}")
        except Exception as e:
            if on_done:
                on_done(False, str(e))

    threading.Thread(target=_run, daemon=True).start()


def _quantize_via_python_api(
    src:         str,
    dst:         str,
    quant_type:  str,
    n_threads:   int,
    on_progress: Optional[Callable[[str], None]],
    on_done:     Optional[Callable[[bool, str], None]],
) -> None:
    """
    Fallback: use llama-cpp-python's Python binding to quantize.
    Does NOT support imatrix — produces a static quant.
    """
    if on_progress:
        on_progress("llama-quantize binary not found — using Python API fallback (static quant, no imatrix).")

    FTYPES = {
        "Q4_K_S": 14, "Q4_K_M": 15,
        "Q5_K_S": 16, "Q5_K_M": 17,
        "Q6_K":   18, "Q8_0":    8,
        "Q2_K":   10, "Q3_K_M": 12,
    }

    try:
        from llama_cpp import llama_cpp as _lib
    except ImportError:
        if on_done:
            on_done(False, "Neither llama-quantize binary nor llama-cpp-python is available.")
        return

    ftype = FTYPES.get(quant_type)
    if ftype is None:
        if on_done:
            on_done(False, f"Python API fallback: unsupported quant type '{quant_type}'")
        return

    try:
        os.makedirs(Path(dst).parent, exist_ok=True)
        params = _lib.llama_model_quantize_default_params()
        params.ftype            = ftype
        params.nthread          = n_threads
        params.allow_requantize = True

        ret = _lib.llama_model_quantize(src.encode(), dst.encode(), params)
        if ret == 0:
            size = os.path.getsize(dst) / 1_048_576
            if on_done:
                on_done(True, f"Done (static, no imatrix): {Path(dst).name}  ({size:.0f} MB)")
        else:
            if on_done:
                on_done(False, f"llama_model_quantize returned code {ret}")
    except Exception as e:
        if on_done:
            on_done(False, f"Python API quantization failed: {e}")


# ------------------------------------------------------------------ #
#  Convenience: run full recipe (download + quantize) synchronously   #
# ------------------------------------------------------------------ #

def run_recipe(
    target:       str       = "q5_k_m",
    quantize_bin: Optional[str] = None,
    n_threads:    int       = 4,
    download_only: bool     = False,
    quantize_only: bool     = False,
    log:          Callable[[str], None] = print,
) -> bool:
    """
    Execute the full recipe synchronously (blocks until complete).

    Returns True on success.
    """
    import threading

    # -------- Step 1: download Q8_0 source --------
    if not quantize_only:
        src_path = _model_path(SOURCE_FILE)
        if src_path.is_file():
            log(f"[1/3] Source already present: {SOURCE_FILE}")
        else:
            log(f"[1/3] Downloading source: {SOURCE_FILE}  (~{SOURCE_MB} MB)...")
            ev   = threading.Event()
            res1: dict = {}

            def _dl_prog(frac: float, msg: str):
                bar = "█" * int(frac * 25) + "░" * (25 - int(frac * 25))
                log(f"\r      [{bar}] {frac*100:4.0f}%  {msg:<30}", end="")

            def _dl_done(ok: bool, msg: str):
                res1["ok"] = ok; res1["msg"] = msg; ev.set()

            _download_file(REPO_ID, SOURCE_FILE, "Q8_0", SOURCE_MB, _dl_prog, _dl_done)
            ev.wait()
            log("")   # newline after progress bar

            if not res1.get("ok"):
                log(f"ERROR: {res1.get('msg')}")
                return False
            log(f"      → {res1['msg']}")

        # -------- Step 2: download imatrix --------
        imat_path = _model_path(IMATRIX_FILE)
        if imat_path.is_file():
            log(f"[2/3] Imatrix already present: {IMATRIX_FILE}")
        else:
            log(f"[2/3] Downloading imatrix: {IMATRIX_FILE}  (~{IMATRIX_MB} MB)...")
            ev2   = threading.Event()
            res2: dict = {}

            def _im_done(ok: bool, msg: str):
                res2["ok"] = ok; res2["msg"] = msg; ev2.set()

            _download_file(REPO_ID, IMATRIX_FILE, "imatrix", IMATRIX_MB, None, _im_done)
            ev2.wait()

            if not res2.get("ok"):
                log(f"WARNING: imatrix download failed — {res2.get('msg')}")
                log("         Proceeding without imatrix (static quant).")
            else:
                log(f"      → {res2['msg']}")
    else:
        log("[1-2/3] Skipping downloads (--quantize-only).")

    if download_only:
        log("\n[3/3] Skipping quantization (--download-only).")
        return True

    # -------- Step 3: quantize --------
    meta = RECIPE_TARGETS.get(target, RECIPE_TARGETS["q5_k_m"])
    log(f"\n[3/3] Quantizing → {meta['quant_type']}  (~{meta['size_mb']} MB)")
    log(f"      Output: {meta['output_file']}")

    ev3   = threading.Event()
    res3: dict = {}

    def _q_prog(msg: str):
        log(f"      {msg}")

    def _q_done(ok: bool, msg: str):
        res3["ok"] = ok; res3["msg"] = msg; ev3.set()

    quantize_to_target(
        target=target,
        quantize_bin=quantize_bin,
        n_threads=n_threads,
        on_progress=_q_prog,
        on_done=_q_done,
    )

    # Spinner while waiting
    import time
    spinner = ["|", "/", "-", "\\"]
    i = 0
    while not ev3.wait(0.2):
        log(f"\r      Working {spinner[i % 4]}", end="")
        i += 1
    log("\r" + " " * 30 + "\r", end="")

    if res3.get("ok"):
        log(f"\n  SUCCESS: {res3['msg']}\n")
        return True
    else:
        log(f"\n  FAILED:  {res3.get('msg', 'unknown error')}\n")
        return False


# ------------------------------------------------------------------ #
#  Status check (for CLI / UI)                                         #
# ------------------------------------------------------------------ #

def recipe_status() -> dict:
    """Return current download / output status for all recipe files."""
    source_path = _model_path(SOURCE_FILE)
    imat_path   = _model_path(IMATRIX_FILE)

    status = {
        "source": {
            "file":        SOURCE_FILE,
            "path":        str(source_path),
            "downloaded":  source_path.is_file(),
            "size_mb":     round(source_path.stat().st_size / 1_048_576, 1) if source_path.is_file() else None,
        },
        "imatrix": {
            "file":        IMATRIX_FILE,
            "path":        str(imat_path),
            "downloaded":  imat_path.is_file(),
            "size_mb":     round(imat_path.stat().st_size / 1_048_576, 1) if imat_path.is_file() else None,
        },
        "targets": {},
    }

    for key, meta in RECIPE_TARGETS.items():
        out = _model_path(meta["output_file"])
        status["targets"][key] = {
            "file":       meta["output_file"],
            "quant_type": meta["quant_type"],
            "size_mb_est": meta["size_mb"],
            "produced":   out.is_file(),
            "size_mb":    round(out.stat().st_size / 1_048_576, 1) if out.is_file() else None,
            "description": meta["description"],
        }

    return status
