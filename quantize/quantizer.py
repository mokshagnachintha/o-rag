"""
quantizer.py — Re-quantize GGUF models using llama-cpp-python's built-in tool.

What is quantization?
─────────────────────
A model like Gemma 3 4B in float16 weighs ~8 GB.  Quantization reduces the
bit-width of each weight:

  Format    Bits/weight  Size (4B model)  Quality loss
  ──────    ───────────  ───────────────  ────────────
  F16       16           ~8.0 GB          none (reference)
  Q8_0       8           ~4.3 GB          negligible
  Q6_K       6.56        ~3.5 GB          very small
  Q5_K_M     5.68        ~3.0 GB          small
  Q4_K_M     4.85        ~2.5 GB          moderate ← good mobile default
  Q4_K_S     4.58        ~2.4 GB          moderate
  Q3_K_M     3.35        ~1.8 GB          noticeable
  Q2_K       2.63        ~1.4 GB          significant
  IQ4_XS     4.25        ~2.2 GB          moderate (importance-weighted)
  IQ2_XS     2.31        ~1.2 GB          high (still usable)

How it works
────────────
llama.cpp ships a `quantize` binary.  llama-cpp-python exposes a Python
wrapper `llama_cpp.llama_cpp.llama_model_quantize()`.

This module provides two paths:
  1.  High-level API  → `quantize_gguf()` – wraps llama-cpp-python
  2.  CLI subprocess  → `quantize_via_binary()` – calls the llama.cpp
      `quantize` binary directly (needed on Android where llama-cpp-python
      may not expose the quantize API).

Usage
─────
    from quantize.quantizer import quantize_gguf, QUANT_TYPES
    quantize_gguf(
        src="models/gemma-3-4b-it-F16.gguf",
        dst="models/gemma-3-4b-it-Q4_K_M.gguf",
        quant_type="Q4_K_M",
        n_threads=4,
    )
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional


# ------------------------------------------------------------------ #
#  Supported quantization types                                        #
# ------------------------------------------------------------------ #

QUANT_TYPES: dict[str, dict] = {
    "Q2_K":    {"ftype": 10, "bpw": 2.63,  "desc": "Very small, significant quality loss"},
    "Q3_K_S":  {"ftype": 11, "bpw": 3.0,   "desc": "Small, high quality loss"},
    "Q3_K_M":  {"ftype": 12, "bpw": 3.35,  "desc": "Small, noticeable quality loss"},
    "Q3_K_L":  {"ftype": 13, "bpw": 3.35,  "desc": "Small, noticeable quality loss"},
    "Q4_K_S":  {"ftype": 14, "bpw": 4.58,  "desc": "Good size/quality — mobile minimum"},
    "Q4_K_M":  {"ftype": 15, "bpw": 4.85,  "desc": "Best mobile default ← recommended"},
    "Q5_K_S":  {"ftype": 16, "bpw": 5.51,  "desc": "High quality, slightly larger"},
    "Q5_K_M":  {"ftype": 17, "bpw": 5.68,  "desc": "High quality"},
    "Q6_K":    {"ftype": 18, "bpw": 6.56,  "desc": "Near-lossless"},
    "Q8_0":    {"ftype": 8,  "bpw": 8.0,   "desc": "Reference quality — large file"},
    "IQ2_XXS": {"ftype": 20, "bpw": 2.06,  "desc": "Importance-weighted, tiny"},
    "IQ2_XS":  {"ftype": 21, "bpw": 2.31,  "desc": "Importance-weighted"},
    "IQ3_XXS": {"ftype": 23, "bpw": 3.06,  "desc": "Importance-weighted"},
    "IQ4_NL":  {"ftype": 25, "bpw": 4.5,   "desc": "Importance-weighted"},
    "IQ4_XS":  {"ftype": 28, "bpw": 4.25,  "desc": "Importance-weighted"},
}


def estimate_output_size_mb(src_path: str, target_quant: str) -> float | None:
    """
    Estimate output file size by reading parameter count from source GGUF.
    Returns None if analysis fails.
    """
    try:
        from quantize.model_analyzer import analyze_gguf
        info = analyze_gguf(src_path)
        params = info.get("total_params")
        bpw    = QUANT_TYPES.get(target_quant, {}).get("bpw")
        if params and bpw:
            return round(params * bpw / 8 / 1_048_576, 1)
    except Exception:
        pass
    return None


# ------------------------------------------------------------------ #
#  Path 1 — llama-cpp-python API                                       #
# ------------------------------------------------------------------ #

def quantize_gguf(
    src:          str,
    dst:          str,
    quant_type:   str = "Q4_K_M",
    n_threads:    int = 4,
    on_progress:  Optional[Callable[[str], None]] = None,
    on_done:      Optional[Callable[[bool, str], None]] = None,
) -> None:
    """
    Re-quantize a GGUF model using llama-cpp-python.
    Runs in a background thread.

    Parameters
    ----------
    src         : path to source GGUF (any quant or F16/F32)
    dst         : output path for quantized GGUF
    quant_type  : one of the keys in QUANT_TYPES (default Q4_K_M)
    n_threads   : CPU threads to use for quantization
    on_progress : callback(status_message: str)
    on_done     : callback(success: bool, message: str)
    """
    if quant_type not in QUANT_TYPES:
        msg = f"Unknown quant type '{quant_type}'. Valid: {list(QUANT_TYPES)}"
        if on_done:
            on_done(False, msg)
        return

    def _run():
        try:
            from llama_cpp import llama_cpp as _lib
        except ImportError:
            if on_done:
                on_done(False, "llama-cpp-python not installed.")
            return

        if on_progress:
            on_progress(f"Starting quantization: {Path(src).name} → {quant_type}")

        ftype  = QUANT_TYPES[quant_type]["ftype"]
        src_b  = src.encode()
        dst_b  = dst.encode()

        try:
            os.makedirs(Path(dst).parent, exist_ok=True)

            # llama_model_quantize_params struct
            params = _lib.llama_model_quantize_default_params()
            params.ftype            = ftype
            params.nthread          = n_threads
            params.allow_requantize = True

            ret = _lib.llama_model_quantize(src_b, dst_b, params)

            if ret == 0:
                size = os.path.getsize(dst) / 1_048_576
                if on_done:
                    on_done(True, f"Done. Output: {Path(dst).name}  ({size:.0f} MB)")
            else:
                if on_done:
                    on_done(False, f"llama_model_quantize returned error code {ret}")

        except Exception as e:
            if on_done:
                on_done(False, f"Quantization failed: {e}")

    threading.Thread(target=_run, daemon=True).start()


# ------------------------------------------------------------------ #
#  Path 2 — subprocess (llama.cpp binary)                              #
# ------------------------------------------------------------------ #

def quantize_via_binary(
    quantize_bin: str,
    src:          str,
    dst:          str,
    quant_type:   str = "Q4_K_M",
    n_threads:    int = 4,
    on_line:      Optional[Callable[[str], None]] = None,
    on_done:      Optional[Callable[[bool, str], None]] = None,
) -> None:
    """
    Run the llama.cpp `quantize` binary as a subprocess.
    Use this when llama-cpp-python's Python bindings don't expose the
    quantize API (e.g. custom Android builds).

    Parameters
    ----------
    quantize_bin : path to the llama.cpp `quantize` (or `llama-quantize`) binary
    src          : source GGUF path
    dst          : destination GGUF path
    quant_type   : e.g. "Q4_K_M"
    n_threads    : threads (passed as --threads argument)
    on_line      : callback(line: str) for each stdout/stderr line
    on_done      : callback(success: bool, message: str)
    """
    def _run():
        cmd = [quantize_bin, src, dst, quant_type, "--threads", str(n_threads)]
        if on_line:
            on_line(f"Running: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                if on_line:
                    on_line(line.rstrip())
            proc.wait()
            if proc.returncode == 0:
                size = os.path.getsize(dst) / 1_048_576
                if on_done:
                    on_done(True, f"Done. {Path(dst).name}  ({size:.0f} MB)")
            else:
                if on_done:
                    on_done(False, f"`quantize` exited with code {proc.returncode}")
        except FileNotFoundError:
            if on_done:
                on_done(False, f"quantize binary not found: {quantize_bin}")
        except Exception as e:
            if on_done:
                on_done(False, str(e))

    threading.Thread(target=_run, daemon=True).start()
