"""
run_quantize.py — CLI for analyzing and re-quantizing Gemma GGUF models.

Examples
--------
# Analyze a downloaded GGUF model:
    python -m quantize.run_quantize analyze ~/models/gemma-3-4b-it-Q4_K_M.gguf

# Re-quantize from Q4_K_M to Q2_K (smaller / faster on-device):
    python -m quantize.run_quantize quantize \\
        --src  ~/models/gemma-3-4b-it-Q4_K_M.gguf \\
        --dst  ~/models/gemma-3-4b-it-Q2_K.gguf   \\
        --type Q2_K

# List supported quantization types:
    python -m quantize.run_quantize types

# Download + analyze the default model (Gemma 3 4B Q4):
    python -m quantize.run_quantize download
"""
from __future__ import annotations

import argparse
import sys
import os
import time


# ------------------------------------------------------------------ #
#  Sub-command: types                                                  #
# ------------------------------------------------------------------ #

def cmd_types(_args) -> int:
    from quantize.quantizer import QUANT_TYPES
    print(f"\n{'Type':<12} {'Bits/weight':<14} Description")
    print("-" * 55)
    for name, meta in QUANT_TYPES.items():
        print(f"  {name:<10} {meta['bpw']:<14.2f} {meta['desc']}")
    print()
    return 0


# ------------------------------------------------------------------ #
#  Sub-command: analyze                                                #
# ------------------------------------------------------------------ #

def cmd_analyze(args) -> int:
    path = args.model
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    from quantize.model_analyzer import analyze_gguf, print_analysis, save_analysis_json
    print(f"\nAnalyzing: {path}")
    info = analyze_gguf(path)
    print_analysis(info)

    if not args.no_save:
        saved = save_analysis_json(info)
        print(f"  Analysis JSON saved → {saved}\n")
    return 0


# ------------------------------------------------------------------ #
#  Sub-command: quantize                                               #
# ------------------------------------------------------------------ #

def cmd_quantize(args) -> int:
    from quantize.quantizer import quantize_gguf, estimate_output_size_mb

    src        = args.src
    dst        = args.dst or _auto_dst(args.src, args.type)
    quant_type = args.type

    if not os.path.isfile(src):
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        return 1

    est = estimate_output_size_mb(src, quant_type)
    if est:
        print(f"\nEstimated output size: ~{est:,.0f} MB")

    print(f"Quantizing  : {os.path.basename(src)}")
    print(f"Target type : {quant_type}")
    print(f"Output      : {dst}")
    print(f"Threads     : {args.threads}\n")

    result: dict = {}
    done_event = __import__("threading").Event()

    def _progress(msg: str):
        print(f"  {msg}")

    def _done(success: bool, message: str):
        result["success"] = success
        result["msg"]     = message
        done_event.set()

    quantize_gguf(
        src=src, dst=dst, quant_type=quant_type,
        n_threads=args.threads,
        on_progress=_progress,
        on_done=_done,
    )

    # Wait with a spinner
    spinner = ["|", "/", "-", "\\"]
    i = 0
    while not done_event.wait(0.2):
        print(f"\r  Working {spinner[i % 4]}", end="", flush=True)
        i += 1
    print("\r" + " " * 20 + "\r", end="")

    if result.get("success"):
        print(f"\n  SUCCESS: {result['msg']}\n")
        return 0
    else:
        print(f"\n  FAILED: {result.get('msg', 'unknown error')}\n", file=sys.stderr)
        return 1


def _auto_dst(src: str, quant_type: str) -> str:
    """Generate output filename from source, replacing quant suffix."""
    import re
    stem = re.sub(r"[_-]?(Q[0-9]|IQ[0-9]|F16|F32|BF16)[^.]*", "", os.path.splitext(src)[0])
    return f"{stem}-{quant_type}.gguf"


# ------------------------------------------------------------------ #
#  Sub-command: recipe                                                 #
# ------------------------------------------------------------------ #

def cmd_recipe(args) -> int:
    """
    Full or partial quantization recipe for the Gemma3 mradermacher repo.

    Steps:
      1. Download Q8_0 source  (~1.1 GB, Apache 2.0, no login)
      2. Download imatrix file (~1.5 MB)
      3. Re-quantize → i1-<TARGET>  (default: Q5_K_M, ~851 MB)

    The resulting file is saved to ~/models/.
    """
    from quantize.gemma3_recipe import (
        run_recipe, recipe_status, RECIPE_TARGETS,
        SOURCE_FILE, IMATRIX_FILE, REPO_ID,
    )

    # --status: just print what's already on disk
    if getattr(args, "status", False):
        st = recipe_status()
        print(f"\nRepo: {REPO_ID}\n")
        print(f"  Source   [{SOURCE_FILE}]")
        print(f"    Downloaded : {st['source']['downloaded']}  "
              f"({st['source']['size_mb']} MB)" if st['source']['downloaded'] else
              f"    Downloaded : False")
        print(f"  Imatrix  [{IMATRIX_FILE}]")
        print(f"    Downloaded : {st['imatrix']['downloaded']}  "
              f"({st['imatrix']['size_mb']} MB)" if st['imatrix']['downloaded'] else
              f"    Downloaded : False")
        print(f"\n  Output targets:")
        for key, tm in st['targets'].items():
            state = f"✓  {tm['size_mb']} MB" if tm['produced'] else f"not yet produced"
            print(f"    [{key:8s}]  {tm['file']}")
            print(f"               {state}  —  {tm['description']}")
        print()
        return 0

    target = (args.target or "q5_k_m").lower()
    if target not in RECIPE_TARGETS:
        print(f"ERROR: unknown target '{target}'. Valid: {list(RECIPE_TARGETS)}",
              file=sys.stderr)
        return 1

    meta = RECIPE_TARGETS[target]
    print(f"\n{'='*60}")
    print(f"  Gemma3 Quantization Recipe")
    print(f"  Target : {meta['quant_type']}  (~{meta['size_mb']} MB)")
    print(f"  Output : {meta['output_file']}")
    print(f"{'='*60}\n")

    ok = run_recipe(
        target        = target,
        quantize_bin  = getattr(args, "quantize_bin", None),
        n_threads     = getattr(args, "threads", 4),
        download_only = getattr(args, "download_only", False),
        quantize_only = getattr(args, "quantize_only", False),
        log           = print,
    )
    return 0 if ok else 1


# ------------------------------------------------------------------ #
#  Sub-command: download                                               #
# ------------------------------------------------------------------ #

def cmd_download(_args) -> int:
    import sys

    # Add app root to path so we can import src.rag.downloader
    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, app_root)

    from src.rag.downloader import DEFAULT_MODEL, download_model, model_dest_path

    dest = model_dest_path(DEFAULT_MODEL["filename"])
    print(f"\nDownloading: {DEFAULT_MODEL['label']}")
    print(f"Destination: {dest}\n")

    done_event = __import__("threading").Event()
    result: dict = {}

    def _prog(frac: float, text: str):
        bar_len = 30
        filled  = int(bar_len * frac)
        bar     = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {frac*100:5.1f}%  {text:<30}", end="", flush=True)

    def _done(ok: bool, msg: str):
        result["ok"]  = ok
        result["msg"] = msg
        done_event.set()

    download_model(
        repo_id     = DEFAULT_MODEL["repo_id"],
        filename    = DEFAULT_MODEL["filename"],
        on_progress = _prog,
        on_done     = _done,
    )

    done_event.wait()
    print()

    if result.get("ok"):
        print(f"\n  Downloaded to: {result['msg']}")
        print(f"\nAnalyzing model...\n")
        from quantize.model_analyzer import analyze_gguf, print_analysis
        info = analyze_gguf(result["msg"])
        print_analysis(info)
        return 0
    else:
        print(f"\n  ERROR: {result['msg']}", file=sys.stderr)
        return 1


# ------------------------------------------------------------------ #
#  Argument parser                                                     #
# ------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m quantize.run_quantize",
        description="Analyze and quantize Gemma GGUF models.",
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    # types
    sp.add_parser("types", help="List supported quantization types.")

    # analyze
    pa = sp.add_parser("analyze", help="Analyze a GGUF model file.")
    pa.add_argument("model", help="Path to .gguf file.")
    pa.add_argument("--no-save", action="store_true",
                    help="Don't save analysis JSON next to the model.")

    # quantize
    pq = sp.add_parser("quantize", help="Re-quantize a GGUF model.")
    pq.add_argument("--src",     required=True,  help="Source GGUF path.")
    pq.add_argument("--dst",     default=None,   help="Output path (auto-generated if omitted).")
    pq.add_argument("--type",    default="Q4_K_M",
                    help="Target quant type (default: Q4_K_M). Run 'types' for list.")
    pq.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4).")

    # download
    sp.add_parser("download", help="Download and analyze the default Gemma 3 1B Q4 model.")

    # recipe
    pr = sp.add_parser(
        "recipe",
        help="Full pipeline: download Q8_0 source + imatrix, then re-quantize to i1-Q5_K_M.",
        description=(
            "Downloads the Q8_0 source and imatrix file from the mradermacher repo\n"
            "(Apache 2.0, no login), then re-quantizes to the requested target.\n"
            "Default target: q5_k_m  (i1-Q5_K_M, ~851 MB)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pr.add_argument(
        "--target",
        default="q5_k_m",
        metavar="TARGET",
        help="Quantization target: q5_k_m (default), q5_k_s, q6_k, q4_k_m",
    )
    pr.add_argument(
        "--quantize-bin",
        default=None,
        metavar="PATH",
        help="Path to llama-quantize binary (auto-detected if omitted).",
    )
    pr.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU threads for quantization (default: 4).",
    )
    pr.add_argument(
        "--download-only",
        action="store_true",
        help="Only download source + imatrix; skip quantization.",
    )
    pr.add_argument(
        "--quantize-only",
        action="store_true",
        help="Skip downloads; only run quantization (source must already exist).",
    )
    pr.add_argument(
        "--status",
        action="store_true",
        help="Show download/output status for all recipe files and exit.",
    )

    return p


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()
    dispatch = {
        "types":    cmd_types,
        "analyze":  cmd_analyze,
        "quantize": cmd_quantize,
        "download": cmd_download,
        "recipe":   cmd_recipe,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
