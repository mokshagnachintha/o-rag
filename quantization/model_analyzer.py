"""
model_analyzer.py — Parse and display GGUF metadata from a model file.

Reads the GGUF binary header directly (no model load, no GPU, minimal RAM).
Works for any GGUF model: Gemma, LLaMA, Mistral, Phi, etc.

Usage (standalone):
    python -m quantize.model_analyzer path/to/model.gguf

Usage (from code):
    from quantization.model_analyzer import analyze_gguf, print_analysis
    info = analyze_gguf("models/gemma-3-4b-it-Q4_K_M.gguf")
    print_analysis(info)
"""
from __future__ import annotations

import os
import struct
import json
from pathlib import Path
from typing import Any


# ------------------------------------------------------------------ #
#  GGUF format constants                                               #
# ------------------------------------------------------------------ #

GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = {1, 2, 3}

# Value type IDs as defined in the GGUF spec
_VTYPE = {
    0:  ("UINT8",   "B", 1),
    1:  ("INT8",    "b", 1),
    2:  ("UINT16",  "H", 2),
    3:  ("INT16",   "h", 2),
    4:  ("UINT32",  "I", 4),
    5:  ("INT32",   "i", 4),
    6:  ("FLOAT32", "f", 4),
    7:  ("BOOL",    "B", 1),
    8:  ("STRING",  None, None),
    9:  ("ARRAY",   None, None),
    10: ("UINT64",  "Q", 8),
    11: ("INT64",   "q", 8),
    12: ("FLOAT64", "d", 8),
}

# Human-readable quantization names keyed by GGML tensor type ID
_QTYPES = {
    0:  "F32",   1:  "F16",   2:  "Q4_0",  3:  "Q4_1",
    6:  "Q5_0",  7:  "Q5_1",  8:  "Q8_0",  9:  "Q8_1",
    10: "Q2_K",  11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K",  19: "Q8_K",   20: "IQ2_XXS", 21: "IQ2_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S",
    27: "IQ2_S",  28: "IQ4_XS", 29: "IQ1_M",  30: "BF16",
    31: "Q4_0_4_4", 32: "Q4_0_4_8", 33: "Q4_0_8_8",
}

# Approximate bits-per-weight for each quant type (for RAM estimate)
_BITS = {
    "F32": 32, "F16": 16, "BF16": 16,
    "Q8_0": 8, "Q8_1": 8, "Q8_K": 8.5,
    "Q6_K": 6.56, "Q5_K_M": 5.68, "Q5_K_S": 5.51, "Q5_0": 5.5, "Q5_1": 5.5,
    "Q4_K_M": 4.85, "Q4_K_S": 4.58, "Q4_0": 4.5, "Q4_1": 4.5,
    "Q3_K_L": 3.35, "Q3_K_M": 3.35, "Q3_K_S": 3.0,
    "Q2_K": 2.63, "IQ4_XS": 4.25, "IQ4_NL": 4.5,
    "IQ3_S": 3.44, "IQ3_XXS": 3.06,
    "IQ2_XS": 2.31, "IQ2_XXS": 2.06, "IQ2_S": 2.5,
    "IQ1_S": 1.56, "IQ1_M": 1.75,
}


# ------------------------------------------------------------------ #
#  Low-level binary reader                                             #
# ------------------------------------------------------------------ #

class _Reader:
    def __init__(self, data: bytes):
        self._d = data
        self._p = 0

    def read(self, n: int) -> bytes:
        chunk = self._d[self._p: self._p + n]
        self._p += n
        return chunk

    def u8(self)  -> int:   return struct.unpack_from("<B", self._d, self._advance(1))[0]
    def u16(self) -> int:   return struct.unpack_from("<H", self._d, self._advance(2))[0]
    def u32(self) -> int:   return struct.unpack_from("<I", self._d, self._advance(4))[0]
    def u64(self) -> int:   return struct.unpack_from("<Q", self._d, self._advance(8))[0]
    def i8(self)  -> int:   return struct.unpack_from("<b", self._d, self._advance(1))[0]
    def i16(self) -> int:   return struct.unpack_from("<h", self._d, self._advance(2))[0]
    def i32(self) -> int:   return struct.unpack_from("<i", self._d, self._advance(4))[0]
    def i64(self) -> int:   return struct.unpack_from("<q", self._d, self._advance(8))[0]
    def f32(self) -> float: return struct.unpack_from("<f", self._d, self._advance(4))[0]
    def f64(self) -> float: return struct.unpack_from("<d", self._d, self._advance(8))[0]
    def bool(self)-> bool:  return bool(self.u8())

    def _advance(self, n: int) -> int:
        pos = self._p; self._p += n; return pos

    def string(self) -> str:
        length = self.u64()
        return self.read(length).decode("utf-8", errors="replace")

    def value(self, vtype: int) -> Any:
        if vtype == 0:  return self.u8()
        if vtype == 1:  return self.i8()
        if vtype == 2:  return self.u16()
        if vtype == 3:  return self.i16()
        if vtype == 4:  return self.u32()
        if vtype == 5:  return self.i32()
        if vtype == 6:  return self.f32()
        if vtype == 7:  return self.bool()
        if vtype == 8:  return self.string()
        if vtype == 10: return self.u64()
        if vtype == 11: return self.i64()
        if vtype == 12: return self.f64()
        if vtype == 9:  # ARRAY
            elem_type = self.u32()
            count     = self.u64()
            # Only collect first 8 elements to keep memory low
            items = []
            for i in range(count):
                v = self.value(elem_type)
                if i < 8:
                    items.append(v)
            return items
        raise ValueError(f"Unknown value type {vtype}")


# ------------------------------------------------------------------ #
#  GGUF parser                                                         #
# ------------------------------------------------------------------ #

def _read_header(r: _Reader) -> dict:
    """Read magic, version, tensor_count, kv_count."""
    magic = r.read(4)
    if magic != GGUF_MAGIC:
        raise ValueError(f"Not a GGUF file (magic={magic!r})")
    version = r.u32()
    if version not in GGUF_VERSION:
        raise ValueError(f"Unsupported GGUF version {version}")
    tensor_count = r.u64()
    kv_count     = r.u64()
    return {"version": version, "tensor_count": tensor_count, "kv_count": kv_count}


def _read_kv_pairs(r: _Reader, kv_count: int) -> dict:
    """Parse all key-value metadata entries."""
    kv: dict[str, Any] = {}
    for _ in range(kv_count):
        key    = r.string()
        vtype  = r.u32()
        value  = r.value(vtype)
        kv[key] = value
    return kv


def _read_tensor_info(r: _Reader, tensor_count: int) -> list[dict]:
    """Read tensor names, shapes, and quantization types."""
    tensors = []
    for _ in range(tensor_count):
        name       = r.string()
        n_dims     = r.u32()
        dims       = [r.u64() for _ in range(n_dims)]
        dtype_id   = r.u32()
        _offset    = r.u64()   # byte offset (not needed for analysis)
        dtype_name = _QTYPES.get(dtype_id, f"TYPE_{dtype_id}")
        n_params   = 1
        for d in dims:
            n_params *= d
        tensors.append({
            "name":    name,
            "dims":    dims,
            "dtype":   dtype_name,
            "n_params": n_params,
        })
    return tensors


# ------------------------------------------------------------------ #
#  High-level analysis                                                 #
# ------------------------------------------------------------------ #

def analyze_gguf(path: str) -> dict:
    """
    Parse a GGUF file and return a structured analysis dict.

    Only the first ~4 MB of the file is read (header + metadata).
    The entire model weights are NOT loaded into RAM.
    """
    file_size = os.path.getsize(path)

    # Read enough bytes to cover header + all metadata KV pairs.
    # For large models, 8 MB is ample for the metadata section.
    read_bytes = min(file_size, 8 * 1024 * 1024)
    with open(path, "rb") as f:
        raw = f.read(read_bytes)

    r = _Reader(raw)

    header     = _read_header(r)
    kv         = _read_kv_pairs(r, header["kv_count"])
    # Tensor info follows KV pairs; read it if it fits in our buffer
    tensors: list[dict] = []
    try:
        tensors = _read_tensor_info(r, header["tensor_count"])
    except Exception:
        pass  # Buffer may not cover all tensors — that's OK

    # ---- derive key metrics ----
    arch = kv.get("general.architecture", "unknown")

    def _kv(key: str, default=None):
        # Try arch-prefixed key first, then bare key
        return kv.get(f"{arch}.{key}", kv.get(key, default))

    ctx_len   = _kv("context_length")
    emb_dim   = _kv("embedding_length")
    n_layers  = _kv("block_count")
    n_heads   = _kv("attention.head_count")
    n_kv_heads= _kv("attention.head_count_kv")
    ffn_dim   = _kv("feed_forward_length")

    # Total parameter count from tensors
    total_params = sum(t["n_params"] for t in tensors) if tensors else None

    # Quant type from the most common tensor dtype (excluding embeddings)
    quant_type = "unknown"
    if tensors:
        from collections import Counter
        # Focus on weight tensors (not biases / norms)
        wt = [t["dtype"] for t in tensors if "weight" in t["name"]
              and "embed" not in t["name"] and "norm" not in t["name"]]
        if wt:
            quant_type = Counter(wt).most_common(1)[0][0]

    # Estimated RAM usage: params * bits_per_weight / 8 bytes + 20% overhead
    ram_est_mb = None
    if total_params and quant_type in _BITS:
        bpw = _BITS[quant_type]
        ram_est_mb = int(total_params * bpw / 8 / 1_048_576 * 1.20)

    return {
        "path":           path,
        "file_size_mb":   round(file_size / 1_048_576, 1),
        "gguf_version":   header["version"],
        "architecture":   arch,
        "model_name":     kv.get("general.name", Path(path).stem),
        "context_length": ctx_len,
        "embedding_dim":  emb_dim,
        "n_layers":       n_layers,
        "n_heads":        n_heads,
        "n_kv_heads":     n_kv_heads,
        "ffn_dim":        ffn_dim,
        "quant_type":     quant_type,
        "total_params":   total_params,
        "ram_est_mb":     ram_est_mb,
        "tensor_count":   header["tensor_count"],
        "kv_metadata":    kv,
    }


def print_analysis(info: dict) -> None:
    """Pretty-print a model analysis dict to stdout."""
    sep = "=" * 54
    print(sep)
    print(f"  Model : {info['model_name']}")
    print(f"  File  : {Path(info['path']).name}  ({info['file_size_mb']} MB)")
    print(sep)
    print(f"  Architecture    : {info['architecture']}")
    print(f"  GGUF version    : {info['gguf_version']}")
    print(f"  Quantization    : {info['quant_type']}")
    if info["total_params"]:
        p = info["total_params"]
        label = f"{p/1e9:.2f} B" if p >= 1e9 else f"{p/1e6:.0f} M"
        print(f"  Parameters      : {label}")
    if info["ram_est_mb"]:
        print(f"  Est. RAM usage  : ~{info['ram_est_mb']:,} MB")
    print(f"  Tensors         : {info['tensor_count']:,}")
    if info["context_length"]:
        print(f"  Context window  : {info['context_length']:,} tokens")
    if info["embedding_dim"]:
        print(f"  Embedding dim   : {info['embedding_dim']:,}")
    if info["n_layers"]:
        print(f"  Layers          : {info['n_layers']}")
    if info["n_heads"]:
        print(f"  Attn heads      : {info['n_heads']}"
              + (f"  (KV: {info['n_kv_heads']})" if info["n_kv_heads"] else ""))
    if info["ffn_dim"]:
        print(f"  FFN dim         : {info['ffn_dim']:,}")
    print(sep)


def save_analysis_json(info: dict, out_path: str | None = None) -> str:
    """
    Save analysis as a JSON file next to the model.
    Returns the path written to.
    """
    if out_path is None:
        out_path = str(Path(info["path"]).with_suffix(".analysis.json"))
    # Remove the large kv_metadata from the saved file to keep it compact
    save_dict = {k: v for k, v in info.items() if k != "kv_metadata"}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2)
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m quantize.model_analyzer <path/to/model.gguf>")
        sys.exit(1)
    result = analyze_gguf(sys.argv[1])
    print_analysis(result)
    saved = save_analysis_json(result)
    print(f"\n  Analysis saved to: {saved}")
