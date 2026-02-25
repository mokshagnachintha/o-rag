"""
llm.py — LLM backend with automatic three-step fallback.

Priority order:
  1. llama-cpp-python (Android / Linux, or Windows with a C++ compiler)
  2. Ollama            (if installed: https://ollama.com)
  3. llama-server      (bundled pre-built Windows CPU binary — zero install)

External interface is identical for all backends:
    load(model_path, ...)  → None
    generate(prompt, ...)  → str
    is_loaded()            → bool
    unload()               → None

Gemma uses the <start_of_turn> / <end_of_turn> instruction format.
"""
from __future__ import annotations

import os
import glob
import json
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

# App root: src/rag/llm.py → ../../..
_APP_ROOT = Path(__file__).resolve().parent.parent.parent

# ------------------------------------------------------------------ #
#  Backend helpers                                                     #
# ------------------------------------------------------------------ #

_llama_mod = None

def _get_llama():
    """Return llama_cpp.Llama class, or raise RuntimeError if not installed."""
    global _llama_mod
    if _llama_mod is None:
        try:
            from llama_cpp import Llama
            _llama_mod = Llama
        except ImportError:
            raise RuntimeError("llama-cpp-python is not installed.")
    return _llama_mod


def _ollama_reachable() -> bool:
    """Return True if the Ollama server is reachable on localhost:11434."""
    try:
        import ollama as _ol
        _ol.list()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ #
#  llama-server subprocess backend                                     #
# ------------------------------------------------------------------ #

_LLAMASERVER_PORT  = 8082
_LLAMASERVER_PROC  = None
_LLAMASERVER_LOCK  = threading.Lock()


def _bin_dir() -> Path:
    return _APP_ROOT / "llamacpp_bin"


def _server_exe():
    for p in [_bin_dir() / "llama-server.exe", _bin_dir() / "llama-server"]:
        if p.exists():
            return p
    return None


def _extract_zip_if_needed() -> bool:
    if _server_exe() is not None:
        return True
    zip_path = _APP_ROOT / "llamacpp_bin.zip"
    if not zip_path.exists():
        return False
    dest = _bin_dir()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[llama-server] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    print("[llama-server] Extraction complete.")
    return _server_exe() is not None


def _wait_for_server(port: int, timeout: int = 120) -> bool:
    import urllib.request
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _start_llama_server(model_path: str, n_ctx: int, n_threads: int) -> bool:
    global _LLAMASERVER_PROC
    exe = _server_exe()
    if exe is None:
        return False
    with _LLAMASERVER_LOCK:
        if _LLAMASERVER_PROC is not None:
            return True
        cmd = [
            str(exe),
            "--model",    model_path,
            "--ctx-size", str(n_ctx),
            "--threads",  str(n_threads),
            "--port",     str(_LLAMASERVER_PORT),
            "--host",     "127.0.0.1",
        ]
        print(f"[llama-server] Starting server (port {_LLAMASERVER_PORT}) ...")
        print(f"  Model: {Path(model_path).name}")
        print("  Loading model into memory, please wait ...")
        cf = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            _LLAMASERVER_PROC = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=cf,
            )
        except Exception as exc:
            print(f"[llama-server] Launch failed: {exc}")
            return False
    ready = _wait_for_server(_LLAMASERVER_PORT, timeout=180)
    if not ready:
        _stop_llama_server()
        print("[llama-server] Timed out waiting for server.")
        return False
    print("[llama-server] Server ready.")
    return True


def _stop_llama_server() -> None:
    global _LLAMASERVER_PROC
    with _LLAMASERVER_LOCK:
        if _LLAMASERVER_PROC is not None:
            try:
                _LLAMASERVER_PROC.terminate()
                _LLAMASERVER_PROC.wait(timeout=5)
            except Exception:
                try:
                    _LLAMASERVER_PROC.kill()
                except Exception:
                    pass
            _LLAMASERVER_PROC = None


def _gen_via_server(
    prompt: str, max_tokens: int, temperature: float,
    top_p: float, stream_cb,
) -> str:
    import urllib.request
    # llama-server native endpoint: /completion  (NOT /v1/completions)
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream_cb is not None,
        "cache_prompt": False,
    }).encode()
    url = f"http://127.0.0.1:{_LLAMASERVER_PORT}/completion"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    if stream_cb is not None:
        full = ""
        with urllib.request.urlopen(req) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    token = json.loads(data).get("content", "")
                    full += token
                    stream_cb(token)
                except Exception:
                    pass
        return full
    else:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
        return body.get("content", "")


# ------------------------------------------------------------------ #
#  Model directory                                                     #
# ------------------------------------------------------------------ #

def _models_dir() -> str:
    base = os.environ.get("ANDROID_PRIVATE", os.path.expanduser("~"))
    d = os.path.join(base, "models")
    os.makedirs(d, exist_ok=True)
    return d


def list_available_models() -> list[str]:
    """Return list of .gguf file paths found in the models directory."""
    pattern = os.path.join(_models_dir(), "*.gguf")
    return sorted(glob.glob(pattern))


# ------------------------------------------------------------------ #
#  LLM singleton                                                       #
# ------------------------------------------------------------------ #

class LlamaCppModel:
    """
    Unified LLM backend — tries each backend in priority order:
      1. llama-cpp-python  (in-process, best performance)
      2. Ollama            (if the server is running on localhost:11434)
      3. llama-server      (auto-extracted from llamacpp_bin.zip)
    """

    DEFAULT_CTX      = 2048
    DEFAULT_MAX_TOK  = 512
    DEFAULT_TEMP     = 0.7
    DEFAULT_TOP_P    = 0.9
    DEFAULT_THREADS  = 4

    def __init__(self) -> None:
        self._model      = None
        self._model_path: Optional[str] = None
        self._lock       = threading.Lock()
        self._backend    = "none"   # "llama_cpp"|"ollama"|"llama_server"|"none"
        self._ollama_name = ""

    # ---------------------------------------------------------------- #
    #  Loading                                                           #
    # ---------------------------------------------------------------- #

    def load(self, model_path: str, n_ctx: int = DEFAULT_CTX,
             n_threads: int = DEFAULT_THREADS, n_gpu_layers: int = 0) -> None:
        with self._lock:
            self._unload_internal()

            # 1. llama-cpp-python
            try:
                Llama = _get_llama()
                self._model = Llama(
                    model_path   = model_path,
                    n_ctx        = n_ctx,
                    n_threads    = n_threads,
                    n_gpu_layers = n_gpu_layers,
                    verbose      = False,
                )
                self._model_path = model_path
                self._backend    = "llama_cpp"
                print("[LLM] Backend: llama-cpp-python")
                return
            except RuntimeError:
                pass

            # 2. Ollama
            if _ollama_reachable():
                try:
                    self._load_via_ollama(model_path)
                    return
                except RuntimeError as e:
                    print(f"[LLM] Ollama failed: {e}")

            # 3. llama-server (bundled binary)
            _extract_zip_if_needed()
            if _start_llama_server(model_path, n_ctx, n_threads):
                self._model_path = model_path
                self._backend    = "llama_server"
                print("[LLM] Backend: llama-server (built-in)")
                return

            raise RuntimeError(
                "No LLM backend available.\n\n"
                "Options:\n"
                "  A) Install Ollama: https://ollama.com/download/windows\n"
                "  B) Place llamacpp_bin.zip in the app folder\n"
                "     (Windows CPU build from https://github.com/ggml-org/llama.cpp/releases)\n"
                "  C) Install llama-cpp-python (requires a C++ compiler)"
            )

    def _load_via_ollama(self, model_path: str) -> None:
        try:
            import ollama as _ol
        except ImportError:
            raise RuntimeError("ollama package not installed.")
        stem  = Path(model_path).stem.lower()
        clean = "".join(c if (c.isalnum() or c == "-") else "-" for c in stem)
        ollama_name = clean[:50].strip("-") or "local-gguf"
        abs_path = str(Path(model_path).resolve())
        print(f"[LLM] Registering '{ollama_name}' with Ollama ...")
        try:
            _ol.create(model=ollama_name, from_=abs_path, stream=False)
        except Exception as exc:
            raise RuntimeError(f"Ollama registration failed: {exc}") from exc
        self._ollama_name = ollama_name
        self._model_path  = model_path
        self._backend     = "ollama"
        print(f"[LLM] Backend: Ollama (model '{ollama_name}')")

    def _unload_internal(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._backend == "llama_server":
            _stop_llama_server()
        self._backend     = "none"
        self._ollama_name = ""

    def unload(self) -> None:
        with self._lock:
            self._unload_internal()

    def is_loaded(self) -> bool:
        return self._backend != "none"

    # ---------------------------------------------------------------- #
    #  Inference                                                         #
    # ---------------------------------------------------------------- #

    def generate(
        self,
        prompt: str,
        max_tokens:  int   = DEFAULT_MAX_TOK,
        temperature: float = DEFAULT_TEMP,
        top_p:       float = DEFAULT_TOP_P,
        stream_cb:   Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Generate a response.  stream_cb (if given) is called with each
        new token fragment as it arrives.  Returns the full response text.
        """
        if self._backend == "none":
            raise RuntimeError("No model loaded. Call load() first.")
        if self._backend == "llama_cpp":
            return self._gen_llama_cpp(prompt, max_tokens, temperature, top_p, stream_cb)
        if self._backend == "ollama":
            return self._gen_ollama(prompt, max_tokens, temperature, top_p, stream_cb)
        return _gen_via_server(prompt, max_tokens, temperature, top_p, stream_cb)

    def _gen_llama_cpp(self, prompt, max_tokens, temp, top_p, stream_cb):
        with self._lock:
            if stream_cb:
                full = ""
                for chunk in self._model(
                    prompt,
                    max_tokens  = max_tokens,
                    temperature = temp,
                    top_p       = top_p,
                    stream      = True,
                ):
                    token = chunk["choices"][0]["text"]
                    full += token
                    stream_cb(token)
                return full
            else:
                out = self._model(
                    prompt,
                    max_tokens  = max_tokens,
                    temperature = temp,
                    top_p       = top_p,
                    stream      = False,
                )
                return out["choices"][0]["text"]

    def _gen_ollama(self, prompt, max_tokens, temp, top_p, stream_cb):
        import ollama as _ol
        options = {
            "temperature": temp,
            "top_p":       top_p,
            "num_predict": max_tokens,
        }
        if stream_cb:
            full = ""
            for chunk in _ol.generate(
                model   = self._ollama_name,
                prompt  = prompt,
                options = options,
                stream  = True,
            ):
                token = chunk.response
                full += token
                stream_cb(token)
            return full
        else:
            resp = _ol.generate(
                model   = self._ollama_name,
                prompt  = prompt,
                options = options,
                stream  = False,
            )
            return resp.response


# ------------------------------------------------------------------ #
#  Prompt builder                                                      #
# ------------------------------------------------------------------ #

def build_rag_prompt(context_chunks: list[str], question: str) -> str:
    """
    Build a RAG prompt using Gemma's <start_of_turn> instruction format.
    Works with all gemma-it / gemma-2-it / gemma-3-it GGUF models.
    """
    ctx_text = "\n\n---\n\n".join(context_chunks)
    system_msg = (
        "You are a helpful assistant. "
        "Answer ONLY based on the provided context. "
        "If the answer is not in the context, say \"I don't know.\""
    )
    return (
        f"<start_of_turn>user\n"
        f"{system_msg}\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {question}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


# Module-level singleton
llm = LlamaCppModel()
