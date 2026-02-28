"""
service/main.py  —  O-RAG Android foreground service.

Runs as a separate process (p4a Android Service) and owns both
llama-server processes.  Keeps them alive between app sessions so
the user never waits for the 45-90 s cold-start again.

Lifecycle:
  1. Service is started by the main app on first launch.
  2. It waits for model files to be extracted (main app does that).
  3. Starts Qwen server (port 8082) + Nomic server (port 8083).
  4. Stays alive forever, restarting servers if they crash.
  5. Android OS keeps it alive because it is a foreground service
     (p4a automatically promotes it and shows a persistent notification).
"""
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional

# ------------------------------------------------------------------ #
#  Android plumbing                                                    #
# ------------------------------------------------------------------ #

def _set_foreground():
    """Promote this service to foreground so Android won't kill it."""
    try:
        from jnius import autoclass  # type: ignore

        PythonService  = autoclass("org.kivy.android.PythonService")
        NotifBuilder   = autoclass("android.app.Notification$Builder")
        NotifManager   = autoclass("android.app.NotificationManager")
        NotifChannel   = autoclass("android.app.NotificationChannel")
        Context        = autoclass("android.content.Context")
        Build          = autoclass("android.os.Build")

        service = PythonService.mService

        CHANNEL_ID = "orag_ai_channel"
        if Build.VERSION.SDK_INT >= 26:
            ch = NotifChannel(
                CHANNEL_ID,
                "O-RAG AI Engine",
                NotifManager.IMPORTANCE_LOW,
            )
            nm = service.getSystemService(Context.NOTIFICATION_SERVICE)
            nm.createNotificationChannel(ch)

        builder = NotifBuilder(service, CHANNEL_ID)
        builder.setContentTitle("O-RAG AI Engine")
        builder.setContentText("AI engine running in background")
        builder.setSmallIcon(service.getApplicationInfo().icon)
        builder.setOngoing(True)

        service.startForeground(1, builder.build())
        print("[service] Foreground notification set.")
    except Exception as exc:
        # Not on Android or jnius unavailable — ignore
        print(f"[service] _set_foreground skipped: {exc}")


# ------------------------------------------------------------------ #
#  Paths                                                               #
# ------------------------------------------------------------------ #

def _models_dir() -> str:
    base = os.environ.get("ANDROID_PRIVATE", os.path.expanduser("~"))
    d = os.path.join(base, "models")
    os.makedirs(d, exist_ok=True)
    return d


def _server_exe() -> Optional[Path]:
    """Locate llama-server binary — same logic as rag/llm.py."""
    # Android: extracted as a native .so to nativeLibraryDir
    try:
        from jnius import autoclass  # type: ignore
        ctx = autoclass("org.kivy.android.PythonActivity").mActivity
        native_dir = ctx.getApplicationInfo().nativeLibraryDir
        so = Path(native_dir) / "libllama_server.so"
        if so.exists():
            return so
    except Exception:
        pass

    # Desktop fallback: project-root/llama-server-arm64
    root = Path(__file__).resolve().parent.parent
    for name in ("llama-server-arm64", "llama-server", "llama-server.exe"):
        p = root / name
        if p.exists():
            return p

    return None


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _optimal_threads() -> int:
    try:
        count = os.cpu_count() or 4
        return max(2, min(8, count // 2))
    except Exception:
        return 4


def _probe(port: int) -> bool:
    """Return True if llama-server is already responding on *port*."""
    import urllib.request
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=1
        ) as r:
            return r.status == 200
    except Exception:
        return False


def _wait(port: int, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _probe(port):
            return True
        time.sleep(1)
    return False


def _launch(model_path: str, port: int,
            n_ctx: int = 2048, extra_flags: Optional[List[str]] = None) -> Optional[subprocess.Popen]:
    exe = _server_exe()
    if exe is None:
        print("[service] llama-server binary not found.")
        return None

    threads = _optimal_threads()
    cmd = [
        str(exe),
        "--model",         model_path,
        "--ctx-size",      str(n_ctx),
        "--threads",       str(threads),
        "--threads-batch", str(threads),
        "--port",          str(port),
        "--host",          "127.0.0.1",
        "--embedding",
        "--flash-attn",    "on",
        "--cache-type-k",  "q8_0",
        "--cache-type-v",  "q8_0",
        "--cont-batching",
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    print(f"[service] Launching llama-server on port {port}: {Path(model_path).name}")
    priv = os.environ.get("ANDROID_PRIVATE", "")
    log_path = os.path.join(priv, f"llama_server_{port}.log") if priv else os.devnull
    try:
        lf = open(log_path, "wb") if log_path != os.devnull else subprocess.DEVNULL
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
        return proc
    except Exception as exc:
        print(f"[service] Popen failed: {exc}")
        return None


# ------------------------------------------------------------------ #
#  Main service loop                                                   #
# ------------------------------------------------------------------ #

QWEN_PORT  = 8082
NOMIC_PORT = 8083
QWEN_FILE  = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
NOMIC_FILE = "nomic-embed-text-v1.5.Q4_K_M.gguf"
MIN_QWEN_BYTES  = 100 * 1024 * 1024   # 100 MB sanity check
MIN_NOMIC_BYTES = 10  * 1024 * 1024   # 10 MB


def _wait_for_models(qwen_path: str, nomic_path: str, timeout: int = 600):
    """Block until Qwen model file is on disk (main app extracts it).
    Nomic is started lazily by the app on first PDF upload.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        qwen_ok = os.path.isfile(qwen_path) and os.path.getsize(qwen_path) > MIN_QWEN_BYTES
        if qwen_ok:
            print("[service] Qwen model file ready.")
            return True
        print("[service] Waiting for Qwen model file…")
        time.sleep(5)
    print("[service] Timed out waiting for Qwen model file.")
    return False


def main():
    print("[service] O-RAG AI service starting.")
    _set_foreground()

    models = _models_dir()
    qwen_path  = os.path.join(models, QWEN_FILE)
    nomic_path = os.path.join(models, NOMIC_FILE)

    if not _wait_for_models(qwen_path, nomic_path):
        print("[service] Models not found — service exiting.")
        return

    qwen_proc  = None

    while True:
        # ── Ensure Qwen server is running ── #
        if qwen_proc is None or qwen_proc.poll() is not None:
            if not _probe(QWEN_PORT):
                print(f"[service] Starting Qwen server (port {QWEN_PORT})…")
                qwen_proc = _launch(qwen_path, QWEN_PORT, n_ctx=512)
                if qwen_proc and _wait(QWEN_PORT, timeout=180):
                    print(f"[service] Qwen server ready on port {QWEN_PORT}.")
                else:
                    print(f"[service] Qwen server failed to start.")
                    qwen_proc = None
            else:
                print("[service] Qwen server already responding — reusing.")

        # ── Heartbeat: check every 10 s ── #
        time.sleep(10)


if __name__ == "__main__":
    main()
