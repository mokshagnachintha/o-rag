"""
cli.py — Interactive offline RAG CLI for testing on desktop (Windows/Linux/macOS).

No Kivy, no Android — pure terminal.  Uses the same pipeline, retriever,
chunker, and LLM code as the Android app.

Usage
─────
    # Default: auto-detect model + empty knowledge base
    python cli.py

    # Specify a model explicitly
    python cli.py --model path/to/model.gguf

    # Pre-load one or more documents at startup
    python cli.py --add docs/manual.pdf --add docs/notes.txt

    # Larger context / more threads for faster desktop CPUs
    python cli.py --ctx 4096 --threads 6 --max-tokens 1024

    # Use without any documents (pure LLM chat, no RAG)
    python cli.py --no-rag

In-session commands
───────────────────
    :add  <path>   Ingest a .txt or .pdf file into the knowledge base
    :docs          List all ingested documents
    :del  <id>     Remove a document by its ID (shown in :docs)
    :clear         Wipe the entire knowledge base (keeps model loaded)
    :model <path>  Load a different GGUF model mid-session
    :status        Show model + retriever status
    :help          Print this help
    :quit  / :q    Exit

Any other input is treated as a question to answer from your documents.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

# ------------------------------------------------------------------ #
#  Make sure src/ is on the path                                       #
# ------------------------------------------------------------------ #
APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT))


# ------------------------------------------------------------------ #
#  ANSI colour helpers (degrade gracefully on plain terminals)         #
# ------------------------------------------------------------------ #
try:
    import colorama
    colorama.init(autoreset=True)
    _C = True
except ImportError:
    _C = False

def _col(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _C or os.name != "nt" else text

def green(t):  return _col("32", t)
def yellow(t): return _col("33", t)
def cyan(t):   return _col("36", t)
def red(t):    return _col("31", t)
def bold(t):   return _col("1",  t)
def dim(t):    return _col("2",  t)


# ------------------------------------------------------------------ #
#  Model auto-detection                                                #
# ------------------------------------------------------------------ #

def _find_model(hint: str | None) -> str | None:
    """
    Locate a .gguf model.  Priority:
      1. Explicit --model argument
      2. Any .gguf in the app directory (the file you just downloaded)
      3. ~/models/*.gguf
    """
    if hint:
        if os.path.isfile(hint):
            return hint
        print(red(f"Model file not found: {hint}"))
        return None

    # App folder first
    app_ggufs = sorted(APP_ROOT.glob("*.gguf"))
    if app_ggufs:
        return str(app_ggufs[0])

    # ~/models/
    models_dir = Path(os.path.expanduser("~")) / "models"
    if models_dir.is_dir():
        found = sorted(models_dir.glob("*.gguf"))
        if found:
            return str(found[0])

    return None


# ------------------------------------------------------------------ #
#  Spinner helper                                                      #
# ------------------------------------------------------------------ #

class Spinner:
    def __init__(self, msg: str = ""):
        self._msg    = msg
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        # Erase spinner line
        print("\r" + " " * (len(self._msg) + 6) + "\r", end="", flush=True)

    def _spin(self):
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while not self._stop.wait(0.1):
            print(f"\r  {chars[i % len(chars)]}  {self._msg}", end="", flush=True)
            i += 1


# ------------------------------------------------------------------ #
#  Core CLI class                                                      #
# ------------------------------------------------------------------ #

class RAGCli:
    def __init__(self, args: argparse.Namespace):
        self.args    = args
        self.no_rag  = args.no_rag

        # Import RAG components (lazy, after path is set)
        from rag.db       import init_db, list_documents, delete_document
        from rag.chunker  import process_document
        from rag.db       import insert_document, insert_chunks, update_doc_chunk_count
        from rag.retriever import HybridRetriever
        from rag.llm      import LlamaCppModel, build_rag_prompt

        self._init_db            = init_db
        self._list_docs          = list_documents
        self._delete_doc         = delete_document
        self._process_doc        = process_document
        self._insert_document    = insert_document
        self._insert_chunks      = insert_chunks
        self._update_chunk_count = update_doc_chunk_count
        self._build_prompt       = build_rag_prompt

        self._retriever = HybridRetriever(alpha=0.5)
        self._llm       = LlamaCppModel()

    # ---------------------------------------------------------------- #
    #  Setup                                                             #
    # ---------------------------------------------------------------- #

    def setup(self) -> bool:
        print(bold("\n  Offline RAG CLI"))
        print(dim("  Type :help for commands, :quit to exit\n"))

        # Init database
        with Spinner("Initialising database..."):
            self._init_db()
            self._retriever.reload()

        # Find model
        model_path = _find_model(self.args.model)
        if not model_path:
            print(red(
                "No .gguf model found.\n"
                "Put the model file in the app folder or ~/models/,\n"
                "or run with:  python cli.py --model path/to/model.gguf"
            ))
            return False

        # Load model
        if not self._load_model(model_path):
            return False

        # Pre-load documents
        for path in (self.args.add or []):
            self._ingest(path)

        return True

    def _load_model(self, model_path: str) -> bool:
        print(f"  Model : {cyan(Path(model_path).name)}")
        print(f"  Ctx   : {self.args.ctx}  |  Threads: {self.args.threads}  |  Max-tokens: {self.args.max_tokens}\n")

        ev  = threading.Event()
        ok  = [True]
        err = [""]

        def _run():
            try:
                self._llm.load(
                    model_path,
                    n_ctx     = self.args.ctx,
                    n_threads = self.args.threads,
                )
            except Exception as e:
                ok[0]  = False
                err[0] = str(e)
            ev.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        with Spinner(f"Loading {Path(model_path).name}..."):
            ev.wait()

        if ok[0]:
            print(green("  ✓ Model loaded.\n"))
        else:
            print(red(f"  ✗ Failed to load model: {err[0]}\n"))
        return ok[0]

    # ---------------------------------------------------------------- #
    #  Document ingestion                                                #
    # ---------------------------------------------------------------- #

    def _ingest(self, file_path: str) -> None:
        p = Path(file_path)
        if not p.is_file():
            print(red(f"  File not found: {file_path}"))
            return

        ev     = threading.Event()
        result = {}

        def _run():
            try:
                doc_id = self._insert_document(p.name, str(p))
                chunks = self._process_doc(str(p))
                self._insert_chunks(doc_id, chunks)
                self._update_chunk_count(doc_id, len(chunks))
                self._retriever.reload()
                result["ok"]  = True
                result["msg"] = f"Ingested '{p.name}' — {len(chunks)} chunks (doc_id={doc_id})"
            except Exception as e:
                result["ok"]  = False
                result["msg"] = str(e)
            ev.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        with Spinner(f"Ingesting {p.name}..."):
            ev.wait()

        if result.get("ok"):
            print(green(f"  ✓ {result['msg']}"))
        else:
            print(red(f"  ✗ Ingest failed: {result['msg']}"))

    # ---------------------------------------------------------------- #
    #  RAG query                                                         #
    # ---------------------------------------------------------------- #

    def _ask(self, question: str) -> None:
        # Retrieve context
        if not self.no_rag:
            if self._retriever.is_empty():
                print(yellow(
                    "\n  [No documents in knowledge base]\n"
                    "  Add files with:  :add path/to/file.pdf\n"
                    "  Or use --no-rag to chat without documents.\n"
                ))
                return

            results = self._retriever.query(question, top_k=self.args.top_k)
            if not results:
                print(yellow("  [No relevant context found for that question]\n"))
                return

            context_chunks = [text for text, _ in results]
            prompt = self._build_prompt(context_chunks, question)

            # Show which chunks were used (verbose mode)
            if self.args.verbose:
                print(dim(f"\n  Retrieved {len(results)} chunks:"))
                for i, (text, score) in enumerate(results, 1):
                    snippet = text[:120].replace("\n", " ")
                    print(dim(f"    [{i}] score={score:.3f}  {snippet}…"))
                print()
        else:
            # Pure LLM mode (no context)
            from rag.llm import build_rag_prompt
            prompt = (
                f"<start_of_turn>user\n{question}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

        # Generate with streaming
        print(f"\n{bold('Assistant:')} ", end="", flush=True)

        ev     = threading.Event()
        result = {}

        def _stream(token: str):
            print(token, end="", flush=True)

        def _run():
            try:
                answer = self._llm.generate(
                    prompt,
                    max_tokens  = self.args.max_tokens,
                    temperature = self.args.temperature,
                    stream_cb   = _stream,
                )
                result["ok"]     = True
                result["answer"] = answer.strip()
            except Exception as e:
                result["ok"]  = False
                result["msg"] = str(e)
            ev.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        ev.wait()

        print("\n")  # newline after streamed tokens

        if not result.get("ok"):
            print(red(f"  ✗ Generation error: {result.get('msg')}\n"))

    # ---------------------------------------------------------------- #
    #  Built-in commands                                                 #
    # ---------------------------------------------------------------- #

    def _cmd_docs(self) -> None:
        docs = self._list_docs()
        if not docs:
            print(yellow("  No documents ingested yet."))
            return
        print(f"\n  {'ID':<5} {'Name':<35} {'Chunks':<8} Path")
        print("  " + "-" * 70)
        for row in docs:
            print(f"  {row['id']:<5} {row['name']:<35} {str(row['num_chunks']):<8} {dim(row['path'])}")
        print()

    def _cmd_del(self, arg: str) -> None:
        try:
            doc_id = int(arg.strip())
        except ValueError:
            print(red("  Usage: :del <id>  (id from :docs)"))
            return
        self._delete_doc(doc_id)
        self._retriever.reload()
        print(green(f"  ✓ Document {doc_id} removed and index reloaded."))

    def _cmd_clear(self) -> None:
        from rag.db import get_conn
        with get_conn() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
        self._retriever.reload()
        print(green("  ✓ Knowledge base cleared."))

    def _cmd_model(self, arg: str) -> None:
        path = arg.strip()
        if not path:
            print(yellow(f"  Current model: {self._llm._model_path or 'none'}"))
            return
        self._llm.unload()
        self._load_model(path)

    def _cmd_status(self) -> None:
        print(f"\n  Model loaded : {green('yes') if self._llm.is_loaded() else red('no')}")
        if self._llm._model_path:
            p = Path(self._llm._model_path)
            sz = p.stat().st_size / 1_048_576 if p.is_file() else 0
            print(f"  Model file   : {cyan(p.name)}  ({sz:.0f} MB)")
        docs      = self._list_docs()
        chunk_cnt = sum(r["num_chunks"] for r in docs)
        print(f"  Documents    : {len(docs)}")
        print(f"  Chunks       : {chunk_cnt}")
        print(f"  Retriever    : {'empty' if self._retriever.is_empty() else green('ready')}")
        print(f"  No-RAG mode  : {self.no_rag}")
        print(f"  Context size : {self.args.ctx}")
        print(f"  Max tokens   : {self.args.max_tokens}")
        print(f"  Temperature  : {self.args.temperature}\n")

    def _cmd_help(self) -> None:
        print(f"""
  {bold('Commands')}
  {cyan(':add')}  <path>   Ingest a .txt or .pdf file
  {cyan(':docs')}           List all ingested documents with IDs
  {cyan(':del')}  <id>     Remove document by ID (from :docs)
  {cyan(':clear')}          Wipe entire knowledge base
  {cyan(':model')} [path]   Show current model or load a new .gguf
  {cyan(':status')}         Show model + retriever status
  {cyan(':help')}           This help text
  {cyan(':quit')}  / :q     Exit

  {bold('Any other input')} → RAG query (searches your documents, then answers)
""")

    # ---------------------------------------------------------------- #
    #  REPL                                                              #
    # ---------------------------------------------------------------- #

    def run(self) -> None:
        while True:
            try:
                raw = input(f"{bold(green('You'))}> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not raw:
                continue

            # ---- built-in commands ----
            if raw.startswith(":"):
                parts = raw.split(None, 1)
                cmd   = parts[0].lower()
                arg   = parts[1] if len(parts) > 1 else ""

                if cmd in (":quit", ":q", ":exit"):
                    print(dim("  Goodbye."))
                    break
                elif cmd == ":help":
                    self._cmd_help()
                elif cmd == ":docs":
                    self._cmd_docs()
                elif cmd == ":status":
                    self._cmd_status()
                elif cmd == ":clear":
                    self._cmd_clear()
                elif cmd == ":add":
                    if not arg:
                        print(yellow("  Usage: :add <path/to/file.txt or file.pdf>"))
                    else:
                        self._ingest(arg.strip())
                elif cmd == ":del":
                    self._cmd_del(arg)
                elif cmd == ":model":
                    self._cmd_model(arg)
                else:
                    print(yellow(f"  Unknown command '{cmd}'.  Type :help for list."))

            else:
                # ---- RAG / chat query ----
                self._ask(raw)


# ------------------------------------------------------------------ #
#  Argument parser                                                     #
# ------------------------------------------------------------------ #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python cli.py",
        description="Interactive offline RAG CLI — test your documents with Gemma locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",      default=None,
                   help="Path to .gguf model (auto-detected if omitted).")
    p.add_argument("--add",        action="append", metavar="FILE", default=[],
                   help="Pre-load file(s) at startup. Repeatable.")
    p.add_argument("--ctx",        type=int, default=4096,
                   help="LLM context window in tokens (default: 4096).")
    p.add_argument("--threads",    type=int, default=4,
                   help="CPU threads for inference (default: 4).")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Max tokens to generate per answer (default: 512).")
    p.add_argument("--temperature",type=float, default=0.7,
                   help="Sampling temperature 0-1 (default: 0.7).")
    p.add_argument("--top-k",      type=int, default=4,
                   help="Number of document chunks to retrieve (default: 4).")
    p.add_argument("--no-rag",     action="store_true",
                   help="Skip retrieval — plain LLM chat (no documents needed).")
    p.add_argument("--verbose",    action="store_true",
                   help="Show retrieved chunks before each answer.")
    return p


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    args = _build_parser().parse_args()
    cli  = RAGCli(args)

    if not cli.setup():
        sys.exit(1)

    cli.run()
