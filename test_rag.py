import sys, os, glob, time
sys.path.insert(0, r"C:/Users/cmoks/Desktop/app")
os.chdir(r"C:/Users/cmoks/Desktop/app")

PDF   = r"C:/Users/cmoks/Desktop/app/1706.03762v7 (1).pdf"
GGUF  = glob.glob(r"C:/Users/cmoks/Desktop/app/*.gguf")
if not GGUF: sys.exit("No .gguf found")
MODEL = GGUF[0]

QUESTIONS = [
    "What is the main contribution of the Transformer architecture?",
    "How does multi-head attention work?",
    "What are the encoder and decoder stacks made of?",
]

print("Model:", os.path.basename(MODEL))
print("PDF  :", os.path.basename(PDF))

from src.rag.pipeline import (init_db, process_document, retriever,
                               insert_document, insert_chunks, update_doc_chunk_count)
from src.rag.llm import llm, build_rag_prompt

print("\nInitialising DB ...")
init_db()

import src.rag.db as _db
conn = _db.get_conn()
cur = conn.execute("SELECT id FROM documents WHERE name=?", (os.path.basename(PDF),))
if cur.fetchone():
    print("PDF already ingested.")
else:
    print("Ingesting PDF ...")
    doc_id = insert_document(os.path.basename(PDF), PDF)
    chunks  = process_document(PDF)
    insert_chunks(doc_id, chunks)
    update_doc_chunk_count(doc_id, len(chunks))
    print("Ingested", len(chunks), "chunks.")

retriever.reload()
print("Retriever ready. Chunks:", len(retriever._chunks))

print("\nLoading model (starting llama-server, please wait up to 2 min) ...")
t0 = time.time()
llm.load(MODEL, n_ctx=4096, n_threads=4)
elapsed = int(time.time() - t0)
print("Backend:", llm._backend, " elapsed:", elapsed, "s")

for i, q in enumerate(QUESTIONS, 1):
    print("\nQ" + str(i) + ": " + q)
    print("-" * 70)
    results = retriever.query(q, top_k=4)
    if not results:
        print("(no relevant chunks found)")
        continue
    prompt = build_rag_prompt([t for t, _ in results], q)
    print(llm.generate(prompt, max_tokens=350).strip())

print("\n=== Done ===")
