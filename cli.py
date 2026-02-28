import sys
from pathlib import Path

# Insert the current directory (check/) so Python can find the 'rag' module
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.db        import init_db, insert_document, insert_chunks, update_doc_chunk_count
from rag.chunker   import process_document
from rag.retriever import HybridRetriever
from rag.llm       import LlamaCppModel, build_rag_prompt

# Use the models natively downloaded to the workspace
model_path = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
embed_path = "nomic-embed-text-v1.5.Q4_K_M.gguf"

# A dummy path for testing. Replace with a real path if you want to test RAG!
doc_path   = "requirements.txt"

init_db()

print("Indexing documents to Database...")
doc_id = insert_document(Path(doc_path).name, doc_path)
chunks = process_document(doc_path)
insert_chunks(doc_id, chunks)
update_doc_chunk_count(doc_id, len(chunks))

llm = LlamaCppModel()

print("Loading Nomic Embedding Model to vectorize chunks...")
llm.load(embed_path)
retriever = HybridRetriever(alpha=0.5)
retriever.reload()

import time
time.sleep(3) # Wait for background embedding thread to process paragraphs

print("Loading Qwen Generation Model for chat...")
llm.unload()
llm.load(model_path)

print("Ready. Type your question (Ctrl+C to quit).\n")

while True:
    question = input("You> ").strip()
    if not question:
        continue
    results = retriever.query(question, top_k=4)
    prompt  = build_rag_prompt([t for t, _ in results], question)
    print("Assistant: ", end="", flush=True)
    llm.generate(prompt, max_tokens=512, temperature=0.7, stream_cb=lambda t: print(t, end="", flush=True))
    print("\n")
