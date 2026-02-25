import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.db        import init_db, insert_document, insert_chunks, update_doc_chunk_count
from rag.chunker   import process_document
from rag.retriever import HybridRetriever
from rag.llm       import LlamaCppModel, build_rag_prompt

model_path = sys.argv[1]
doc_path   = sys.argv[2]

init_db()

llm = LlamaCppModel()
llm.load(model_path)

doc_id = insert_document(Path(doc_path).name, doc_path)
chunks = process_document(doc_path)
insert_chunks(doc_id, chunks)
update_doc_chunk_count(doc_id, len(chunks))

retriever = HybridRetriever(alpha=0.5)
retriever.reload()

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
