import os
from huggingface_hub import hf_hub_download

TARGET_DIR = os.path.dirname(os.path.abspath(__file__))

print("Downloading Nomic Embed Text v1.5 (Q4)...")
nomic_path = hf_hub_download(
    repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
    filename="nomic-embed-text-v1.5.Q4_K_M.gguf",
    local_dir=TARGET_DIR
)
print(f"Saved to: {nomic_path}")

print("\nDownloading Qwen 2.5 1.5B Instruct (Q4)...")
qwen_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    local_dir=TARGET_DIR
)
print(f"Saved to: {qwen_path}")

print("\nDone! Both models are now in your current directory.")
