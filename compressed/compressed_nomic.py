import os
import time
import llama_cpp

def quantize_model(input_path: str, output_path: str, qtype: str = "q4_k_m"):
    """
    Quantizes a dense embedding model (Nomic) to a lower bit precision 
    suitable for rapid mobile document indexing.
    """
    if not os.path.exists(input_path):
        print(f"❌ Cannot find input model: {input_path}")
        return False
        
    print(f"\\n--- Quantizing Nomic Embedding Model ---")
    print(f"Input:    {input_path} ({os.path.getsize(input_path) / (1024*1024):.1f} MB)")
    print(f"Output:   {output_path}")
    print(f"Format:   {qtype.upper()}")
    
    start_time = time.time()
    
    # Utilize direct Python bindings to memory map and compress the GGUF weights
    llama_cpp.llama_model_quantize(
        input_path.encode("utf-8"),
        output_path.encode("utf-8"),
        llama_cpp.llama_model_quantize_default_params() 
    )

    end_time = time.time()
    if os.path.exists(output_path):
        final_size = os.path.getsize(output_path) / (1024*1024)
        print(f"✅ Nomic Quantization Complete in {end_time - start_time:.1f} seconds!")
        print(f"✅ Final Size: {final_size:.1f} MB")
        return True
    return False

if __name__ == "__main__":
    # Base Nomic Embed Text V1.5 file path
    base_nomic = r"c:\\Users\\cmoks\\Desktop\\check\\nomic-embed-text-v1.5.f16.gguf"
    
    # Target Q4 destination
    out_nomic = r"c:\\Users\\cmoks\\Desktop\\check\\quantization\\nomic-embed-text-v1.5-CUSTOM-Q4_K_M.gguf"
    
    quantize_model(input_path=base_nomic, output_path=out_nomic, qtype="Q4_K_M")
