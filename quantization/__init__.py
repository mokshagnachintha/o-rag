"""
quantize/ — Tools for inspecting and re-quantizing Gemma GGUF models.

Modules
-------
model_analyzer   Reads GGUF metadata without loading the model into RAM.
                 Prints architecture, context length, layer count, embedding
                 dimensions, quantization type, and estimated RAM usage.

quantizer        Converts a float16/bfloat16 or existing GGUF model into
                 a different quantization level using llama-cpp-python's
                 built-in quantize utility.

gemma3_recipe    End-to-end recipe for the mradermacher Gemma-3-1B uncensored
                 repo (Apache 2.0, no HF login).  Downloads the Q8_0 source
                 + imatrix calibration file, then produces i1-Q5_K_M
                 (or any other target via --target flag).

                 Quick start:
                   python -m quantize.run_quantize recipe --target q5_k_m

run_quantize     CLI entry point.  Run:
                   python -m quantize.run_quantize --help

                 Subcommands:
                   types        List supported quantization type names
                   analyze      Parse and display GGUF header metadata
                   quantize     Re-quantize any GGUF (generic)
                   download     Download the default app model
                   recipe       Full i1-Q5_K_M pipeline for the Gemma3 repo
"""
