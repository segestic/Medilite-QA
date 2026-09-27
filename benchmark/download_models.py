"""
download_models.py
==============================
Standalone script to download the necessary GGUF files into the 
local 'gguf_models/' directory.

Usage:
    pip install huggingface-hub

    # Standard download
    python download_models.py

    # With a token for gated repos (Llama 3, Gemma)
    python download_models.py --hf-token "hf_YourTokenHere..."

    # Include extra models (MedAlpaca, Qwen)
    python download_models.py --include-extra
"""

import os
import argparse
from huggingface_hub import hf_hub_download, login
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

# ------------------------------------------------------------------------------
# Hugging Face Repo & Filename definitions
# ------------------------------------------------------------------------------
MODEL_SOURCES = {
    "Phi-3.5 Mini (base)": ("QuantFactory/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct.Q4_K_M.gguf"),
    "LLaMA 3 8B": ("QuantFactory/Meta-Llama-3-8B-GGUF", "Meta-Llama-3-8B.Q4_K_M.gguf"),
    "OpenBio": ("bartowski/OpenBioLLM-Llama3-8B-GGUF", "OpenBioLLM-Llama3-8B-Q4_K_M.gguf"),
    "Medilite-QA": ("segestic/MediLITE-QA-GGUF", "MediLITE-QA.Q4_K_M.gguf"),
    "BioMistral-7B": ("BioMistral/BioMistral-7B-GGUF", "ggml-model-Q4_K_M.gguf"),
    "Apollo-7B": ("FreedomIntelligence/Apollo-7B-GGUF", "Apollo-7B.Q4_K_M.gguf"),
    "AlpaCare-llama2-7b": ("mradermacher/AlpaCare-llama2-7b-CT_I-II-III_efficient-GGUF", "AlpaCare-llama2-7b-CT_I-II-III_efficient.Q4_K_M.gguf"),
    "ClinicalGPT": ("QuantFactory/ClinicalGPT-base-zh-GGUF", "ClinicalGPT-base-zh.Q4_K_M.gguf"),
    "Gemma 2B": ("mlabonne/gemma-2b-GGUF", "gemma-2b.Q4_K_M.gguf"),
}

EXTRA_MODEL_SOURCES = {
    "MedAlpaca-13b": ("TheBloke/medalpaca-13B-GGUF", "medalpaca-13b.Q4_K_M.gguf"),
    "Qwen2.5-3b": ("QuantFactory/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct.Q4_K_M.gguf"),
}

MODEL_DIR = "gguf_models"

def download_model(label, repo_id, filename, dest_dir=MODEL_DIR, token=None):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\n[{label}] Fetching '{filename}'...")
    print(f"  Repo: {repo_id}")
    
    try:
        # local_dir forces HF to place the file directly in the specified folder
        dest_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest_dir,
            token=token,
        )
        print(f"  -> Success: {dest_path}")
        return dest_path
    except GatedRepoError:
        print(f"  [ERROR] Gated repository.")
        print(f"    -> You must pass a valid token using --hf-token for {repo_id}")
        return None
    except RepositoryNotFoundError:
        print(f"  [ERROR] Repository or file not found (or it's private and missing a token).")
        return None
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download GGUF models for benchmarking")
    parser.add_argument("--include-extra", action="store_true",
                        help="Also fetch optional models like MedAlpaca and Qwen.")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Hugging Face token for gated models (like LLaMA 3/Gemma).")
    args = parser.parse_args()

    if args.hf_token:
        print("Logging in to Hugging Face...")
        login(token=args.hf_token)

    # Combine dicts if extras are requested
    source_set = dict(MODEL_SOURCES)
    if args.include_extra:
        source_set.update(EXTRA_MODEL_SOURCES)
        
    print(f"=== Initiating download for {len(source_set)} models into ./{MODEL_DIR}/ ===")
    
    success_count = 0
    for label, (repo_id, filename) in source_set.items():
        res = download_model(label, repo_id, filename, dest_dir=MODEL_DIR, token=args.hf_token)
        if res:
            success_count += 1
            
    print(f"\n=== Download Complete: {success_count}/{len(source_set)} files successfully acquired ===")

if __name__ == "__main__":
    main()
