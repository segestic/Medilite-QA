import os
import subprocess
import argparse
from huggingface_hub import HfApi, login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def run_cmd(cmd):
    """Utility to run shell commands safely."""
    print(f"🚀 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

def merge_lora(base_model_id, lora_id, output_dir):
    """Loads base model, applies LoRA, and saves the merged float16 weights."""
    print(f"🔄 Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="cpu", # Safest for merging to avoid OOM
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    print(f"🔌 Loading and applying LoRA adapter: {lora_id}")
    ft_model = PeftModel.from_pretrained(base_model, lora_id)
    
    print("🔀 Merging weights (this may take a moment)...")
    merged_model = ft_model.merge_and_unload()
    
    print(f"💾 Saving merged model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    
    print("💾 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print("✅ Merge complete.")

def convert_and_quantize(merged_dir, gguf_dir, model_name, quants):
    """Converts merged HF model to F32 GGUF, then generates smaller quantizations."""
    os.makedirs(gguf_dir, exist_ok=True)
    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", "/workspace/llama.cpp")
    
    # 1. Convert to Base F32 GGUF (Matches your 7.64 GB MediLITE-QA.F32.gguf)
    f32_gguf_path = os.path.join(gguf_dir, f"{model_name}.F32.gguf")
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    
    print(f"🔄 Converting HuggingFace to base F32 GGUF: {f32_gguf_path}")
    run_cmd(f"python {convert_script} {merged_dir} --outfile {f32_gguf_path} --outtype f32")
    
    # 2. Generate specified Quantizations
    quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
    for qtype in quants:
        quant_path = os.path.join(gguf_dir, f"{model_name}.{qtype}.gguf")
        print(f"🔧 Quantizing to {qtype} -> {quant_path}")
        run_cmd(f"{quantize_bin} {f32_gguf_path} {quant_path} {qtype}")

    print("✅ All quantizations complete.")

def push_to_hub(gguf_dir, repo_id, token):
    """Pushes the generated GGUF files to a Hugging Face repository."""
    if not token:
        print("⚠️ No HF_HUB_TOKEN provided. Skipping upload.")
        return
        
    print(f"☁️ Uploading GGUF files to Hugging Face Hub: {repo_id}")
    login(token=token)
    api = HfApi()
    
    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
    
    api.upload_folder(
        repo_id=repo_id,
        folder_path=gguf_dir,
        repo_type="model",
        commit_message="Upload MediLITE-QA quantized GGUF models"
    )
    print("✅ Upload complete.")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and Quantize to GGUF")
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--lora_adapter", type=str, required=True, help="Path or HF ID of the trained LoRA")
    parser.add_argument("--merged_dir", type=str, default="./outputs/merged_hf")
    parser.add_argument("--gguf_dir", type=str, default="./outputs/gguf")
    
    # Updated Defaults based on your specific repo
    parser.add_argument("--model_name", type=str, default="MediLITE-QA")
    parser.add_argument("--quants", nargs="+", default=["Q8_0", "Q5_K_M", "Q4_K_M"])
    parser.add_argument("--push_repo", type=str, help="HF repo to push GGUFs to (e.g., segestic/MediLITE-QA-GGUF)")
    
    args = parser.parse_args()
    
    # 1. Merge
    merge_lora(args.base_model, args.lora_adapter, args.merged_dir)
    
    # 2. Convert & Quantize
    convert_and_quantize(args.merged_dir, args.gguf_dir, args.model_name, args.quants)
    
    # 3. Push
    if args.push_repo:
        push_to_hub(args.gguf_dir, args.push_repo, os.getenv("HF_HUB_TOKEN"))

if __name__ == "__main__":
    main()
