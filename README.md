# Medilite-QA

This repository contains the pipeline for instruction fine-tuning a Phi-3.5-mini model on medical data (MedMCQA, PubMedQA, and MedQA-USMLE) using 4-bit QLoRA.

## Publication

This repository accompanies the paper:

> Odewole, O.O.; Başaran, S. **MediLITE-QA: Energy-Efficient Medical Question Answering with a Quantized
> Domain-Specific Language Model on a Commodity Laptop CPU.** *Sustainability* **2026**, 18(21), 11245.
> https://doi.org/10.3390/su182111245

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fsu182111245-blue)](https://doi.org/10.3390/su182111245)

Published 4 November 2026 in *Sustainability* (MDPI), Section Sustainable Engineering and Science.
Paper: https://www.mdpi.com/2071-1050/18/21/11245

### Released models

| Artifact | Repository |
|---|---|
| LoRA adapter | [`segestic/phi3.5-mini-4k-qlora-medical-seg-vall_med`](https://huggingface.co/segestic/phi3.5-mini-4k-qlora-medical-seg-vall_med) |
| Merged model (Transformers) | [`segestic/MediliteQA-phi3.5-mini-all-med`](https://huggingface.co/segestic/MediliteQA-phi3.5-mini-all-med) |
| GGUF quantizations (CPU inference) | [`segestic/MediLITE-QA-GGUF`](https://huggingface.co/segestic/MediLITE-QA-GGUF) |

### Citation

```bibtex
@article{odewole2026medilite,
  author  = {Odewole, Olusegun Oladele and Başaran, Seren},
  title   = {MediLITE-QA: Energy-Efficient Medical Question Answering with a Quantized
             Domain-Specific Language Model on a Commodity Laptop CPU},
  journal = {Sustainability},
  volume  = {18},
  number  = {21},
  pages   = {11245},
  year    = {2026},
  doi     = {10.3390/su182111245},
  url     = {https://www.mdpi.com/2071-1050/18/21/11245}
}
```

## 1. Directory Structure

```text
Medilite-QA/
├── Dockerfile
├── README.md
├── requirements.txt
└── src/
    ├── data_loader.py       # Strict train-split loading and ChatML formatting
    ├── __init__.py
    ├── train.py             # QLoRA configuration and SFTTrainer execution
    └── convert.py           # Merges LoRA, converts to GGUF, quantizes, and pushes

```

## 2. Pipeline Overview

* **`src/data_loader.py`**: Ensures strictly the `train` splits are loaded from MedMCQA (General Medicine), PubMedQA (Biomedical Research), and MedQA-USMLE (Clinical Vignettes) to prevent test-set data leakage. It formats all data into standard Phi-3 ChatML (`<|user|>` / `<|assistant|>`).
* **`src/train.py`**: Configures the model in 4-bit precision (NF4) using `bitsandbytes`, applies LoRA adapters (Rank 32, Alpha 256) to target projection modules, and executes training using `SFTTrainer` with a Paged AdamW 8-bit optimizer and a linear warmup schedule.
* **`src/convert.py`**: Merges the trained LoRA adapter into the base model, converts it to a base F32 GGUF, and quantizes it into smaller formats (Q8_0, Q5_K_M, Q4_K_M) for edge inference.

## 3. Set your API Keys (Prerequisites)

The scripts require your Hugging Face token (to download the base model and push the final adapters/GGUFs) and your Weights & Biases API key (for logging). Export these on your host machine before running:

```bash
export HF_HUB_TOKEN="hf_your_token_here"
export WANDB_API_KEY="your_wandb_key_here"

```

*(Alternatively, you can place these in a `.env` file in your root directory and pass `--env-file .env` to your Docker run command).*

---

## 4. Option A: Run via Docker (Recommended)

The `Dockerfile` handles installing PyTorch (CUDA 13.1), Flash Attention 2, the Hugging Face ecosystem, and compiles `llama.cpp` for quantization.

### Step 1: Build the Docker Image

```bash
docker build -t medilite-qa:latest .

```

### Step 2: Run Training

Execute the fine-tuning pipeline inside the container. We mount the current directory so that the trained model weights are saved to your local machine. *Note: The container automatically runs `src.train` by default.*

```bash
docker run --gpus all --ipc=host -it --rm \
  -v $(pwd):/workspace \
  -e HF_HUB_TOKEN=$HF_HUB_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  medilite-qa:latest

```

### Step 3: Merge and Convert to GGUF

Because `llama.cpp` is pre-compiled inside the Docker image, you can run the conversion script immediately. **Note the `--entrypoint python` flag**, which overrides the default training script.

```bash
docker run --ipc=host -it --rm \
  --entrypoint python \
  -v $(pwd):/workspace \
  -e HF_HUB_TOKEN=$HF_HUB_TOKEN \
  medilite-qa:latest \
  src/convert.py \
    --lora_adapter "segestic/phi3.5-mini-4k-qlora-medical-seg-vall_med" \
    --push_repo "segestic/MediLITE-QA-GGUF"

```

---

## 5. Option B: Run via Conda (Without Docker)

If you prefer to run the pipeline locally without Docker, you can set up a Conda environment. Ensure you have NVIDIA drivers installed on your host machine.

### Step 1: Environment & Dependencies

```bash
conda create -n medilite-qa python=3.10 -y
conda activate medilite-qa

pip install -r requirements.txt
pip install transformers==4.57.6 trl==0.19.1 peft==0.19.1 accelerate==1.13.0
pip install flash-attn==2.8.3 --no-build-isolation

```

### Step 2: Run Training

```bash
python -m src.train

```

### Step 3: Compile `llama.cpp` (Required for Conversion)

To run the quantization script locally, you must first clone and build `llama.cpp`:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=OFF
cmake --build build --config Release -j "$(nproc)"
cd ..
export LLAMA_CPP_DIR="$(pwd)/llama.cpp"

```

### Step 4: Merge and Convert to GGUF

```bash
python src/convert.py \
    --base_model "microsoft/Phi-3.5-mini-instruct" \
    --lora_adapter "segestic/phi3.5-mini-4k-qlora-medical-seg-vall_med" \
    --model_name "MediLITE-QA" \
    --quants Q8_0 Q5_K_M Q4_K_M \
    --push_repo "segestic/MediLITE-QA-GGUF"
```
