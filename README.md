# Medilite-QA

This repository contains the pipeline for instruction fine-tuning a Phi-3.5-mini model on medical data (MedMCQA, PubMedQA, and MedDialog) using 4-bit QLoRA.

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
    └── convert.py           # NEW: Merges LoRA, converts to GGUF, quantizes, and pushes
```

## 2. Pipeline Overview

* **`src/data_loader.py`**: Ensures strictly the `train` splits are loaded from MedMCQA (General Medicine), PubMedQA (Biomedical Research), and MedDialog (Clinical Dialogue) to prevent test-set data leakage. It formats all data into standard Phi-3 ChatML (`<|user|>` / `<|assistant|>`).
* **`src/train.py`**: Configures the model in 4-bit precision (NF4) using `bitsandbytes`, applies LoRA adapters (Rank 32, Alpha 256) to target projection modules, and executes training using `SFTTrainer` with a Paged AdamW 8-bit optimizer and a linear warmup schedule.

## 3. Set your API Keys (Prerequisites)

The training script requires your Hugging Face token to download the base model and push the final adapter, and your Weights & Biases API key for logging. You can export these on your host machine before running via Docker or Conda:

```bash
export HF_HUB_TOKEN="hf_your_token_here"
export WANDB_API_KEY="your_wandb_key_here"

```

*(Alternatively, you can place these in a `.env` file in your root directory and pass `--env-file .env` to your Docker run command).*

---

## 4. Run Training (Choose Docker or Conda)

### Option A: Run via Docker (Recommended)

The `Dockerfile` handles installing PyTorch (CUDA 13.1), Flash Attention 2, and the Hugging Face ecosystem (Transformers, PEFT, TRL, etc.).

**1. Build the Docker Image**

```bash
docker build -t medilite-qa:latest .

```

**2. Run Training**
Execute the fine-tuning pipeline inside the container. We mount the current directory so that the trained model weights are saved to your local machine rather than getting trapped inside the container.

```bash
docker run --gpus all --ipc=host -it --rm \
  -v $(pwd):/workspace \
  -e HF_HUB_TOKEN=$HF_HUB_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  medilite-qa:latest

```

*Note: The `Dockerfile` has `ENTRYPOINT ["python", "src/train.py"]`, so running the container automatically kicks off the training script.*

### Option B: Run via Conda (Without Docker)

If you prefer to run the pipeline locally without Docker, you can set up a Conda environment. Ensure you have NVIDIA drivers installed on your host machine.

**1. Create and activate a new Conda environment:**

```bash
conda create -n medilite-qa python=3.10 -y
conda activate medilite-qa

```

**2. Install dependencies:**
Because our `requirements.txt` already specifies the PyTorch `cu128` index, you can simply use pip to install everything in one go:

```bash
pip install -r requirements.txt

# Pin core AI toolchains (grouped together so pip cannot silently downgrade them)
pip install transformers==4.57.6 trl==0.19.1 peft==0.19.1 accelerate==1.13.0

# Install Flash Attention
pip install flash-attn==2.8.3 --no-build-isolation

```

**3. Run the training pipeline:**
*(Assuming you already exported the API keys in Step 3)*

```bash
python -m src.train

```
## 5. Merge and Convert to GGUF

Once your model is fine-tuned, you can merge the LoRA adapter into the base model, convert it to GGUF format, and generate quantized versions (like `Q4_K_M` or `Q8_0`) for fast CPU/edge inference. 

Because `llama.cpp` is pre-compiled inside the Docker image, you can run this script immediately.

```bash
docker run --ipc=host -it --rm \
  -v $(pwd):/workspace \
  -e HF_HUB_TOKEN=$HF_HUB_TOKEN \
  medilite-qa:latest \
  python src/convert.py \
    --lora_adapter "segestic/phi3.5-mini-4k-qlora-medical-seg-vall_med" \
    --push_repo "segestic/MediLITE-QA-GGUF"
```
