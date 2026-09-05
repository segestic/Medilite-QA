import os
import torch
from datetime import datetime
from dotenv import load_dotenv

import wandb
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)

from src.data_loader import get_prepared_datasets

def main():
    # Load .env file for HF_HUB_TOKEN and WANDB_API_KEY
    load_dotenv()
    
    # 1. Global Parameters
    model_name = "microsoft/Phi-3.5-mini-instruct"
    new_model = "phi3.5-mini-4k-qlora-medical-seg-vall_med"
    hf_model_repo = f"segestic/{new_model}"
    seed = 1234
    set_seed(seed)

    # 2. Identify GPU capabilities
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'
        
    print(f"Using attention implementation: {attn_implementation}")
    print(f"Using compute dtype: {compute_dtype}")

    # 3. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        add_eos_token=True, 
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'left'

    # 4. Initialize Data
    train_dataset, eval_dataset = get_prepared_datasets(tokenizer)

    # 5. Load Model & Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"": 0},
        quantization_config=bnb_config,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 6. LoRA Configuration
    target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
    peft_config = LoraConfig(
        lora_alpha=256,
        lora_dropout=0.10,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

    # 7. Weights & Biases
    if os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        day_ = f"{datetime.now().strftime('%Y-%m-%d')}"
        wandb_project = os.getenv("WANDB_PROJECT", f"medical-finetune-3_5_{day_}")
        os.environ["WANDB_PROJECT"] = wandb_project
        run_name = f"phi3.5-microsoft-{wandb_project}full_med-{datetime.now().strftime('%H-%M')}"
    else:
        run_name = "phi3.5-medical-finetune"

    # 8. Training Arguments (Updated to match thesis text exact constraints)
    args = TrainingArguments(
        output_dir=f"./{new_model}",
        eval_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_steps=100,
        num_train_epochs=3,
        
        # --- ALIGNED WITH THESIS TEXT ---
        weight_decay=0.1,    # Text: "weight decay of 0.1"
        warmup_steps=100,    # Text: "100-step linear warmup"
        lr_scheduler_type="linear",
        # --------------------------------
        
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        seed=seed,
        run_name=run_name,
    )

    # 9. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=args,
    )

    # 10. Train and Push
    print("Starting Training...")
    trainer.train()
    
    if wandb.run is not None:
        wandb.finish()

    print(f"Pushing to Hub: {hf_model_repo}")
    trainer.push_to_hub(hf_model_repo, token=os.getenv("HF_HUB_TOKEN"))

if __name__ == "__main__":
    main()
