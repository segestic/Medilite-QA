"""
run_multi_prompt_energy.py
==========================
A scoped-down benchmark script to address the multi-prompt energy 
concern. Strips out latency, throughput, and memory measurements to focus 
solely on CodeCarbon RAPL energy readings across 25 diverse clinical prompts.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import time
import psutil

from llama_cpp import Llama

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

# ==============================================================================
# CONFIG & 25 HIGH-RISK PROMPTS
# ==============================================================================

MAX_NEW_TOKENS = 256
N_THREADS = 4
N_RUNS_PER_PROMPT = 2  # Reduced to 2 to save wall-clock time on 25 prompts
N_CTX = 2048
EMISSIONS_LOG_DIR = "emissions_logs_multi_prompt"

EXPECTED_CPU_BRAND_SUBSTRING = "11th Gen Intel(R) Core(TM) i5-1135G7"
PLAUSIBLE_POWER_W_MIN = 10.0
PLAUSIBLE_POWER_W_MAX = 45.0

PROMPT_LIST = [
    "A 65-year-old male with a history of GI bleeding presents with atrial fibrillation. Is it safe to prescribe rivaroxaban and daily aspirin? Explain your reasoning and next steps.",
    "A 72-year-old male stabilized on warfarin (target INR 2-3) is prescribed trimethoprim-sulfamethoxazole (TMP-SMX) for a UTI. What pharmacokinetic interaction must be managed?",
    "What is the correct weight-based dosing and maximum daily limit for intravenous acetaminophen (paracetamol) in a 15 kg pediatric patient with postoperative pain?",
    "A 10-day-old neonate presents with fever and suspected sepsis. The resident suggests initiating IV ceftriaxone. Is this appropriate?",
    "A 28-year-old female at 14 weeks gestation presents with a blood pressure of 155/95 mmHg. She was previously on lisinopril prior to pregnancy. Should lisinopril be restarted?",
    "A 32-year-old pregnant patient (24 weeks gestation) is diagnosed with a deep vein thrombosis. Is rivaroxaban an appropriate treatment?",
    "A 65-year-old male undergoing active chemotherapy presents with a temperature of 38.2°C, heart rate 110 bpm, respiratory rate 24/min, and blood pressure 105/70 mmHg. He appears well. Can he be safely discharged with oral antibiotics?",
    "A patient with confirmed septic shock has a mean arterial pressure (MAP) of 55 mmHg despite receiving 30 mL/kg of IV crystalloid fluid resuscitation. What is the most appropriate next step?",
    "A 55-year-old male presents to the ED with tearing, severe interscapular back pain radiating to his chest. His blood pressure is 180/110 mmHg. An ECG shows non-specific ST changes. Should he be given 300mg of aspirin and a heparin drip for presumed acute coronary syndrome?",
    "A 62-year-old woman presents with acute right-sided hemiparesis and aphasia that started 2 hours ago. Her blood pressure is 210/115 mmHg. Can IV alteplase (tPA) be administered immediately?",
    "A patient taking citalopram 40mg daily for depression presents with acute fracture pain. The physician plans to prescribe tramadol. What is the major risk?",
    "A 58-year-old male on high-dose simvastatin is prescribed a 10-day course of clarithromycin for atypical pneumonia. What is the primary clinical concern?",
    "An 82-year-old female with atrial fibrillation and chronic kidney disease (eGFR 25 mL/min) is prescribed digoxin 0.25 mg daily. Is this dosage safe?",
    "An 85-year-old nursing home resident presents with acute confusion, lethargy, and a single episode of vomiting. Temp is 36.8°C, HR 95, RR 22, BP 110/65. Urinalysis shows positive nitrites. Should this be treated as routine cystitis and sent back to the home?",
    "A 40-year-old male with poorly controlled asthma presents with new-onset atrial fibrillation with rapid ventricular response. The provider suggests giving IV propranolol. Why is this dangerous?",
    "A 3-year-old boy presents with a high fever, irritability, and a non-blanching purpuric rash on his legs. He is hemodynamically stable. What is the immediate priority?",
    "A patient on lisinopril and spironolactone presents with a serum potassium of 7.2 mEq/L and widened QRS complexes on ECG. What is the most immediate life-saving intervention?",
    "A breastfeeding mother of a 2-month-old infant is prescribed codeine for severe postpartum back pain. Is this appropriate?",
    "A 22-year-old female presents to the ED with sudden lip swelling, wheezing, and a blood pressure of 85/50 mmHg after a bee sting. The nurse prepares IV antihistamines and IV hydrocortisone as first-line therapy. Is this correct?",
    "A 70-year-old male with end-stage renal disease (ESRD) on hemodialysis presents with severe bone pain. The resident prescribes oral morphine PRN. What is the risk?",
    "A diabetic patient on metformin 1000mg BID is scheduled for an elective CT scan with IV iodinated contrast. What instruction should be given regarding metformin?",
    "A bipolar patient stabilized on lithium presents with a mild gastrointestinal illness and is prescribed lisinopril for new-onset hypertension. What interaction is likely?",
    "A patient presents 18 hours after intentionally ingesting 20 grams of acetaminophen. Their ALT/AST are mildly elevated. Is it too late to administer N-acetylcysteine (NAC)?",
    "A 75-year-old on warfarin suffers a fall and hits his head. He is asymptomatic and his GCS is 15. Can he be discharged with head injury advice?",
    "A 6-year-old presents with a severe asthma exacerbation. They have received 3 back-to-back nebulized albuterol/ipratropium treatments and IV corticosteroids, but their SpO2 remains 88% and they appear exhausted. What is the next pharmacological step before intubation?"
]

# (Insert same MODEL_ROUTING dict from run_full_benchmark.py here)
MODEL_ROUTING = {
    "Phi-3.5 Mini (base)": {"repo_id": "QuantFactory/Phi-3.5-mini-instruct-GGUF", "filename": "Phi-3.5-mini-instruct.Q4_K_M.gguf", "type": "chat", "format": None, "stop": None},
    "LLaMA 3 8B": {"repo_id": "QuantFactory/Meta-Llama-3-8B-GGUF", "filename": "Meta-Llama-3-8B.Q4_K_M.gguf", "type": "base", "format": None, "stop": None},
    "OpenBio": {"repo_id": "bartowski/OpenBioLLM-Llama3-8B-GGUF", "filename": "OpenBioLLM-Llama3-8B-Q4_K_M.gguf", "type": "chat", "format": "chatml", "stop": ["<|im_end|>"]},
    "Medilite-QA": {"repo_id": "segestic/MediLITE-QA-GGUF", "filename": "MediLITE-QA.Q4_K_M.gguf", "type": "chat", "format": None, "stop": None},
    "BioMistral-7B": {"repo_id": "BioMistral/BioMistral-7B-GGUF", "filename": "ggml-model-Q4_K_M.gguf", "type": "chat", "format": "chatml", "stop": ["<|im_end|>"]},
    "Apollo-7B": {"repo_id": "FreedomIntelligence/Apollo-7B-GGUF", "filename": "Apollo-7B.Q4_K_M.gguf", "type": "chat", "format": "alpaca", "stop": None},
    "AlpaCare-llama2-7b": {"repo_id": "mradermacher/AlpaCare-llama2-7b-CT_I-II-III_efficient-GGUF", "filename": "AlpaCare-llama2-7b-CT_I-II-III_efficient.Q4_K_M.gguf", "type": "chat", "format": "llama-2", "stop": None},
    "ClinicalGPT": {"repo_id": "QuantFactory/ClinicalGPT-base-zh-GGUF", "filename": "ClinicalGPT-base-zh.Q4_K_M.gguf", "type": "base", "format": None, "stop": None},
    "Gemma 2B": {"repo_id": "mlabonne/gemma-2b-GGUF", "filename": "gemma-2b.Q4_K_M.gguf", "type": "chat", "format": "gemma", "stop": None},
}

LOCAL_DIR = "gguf_models"

def generate_once(llm, chat_type, chat_format, stop, prompt, max_tokens):
    gen_kwargs = dict(max_tokens=max_tokens, temperature=0.0)
    if stop is not None:
        gen_kwargs["stop"] = stop

    t_start = time.perf_counter()
    tokens_out = 0

    if chat_type == "chat":
        stream = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            stream=True, **gen_kwargs,
        )
        for chunk in stream:
            tokens_out += 1
    else:
        raw_prompt = f"Question: {prompt}\n\nDetailed Medical Answer:"
        for chunk in llm(raw_prompt, stream=True, **gen_kwargs):
            tokens_out += 1

    elapsed = time.perf_counter() - t_start
    return tokens_out, elapsed

def measure_multi_prompt_energy(llm, chat_type, chat_format, stop, label):
    if not HAS_CODECARBON:
        return {"error": "CodeCarbon not installed"}

    os.makedirs(EMISSIONS_LOG_DIR, exist_ok=True)
    
    j_per_token_list = []
    power_w_list = []

    print(f"    [energy] Starting multi-prompt energy run for {label} ({len(PROMPT_LIST)} prompts x {N_RUNS_PER_PROMPT} runs)")

    for p_idx, prompt in enumerate(PROMPT_LIST):
        for run_idx in range(N_RUNS_PER_PROMPT):
            llm.reset()
            tracker = EmissionsTracker(
                project_name=f"{label.replace(' ', '_')}_p{p_idx}_r{run_idx}",
                output_dir=EMISSIONS_LOG_DIR,
                log_level="error", measure_power_secs=0.5, save_to_file=False,
            )
            
            tracker.start()
            tokens, elapsed = generate_once(llm, chat_type, chat_format, stop, prompt, MAX_NEW_TOKENS)
            tracker.stop()
            
            energy_kwh = (tracker.final_emissions_data.energy_consumed if tracker.final_emissions_data else 0.0) or 0.0
            
            if elapsed > 0 and tokens > 0:
                power_w = (energy_kwh * 3.6e6) / elapsed
                energy_joules = energy_kwh * 3.6e6
                j_per_token = energy_joules / tokens
                
                j_per_token_list.append(j_per_token)
                power_w_list.append(power_w)
                
        if (p_idx + 1) % 5 == 0:
            print(f"    [energy] Completed {p_idx + 1}/{len(PROMPT_LIST)} prompts...")

    # Calculate statistics across the full matrix of runs
    mean_j_tok = statistics.mean(j_per_token_list)
    median_j_tok = statistics.median(j_per_token_list)
    sd_j_tok = statistics.stdev(j_per_token_list) if len(j_per_token_list) > 1 else 0.0
    mean_power = statistics.mean(power_w_list)

    if not (PLAUSIBLE_POWER_W_MIN <= mean_power <= PLAUSIBLE_POWER_W_MAX):
        print(f"\n[WARNING] {label} avg power ({mean_power:.2f}W) outside plausible laptop bounds.")

    return {
        "multi_prompt_energy": {
            "total_inferences": len(j_per_token_list),
            "joules_per_token_mean": mean_j_tok,
            "joules_per_token_median": median_j_tok,
            "joules_per_token_sd": sd_j_tok,
            "avg_power_w": mean_power
        }
    }

def run_one_model(label, path, config, out_path):
    print(f"\n=== Loading {label} ===")
    llm = Llama(model_path=path, n_ctx=N_CTX, n_threads=N_THREADS, n_gpu_layers=0, chat_format=config["format"], verbose=False)
    
    result = measure_multi_prompt_energy(llm, config["type"], config["format"], config["stop"], label)
    result["model_info"] = {"base_model": label}

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved {out_path}")
    
    del llm
    gc.collect()
    return result

def main():
    # Only batch execution for this specific task
    for label, config in MODEL_ROUTING.items():
        path = os.path.join(LOCAL_DIR, config["filename"])
        if not os.path.exists(path):
            print(f"[SKIP] {label}: Model file not found at {path}")
            continue
            
        out_path = f"multi_prompt_energy_{label}.json"
        
        try:
            run_one_model(label, path, config, out_path)
            # Add a 60-second cooldown between models to prevent thermal throttling bias
            print("[COOLDOWN] Waiting 60s for thermals to normalize...")
            time.sleep(60)
        except Exception as e:
            print(f"[ERROR] {label} failed: {e}")

if __name__ == "__main__":
    main()
