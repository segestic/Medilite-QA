"""
run_multi_prompt_energy.py  (revision for MDPI Sustainability, R1.2 / R3.4)
=============================================================================
CodeCarbon/RAPL energy across 25 high-risk clinical prompts, in two modes:

  natural : model may stop at EOS (as in the submitted manuscript)
  fixed   : EOS / end-of-turn tokens banned via logit_bias, so every
            generation runs to exactly MAX_NEW_TOKENS (Reviewer 1, point 2)

Changes vs. the submitted version:
  * host + RAPL-readability guards (imported from run_full_benchmark.py)
  * exact token counts from llama.cpp `usage` (no stream-chunk counting)
  * one discarded warm-up generation per model (removes cold-start energy)
  * idle-power baseline per model; raw and idle-subtracted energy reported
  * CPU-package and RAM energy recorded separately (CodeCarbon's
    energy_consumed = cpu + ram + gpu; RAM is a model-based estimate)
  * every individual run saved (raw records) for later re-analysis
  * prompt-clustered bootstrap 95% CI, pooled J/token, and %-at-full-length
  * effective sampling parameters + library versions logged (Reviewer 1, point 4)

Usage:
    sudo $(which python3) run_multi_prompt_energy.py --mode both
    sudo $(which python3) run_multi_prompt_energy.py --model Medilite-QA --mode fixed
"""

import argparse
import gc
import inspect
import json
import os
import random
import re
import statistics
import time

import llama_cpp
from llama_cpp import Llama
from codecarbon import EmissionsTracker
import codecarbon

from run_full_benchmark import (
    MODEL_ROUTING, LOCAL_DIR, N_THREADS, N_CTX,
    assert_correct_host, assert_rapl_readable, check_power_plausibility,
    cpu_brand,
)

# ==============================================================================
# CONFIG
# ==============================================================================

MAX_NEW_TOKENS = 256
N_RUNS_PER_PROMPT = 2
IDLE_BASELINE_S = 30
COOLDOWN_S = 60
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20261007
OUT_DIR = "energy_multi_prompt_revision"

# Candidate end-of-turn strings; only those that map to ONE token in a given
# model's vocabulary are banned (multi-token matches are left alone).
END_TOKEN_STRINGS = ["<|end|>", "<|eot_id|>", "<|im_end|>", "<end_of_turn>",
                     "</s>", "<|endoftext|>", "<|end_of_text|>"]

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
    "A 6-year-old presents with a severe asthma exacerbation. They have received 3 back-to-back nebulized albuterol/ipratropium treatments and IV corticosteroids, but their SpO2 remains 88% and they appear exhausted. What is the next pharmacological step before intubation?",
]

KWH_TO_J = 3.6e6


# ==============================================================================
# HELPERS
# ==============================================================================

def safe_name(label):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")


def sampling_defaults():
    """Effective llama-cpp-python defaults for the parameters we do not set,
    so the paper can report them exactly (Reviewer 1, point 4)."""
    sig = inspect.signature(Llama.create_completion).parameters
    keep = ["top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty",
            "frequency_penalty", "typical_p", "seed"]
    return {k: sig[k].default for k in keep if k in sig}


def end_token_ids(llm):
    ids = {llm.token_eos()}
    for s in END_TOKEN_STRINGS:
        try:
            toks = llm.tokenize(s.encode("utf-8"), add_bos=False, special=True)
        except TypeError:  # very old llama-cpp-python without `special`
            continue
        if len(toks) == 1:
            ids.add(toks[0])
    return sorted(i for i in ids if i is not None and i >= 0)


def make_tracker(project):
    return EmissionsTracker(project_name=project, output_dir=OUT_DIR,
                            log_level="error", measure_power_secs=0.5,
                            save_to_file=False)


def read_tracker(tracker):
    d = tracker.final_emissions_data
    if d is None:
        return None
    return {
        "duration_s": float(d.duration),
        "energy_total_kwh": float(d.energy_consumed or 0.0),
        "energy_cpu_kwh": float(d.cpu_energy or 0.0),
        "energy_ram_kwh": float(d.ram_energy or 0.0),
        "energy_gpu_kwh": float(d.gpu_energy or 0.0),
        "cpu_power_w_reported": float(d.cpu_power or 0.0),
        "ram_power_w_reported": float(d.ram_power or 0.0),
    }


def measure_idle(label):
    gc.collect()
    tracker = make_tracker(f"{safe_name(label)}_idle")
    tracker.start()
    time.sleep(IDLE_BASELINE_S)
    tracker.stop()
    r = read_tracker(tracker)
    r["idle_cpu_power_w"] = r["energy_cpu_kwh"] * KWH_TO_J / r["duration_s"]
    r["idle_total_power_w"] = r["energy_total_kwh"] * KWH_TO_J / r["duration_s"]
    return r


def generate(llm, config, prompt, mode, banned_ids):
    """Non-streaming generation -> exact completion token count from `usage`."""
    kw = dict(max_tokens=MAX_NEW_TOKENS, temperature=0.0)
    if mode == "fixed":
        kw["logit_bias"] = {tid: -1e9 for tid in banned_ids}
        # no user stop strings in fixed mode; any template stop that still
        # fires is detected below via completion_tokens < MAX_NEW_TOKENS
    elif config["stop"] is not None:
        kw["stop"] = config["stop"]

    llm.reset()
    t0 = time.perf_counter()
    if config["type"] == "chat":
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}], **kw)
    else:
        out = llm(f"Question: {prompt}\n\nDetailed Medical Answer:", **kw)
    elapsed = time.perf_counter() - t0
    usage = out.get("usage", {})
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": out["choices"][0].get("finish_reason"),
        "wall_s": elapsed,
    }


def bootstrap_ci_by_prompt(records, key):
    """95% CI of the mean of per-prompt means, resampling prompts
    (runs of the same prompt are not independent)."""
    by_prompt = {}
    for r in records:
        if r.get(key) is not None:
            by_prompt.setdefault(r["prompt_idx"], []).append(r[key])
    means = [statistics.mean(v) for v in by_prompt.values()]
    if len(means) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    boots = sorted(statistics.mean(rng.choices(means, k=len(means)))
                   for _ in range(BOOTSTRAP_REPS))
    return [boots[int(0.025 * BOOTSTRAP_REPS)], boots[int(0.975 * BOOTSTRAP_REPS) - 1]]


def summarize(records):
    ok = [r for r in records if r["completion_tokens"]]
    s = {"n_inferences": len(ok)}
    for key in ["j_per_token_cpu", "j_per_token_cpu_idle_sub", "j_per_token_total"]:
        vals = [r[key] for r in ok]
        s[key] = {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "ci95_prompt_bootstrap": bootstrap_ci_by_prompt(ok, key),
        }
    tot_tok = sum(r["completion_tokens"] for r in ok)
    s["pooled_j_per_token_cpu"] = sum(r["energy_cpu_kwh"] for r in ok) * KWH_TO_J / tot_tok
    s["pooled_j_per_token_cpu_idle_sub"] = sum(r["energy_cpu_j_idle_sub"] for r in ok) / tot_tok
    s["mean_cpu_power_w"] = statistics.mean(r["cpu_power_w"] for r in ok)
    s["sd_cpu_power_w"] = statistics.stdev(r["cpu_power_w"] for r in ok) if len(ok) > 1 else 0.0
    s["mean_completion_tokens"] = statistics.mean(r["completion_tokens"] for r in ok)
    s["pct_runs_at_max_tokens"] = 100.0 * sum(
        r["completion_tokens"] == MAX_NEW_TOKENS for r in ok) / len(ok)
    return s


# ==============================================================================
# MAIN LOOP
# ==============================================================================

def run_model(label, path, config, modes):
    print(f"\n=== {label} ===")
    llm = Llama(model_path=path, n_ctx=N_CTX, n_threads=N_THREADS, n_gpu_layers=0,
                chat_format=config["format"], verbose=False)
    banned = end_token_ids(llm)

    print("  warm-up generation (discarded)")
    generate(llm, config, PROMPT_LIST[0], "natural", banned)

    print(f"  idle baseline ({IDLE_BASELINE_S}s, model loaded)")
    idle = measure_idle(label)
    print(f"    idle CPU-package power: {idle['idle_cpu_power_w']:.2f} W")

    result = {
        "model": label, "gguf": os.path.basename(path), "routing": config,
        "banned_end_token_ids_fixed_mode": banned,
        "idle_baseline": idle, "modes": {},
    }

    for mode in modes:
        records = []
        for p_idx, prompt in enumerate(PROMPT_LIST):
            for run_idx in range(N_RUNS_PER_PROMPT):
                tracker = make_tracker(f"{safe_name(label)}_{mode}_p{p_idx}_r{run_idx}")
                tracker.start()
                g = generate(llm, config, prompt, mode, banned)
                tracker.stop()
                e = read_tracker(tracker)
                if e is None or not g["completion_tokens"]:
                    print(f"    [WARN] p{p_idx} r{run_idx}: no energy or 0 tokens")
                    continue
                cpu_j = e["energy_cpu_kwh"] * KWH_TO_J
                cpu_j_sub = max(cpu_j - idle["idle_cpu_power_w"] * e["duration_s"], 0.0)
                n = g["completion_tokens"]
                records.append({
                    "mode": mode, "prompt_idx": p_idx, "run_idx": run_idx, **g, **e,
                    "cpu_power_w": cpu_j / e["duration_s"],
                    "energy_cpu_j_idle_sub": cpu_j_sub,
                    "j_per_token_cpu": cpu_j / n,
                    "j_per_token_cpu_idle_sub": cpu_j_sub / n,
                    "j_per_token_total": e["energy_total_kwh"] * KWH_TO_J / n,
                })
            if (p_idx + 1) % 5 == 0:
                print(f"    [{mode}] {p_idx + 1}/{len(PROMPT_LIST)} prompts")

        summ = summarize(records)
        check_power_plausibility(f"{label} [{mode}]", summ["mean_cpu_power_w"])
        if mode == "fixed" and summ["pct_runs_at_max_tokens"] < 100:
            print(f"    [NOTE] only {summ['pct_runs_at_max_tokens']:.0f}% of fixed-mode "
                  f"runs reached {MAX_NEW_TOKENS} tokens (a multi-token stop string "
                  f"still fired) -- report this percentage alongside the result.")
        result["modes"][mode] = {"summary": summ, "runs": records}

    del llm
    gc.collect()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=["natural", "fixed", "both"], default="both")
    ap.add_argument("--force-host", action="store_true")
    args = ap.parse_args()

    assert_correct_host(force=args.force_host)
    assert_rapl_readable(force=args.force_host)
    os.makedirs(OUT_DIR, exist_ok=True)

    modes = ["natural", "fixed"] if args.mode == "both" else [args.mode]
    routing = MODEL_ROUTING if args.model is None else {args.model: MODEL_ROUTING[args.model]}

    run_meta = {
        "cpu_brand": cpu_brand(),
        "llama_cpp_python_version": llama_cpp.__version__,
        "codecarbon_version": codecarbon.__version__,
        "max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0,
        "n_ctx": N_CTX, "n_threads": N_THREADS,
        "n_prompts": len(PROMPT_LIST), "n_runs_per_prompt": N_RUNS_PER_PROMPT,
        "library_sampling_defaults": sampling_defaults(),
        "idle_baseline_s": IDLE_BASELINE_S, "cooldown_s": COOLDOWN_S,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(run_meta, indent=2, default=str))
    with open(os.path.join(OUT_DIR, "run_metadata.json"), "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    for i, (label, config) in enumerate(routing.items()):
        path = os.path.join(LOCAL_DIR, config["filename"])
        if not os.path.exists(path):
            print(f"[SKIP] {label}: {path} not found")
            continue
        if i > 0:
            print(f"[COOLDOWN] {COOLDOWN_S}s")
            time.sleep(COOLDOWN_S)
        try:
            res = run_model(label, path, config, modes)
            out = os.path.join(OUT_DIR, f"multi_prompt_energy_{safe_name(label)}.json")
            with open(out, "w") as f:
                json.dump(res, f, indent=2, default=str)
            print(f"Saved {out}")
        except Exception as e:
            print(f"[ERROR] {label}: {e}")


if __name__ == "__main__":
    main()
