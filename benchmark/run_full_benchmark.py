"""
run_full_benchmark.py
========================
Generates benchmark_summary_<model>.json for a GGUF Q4_K_M model, in the
exact schema used throughout the paper. 

Usage:
    pip install llama-cpp-python psutil codecarbon huggingface_hub py-cpuinfo --break-system-packages

    # Run ALL 8 models, downloading any missing, one after another:
    python3 run_full_benchmark.py

    # Just one model:
    python3 run_full_benchmark.py --model MediLITE-QA

    # Only use GGUF files already on disk:
    python3 run_full_benchmark.py --skip-download

    # Explicit local file outside MODEL_ROUTING entirely:
    python3 run_full_benchmark.py --model SomeModel --path gguf_models/some.gguf --type chat --format chatml

    # Skip the host-brand guard (NOT recommended -- see WARNING it prints):
    python3 run_full_benchmark.py --force-host
"""

import argparse
import gc
import glob
import json
import os
import platform
import statistics
import threading
import time

import psutil
from llama_cpp import Llama

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

try:
    import cpuinfo
    HAS_CPUINFO = True
except ImportError:
    HAS_CPUINFO = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_GPU = True
except Exception:
    HAS_GPU = False


# ==============================================================================
# CONFIG
# ==============================================================================

SAMPLE_PROMPT = (
    "A 65-year-old male with a history of GI bleeding presents with atrial "
    "fibrillation. Is it safe to prescribe rivaroxaban and daily aspirin? "
    "Explain your reasoning and next steps."
)
MAX_NEW_TOKENS = 256
N_THREADS = 4                    # physical cores on the pinned laptop (i5-1135G7)
N_RUNS = 3
BATCH_SIZES = [1, 2, 4, 8, 16]
CONTEXT_LENGTHS_FOR_MEMORY = [512, 1024, 1536, 2000]
N_CTX = 2048
EMISSIONS_LOG_DIR = "emissions_logs_full_benchmark"


EXPECTED_CPU_BRAND_SUBSTRING = "11th Gen Intel(R) Core(TM) i5-1135G7"

# --- Power plausibility guard (fix #7) ---
# RAPL-measured power draw for this laptop (i5-1135G7) under load.
PLAUSIBLE_POWER_W_MIN = 10.0
PLAUSIBLE_POWER_W_MAX = 45.0


# ==============================================================================
# HOST GUARD
# ==============================================================================

def assert_correct_host(force=False):
    brand = cpu_brand()
    print(f"[HOST CHECK] Detected CPU brand: {brand!r}")
    if EXPECTED_CPU_BRAND_SUBSTRING not in brand:
        msg = (
            f"\n[HOST MISMATCH] Expected a CPU brand containing "
            f"'{EXPECTED_CPU_BRAND_SUBSTRING}' (the laptop this project "
            f"standardized on after the cloud provider's Intel Xeon / AMD "
            f"EPYC host-swap and implausible power-draw incidents), but "
            f"detected {brand!r} instead.\n"
            f"Running here would reintroduce exactly the cross-host "
            f"inconsistency this rewrite exists to eliminate.\n"
        )
        if force:
            print(msg + "[--force-host specified: continuing anyway. Do "
                  "NOT mix these results with the laptop dataset without "
                  "flagging the different host explicitly.]\n")
        else:
            raise SystemExit(msg + "Aborting. Re-run with --force-host only "
                              "if you understand and accept this.\n")
    else:
        print("[HOST CHECK] OK -- matches the pinned laptop.\n")


def assert_rapl_readable(force=False):
    """
    Confirms RAPL is actually readable BEFORE running the full benchmark
    -- not after, when you discover the energy numbers are wrong. This
    check exists because an earlier run without root silently produced
    a ~34-35W TDP-fallback estimate instead of the real ~15W RAPL
    measurement, and that error was only caught after the fact by a
    separate diagnostic script. It should not be possible to repeat
    that mistake with this script.
    """
    rapl_paths = glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj")
    if not rapl_paths:
        print("[RAPL CHECK] No RAPL counters found on this system at all "
              "-- CodeCarbon will use a TDP-based estimate regardless of "
              "privileges. This is expected on non-Intel or virtualized "
              "hosts; on the pinned laptop it should not happen.")
        return

    readable = []
    for p in rapl_paths:
        try:
            with open(p) as f:
                f.read()
            readable.append(p)
        except PermissionError:
            pass

    if not readable:
        msg = (
            f"\n[RAPL BLOCKED] Found RAPL counter(s) at {rapl_paths} but "
            f"none are readable by this process -- this is almost always "
            f"a missing-root problem, not a hardware problem. Without "
            f"root, CodeCarbon silently falls back to a TDP-based "
            f"estimate which masks true hardware metrics. Re-run this script with:\n"
            f"    sudo python3 {os.path.basename(__file__)} [your args]\n"
            f"(if using a virtualenv, use 'sudo $(which python3) ...' so "
            f"sudo doesn't drop back to the system Python.)\n"
        )
        if force:
            print(msg + "[--force-host specified: continuing anyway. "
                  "Energy figures from this run will likely be TDP "
                  "estimates, not real measurements -- do not present "
                  "them as RAPL-measured in the paper.]\n")
        else:
            raise SystemExit(msg + "Aborting. Re-run with sudo, or pass "
                              "--force-host to proceed anyway (not "
                              "recommended).\n")
    else:
        print(f"[RAPL CHECK] OK -- readable: {readable}\n")


def check_power_plausibility(label, avg_power_w):
    if avg_power_w is None:
        return
    if not (PLAUSIBLE_POWER_W_MIN <= avg_power_w <= PLAUSIBLE_POWER_W_MAX):
        print(
            f"\n[POWER IMPLAUSIBLE] {label}: avg_power_w = {avg_power_w:.2f} W "
            f"is outside the expected {PLAUSIBLE_POWER_W_MIN}-"
            f"{PLAUSIBLE_POWER_W_MAX} W range for the pinned laptop "
            f"(RAPL-validated steady-state baseline: ~34-36 W under load). This is the "
            f"same failure signature as the cloud provider's earlier "
            f"243-249 W misattribution -- do NOT use this figure in the "
            f"paper without investigating (check RAPL is actually being "
            f"read, not a TDP fallback estimate; see the RAPL-availability "
            f"check pattern from rapl_validation_laptop.py).\n"
        )
    else:
        print(f"[POWER CHECK] {label}: {avg_power_w:.2f} W -- within plausible range.")


# ==============================================================================
# UTILITIES
# ==============================================================================

def mean_stdev(values):
    values = list(values)
    return {"mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


def pctl(values, p):
    """Percentile helper. With only N_RUNS=3 samples this is a coarse
    estimate, not a statistically rigorous tail estimate -- fine for
    descriptive reporting, but do not oversell 'p99' from n=3."""
    if len(values) <= 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


def cpu_brand():
    if HAS_CPUINFO:
        try:
            brand = cpuinfo.get_cpu_info().get("brand_raw")
            if brand:
                return brand
        except Exception:
            pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.strip().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor() or "unknown"


def gpu_device_name():
    """Passive disclosure only -- always n_gpu_layers=0; never affects
    any measured metric."""
    if HAS_GPU:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            return name.decode() if isinstance(name, bytes) else name
        except Exception:
            return "GPU present but query failed"
    return "none detected (CPU-only run, n_gpu_layers=0)"


class UtilSampler:
    def __init__(self, poll_interval_s=0.1):
        self.poll_interval_s = poll_interval_s
        self._stop_event = threading.Event()
        self._thread = None
        self.samples = []

    def _run(self):
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            self.samples.append(psutil.cpu_percent(interval=None))
            time.sleep(self.poll_interval_s)

    def start(self):
        self.samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        return self.samples or [0.0]


class RSSSampler:
    def __init__(self, poll_interval_s=0.05):
        self.poll_interval_s = poll_interval_s
        self.process = psutil.Process(os.getpid())
        self.peak_rss_mb = 0.0
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop_event.is_set():
            rss_mb = self.process.memory_info().rss / (1024 ** 2)
            self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
            time.sleep(self.poll_interval_s)

    def start(self):
        self.peak_rss_mb = self.process.memory_info().rss / (1024 ** 2)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        return self.peak_rss_mb


def generate_once(llm, chat_type, chat_format, stop, prompt, max_tokens):
    gen_kwargs = dict(max_tokens=max_tokens, temperature=0.0)
    if stop is not None:
        gen_kwargs["stop"] = stop

    t_first = None
    t_start = time.perf_counter()
    tokens_out = 0

    if chat_type == "chat":
        stream = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            stream=True, **gen_kwargs,
        )
        for chunk in stream:
            if t_first is None and chunk["choices"][0].get("delta", {}).get("content"):
                t_first = time.perf_counter() - t_start
            tokens_out += 1
    else:
        raw_prompt = f"Question: {prompt}\n\nDetailed Medical Answer:"
        for chunk in llm(raw_prompt, stream=True, **gen_kwargs):
            if t_first is None and chunk["choices"][0].get("text"):
                t_first = time.perf_counter() - t_start
            tokens_out += 1

    elapsed = time.perf_counter() - t_start
    return tokens_out, elapsed, (t_first or elapsed)


# ==============================================================================
# MEASUREMENT STAGES
# ==============================================================================

def measure_streaming_latency(llm, chat_type, chat_format, stop):
    """
    FIX: Run 0 (immediately after model load, or after any long idle gap)
    is consistently and reproducibly ~6-7x slower on TTFT than steady
    state across every diagnostic run performed on this hardware --
    almost certainly a cold cache-load penalty for pulling a multi-GB
    checkpoint into a memory-constrained (6.96 GB) system. Blending it
    into the same percentile distribution as warm-state runs distorts
    p90/p99 in a way that conflates two genuinely different real-world
    scenarios (first query after a cold boot vs. a query in an
    already-running session). Both are now reported explicitly and
    separately instead.
    """
    ttfts, tpots, prompt_times, overall_spts = [], [], [], []

    for i in range(N_RUNS):
        llm.reset()
        tokens, elapsed, ttft = generate_once(
            llm, chat_type, chat_format, stop, SAMPLE_PROMPT, MAX_NEW_TOKENS
        )
        tpot = (elapsed - ttft) / max(tokens - 1, 1)
        overall_spt = elapsed / max(tokens, 1)

        ttfts.append(ttft)
        tpots.append(tpot)
        prompt_times.append(ttft)
        overall_spts.append(overall_spt)
        print(f"    [latency] run {i}: {tokens} tok, {elapsed:.2f}s, TTFT={ttft:.2f}s, TPOT={tpot:.4f}s")

    # Split cold-start (run 0) from warm/steady-state (runs 1+) explicitly.
    cold_start_ttft = ttfts[0]
    warm_ttfts = ttfts[1:] if len(ttfts) > 1 else ttfts
    warm_tpots = tpots[1:] if len(tpots) > 1 else tpots
    warm_overall_spts = overall_spts[1:] if len(overall_spts) > 1 else overall_spts

    if len(warm_ttfts) != len(ttfts):
        print(f"    [NOTE] Cold-start TTFT (run 0): {cold_start_ttft:.2f}s -- "
              f"excluded from warm-state percentiles below. Steady-state "
              f"mean TTFT (runs 1-{N_RUNS-1}): {statistics.mean(warm_ttfts):.2f}s.")

    ttft_p50, ttft_p90, ttft_p99 = pctl(warm_ttfts, 50), pctl(warm_ttfts, 90), pctl(warm_ttfts, 99)
    tpot_p50, tpot_p90, tpot_p99 = pctl(warm_tpots, 50), pctl(warm_tpots, 90), pctl(warm_tpots, 99)
    spt_p50, spt_p90, spt_p99 = pctl(warm_overall_spts, 50), pctl(warm_overall_spts, 90), pctl(warm_overall_spts, 99)

    return {
        "total_completion_time_metrics": {
            "mean_total_prompt_time": mean_stdev(prompt_times),
            "mean_overall_sec_per_token": mean_stdev(overall_spts),
            "p50_overall_spt": {"mean": spt_p50, "stdev": 0.0},
            "p90_overall_spt": {"mean": spt_p90, "stdev": 0.0},
            "p99_overall_spt": {"mean": spt_p99, "stdev": 0.0},
        },
        "streaming_latency_metrics": {
            # NEW: cold-start TTFT reported as its own explicit field,
            # not blended into the warm-state distribution below.
            "cold_start_ttft_s": cold_start_ttft,
            "mean_ttft": mean_stdev(ttfts),            # includes cold start (for backward compatibility)
            "mean_ttft_warm_only": mean_stdev(warm_ttfts),  # excludes run 0
            "p50_ttft": {"mean": ttft_p50, "stdev": 0.0},   # warm-only percentiles
            "p90_ttft": {"mean": ttft_p90, "stdev": 0.0},
            "p99_ttft": {"mean": ttft_p99, "stdev": 0.0},
            "mean_tpot": mean_stdev(tpots),
            "p50_tpot": {"mean": tpot_p50, "stdev": 0.0},
            "p90_tpot": {"mean": tpot_p90, "stdev": 0.0},
            "p99_tpot": {"mean": tpot_p99, "stdev": 0.0},
        },
    }


def measure_throughput(llm, chat_type, chat_format, stop):
    results = {}
    for batch_size in BATCH_SIZES:
        run_tokens_per_sec, run_cpu_util, run_gpu_util = [], [], []

        for run in range(N_RUNS):
            llm.reset()
            util_sampler = UtilSampler()
            util_sampler.start()

            total_tokens = 0
            t_batch_start = time.perf_counter()
            for _ in range(batch_size):
                tokens, elapsed, ttft = generate_once(
                    llm, chat_type, chat_format, stop, SAMPLE_PROMPT, MAX_NEW_TOKENS
                )
                total_tokens += tokens
            t_batch_elapsed = time.perf_counter() - t_batch_start

            cpu_samples = util_sampler.stop()
            tokens_per_sec = total_tokens / t_batch_elapsed if t_batch_elapsed > 0 else 0.0

            assert not (total_tokens > 0 and tokens_per_sec == 0.0), (
                f"[BUG] batch_size={batch_size}, run={run}: generated "
                f"{total_tokens} tokens but computed 0.0 tokens/sec "
                f"(elapsed={t_batch_elapsed}). Investigate before trusting "
                f"this result."
            )

            run_tokens_per_sec.append(tokens_per_sec)
            run_cpu_util.append(statistics.mean(cpu_samples))
            run_gpu_util.append(0.0)

            print(f"    [batch_{batch_size}] run {run}: {total_tokens} tok in "
                  f"{t_batch_elapsed:.2f}s = {tokens_per_sec:.3f} tok/s")

        results[f"batch_{batch_size}"] = {
            "tokens_per_sec": mean_stdev(run_tokens_per_sec),
            "avg_gpu_util_pct": mean_stdev(run_gpu_util),
            "avg_cpu_util_pct": mean_stdev(run_cpu_util),
        }

    return {"throughput_metrics": results}


def measure_memory_vs_context(model_path, chat_type, chat_format, stop):
    """
    FIX #5: now takes chat_type explicitly and dispatches on it directly,
    the same way measure_streaming_latency()/measure_throughput() do --
    no more "try chat, fall back to raw on exception" guessing, which
    could silently succeed with the wrong prompt format for base-type
    models instead of throwing.
    """
    results = {}
    for ctx_len in CONTEXT_LENGTHS_FOR_MEMORY:
        peaks = []
        for run in range(N_RUNS):
            gc.collect()
            llm = Llama(model_path=model_path, n_ctx=ctx_len, n_threads=N_THREADS,
                        chat_format=chat_format, verbose=False)
            sampler = RSSSampler()
            sampler.start()

            max_tok = min(MAX_NEW_TOKENS, ctx_len // 4)
            gen_kwargs = dict(max_tokens=max_tok, temperature=0.0)
            if stop is not None:
                gen_kwargs["stop"] = stop

            if chat_type == "chat":
                llm.create_chat_completion(
                    messages=[{"role": "user", "content": SAMPLE_PROMPT}], **gen_kwargs,
                )
            else:
                raw_prompt = f"Question: {SAMPLE_PROMPT}\n\nDetailed Medical Answer:"
                llm(raw_prompt, **gen_kwargs)

            peak = sampler.stop()
            peaks.append(peak)
            del llm
            gc.collect()
            time.sleep(0.5)
        results[str(ctx_len)] = mean_stdev(peaks)
        print(f"    [memory] ctx={ctx_len}: {mean_stdev(peaks)}")
    return {"memory_context_metrics": results}


def measure_environmental_impact(llm, chat_type, chat_format, stop, label):
    if not HAS_CODECARBON:
        print("    [energy] codecarbon not installed -- skipping environmental_impact.")
        return {"environmental_impact": {
            "avg_power_w": None, "energy_per_token_kwh": None,
            "emissions_per_token_kg_co2eq": None,
        }}

    os.makedirs(EMISSIONS_LOG_DIR, exist_ok=True)
    energies, powers, tokens_list = [], [], []
    for i in range(N_RUNS):
        llm.reset()
        tracker = EmissionsTracker(
            project_name=f"{label.replace(' ', '_')}_env_impact_run{i}",
            output_dir=EMISSIONS_LOG_DIR,
            log_level="error", measure_power_secs=0.5, save_to_file=True,
        )
        tracker.start()
        tokens, elapsed, _ = generate_once(
            llm, chat_type, chat_format, stop, SAMPLE_PROMPT, MAX_NEW_TOKENS
        )
        tracker.stop()
        energy_kwh = (tracker.final_emissions_data.energy_consumed
                      if tracker.final_emissions_data else 0.0) or 0.0
        power_w = (energy_kwh * 3.6e6) / elapsed if elapsed > 0 else 0.0
        energies.append(energy_kwh / max(tokens, 1))
        powers.append(power_w)
        tokens_list.append(tokens)

    mean_energy_per_token = statistics.mean(energies)
    avg_power_w = statistics.mean(powers)

    # FIX #7: check plausibility immediately, right where the number is
    # computed, so it can't be missed downstream in a table somewhere.
    check_power_plausibility(label, avg_power_w)

    return {
        "environmental_impact": {
            "avg_power_w": avg_power_w,
            "energy_per_token_kwh": mean_energy_per_token,
            "emissions_per_token_kg_co2eq": mean_energy_per_token,
        },
        "figure_3_4_metrics": {
            "energy_kwh": mean_stdev(energies),
            "latency_per_token": mean_stdev([1.0 / max(t, 1) for t in tokens_list]),
            "util_pct": mean_stdev(powers),
        },
    }


# ==============================================================================
# MODEL ROUTING
# FIX #2: all 8 core models restored (none commented out).
# FIX #3: repo_id/filename corrected against sources already verified
# working earlier in this project.
# FIX #4: each model carries its own "accuracy" field (9-benchmark
# average from the verified comparison spreadsheet's "seg_calc" column),
# rather than one shared value applied to every model in batch mode.
# Two models (LLaMA 3 8B, Gemma 2B) do not appear in that spreadsheet at
# all -- their accuracy is left as None rather than guessed; fill these
# in from your own verified source before using them in the paper.
# ==============================================================================

MODEL_ROUTING = {
    "Phi-3.5 Mini (base)": {
        "repo_id": "QuantFactory/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct.Q4_K_M.gguf",
        "type": "chat", "format": None, "stop": None,
        "accuracy": 68.56555556,
    },
    "LLaMA 3 8B": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-GGUF",
        "filename": "Meta-Llama-3-8B.Q4_K_M.gguf",
        "type": "base", "format": None, "stop": None,
        "accuracy": None,  # not present in the verified comparison spreadsheet -- do not guess
    },
    "OpenBio": {
        "repo_id": "bartowski/OpenBioLLM-Llama3-8B-GGUF",
        "filename": "OpenBioLLM-Llama3-8B-Q4_K_M.gguf",
        "type": "chat", "format": "chatml", "stop": ["<|im_end|>"],
        "accuracy": 72.5,
    },
    "Medilite-QA": {
        "repo_id": "segestic/MediLITE-QA-GGUF",
        "filename": "MediLITE-QA.Q4_K_M.gguf",
        "type": "chat", "format": None, "stop": None,
        "accuracy": 73.06888889,
    },
    "BioMistral-7B": {
        "repo_id": "BioMistral/BioMistral-7B-GGUF",
        "filename": "ggml-model-Q4_K_M.gguf",
        "type": "chat", "format": "chatml", "stop": ["<|im_end|>"],
        # NOTE: verified spreadsheet shows two different figures for this
        # model -- a stated average of 57.3 vs. a recalculated ("seg_calc")
        # average of 58.96666667 from its own per-benchmark row. Flagging
        # explicitly rather than silently picking one; resolve which is
        # correct before using in the paper.
        "accuracy": None,  # SEE NOTE ABOVE -- discrepancy unresolved (57.3 vs 58.97)
    },
    "Apollo-7B": {
        "repo_id": "FreedomIntelligence/Apollo-7B-GGUF",
        "filename": "Apollo-7B.Q4_K_M.gguf",
        "type": "chat", "format": "alpaca", "stop": None,
        "accuracy": 59.99666667,
    },
    "AlpaCare-llama2-7b": {
        "repo_id": "mradermacher/AlpaCare-llama2-7b-CT_I-II-III_efficient-GGUF",
        "filename": "AlpaCare-llama2-7b-CT_I-II-III_efficient.Q4_K_M.gguf",
        "type": "chat", "format": "llama-2", "stop": None,
        # NOTE: 45.36555556 is the true 9-benchmark average. The value
        # 49.81 seen in an earlier JSON was a mislabeling bug -- that
        # number is actually this model's single Clinical Knowledge
        # subscore, not its overall accuracy. Do not reuse 49.81.
        "accuracy": 45.36555556,
    },
    "ClinicalGPT": {
        "repo_id": "QuantFactory/ClinicalGPT-base-zh-GGUF",
        "filename": "ClinicalGPT-base-zh.Q4_K_M.gguf",
        "type": "base", "format": None, "stop": None,
        "accuracy": 30.52666667,
    },
    "Gemma 2B": {
        "repo_id": "mlabonne/gemma-2b-GGUF",
        "filename": "gemma-2b.Q4_K_M.gguf",
        "type": "chat", "format": "gemma", "stop": None,
        "accuracy": None,  # spreadsheet only has "gemma-7b", a different model -- do not conflate
    },
}

EXTRA_MODELS = {
    "MedAlpaca-7B": {
        "repo_id": "TheBloke/medalpaca-7B-GGUF",
        "filename": "medalpaca-7b.Q4_K_M.gguf",
        "type": "chat", "format": "alpaca", "stop": None,
        "accuracy": 58.03555556,
    },
    "Qwen1.5-7B-Chat": {
        "repo_id": "Qwen/Qwen1.5-7B-Chat-GGUF",
        "filename": "qwen1_5-7b-chat-q4_k_m.gguf",
        "type": "chat", "format": "chatml", "stop": ["<|im_end|>"],
        "accuracy": None,
    },
}

LOCAL_DIR = "gguf_models"


def download_model_if_missing(label, config, local_dir, skip_download, hf_token):
    os.makedirs(local_dir, exist_ok=True)
    file_path = os.path.join(local_dir, config["filename"])
    if os.path.exists(file_path):
        return file_path
    if skip_download:
        print(f"[SKIPPED DOWNLOAD] {label}: '{config['filename']}' not found "
              f"and --skip-download is set.")
        return None
    print(f"[DOWNLOAD] {label}: fetching '{config['filename']}' from "
          f"{config['repo_id']} ...")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename"],
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        print(f"[DOWNLOAD SUCCESS] {label} -> {file_path}")
        return file_path
    except Exception as e:
        print(f"[DOWNLOAD ERROR] {label}: {e}")
        return None


# ==============================================================================
# MAIN
# ==============================================================================

def run_one_model(label, path, model_type, chat_format, stop, accuracy, out_path):
    print(f"\n=== Loading {label} from {path} ===")
    llm = Llama(model_path=path, n_ctx=N_CTX, n_threads=N_THREADS,
                n_gpu_layers=0, chat_format=chat_format, verbose=False)

    result = {
        "model_info": {"base_model": os.path.basename(path), "accuracy": accuracy},
        "system_info": {
            "platform": platform.platform(),
            "cpu_brand": cpu_brand(),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "gpu_device_name": gpu_device_name(),
        },
        "n_runs": N_RUNS,
    }

    print("--- Streaming latency (TTFT/TPOT) ---")
    result.update(measure_streaming_latency(llm, model_type, chat_format, stop))

    print("--- Throughput across batch sizes (including batch_1) ---")
    result.update(measure_throughput(llm, model_type, chat_format, stop))

    print("--- Environmental impact ---")
    result.update(measure_environmental_impact(llm, model_type, chat_format, stop, label))

    del llm
    gc.collect()

    print("--- Memory vs. context length ---")
    result.update(measure_memory_vs_context(path, model_type, chat_format, stop))

    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved {out_path}")

    b1 = result["throughput_metrics"]["batch_1"]["tokens_per_sec"]["mean"]
    print(f"[SANITY CHECK] {label}: batch_1 tokens_per_sec = {b1:.4f} "
          f"({'OK -- nonzero' if b1 > 0 else 'STILL ZERO -- investigate generate_once()'})")
    if accuracy is None:
        print(f"[ACCURACY NOTE] {label}: accuracy is None in this output -- "
              f"not present/resolved in the verified comparison data. Fill "
              f"in manually from a confirmed source before using in the paper.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--type", default="chat", choices=["chat", "base"])
    parser.add_argument("--format", default=None)
    parser.add_argument("--stop", nargs="*", default=None)
    parser.add_argument("--accuracy", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--local-dir", default=LOCAL_DIR)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--include-extra", action="store_true")
    parser.add_argument("--force-host", action="store_true",
                         help="Skip the CPU-brand host guard (not recommended).")
    parser.add_argument("--cooldown-seconds", type=int, default=60,
                         help="Pause between models in batch mode, to let swap/cache "
                              "state settle rather than carrying degraded state from "
                              "one model's run into the next model's cold-start "
                              "measurement (default 60s; set 0 to disable).")
    args = parser.parse_args()

    assert_correct_host(force=args.force_host)
    assert_rapl_readable(force=args.force_host)

    # --- Single explicit local file, outside MODEL_ROUTING ---
    if args.path is not None:
        label = args.model or os.path.splitext(os.path.basename(args.path))[0]
        out_path = args.out or f"benchmark_summary_{label}.json"
        run_one_model(label, args.path, args.type, args.format, args.stop, args.accuracy, out_path)
        return

    # --- Batch mode ---
    routing = dict(MODEL_ROUTING)
    if args.include_extra:
        routing.update(EXTRA_MODELS)
    if args.model is not None:
        if args.model not in routing:
            raise SystemExit(f"'{args.model}' not found in MODEL_ROUTING"
                              f"{' + EXTRA_MODELS' if args.include_extra else ''}. "
                              f"Known labels: {sorted(routing)}")
        routing = {args.model: routing[args.model]}

    print(f"=== Batch run: {len(routing)} model(s) ===")
    all_results, failures = {}, []
    for idx, (label, config) in enumerate(routing.items()):
        if idx > 0 and args.cooldown_seconds > 0:
            print(f"\n[COOLDOWN] Waiting {args.cooldown_seconds}s before starting "
                  f"'{label}' -- lets swap/cache state settle so this model's "
                  f"cold-start measurement isn't inherited degradation from the "
                  f"previous model's run.")
            gc.collect()
            time.sleep(args.cooldown_seconds)

        path = download_model_if_missing(label, config, args.local_dir,
                                          args.skip_download, args.hf_token)
        if not path or not os.path.exists(path):
            print(f"[SKIP] {label}: no local file available.")
            failures.append(label)
            continue

        out_path = f"benchmark_summary_{label}.json"
        # FIX #4: use each model's own accuracy from MODEL_ROUTING,
        # falling back to --accuracy only in explicit single-model
        # --path mode (handled above), never silently shared across
        # a whole batch.
        accuracy = config.get("accuracy")
        try:
            all_results[label] = run_one_model(
                label, path, config["type"], config["format"], config["stop"],
                accuracy, out_path,
            )
        except Exception as e:
            print(f"[ERROR] {label} failed: {e}")
            failures.append(label)

    print(f"\n=== Batch run complete: {len(all_results)} succeeded, {len(failures)} failed/skipped ===")
    if failures:
        print(f"Failed/skipped: {failures}")

    missing_accuracy = [l for l, cfg in routing.items()
                         if l in all_results and cfg.get("accuracy") is None]
    if missing_accuracy:
        print(f"\n[ACCURACY GAPS] These models completed successfully but have "
              f"accuracy=None in their JSON -- resolve before using in the paper: "
              f"{missing_accuracy}")


if __name__ == "__main__":
    main()
