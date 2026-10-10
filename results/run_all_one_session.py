"""
run_all_one_session.py  --  everything for the accuracy side in ONE Kaggle session
==================================================================================
Kaggle GPU T4 x2. One command:

    !python run_all_one_session.py                    # base + MediLITE: fp16 then Q4
    !python run_all_one_session.py --with-comparison  # + OpenBioLLM-8B, BioMistral-7B,
                                                      #   LLaMA 3 8B, Gemma 2B (fp16, long)

Stages (each skipped if its output already exists, so a re-run resumes):
  0. install pinned packages (same versions as your original lm-eval cell)
  1. fp16 lm-eval, base on GPU 0 and MediLITE-QA on GPU 1 IN PARALLEL
     (identical settings to your original cell: 0-shot, fp16, batch 4,
     --log_samples) -> also checks the new numbers reproduce Table 4
  2. Q4_K_M GGUF scoring of the SAME logged requests, base on GPU 0 and
     MediLITE-QA on GPU 1 in parallel (in-process llama.cpp; no lm-eval
     gguf backend, which gave a 30% artefact)
  3. optional: comparison models, fp16, one at a time across both GPUs
  4. stats_lm_eval.py if present, then zip everything to
     /kaggle/working/revision_outputs.zip

Before running: set HF_TOKEN (your secrets cell). At the end use
"Save Version -> Save & Run All" or download the zip: /kaggle/working is wiped
when the session ends.
"""


import argparse
import glob
import json
import math
import os
import re
import time

import numpy as np

try:
    import llama_cpp
except ImportError:
    llama_cpp = None

SMOKE_TASK = "mmlu_medical_genetics"


# ----------------------------------------------------------- parse fp16 logs
def task_of(path):
    name = os.path.basename(path)[len("samples_"):-len(".jsonl")]
    return re.sub(r"_\d{4}-\d{2}-\d{2}T.*$", "", name)


def latest_files(log_dir):
    by_task = {}
    for f in glob.glob(os.path.join(log_dir, "**", "samples_*.jsonl"), recursive=True):
        t = task_of(f)
        if t not in by_task or os.path.basename(f) > os.path.basename(by_task[t]):
            by_task[t] = f
    return by_task


def get_requests(rec):
    """[(context, continuation), ...] from the lm-eval 'arguments' field."""
    args = rec["arguments"]
    if isinstance(args, dict):  # 0.4.3+: {"gen_args_0": {"arg_0": ctx, "arg_1": cont}, ...}
        keys = sorted(args, key=lambda k: int(re.findall(r"\d+", k)[-1]))
        return [(args[k]["arg_0"], args[k]["arg_1"]) for k in keys]
    return [(a[0], a[1]) for a in args]  # older: [[ctx, cont], ...]


def first_number(x):
    while isinstance(x, (list, tuple)):
        x = x[0]
    return float(x)


def gold_index(rec, reqs):
    t = rec["target"]
    if isinstance(t, list):
        t = t[0]
    try:
        return int(t)
    except (TypeError, ValueError):
        conts = [c.strip() for _, c in reqs]
        return conts.index(str(t).strip())


def choice_len(cont):
    return float(len(cont[1:] if cont.startswith(" ") else cont))


def score_from(lls, reqs, gold):
    lls = np.asarray(lls, dtype=float)
    acc = float(int(np.argmax(lls)) == gold)
    acc_norm = float(int(np.argmax(lls / np.array([choice_len(c) for _, c in reqs]))) == gold)
    return acc, acc_norm


def parsing_check(task, records):
    agree = n = 0
    for rec in records:
        if "acc" not in rec or "resps" not in rec:
            continue
        reqs = get_requests(rec)
        lls = [first_number(r) for r in rec["resps"]]
        acc, _ = score_from(lls, reqs, gold_index(rec, reqs))
        agree += acc == float(rec["acc"])
        n += 1
    rate = agree / n if n else 0.0
    print(f"  [check 1] {task}: fp16 acc re-derived from stored log-likelihoods "
          f"matches stored acc on {agree}/{n} ({100 * rate:.2f}%)")
    if n == 0 or rate < 0.995:
        raise SystemExit("Parsing check failed -- targets/arguments not read correctly. Stop.")


# ----------------------------------------------------------- scoring
class Scorer:
    def __init__(self, model_path, n_ctx, add_bos):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1,
                         logits_all=False, verbose=False)
        self.n_ctx = n_ctx
        self.add_bos = add_bos
        self.cache = True
        self.n_vocab = self.llm.n_vocab()
        self.model_path = model_path

    def reference_lls(self, reqs):
        """Independent reference: a separate logits_all=True instance, whole
        sequence evaluated at once, log-probs read from global rows."""
        from llama_cpp import Llama
        if not hasattr(self, "_ref"):
            self._ref = Llama(model_path=self.model_path, n_ctx=self.n_ctx, n_gpu_layers=-1,
                              logits_all=True, verbose=False)
        out = []
        for ctx, cont in reqs:
            ctx_t, cont_t = self.encode_pair(ctx, cont)
            ctx_t = ctx_t[-(self.n_ctx - len(cont_t) - 1):]
            self._ref.reset()
            self._ref.eval(ctx_t + cont_t)
            sc = np.asarray(self._ref.scores, dtype=np.float64)
            ll = 0.0
            for j, t in enumerate(cont_t):
                row = sc[len(ctx_t) + j - 1]
                m = row.max()
                ll += row[t] - (m + math.log(np.exp(row - m).sum()))
            out.append(ll)
        return out

    def drop_reference(self):
        if hasattr(self, "_ref"):
            del self._ref

    def tok(self, text):
        return self.llm.tokenize(text.encode("utf-8"), add_bos=self.add_bos, special=False)

    def encode_pair(self, ctx, cont):
        n_sp = len(ctx) - len(ctx.rstrip())  # lm-eval moves trailing spaces to the continuation
        if n_sp:
            cont, ctx = ctx[-n_sp:] + cont, ctx[:-n_sp]
        whole = self.tok(ctx + cont)
        ctx_t = self.tok(ctx)
        return ctx_t, whole[len(ctx_t):]

    def _last_logprobs(self):
        # Read the last token's logits straight from llama.cpp. NOT llm.scores:
        # with logits_all=False that buffer has only n_batch rows and is
        # written at a batch-local index, so reading row n_tokens-1 returns
        # stale logits (this produced the near-chance 32% MedMCQA result).
        ptr = llama_cpp.llama_get_logits_ith(self.llm._ctx.ctx, -1)
        logits = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).astype(np.float64)
        m = logits.max()
        return logits - (m + math.log(np.exp(logits - m).sum()))

    def _cont_ll(self, cont_t, first_lp):
        ll, lp = 0.0, first_lp
        for i, t in enumerate(cont_t):
            if i > 0:
                self.llm.eval([cont_t[i - 1]])
                lp = self._last_logprobs()
            ll += lp[t]
        return ll

    def score_doc(self, reqs, use_cache=None):
        use_cache = self.cache if use_cache is None else use_cache
        enc = [self.encode_pair(c, k) for c, k in reqs]
        out = []
        if use_cache and len({tuple(c) for c, _ in enc}) == 1:
            ctx_t = enc[0][0]
            budget = self.n_ctx - max(len(k) for _, k in enc) - 1
            ctx_t = ctx_t[-budget:]  # left-truncate like lm-eval
            self.llm.reset()
            self.llm.eval(ctx_t)
            ctx_len, first_lp = self.llm.n_tokens, self._last_logprobs()
            for _, cont_t in enc:
                self.llm.n_tokens = ctx_len  # reuse cached prefix
                out.append(self._cont_ll(cont_t, first_lp))
        else:
            for ctx_t, cont_t in enc:
                ctx_t = ctx_t[-(self.n_ctx - len(cont_t) - 1):]
                self.llm.reset()
                self.llm.eval(ctx_t)
                out.append(self._cont_ll(cont_t, self._last_logprobs()))
        return out


def reference_check(scorer, records, k=40):
    """Fast scorer vs an independent full-sequence (logits_all) reference.
    Small |dLL| (~0.1) is expected: llama.cpp uses different CUDA kernels for
    different batch shapes. The stale-logits bug this guards against gives
    multi-nat errors and random picks, so we test pick agreement + a loose bound."""
    diffs, agree = [], 0
    for rec in records[:k]:
        reqs = get_requests(rec)
        a = scorer.score_doc(reqs, use_cache=False)
        b = scorer.reference_lls(reqs)
        diffs += [abs(x - y) for x, y in zip(a, b)]
        agree += int(np.argmax(a) == np.argmax(b))
    scorer.drop_reference()
    print(f"  [check 3] fast vs independent reference on {k} items: same pick {agree}/{k}, "
          f"|dLL| mean {np.mean(diffs):.3f}, max {np.max(diffs):.3f}")
    if agree < 0.95 * k or np.max(diffs) > 0.5:
        raise SystemExit("Reference check failed -- scorer disagrees with full-sequence reference. Stop.")


def cache_check(scorer, records, k=20):
    worst = 0.0
    for rec in records[:k]:
        reqs = get_requests(rec)
        a = scorer.score_doc(reqs, use_cache=True)
        b = scorer.score_doc(reqs, use_cache=False)
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
    scorer.cache = worst < 1e-2
    print(f"  [check 2] cached vs uncached max |dLL| = {worst:.2e} -> "
          f"prefix cache {'ON' if scorer.cache else 'OFF (slower, still correct)'}")


# ----------------------------------------------------------- main
def q4_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16-logs", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--add-bos", action="store_true",
                    help="prepend BOS (lm-eval's HF backend does not by default)")
    ap.add_argument("--smoke", action="store_true", help=f"only {SMOKE_TASK}")
    args = ap.parse_args(argv)

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=args.repo, filename=args.file, local_dir="gguf_models")

    files = latest_files(args.fp16_logs)
    if not files:
        raise SystemExit(
            f"No samples_*.jsonl found under '{args.fp16_logs}' (searched recursively).\n"
            f"The fp16 logs from your earlier lm-eval run are not in this session. "
            f"Attach them as a Kaggle dataset and pass e.g. "
            f"--fp16-logs /kaggle/input/<dataset-name>/medilite")
    print(f"found fp16 logs for: {sorted(files)}")
    if args.smoke:
        key = next((t for t in files if t.endswith("medical_genetics")), None)
        if key is None:
            raise SystemExit(f"no Medical Genetics log among {sorted(files)}")
        files = {key: files[key]}
    print(f"tasks: {sorted(files)}")

    scorer = Scorer(path, args.n_ctx, args.add_bos)
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S.000000")
    out_dir = os.path.join(args.out, os.path.splitext(args.file)[0])
    os.makedirs(out_dir, exist_ok=True)

    checked_cache = False
    for task, f in sorted(files.items()):
        with open(f) as fh:
            records = [json.loads(line) for line in fh]
        parsing_check(task, records)
        if not checked_cache:
            reference_check(scorer, records)
            cache_check(scorer, records)
            checked_cache = True

        t0, accs, fp16_accs = time.time(), [], []
        out_path = os.path.join(out_dir, f"samples_{task}_{stamp}.jsonl")
        with open(out_path, "w") as fo:
            for i, rec in enumerate(records):
                reqs = get_requests(rec)
                gold = gold_index(rec, reqs)
                lls = scorer.score_doc(reqs)
                acc, acc_norm = score_from(lls, reqs, gold)
                accs.append(acc)
                fp16_accs.append(float(rec.get("acc", float("nan"))))
                fo.write(json.dumps({"doc_id": rec["doc_id"], "doc": rec["doc"],
                                     "target": rec["target"], "resps": lls,
                                     "acc": acc, "acc_norm": acc_norm}) + "\n")
                if (i + 1) % 500 == 0:
                    print(f"    {task}: {i + 1}/{len(records)}")
        q4a, fpa = 100 * np.mean(accs), 100 * np.nanmean(fp16_accs)
        print(f"  {task}: Q4 acc {q4a:.2f}%  vs fp16 {fpa:.2f}%  "
              f"(n={len(records)}, {(time.time() - t0) / 60:.1f} min)")
        if q4a < fpa - 10:
            print(f"  [WARNING] {task}: Q4 is {fpa - q4a:.1f} points below fp16 -- "
                  f"not a normal Q4_K_M loss; investigate before using")
    print(f"\nwritten to {out_dir}")




# =============================================================== PIPELINE
import shutil
import subprocess
import sys

SAVE = "lm_eval_logs"
TASKS = ("mmlu_anatomy,mmlu_professional_medicine,mmlu_clinical_knowledge,"
         "mmlu_medical_genetics,mmlu_college_biology,mmlu_college_medicine,"
         "medqa_4options,medmcqa,pubmedqa")
BASE = "microsoft/Phi-3.5-mini-instruct"
ADAPTER = "segestic/phi3.5-mini-4k-qlora-medical-seg-vall_med"
GGUF = {
    "base":     ("QuantFactory/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct.Q4_K_M.gguf"),
    "medilite": ("segestic/MediLITE-QA-GGUF",               "MediLITE-QA.Q4_K_M.gguf"),
}
COMPARISON = {
    "openbiollm8b": "aaditya/Llama3-OpenBioLLM-8B",
    "biomistral7b": "BioMistral/BioMistral-7B",
    "llama3_8b":    "meta-llama/Meta-Llama-3-8B",
    "gemma2b":      "google/gemma-2b",
}
# submitted Table 4 values, for the reproduction check
PAPER = {
    "medilite": {"mmlu_clinical_knowledge": 80.00, "mmlu_medical_genetics": 80.00, "mmlu_anatomy": 65.93,
                 "mmlu_professional_medicine": 76.84, "mmlu_college_biology": 88.89,
                 "mmlu_college_medicine": 67.63, "medqa_4options": 59.86, "pubmedqa": 75.60, "medmcqa": 62.87},
    "base":     {"mmlu_clinical_knowledge": 76.60, "mmlu_medical_genetics": 75.00, "mmlu_anatomy": 65.19,
                 "mmlu_professional_medicine": 68.75, "mmlu_college_biology": 84.03,
                 "mmlu_college_medicine": 65.90, "medqa_4options": 49.10, "pubmedqa": 77.20, "medmcqa": 55.32},
}


def sh(cmd, env=None):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def has_samples(d, n_tasks=9):
    return len(latest_files(d)) >= n_tasks if os.path.isdir(d) else False


def gpu_env(gpu):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    return env


def lm_eval_cmd(model_args, out):
    return [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", model_args,
            "--tasks", TASKS, "--num_fewshot", "0", "--batch_size", "4",
            "--log_samples", "--output_path", out]


N_GPUS = 2


def run_parallel(jobs):
    """jobs: [(name, cmd, env, logfile)] -- in parallel if 2+ GPUs, else sequentially."""
    if N_GPUS < 2 and len(jobs) > 1:
        for job in jobs:
            run_parallel([job])
        return
    procs = []
    for name, cmd, env, log in jobs:
        print(f"[start] {name}  (log: {log})", flush=True)
        procs.append((name, subprocess.Popen(cmd, env=env, stdout=open(log, "w"),
                                             stderr=subprocess.STDOUT), log))
    failed = []
    for name, p, log in procs:
        rc = p.wait()
        print(f"[done]  {name}: exit {rc}", flush=True)
        if rc != 0:
            failed.append(name)
            print(open(log).read()[-3000:])
    if failed:
        raise SystemExit(f"failed: {failed} -- see logs above")


def task_acc(d):
    out = {}
    for t, f in latest_files(d).items():
        vals = [float(json.loads(l)["acc"]) for l in open(f)]
        out[t] = 100 * sum(vals) / len(vals)
    return out


def stage_install():
    # Pin the torch already installed so no package (bitsandbytes, accelerate,
    # lm-eval) can upgrade it and break torchvision/torchaudio binaries.
    pins = subprocess.run([sys.executable, "-c",
        "import importlib.metadata as m\n"
        "for p in ('torch','torchvision'):\n"
        "    try: print(f'{p}=={m.version(p)}')\n"
        "    except Exception: pass"], capture_output=True, text=True).stdout
    with open("torch_constraints.txt", "w") as f:
        f.write(pins)
    print("[pins]", pins.strip().replace("\n", ", "))
    pip = [sys.executable, "-m", "pip", "install", "-q", "-c", "torch_constraints.txt"]
    sh(pip + ["lm-eval==0.4.11"])
    # One mutually compatible set, installed together LAST:
    #  - datasets 3.2.0: 4.x dropped dataset scripts (pubmedqa needs pubmed_qa.py)
    #  - transformers 4.56.x: lm-eval 0.4.11 passes `dtype=` to from_pretrained,
    #    which transformers only accepts from 4.56 on
    #  - huggingface_hub < 1.0: 1.x rejects legacy un-namespaced dataset ids
    #    such as "medmcqa" used by lm-eval's task configs (4.56 requires <1.0 too)
    sh(pip + ["datasets==3.2.0", "transformers>=4.56,<4.57", "huggingface_hub>=0.34,<1.0",
              "peft>=0.17,<0.18", "accelerate>=1.10,<1.11"])
    # torchaudio is not used here; a build that doesn't match torch crashes
    # `import transformers`, so remove it
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"])
    check_versions()


def check_versions():
    """Abort early on a bad environment; record versions for the paper (R1-4)."""
    code = ("import json, datasets, transformers, torch, peft, llama_cpp, huggingface_hub, importlib.metadata as m;"
            "print(json.dumps({'lm_eval': m.version('lm-eval'), 'datasets': datasets.__version__,"
            "'huggingface_hub': huggingface_hub.__version__,"
            "'transformers': transformers.__version__, 'torch': torch.__version__,"
            "'peft': peft.__version__, 'llama_cpp_python': llama_cpp.__version__,"
            "'llama_cpp_gpu_offload': bool(llama_cpp.llama_supports_gpu_offload()),"
            "'cuda_devices': torch.cuda.device_count()}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("version check failed:\n" + out.stderr[-2000:])
    v = json.loads(out.stdout.strip().splitlines()[-1])
    print("[versions]", v)
    with open("environment_versions.json", "w") as f:
        json.dump(v, f, indent=2)
    problems = []
    if int(v["datasets"].split(".")[0]) >= 4:
        problems.append(f"datasets {v['datasets']} >= 4 (pubmedqa needs < 4)")
    ver = lambda x: tuple(int(p) for p in re.findall(r"\d+", x)[:2])
    if ver(v["huggingface_hub"]) >= (1, 0):
        problems.append(f"huggingface_hub {v['huggingface_hub']} >= 1.0 (rejects 'medmcqa' dataset id)")
    if ver(v["transformers"]) < (4, 56):
        problems.append(f"transformers {v['transformers']} < 4.56 (lm-eval 0.4.11 passes dtype=)")
    if not v["llama_cpp_gpu_offload"]:
        problems.append("llama-cpp-python is a CPU-only build")
    global N_GPUS
    N_GPUS = v["cuda_devices"]
    if N_GPUS < 1:
        problems.append("no CUDA GPU visible")
    elif N_GPUS == 1:
        print("[note] 1 GPU visible: jobs run one after another instead of in parallel")
    if problems:
        raise SystemExit("[ENV PROBLEM] " + "; ".join(problems))


def stage_fp16():
    # download every dataset once, so the two parallel runs don't race on the cache
    if not (has_samples(f"{SAVE}/base") and has_samples(f"{SAVE}/medilite")):
        sh([sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args",
            f"pretrained={BASE},dtype=float16", "--tasks", TASKS, "--num_fewshot", "0",
            "--batch_size", "1", "--limit", "1", "--output_path", "_cache_warmup"],
           env=gpu_env(0))
    jobs = []
    for name, margs in [("base", f"pretrained={BASE},dtype=float16"),
                        ("medilite", f"pretrained={BASE},peft={ADAPTER},dtype=float16")]:
        out = f"{SAVE}/{name}"
        if has_samples(out):
            print(f"[skip] fp16 {name}: logs present")
            continue
        jobs.append((f"fp16 {name}", lm_eval_cmd(margs, out),
                     gpu_env(0 if name == "base" or N_GPUS < 2 else 1), f"log_fp16_{name}.txt"))
    if jobs:
        run_parallel(jobs)

    print("\n=== Reproduction check vs submitted Table 4 (acc, %) ===")
    for name in ("base", "medilite"):
        got = task_acc(f"{SAVE}/{name}")
        for t, ref in PAPER[name].items():
            g = got.get(t)
            flag = "" if g is not None and abs(g - ref) < 0.5 else "   <-- differs"
            print(f"  {name:<9}{t:<28} paper {ref:6.2f}  now {g if g is None else round(g, 2)}{flag}")
    print("  (differences mean Table 4 did not come from this exact setup -- report the new numbers)")


def stage_q4():
    jobs = []
    for i, (name, (repo, fname)) in enumerate(GGUF.items()):
        out = f"{SAVE}/{name}_q4"
        if has_samples(out):
            print(f"[skip] Q4 {name}: logs present")
            continue
        argv = ["q4", "--fp16-logs", f"{SAVE}/{name}", "--repo", repo, "--file", fname, "--out", out]
        jobs.append((f"Q4 {name}", [sys.executable, os.path.abspath(__file__)] + argv,
                     gpu_env(i if N_GPUS >= 2 else 0), f"log_q4_{name}.txt"))
    if jobs:
        run_parallel(jobs)
    print("\n=== fp16 vs Q4_K_M (acc, %) ===")
    for name in ("base", "medilite"):
        a, b = task_acc(f"{SAVE}/{name}"), task_acc(f"{SAVE}/{name}_q4")
        for t in sorted(a):
            if t in b:
                print(f"  {name:<9}{t:<28} fp16 {a[t]:6.2f}  Q4 {b[t]:6.2f}  diff {b[t] - a[t]:+6.2f}")


def stage_comparison():
    for name, repo in COMPARISON.items():
        out = f"{SAVE}/{name}"
        if has_samples(out):
            print(f"[skip] {name}: logs present")
            continue
        env = gpu_env("0,1" if N_GPUS >= 2 else "0")
        try:
            sh([sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args",
                f"pretrained={repo},dtype=float16,parallelize=True",
                "--tasks", TASKS, "--num_fewshot", "0", "--batch_size", "4",
                "--log_samples", "--output_path", out], env=env)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {name}: {e} -- continuing")


def stage_stats(with_comparison):
    if not os.path.exists("stats_lm_eval.py"):
        print("[skip] stats_lm_eval.py not in working dir")
        return
    runs = ["--run", f"MediLITE-QA={SAVE}/medilite", "--run", f"Phi-3.5 base={SAVE}/base"]
    if with_comparison:
        labels = {"openbiollm8b": "OpenBioLLM-8B", "biomistral7b": "BioMistral-7B",
                  "llama3_8b": "LLaMA 3 8B", "gemma2b": "Gemma 2B"}
        runs += sum([["--run", f"{labels[n]}={SAVE}/{n}"] for n in COMPARISON if has_samples(f"{SAVE}/{n}")], [])
    sh([sys.executable, "stats_lm_eval.py", "--ref", "MediLITE-QA", "--out", "stats_out"] + runs)
    sh([sys.executable, "stats_lm_eval.py", "--ref", "MediLITE-Q4", "--out", "stats_out_ablation",
        "--run", f"MediLITE-Q4={SAVE}/medilite_q4", "--run", f"Base-Q4={SAVE}/base_q4",
        "--run", f"MediLITE-fp16={SAVE}/medilite", "--run", f"Base-fp16={SAVE}/base"])


def pipeline_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-comparison", action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if not os.getenv("HF_TOKEN"):
        print("[WARN] HF_TOKEN not set -- gated models (LLaMA 3, Gemma) will fail")
    if not args.skip_install:
        stage_install()
    else:
        check_versions()
    for name, fn in [("fp16 lm-eval", stage_fp16), ("Q4 scoring", stage_q4)]:
        s = time.time()
        fn()
        print(f"[time] {name}: {(time.time() - s) / 3600:.2f} h")
    if args.with_comparison:
        s = time.time()
        stage_comparison()
        print(f"[time] comparison models: {(time.time() - s) / 3600:.2f} h")
    stage_stats(args.with_comparison)
    # zip results only (NOT the multi-GB model files)
    stage = "revision_outputs"
    os.makedirs(stage, exist_ok=True)
    for item in [SAVE, "stats_out", "stats_out_ablation", "environment_versions.json"] + glob.glob("log_*.txt"):
        if os.path.isdir(item):
            shutil.copytree(item, os.path.join(stage, item), dirs_exist_ok=True)
        elif os.path.isfile(item):
            shutil.copy2(item, stage)
    shutil.make_archive("revision_outputs", "zip", stage)
    print(f"\nALL DONE in {(time.time() - t0) / 3600:.2f} h -> revision_outputs.zip "
          f"(Save Version or download before the session ends)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "q4":
        q4_main(sys.argv[2:])
    else:
        pipeline_main()
