"""
stats_lm_eval.py  --  R1-3 (CIs), R1-7 (tests vs directly evaluated models),
                     R3-2 (PubMedQA dominance, ablation)
==========================================================================
Reads lm-eval 0.4.x samples_<task>_<timestamp>.jsonl files and reports:

  1. per-benchmark accuracy, n, Wilson 95% CI                      (R1-3)
  2. 9-benchmark macro average with item-bootstrap 95% CI,
     plus the average EXCLUDING PubMedQA                            (R1-3, R3-2)
  3. reference model vs every other model:
       exact McNemar per benchmark, Holm-corrected across benchmarks,
       paired bootstrap 95% CI of the macro-average difference      (R1-7)
  4. integrity checks: identical doc_id sets across models, and
     McNemar discordant counts == difference in correct counts
     (the check that the submitted Tables 3/4 fail)
  5. optional clean-subset results, excluding items flagged by
     contamination_check.py                                         (R2-2)

Usage:
    python stats_lm_eval.py \
        --run MediLITE-QA=lm_eval_logs/medilite \
        --run "Phi-3.5 base"=lm_eval_logs/base \
        --run OpenBioLLM-8B=lm_eval_logs/openbiollm8b \
        --run BioMistral-7B=lm_eval_logs/biomistral7b \
        --run "LLaMA 3 8B"=lm_eval_logs/llama3_8b \
        --run "Gemma 2B"=lm_eval_logs/gemma2b \
        --ref MediLITE-QA --out stats_out
    # ablation:  --run MediLITE-Q4=lm_eval_logs/medilite_q4 --run Base-Q4=lm_eval_logs/base_q4
    # clean subset: add  --exclude contamination_out/contaminated_items.json
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, norm

TASKS = ["mmlu_clinical_knowledge", "mmlu_medical_genetics", "mmlu_anatomy",
         "mmlu_professional_medicine", "mmlu_college_biology", "mmlu_college_medicine",
         "medqa_4options", "pubmedqa", "medmcqa"]          # order of Table 4
SHORT = dict(zip(TASKS, ["CK", "MG", "An", "PM", "CB", "CM", "MQA", "PQA", "MCQA"]))
N_BOOT = 10000
SEED = 20261007


# ---------------------------------------------------------------- loading
def load_run(directory, metric, exclude):
    files = glob.glob(os.path.join(directory, "**", "samples_*.jsonl"), recursive=True)
    latest = {}
    for f in files:
        m = re.match(r"samples_(.+)_(\d{4}-\d{2}-\d{2}T[\d\-\.]+)\.jsonl$", os.path.basename(f))
        task, stamp = (m.group(1), m.group(2)) if m else (os.path.basename(f)[8:-6], "")
        if task in latest and latest[task][0] != stamp:
            print(f"  [note] {directory}: several runs of {task}; using the latest")
        if task not in latest or stamp > latest[task][0]:
            latest[task] = (stamp, f)

    out = {}
    for task in TASKS:
        if task not in latest:
            print(f"  [WARN] {directory}: no samples for {task}")
            continue
        drop = set(exclude.get(task, []))
        scores = {}
        with open(latest[task][1]) as fh:
            for line in fh:
                r = json.loads(line)
                if metric not in r:
                    raise SystemExit(f"metric '{metric}' not in {latest[task][1]}; "
                                     f"available: {[k for k in r if k.startswith('acc')]}")
                if r["doc_id"] in drop:
                    continue
                scores[r["doc_id"]] = int(round(float(r[metric])))
        out[task] = scores
    return out


# ---------------------------------------------------------------- statistics
def wilson(k, n, conf=0.95):
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def boot_macro(vectors, tasks, rng):
    """Bootstrap the unweighted mean of per-task accuracies, resampling items
    within each task. `vectors[task]` = 1-D array (accuracy) or 2-D (paired)."""
    acc = np.zeros(N_BOOT)
    for t in tasks:
        v = vectors[t]
        idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
        acc += v[idx].mean(axis=1)
    return acc / len(tasks)


def pct(x):
    return 100.0 * x


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--metric", default="acc", help="acc or acc_norm (use what Table 4 used)")
    ap.add_argument("--exclude", default=None, help="contaminated_items.json")
    ap.add_argument("--out", default="stats_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    exclude = {}
    if args.exclude:
        with open(args.exclude) as f:
            exclude = {t: set(v) for t, v in json.load(f).items()}
        print(f"Excluding flagged items: { {t: len(v) for t, v in exclude.items()} }")

    runs = {}
    for spec in args.run:
        name, d = spec.split("=", 1)
        print(f"Loading {name} from {d}")
        runs[name] = load_run(d, args.metric, exclude)
    if args.ref not in runs:
        raise SystemExit(f"--ref {args.ref} not among --run names")

    tasks = [t for t in TASKS if all(t in r for r in runs.values())]
    missing = [t for t in TASKS if t not in tasks]
    if missing:
        print(f"[WARN] tasks missing for at least one model, left out: {missing}")

    # ---- integrity: identical item sets
    for t in tasks:
        ref_ids = set(runs[args.ref][t])
        for name, r in runs.items():
            if set(r[t]) != ref_ids:
                print(f"[WARN] {t}: {name} has {len(r[t])} items vs ref {len(ref_ids)}; "
                      f"paired tests use the intersection only")

    # ---- 1 & 2: accuracy table
    acc_rows = []
    for name, r in runs.items():
        row = {"model": name}
        vecs = {}
        for t in tasks:
            v = np.array(list(r[t].values()))
            vecs[t] = v
            k, n = int(v.sum()), len(v)
            lo, hi = wilson(k, n)
            row[f"{SHORT[t]}_acc"] = pct(k / n)
            row[f"{SHORT[t]}_lo"], row[f"{SHORT[t]}_hi"] = pct(lo), pct(hi)
            row[f"{SHORT[t]}_n"] = n
        for label, ts in [("avg", tasks), ("avg_noPQA", [t for t in tasks if t != "pubmedqa"])]:
            point = np.mean([vecs[t].mean() for t in ts])
            b = boot_macro(vecs, ts, rng)
            row[label] = pct(point)
            row[f"{label}_lo"], row[f"{label}_hi"] = pct(np.percentile(b, 2.5)), pct(np.percentile(b, 97.5))
        acc_rows.append(row)

    with open(os.path.join(args.out, "accuracy_ci.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(acc_rows[0]))
        w.writeheader()
        w.writerows(acc_rows)

    print("\n=== Accuracy (%) with Wilson 95% CI; average with item-bootstrap 95% CI ===")
    for row in acc_rows:
        cells = [f"{SHORT[t]} {row[f'{SHORT[t]}_acc']:.2f} [{row[f'{SHORT[t]}_lo']:.1f}, "
                 f"{row[f'{SHORT[t]}_hi']:.1f}] (n={row[f'{SHORT[t]}_n']})" for t in tasks]
        print(f"\n{row['model']}:\n  " + "\n  ".join(cells))
        print(f"  AVG       {row['avg']:.2f} [{row['avg_lo']:.2f}, {row['avg_hi']:.2f}]")
        print(f"  AVG noPQA {row['avg_noPQA']:.2f} [{row['avg_noPQA_lo']:.2f}, {row['avg_noPQA_hi']:.2f}]")

    # ---- 3 & 4: paired comparisons vs ref
    pair_rows, avg_rows = [], []
    ref = runs[args.ref]
    for name, r in runs.items():
        if name == args.ref:
            continue
        per_task, pvals, paired = [], [], {}
        for t in tasks:
            ids = sorted(set(ref[t]) & set(r[t]))
            a = np.array([ref[t][i] for i in ids])
            o = np.array([r[t][i] for i in ids])
            b = int(((a == 1) & (o == 0)).sum())
            c = int(((a == 0) & (o == 1)).sum())
            assert b - c == int(a.sum() - o.sum()), "discordant/accuracy mismatch"
            p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
            pvals.append(p)
            paired[t] = (a - o).astype(float)
            per_task.append(dict(comparison=f"{args.ref} vs {name}", benchmark=SHORT[t],
                                 n=len(ids), ref_only=b, other_only=c,
                                 diff_pp=pct((a.sum() - o.sum()) / len(ids)), p_raw=p))
        for row, padj in zip(per_task, holm(np.array(pvals))):
            row["p_holm"] = padj
            row["sig_0.05"] = "*" if padj < 0.05 else ""
        pair_rows += per_task

        point = np.mean([paired[t].mean() for t in tasks])
        bd = boot_macro(paired, tasks, rng)
        lo, hi = np.percentile(bd, [2.5, 97.5])
        avg_rows.append(dict(comparison=f"{args.ref} vs {name}", avg_diff_pp=pct(point),
                             ci_lo=pct(lo), ci_hi=pct(hi),
                             excludes_zero="yes" if lo > 0 or hi < 0 else "no"))

    for fname, rows in [("mcnemar_holm.csv", pair_rows), ("avg_difference_bootstrap.csv", avg_rows)]:
        if rows:
            with open(os.path.join(args.out, fname), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    print("\n=== Exact McNemar, Holm-corrected across benchmarks (per comparison) ===")
    for row in pair_rows:
        print(f"  {row['comparison']:<35} {row['benchmark']:<5} n={row['n']:<5} "
              f"ref-only={row['ref_only']:<4} other-only={row['other_only']:<4} "
              f"diff={row['diff_pp']:+6.2f}pp  p_holm={row['p_holm']:.4g} {row['sig_0.05']}")
    print("\n=== Macro-average difference, paired item-bootstrap 95% CI ===")
    for row in avg_rows:
        print(f"  {row['comparison']:<35} {row['avg_diff_pp']:+.2f}pp "
              f"[{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}]  excludes 0: {row['excludes_zero']}")
    print(f"\nCSV files written to {args.out}/")


if __name__ == "__main__":
    main()
