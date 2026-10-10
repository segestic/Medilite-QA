"""
diag_q4.py  --  why is MediLITE-QA Q4 near chance?  (no GPU needed; reads existing logs)

For each model (base, medilite) and task, compares the fp16 lm-eval log with the
Q4 log written by the pipeline:
  * accuracy fp16 vs Q4
  * how often fp16 and Q4 pick the SAME option
  * Pearson correlation of per-option log-likelihoods (fp16 vs Q4)
  * distribution of the option Q4 picks (a collapsed model picks one letter)

Reading the result:
  base Q4 ~ base fp16 (agreement high, corr > ~0.9)  AND  medilite Q4 near chance
      -> the scorer is fine; the MediLITE-QA GGUF file is the problem
  base Q4 ALSO near chance / low corr
      -> scoring bug in the Q4 path, not the model files

Usage:  python diag_q4.py            (run from the folder containing lm_eval_logs/)
"""

import glob
import json
import os
import re
from collections import Counter

import numpy as np

SAVE = "lm_eval_logs"


def task_of(path):
    name = os.path.basename(path)[len("samples_"):-len(".jsonl")]
    return re.sub(r"_\d{4}-\d{2}-\d{2}T.*$", "", name)


def latest(d):
    out = {}
    for f in glob.glob(os.path.join(d, "**", "samples_*.jsonl"), recursive=True):
        t = task_of(f)
        if t not in out or os.path.basename(f) > os.path.basename(out[t]):
            out[t] = f
    return out


def first_number(x):
    while isinstance(x, (list, tuple)):
        x = x[0]
    return float(x)


def load(f, q4):
    recs = {}
    for line in open(f):
        r = json.loads(line)
        lls = [float(v) for v in r["resps"]] if q4 else [first_number(v) for v in r["resps"]]
        recs[r["doc_id"]] = (np.array(lls), float(r["acc"]))
    return recs


def main():
    for model in ("base", "medilite"):
        fp, q4 = latest(f"{SAVE}/{model}"), latest(f"{SAVE}/{model}_q4")
        if not q4:
            print(f"\n[{model}] no Q4 logs yet")
            continue
        print(f"\n=== {model} ===")
        print(f"{'task':<28}{'n':>6}{'fp16':>8}{'Q4':>8}{'same pick':>11}{'LL corr':>9}  Q4 pick distribution")
        for t in sorted(q4):
            if t not in fp:
                continue
            a, b = load(fp[t], False), load(q4[t], True)
            ids = sorted(set(a) & set(b))
            same, xs, ys, picks = 0, [], [], Counter()
            for i in ids:
                la, lb = a[i][0], b[i][0]
                if len(la) != len(lb):
                    continue
                same += int(np.argmax(la) == np.argmax(lb))
                picks[int(np.argmax(lb))] += 1
                xs.extend(la)
                ys.extend(lb)
            corr = np.corrcoef(xs, ys)[0, 1] if len(xs) > 2 else float("nan")
            acc_a = 100 * np.mean([a[i][1] for i in ids])
            acc_b = 100 * np.mean([b[i][1] for i in ids])
            dist = " ".join(f"{k}:{100 * v / len(ids):.0f}%" for k, v in sorted(picks.items()))
            print(f"{t:<28}{len(ids):>6}{acc_a:>8.2f}{acc_b:>8.2f}{100 * same / len(ids):>10.1f}%{corr:>9.3f}  {dist}")
        # one example, raw numbers
        t = next(iter(sorted(q4)))
        a, b = load(fp[t], False), load(q4[t], True)
        i = sorted(set(a) & set(b))[0]
        print(f"  example {t} doc {i}: fp16 LL {np.round(a[i][0], 2)}  Q4 LL {np.round(b[i][0], 2)}")


if __name__ == "__main__":
    main()
