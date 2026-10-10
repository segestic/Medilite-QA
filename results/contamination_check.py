"""
contamination_check.py  --  R2-2 (and R3-2 / R3-3 dataset bias, deduplication)
=============================================================================
Word n-gram overlap between the fine-tuning corpus and the EXACT items that
lm-eval scored. Test items are read from the `doc` field of the
samples_*.jsonl logs, so there is no ambiguity about splits.

Method (report this in Section 2.4):
  * normalisation: Unicode NFKC, lower-case, punctuation -> space, collapse spaces
  * 13-gram matching; per item, TOKEN COVERAGE = share of its words inside
    any 13-gram also found in training (Llama-2-style). dirty >= 80% coverage
    (or full question-stem duplicate) -> contaminated_items.json;
    not clean >= 20% -> not_clean_items.json (sensitivity analysis).
    "any 13-gram" is reported too but over-flags templated exam vignettes.
  * secondary: % of each item's 8-grams found in training (graded measure)
  * question-stem duplicate: every n-gram of the question stem occurs in training
  * the 25 most frequent matching 13-grams are listed for MANUAL inspection:
    generic exam phrasing ("which of the following is the most likely ...")
    can cause false positives and must be judged by a human.

Training corpus input (repeatable --train):
    --train hf:<dataset_id>[:<config>][@<split>]     e.g. hf:segestic/med_train@train
    --train path/to/file.jsonl|.json|.csv|.parquet
All string fields are used unless --train-fields is given.

Usage:
    python contamination_check.py --train hf:<your_training_dataset>@train \
        --eval-logs lm_eval_logs/medilite --out contamination_out
Outputs:
    contamination_out/summary.csv            per-benchmark rates
    contamination_out/contaminated_items.json  {task: [doc_id,...]}  -> stats_lm_eval.py --exclude
    contamination_out/top_ngrams.txt         for manual review
"""

import argparse
import csv
import glob
import json
import os
import re
import unicodedata
from collections import Counter

import numpy as np

N_MAIN, N_SEC = 13, 8
CLEAN, DIRTY = 0.20, 0.80   # Llama-2-style token-coverage thresholds
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalise(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()


def ngrams(words, n):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def strings_in(obj, fields=None):
    """All string values in a (possibly nested) record."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if fields is None or k in fields:
                yield from strings_in(v, None)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from strings_in(v, None)


QUESTION_KEYS = ["question", "QUESTION", "sent1", "query"]


# ------------------------------------------------------------- eval items
def load_eval_items(log_dir):
    items = {}
    for f in glob.glob(os.path.join(log_dir, "**", "samples_*.jsonl"), recursive=True):
        task = re.sub(r"_\d{4}-\d{2}-\d{2}T.*$", "", os.path.basename(f)[len("samples_"):-len(".jsonl")])
        recs = {}
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                doc = r["doc"]
                full = normalise(" ".join(strings_in(doc)))
                q = next((doc[k] for k in QUESTION_KEYS if isinstance(doc, dict) and k in doc), "")
                recs[r["doc_id"]] = {"full": full, "question": normalise(q)}
        items[task] = recs
    if not items:
        raise SystemExit(f"no samples_*.jsonl under {log_dir}")
    return items


# ------------------------------------------------------------- training
def iter_training(spec, fields):
    if spec.startswith("hf:"):
        from datasets import load_dataset
        body = spec[3:]
        split = "train"
        if "@" in body:
            body, split = body.split("@", 1)
        name, config = (body.split(":", 1) + [None])[:2]
        ds = load_dataset(name, config, split=split)
        for r in ds:
            yield " ".join(strings_in(r, fields))
        return
    ext = os.path.splitext(spec)[1].lower()
    if ext == ".jsonl":
        with open(spec) as fh:
            for line in fh:
                if line.strip():
                    yield " ".join(strings_in(json.loads(line), fields))
    elif ext == ".json":
        with open(spec) as fh:
            data = json.load(fh)
        for r in (data if isinstance(data, list) else data.get("data", [])):
            yield " ".join(strings_in(r, fields))
    elif ext in (".csv", ".parquet"):
        import pandas as pd
        df = pd.read_csv(spec) if ext == ".csv" else pd.read_parquet(spec)
        cols = [c for c in df.columns if (fields is None or c in fields) and df[c].dtype == object]
        for row in df[cols].itertuples(index=False):
            yield " ".join(str(x) for x in row if isinstance(x, str))
    else:
        raise SystemExit(f"unsupported training file: {spec}")


def question_duplicated(question, hit13, hit8):
    """Question stem fully present in training: every 13-gram (or, for
    8-12-word stems, every 8-gram) of the stem occurs in the corpus.
    Stems under 8 words are not assessed (too generic)."""
    w = question.split()
    if len(w) >= N_MAIN:
        g = ngrams(w, N_MAIN)
        return all(x in hit13 for x in g)
    if len(w) >= N_SEC:
        return ngrams(w, N_SEC) <= hit8
    return False


# ------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--train-fields", nargs="*", default=None)
    ap.add_argument("--eval-logs", required=True, help="one model's lm-eval output dir")
    ap.add_argument("--out", default="contamination_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fields = set(args.train_fields) if args.train_fields else None

    items = load_eval_items(args.eval_logs)
    print({t: len(v) for t, v in items.items()})

    # index the (small) test side; stream the (large) training side
    main_index, sec_index, short_full = set(), set(), set()
    item_ngrams = {}
    for task, recs in items.items():
        for did, rec in recs.items():
            w = rec["full"].split()
            g13, g8 = ngrams(w, N_MAIN), ngrams(w, N_SEC)
            item_ngrams[(task, did)] = (g13, g8)
            main_index |= g13
            sec_index |= g8
            if len(w) < N_MAIN:
                short_full.add(rec["full"])

    hit13, hit8, hit_short = Counter(), set(), set()
    n_docs = n_words = 0
    for spec in args.train:
        print(f"streaming training corpus {spec} ...")
        for text in iter_training(spec, fields):
            t = normalise(text)
            w = t.split()
            n_docs += 1
            n_words += len(w)
            for g in ngrams(w, N_MAIN):
                if g in main_index:
                    hit13[g] += 1
            hit8 |= ngrams(w, N_SEC) & sec_index
            if t in short_full:
                hit_short.add(t)
            if n_docs % 20000 == 0:
                print(f"  {n_docs} training records")
    print(f"training corpus: {n_docs} records, {n_words} words")

    rows, flagged, not_clean = [], {}, {}
    for task, recs in items.items():
        f13 = fq = n_dirty = n_notclean = 0
        sec_rates, covs, dirty_ids, nc_ids = [], [], [], []
        for did, rec in recs.items():
            g13, g8 = item_ngrams[(task, did)]
            w = rec["full"].split()
            is13 = any(g in hit13 for g in g13) or (not g13 and rec["full"] in hit_short)
            isq = question_duplicated(rec["question"], hit13, hit8)
            # token coverage: share of the item's words inside any matched 13-gram
            covered = np.zeros(len(w), dtype=bool)
            for i in range(len(w) - N_MAIN + 1):
                if " ".join(w[i:i + N_MAIN]) in hit13:
                    covered[i:i + N_MAIN] = True
            if not g13 and rec["full"] in hit_short:
                covered[:] = True
            cov = covered.mean() if len(w) else 0.0
            covs.append(cov)
            f13 += is13
            fq += isq
            if g8:
                sec_rates.append(len(g8 & hit8) / len(g8))
            if cov >= DIRTY or isq:
                dirty_ids.append(did)
            if cov >= CLEAN or isq:
                nc_ids.append(did)
        flagged[task], not_clean[task] = sorted(dirty_ids), sorted(nc_ids)
        n = len(recs)
        rows.append(dict(benchmark=task, n_items=n,
                         any_13gram=f13, pct_any_13gram=100 * f13 / n,
                         mean_pct_tokens_covered=100 * float(np.mean(covs)),
                         pct_not_clean_cov20=100 * len(nc_ids) / n,
                         pct_dirty_cov80=100 * len(dirty_ids) / n,
                         question_stem_dup=fq, pct_question_dup=100 * fq / n,
                         mean_pct_8gram_overlap=100 * sum(sec_rates) / max(len(sec_rates), 1)))

    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "contaminated_items.json"), "w") as f:
        json.dump(flagged, f, indent=1)          # dirty: coverage >= 80% or stem duplicate
    with open(os.path.join(args.out, "not_clean_items.json"), "w") as f:
        json.dump(not_clean, f, indent=1)        # coverage >= 20% or stem duplicate
    with open(os.path.join(args.out, "top_ngrams.txt"), "w") as f:
        f.write(f"training records={n_docs}, words={n_words}\n"
                f"25 most frequent matching {N_MAIN}-grams (inspect for generic phrasing):\n\n")
        for g, c in hit13.most_common(25):
            f.write(f"{c:>7}  {g}\n")

    print(f"\n{'benchmark':<28}{'n':>6}{'any13 %':>9}{'cov %':>7}{'>=20% %':>9}{'>=80% %':>9}{'Q-dup %':>9}")
    for r in rows:
        print(f"{r['benchmark']:<28}{r['n_items']:>6}{r['pct_any_13gram']:>9.2f}"
              f"{r['mean_pct_tokens_covered']:>7.2f}{r['pct_not_clean_cov20']:>9.2f}"
              f"{r['pct_dirty_cov80']:>9.2f}{r['pct_question_dup']:>9.2f}")
    print(f"\nwritten to {args.out}/  -- read top_ngrams.txt before drawing conclusions")


if __name__ == "__main__":
    main()
