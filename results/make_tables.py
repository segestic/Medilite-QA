"""
make_tables.py  --  all latency / memory / energy (+ optional accuracy) tables
==============================================================================
Point it at your folders; it searches them RECURSIVELY, so the layout inside
does not matter, and partial results are fine (missing items are listed).

  benchmarks/  ... benchmark_summary_<model>.json      (run_full_benchmark.py)
  energy/      ... multi_prompt_energy_<model>.json    (run_multi_prompt_energy.py)
               ... run_metadata.json                    (optional)
  accuracy/    ... accuracy_ci.csv, avg_difference_bootstrap.csv   (optional;
                   the folder holding stats_out/ and stats_out_ablation/)

Usage:
    python3 make_tables.py --bench benchmarks --energy energy
    python3 make_tables.py --bench benchmarks --energy energy --acc accuracy \
                           --script run_full_benchmark.py
If a model has several files, the NEWEST is used and the others are listed.

Writes tables/*.csv, *.md, *.tex and tables/CHECKS.txt  (paste CHECKS.txt +
tables/ALL_TABLES.md back).
"""

import argparse
import ast
import csv
import datetime as dt
import glob
import json
import os
import random
import statistics as st

KWH_J = 3.6e6
N_BOOT, SEED = 10000, 20261007
MAX_TOKENS = 256
FT_ENERGY_KWH = 138.75
REF = "MediLITE-QA"

CANON = {
    "Medilite-QA": "MediLITE-QA", "MediLITE-QA": "MediLITE-QA", "MediLITE-Q4": "MediLITE-QA",
    "Phi-3.5 Mini (base)": "Phi-3.5-mini (base)", "Phi-3.5_Mini_base": "Phi-3.5-mini (base)",
    "Phi-3.5 base": "Phi-3.5-mini (base)", "Base-Q4": "Phi-3.5-mini (base)",
    "OpenBio": "OpenBioLLM-8B", "OpenBioLLM-8B": "OpenBioLLM-8B",
    "BioMistral-7B": "BioMistral-7B", "LLaMA 3 8B": "LLaMA 3 8B", "LLaMA_3_8B": "LLaMA 3 8B",
    "Gemma 2B": "Gemma 2B", "Gemma_2B": "Gemma 2B", "Apollo-7B": "Apollo-7B",
    "AlpaCare-llama2-7b": "AlpaCare-7B", "ClinicalGPT": "ClinicalGPT",
}
ORDER = ["MediLITE-QA", "Phi-3.5-mini (base)", "OpenBioLLM-8B", "LLaMA 3 8B", "BioMistral-7B",
         "Apollo-7B", "AlpaCare-7B", "ClinicalGPT", "Gemma 2B"]
WRONG_TEMPLATES = {"OpenBio": {"chatml"}, "BioMistral-7B": {"chatml"}}


def canon(n):
    return CANON.get(n.strip(), n.strip())


def fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def mtime(p):
    return dt.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")


def boot(values, stat=st.mean):
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = random.Random(SEED)
    b = sorted(stat(rng.choices(values, k=len(values))) for _ in range(N_BOOT))
    return b[int(0.025 * N_BOOT)], b[int(0.975 * N_BOOT) - 1]


def newest_by_model(paths, prefix, checks, kind):
    found = {}
    for p in paths:
        label = os.path.basename(p)[len(prefix):-len(".json")]
        found.setdefault(canon(label), []).append(p)
    chosen = {}
    for m, ps in found.items():
        ps.sort(key=os.path.getmtime, reverse=True)
        chosen[m] = ps[0]
        if len(ps) > 1:
            checks.append(f"  [{kind}] {m}: {len(ps)} files, using newest {ps[0]} ({mtime(ps[0])}); "
                          f"ignored: {', '.join(ps[1:])}")
    return chosen


def write_table(out, name, header, rows, nd_map=None):
    with open(os.path.join(out, name + ".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    esc = lambda s: str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("±", r"$\pm$")
    tex = [r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\toprule",
           " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, name + ".tex"), "w").write("\n".join(tex) + "\n")
    return f"### {name}\n\n" + "\n".join(md) + "\n"


def order_key(m):
    return ORDER.index(m) if m in ORDER else len(ORDER)


# ======================================================================= ENERGY
def energy_tables(files, out, checks):
    sections = []
    per_model = {}
    for m, p in files.items():
        d = json.load(open(p))
        if "modes" not in d:
            checks.append(f"  [energy] {m}: {p} is from the OLD script (no per-run data) -- skipped")
            continue
        per_model[m] = d
    if not per_model:
        return sections, {}

    summary = {}
    for mode in ("fixed", "natural"):
        rows = []
        for m in sorted(per_model, key=order_key):
            runs = per_model[m]["modes"].get(mode, {}).get("runs", [])
            if not runs:
                continue
            by_p, by_p_resp = {}, {}
            for r in runs:
                by_p.setdefault(r["prompt_idx"], []).append(r["j_per_token_cpu"])
                by_p_resp.setdefault(r["prompt_idx"], []).append(r["energy_cpu_kwh"] * KWH_J)
            pp = [st.mean(v) for v in by_p.values()]
            pr = [st.mean(v) for v in by_p_resp.values()]
            jt = st.mean(r["j_per_token_cpu"] for r in runs)
            lo, hi = boot(pp)
            resp = st.mean(pr)
            rlo, rhi = boot(pr)
            toks = [r["completion_tokens"] for r in runs]
            pw = [r["cpu_power_w"] for r in runs]
            pct_max = 100 * sum(t == MAX_TOKENS for t in toks) / len(toks)
            idle = per_model[m].get("idle_baseline", {}).get("idle_cpu_power_w")
            summary[(m, mode)] = dict(jt=jt, resp=resp, pp=by_p, pct_max=pct_max)
            rows.append([m, len(runs), fmt(st.mean(toks), 0), fmt(pct_max, 0),
                         f"{fmt(jt, 3)} [{fmt(lo, 3)}, {fmt(hi, 3)}]", fmt(st.median(r['j_per_token_cpu'] for r in runs), 3),
                         fmt(st.mean(r['j_per_token_cpu_idle_sub'] for r in runs), 3),
                         f"{fmt(resp, 0)} [{fmt(rlo, 0)}, {fmt(rhi, 0)}]",
                         f"{fmt(st.mean(pw), 1)} ± {fmt(st.stdev(pw) if len(pw) > 1 else 0, 1)}",
                         fmt(idle, 1), fmt(st.mean(t / r['duration_s'] for t, r in zip(toks, runs)), 2)])
            if mode == "fixed" and pct_max < 100:
                checks.append(f"  [energy] {m}: only {pct_max:.0f}% of fixed-length runs reached {MAX_TOKENS} tokens")
        hdr = ["Model", "Runs", "Mean tokens", "% at 256", "J/token [95% CI]", "Median J/token",
               "Idle-sub J/token", "J/response [95% CI]", "CPU power W (mean ± SD)", "Idle W", "tok/s"]
        sections.append(write_table(out, f"energy_{mode}", hdr, rows))

    # paired ratios vs reference
    if (REF, "fixed") in summary:
        rows = []
        for mode in ("fixed", "natural"):
            ref = summary.get((REF, mode))
            if not ref:
                continue
            ref_pp = {p: st.mean(v) for p, v in ref["pp"].items()}
            for m in sorted(per_model, key=order_key):
                if m == REF or (m, mode) not in summary:
                    continue
                o_pp = {p: st.mean(v) for p, v in summary[(m, mode)]["pp"].items()}
                common = sorted(set(ref_pp) & set(o_pp))
                pairs = [(ref_pp[p], o_pp[p]) for p in common]
                ratio = lambda ps: st.mean(o for _, o in ps) / st.mean(r for r, _ in ps)
                rng = random.Random(SEED)
                b = sorted(ratio(rng.choices(pairs, k=len(pairs))) for _ in range(N_BOOT))
                rows.append([mode, m, len(common), f"{ratio(pairs):.2f}×",
                             f"[{b[int(.025 * N_BOOT)]:.2f}, {b[int(.975 * N_BOOT) - 1]:.2f}]"])
        sections.append(write_table(out, "energy_ratio_vs_medilite",
                                    ["Mode", "Model", "Prompts", "J/token ratio (model / MediLITE-QA)", "95% CI"], rows))

        # break-even
        ref_jt = summary[(REF, "fixed")]["jt"]
        rows = []
        for m in sorted(per_model, key=order_key):
            if m == REF or (m, "fixed") not in summary:
                continue
            save = (summary[(m, "fixed")]["jt"] - ref_jt) * MAX_TOKENS
            q = FT_ENERGY_KWH * KWH_J / save if save > 0 else None
            rows.append([m, fmt(save, 1), f"{q:,.0f}" if q else "never"])
        sections.append(write_table(out, "finetune_break_even",
                                    ["Model", "Saving per 256-token response (J)",
                                     f"Responses to recover {FT_ENERGY_KWH} kWh"], rows))
    else:
        checks.append(f"  [energy] reference {REF} missing -- ratios and break-even skipped")
    return sections, summary


# ================================================================== BENCHMARKS
def bench_tables(files, out, checks):
    sections, lat_rows, mem_rows, thr_rows = [], [], [], []
    ctx_all = set()
    data = {}
    for m, p in files.items():
        d = json.load(open(p))
        data[m] = d
        ctx_all |= set(d.get("memory_context_metrics", {}))
    ctx_sorted = sorted(ctx_all, key=int)
    for m in sorted(data, key=order_key):
        d = data[m]
        sl = d.get("streaming_latency_metrics", {})
        warm = sl.get("mean_ttft_warm_only", {})
        cold = sl.get("cold_start_ttft_s")
        tp = d.get("throughput_metrics", {})
        b1 = tp.get("batch_1", {}).get("tokens_per_sec", {})
        mem = d.get("memory_context_metrics", {})
        ram_gb = d.get("system_info", {}).get("memory_total_gb")
        peak_ctx = max(mem, key=int) if mem else None
        peak_gb = mem[peak_ctx]["mean"] / 1024 if peak_ctx else None
        n = d.get("n_runs")
        lat_rows.append([m, fmt(cold), f"{fmt(warm.get('mean'))} ± {fmt(warm.get('stdev'))}",
                         fmt(cold / warm["mean"] if cold and warm.get("mean") else None, 1),
                         fmt((sl.get("p50_tpot", {}).get("mean") or 0) * 1000 or None, 0),
                         f"{fmt(b1.get('mean'))} ± {fmt(b1.get('stdev'))}",
                         f"{fmt(peak_gb)} ({fmt(100 * peak_gb / ram_gb if peak_gb and ram_gb else None, 0)}%)",
                         f"{n} ({(n or 1) - 1} warm)"])
        mem_rows.append([m] + [f"{fmt(mem[c]['mean'] / 1024)} ± {fmt(mem[c]['stdev'] / 1024)}"
                               if c in mem else "--" for c in ctx_sorted])
        thr_rows.append([m] + [fmt(tp[k]["tokens_per_sec"]["mean"])
                               for k in sorted(tp, key=lambda k: int(k.split("_")[1]))])
        checks.append(f"  [bench] {m:<22} {files[m]}  modified {mtime(files[m])}")
    sections.append(write_table(out, "latency_memory",
                                ["Model", "Cold-start TTFT (s, n=1)", "Warm TTFT (s, mean ± SD)",
                                 "Cold/warm", "Warm TPOT (ms)", "Throughput (tok/s, mean ± SD)",
                                 "Peak RSS GB (% RAM)", "Runs"], lat_rows))
    sections.append(write_table(out, "memory_vs_context",
                                ["Model"] + [f"ctx {c} (GB)" for c in ctx_sorted], mem_rows))
    if thr_rows:
        ks = sorted(next(iter(data.values())).get("throughput_metrics", {}), key=lambda k: int(k.split("_")[1]))
        sections.append(write_table(out, "sequential_throughput",
                                    ["Model"] + [f"{k.replace('batch_', '')} sequential req. (tok/s)" for k in ks],
                                    thr_rows))
    checks.append("  [bench] NOTE: N_RUNS=3 -> warm values rest on 2 runs, cold start on 1; "
                  "do not report p90/p99. 'batch' = sequential requests, not batching.")
    return sections, data


# ==================================================================== ACCURACY
def accuracy_table(acc_root, out, checks):
    if not acc_root or not os.path.isdir(acc_root):
        return []
    acc_files = glob.glob(os.path.join(acc_root, "**", "accuracy_ci.csv"), recursive=True)
    diff_files = glob.glob(os.path.join(acc_root, "**", "avg_difference_bootstrap.csv"), recursive=True)
    acc, q4, diff = {}, {}, {}
    for f in acc_files:
        for r in csv.DictReader(open(f)):
            if r["model"] in ("MediLITE-Q4", "Base-Q4"):
                q4[canon(r["model"])] = r
            elif r["model"] not in ("MediLITE-fp16", "Base-fp16"):
                acc[canon(r["model"])] = r
    for f in diff_files:
        for r in csv.DictReader(open(f)):
            a, b = r["comparison"].split(" vs ", 1)
            if canon(a) == REF and b not in ("Base-Q4", "MediLITE-fp16", "Base-fp16"):
                diff[canon(b)] = r
    if not acc:
        checks.append("  [acc] no accuracy_ci.csv found under " + acc_root)
        return []
    rows = []
    for m in sorted(acc, key=order_key):
        a = acc[m]
        d = diff.get(m)
        rows.append([m, f"{fmt(a['avg'])} [{fmt(a['avg_lo'])}, {fmt(a['avg_hi'])}]",
                     f"{fmt(q4[m]['avg'])} [{fmt(q4[m]['avg_lo'])}, {fmt(q4[m]['avg_hi'])}]" if m in q4 else "--",
                     fmt(a.get("avg_noPQA")),
                     "ref" if m == REF else (f"{-float(d['avg_diff_pp']):+.2f} [{-float(d['ci_hi']):+.2f}, "
                                             f"{-float(d['ci_lo']):+.2f}]" + ("" if d["excludes_zero"] == "yes" else " ns")
                                             if d else "--")])
    n_cmp = len(diff)
    if n_cmp < 5:
        checks.append(f"  [acc] only {n_cmp} model comparisons found -- is this the 6-model stats_out?")
    return [write_table(out, "accuracy", ["Model", "Accuracy fp16 % [95% CI]", "Accuracy Q4_K_M % [95% CI]",
                                          "Avg excl. PubMedQA", "Model − MediLITE-QA (pp) [95% CI]"], rows)]


# ======================================================================== MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="benchmarks")
    ap.add_argument("--energy", default="energy")
    ap.add_argument("--acc", default=None)
    ap.add_argument("--script", default="run_full_benchmark.py")
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    checks, sections = ["=== FILES & CHECKS ==="], []

    # routing
    if os.path.exists(args.script):
        tree = ast.parse(open(args.script).read())
        routing = next((ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                        and any(getattr(t, "id", None) == "MODEL_ROUTING" for t in n.targets)), {})
        checks.append(f"--- chat templates in {args.script}")
        for lab, c in routing.items():
            bad = c.get("format") in WRONG_TEMPLATES.get(lab, set())
            checks.append(f"  {lab:<22} type={c.get('type')}  format={c.get('format')}  stop={c.get('stop')}"
                          + ("   <-- STILL WRONG" if bad else ""))
    else:
        checks.append(f"--- {args.script} not found: template check skipped (use --script)")

    sections += accuracy_table(args.acc, args.out, checks)

    bfiles = glob.glob(os.path.join(args.bench, "**", "benchmark_summary_*.json"), recursive=True)
    checks.append(f"--- benchmarks: {len(bfiles)} file(s) under {args.bench}")
    if bfiles:
        s, bdata = bench_tables(newest_by_model(bfiles, "benchmark_summary_", checks, "bench"), args.out, checks)
        sections += s
    else:
        bdata = {}

    efiles = glob.glob(os.path.join(args.energy, "**", "multi_prompt_energy_*.json"), recursive=True)
    checks.append(f"--- energy: {len(efiles)} file(s) under {args.energy}")
    edata = {}
    if efiles:
        s, edata = energy_tables(newest_by_model(efiles, "multi_prompt_energy_", checks, "energy"), args.out, checks)
        sections += s
    metas = glob.glob(os.path.join(args.energy, "**", "run_metadata.json"), recursive=True)
    if metas:
        meta = json.load(open(max(metas, key=os.path.getmtime)))
        checks.append("--- energy run settings: " + json.dumps(
            {k: meta.get(k) for k in ("cpu_brand", "llama_cpp_python_version", "codecarbon_version",
                                      "n_threads", "n_ctx", "max_new_tokens", "temperature",
                                      "library_sampling_defaults")}, default=str))

    checks.append("--- coverage (A=accuracy file not checked here, B=benchmark, E=energy)")
    have_e = {m for m, _ in edata}
    for m in ORDER:
        checks.append(f"  {m:<22} B:{'yes' if m in bdata else 'MISSING'}  E:{'yes' if m in have_e else 'MISSING'}")

    text = "\n".join(checks)
    open(os.path.join(args.out, "CHECKS.txt"), "w").write(text + "\n")
    open(os.path.join(args.out, "ALL_TABLES.md"), "w").write("\n".join(sections))
    print(text + "\n\n" + "\n".join(sections))
    print(f"\nwritten to {args.out}/ -- paste CHECKS.txt and ALL_TABLES.md back")


if __name__ == "__main__":
    main()
