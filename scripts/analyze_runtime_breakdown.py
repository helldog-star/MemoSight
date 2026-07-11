#!/usr/bin/env python
"""Analyze the MTP decode-time runtime breakdown: prediction / verification /
compression / other.

Consumes the inference jsonl produced by LightThinker/inference.py when run with
`--spec_decode true --profile_breakdown true`. Each record then carries a
`mtp_stats.runtime_breakdown` block with GPU-synchronized per-phase seconds:

    t_predict   main forward + draft-token sampling
    t_verify    verify forward + verify sampling + KV trims
    t_compress  compression-branch cache/input-ids reduction
    t_total     whole decode-step body

`other = t_total - (t_predict + t_verify + t_compress)` captures mask
construction, register bookkeeping and Python overhead. Seconds are POOLED
across all samples, then turned into fractions (correct pooling: sum then
divide) — so long samples weigh more, which is what you want for a
"where does decode time go" figure.

WARNING: profiling inserts torch.cuda.synchronize() around each phase, so a
profiled run's tokens/s is NOT a valid throughput number. Run profiling
separately from the end-to-end speed measurement.

Usage:
  # one or more result files / globs
  python scripts/analyze_runtime_breakdown.py results/dl2/**/*.jsonl

  # compare across datasets (or draft lengths), emit table + stacked-bar plot
  python scripts/analyze_runtime_breakdown.py \
      --group gsm8k=results/dl2/gsm8k/*.jsonl \
      --group bbh=results/dl2/bbh/*.jsonl \
      --plot runtime_breakdown.png --csv runtime_breakdown.csv
"""
import os
import csv
import glob
import json
import argparse

PHASES = ("predict", "verify", "compress", "other")


def _iter_records(paths):
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def aggregate(paths):
    """Pool per-phase seconds across all samples. Returns None if no record
    carries a runtime_breakdown (i.e. not run with --profile_breakdown)."""
    agg = dict(
        n_samples=0, n_prof_samples=0,
        t_predict=0.0, t_verify=0.0, t_compress=0.0, t_total=0.0,
        decode_steps=0, comp_steps=0, forward_passes=0,
        committed_tokens=0, output_len=0, infer_time=0.0,
    )
    for rec in _iter_records(paths):
        agg["n_samples"] += 1
        if "output_len" in rec:
            agg["output_len"] += int(rec["output_len"])
        if "infer_time" in rec:
            agg["infer_time"] += float(rec["infer_time"])
        st = rec.get("mtp_stats") or {}
        rb = st.get("runtime_breakdown")
        if not rb:
            continue
        agg["n_prof_samples"] += 1
        for k in ("t_predict", "t_verify", "t_compress", "t_total"):
            agg[k] += float(rb.get(k, 0.0))
        for k in ("decode_steps", "comp_steps", "forward_passes",
                  "committed_tokens"):
            agg[k] += int(st.get(k, 0))
    if agg["n_prof_samples"] == 0:
        return None
    return agg


def derive(agg):
    total = agg["t_total"]
    other = max(0.0, total - agg["t_predict"] - agg["t_verify"] - agg["t_compress"])
    denom = total if total > 0 else 1.0
    secs = dict(
        predict=agg["t_predict"], verify=agg["t_verify"],
        compress=agg["t_compress"], other=other,
    )
    frac = {p: secs[p] / denom for p in PHASES}
    steps = max(agg["decode_steps"], 1)
    return dict(
        n_samples=agg["n_samples"],
        n_prof_samples=agg["n_prof_samples"],
        decode_steps=agg["decode_steps"],
        comp_steps=agg["comp_steps"],
        comp_step_frac=agg["comp_steps"] / steps,
        forward_passes=agg["forward_passes"],
        committed_tokens=agg["committed_tokens"],
        t_total=total,
        seconds=secs,
        fraction=frac,
        ms_per_step={p: 1000.0 * secs[p] / steps for p in PHASES},
    )


def print_report(name, m):
    print(f"\n{'='*64}\n[{name}]  samples={m['n_samples']} "
          f"(profiled={m['n_prof_samples']})  steps={m['decode_steps']} "
          f"(comp {m['comp_step_frac']*100:.1f}%)\n{'='*64}")
    print(f"  total decode time (summed, GPU-synced): {m['t_total']:.2f} s")
    print(f"  {'phase':<10}{'seconds':>12}{'share':>9}{'ms/step':>10}")
    for p in PHASES:
        bar = "#" * int(round(m["fraction"][p] * 40))
        print(f"  {p:<10}{m['seconds'][p]:>12.2f}{m['fraction'][p]*100:>8.1f}%"
              f"{m['ms_per_step'][p]:>10.3f}  {bar}")


def write_csv(path, rows):
    fields = ["group", "n_samples", "decode_steps", "comp_step_frac",
              "t_total", "predict_s", "verify_s", "compress_s", "other_s",
              "predict_pct", "verify_pct", "compress_pct", "other_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, m in rows:
            w.writerow({
                "group": name,
                "n_samples": m["n_samples"],
                "decode_steps": m["decode_steps"],
                "comp_step_frac": round(m["comp_step_frac"], 4),
                "t_total": round(m["t_total"], 3),
                "predict_s": round(m["seconds"]["predict"], 3),
                "verify_s": round(m["seconds"]["verify"], 3),
                "compress_s": round(m["seconds"]["compress"], 3),
                "other_s": round(m["seconds"]["other"], 3),
                "predict_pct": round(m["fraction"]["predict"] * 100, 2),
                "verify_pct": round(m["fraction"]["verify"] * 100, 2),
                "compress_pct": round(m["fraction"]["compress"] * 100, 2),
                "other_pct": round(m["fraction"]["other"] * 100, 2),
            })
    print(f"\n[csv] wrote {path}")


def plot(path, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib unavailable, skipping")
        return
    names = [n for n, _ in rows]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(names) + 2), 4.2))
    bottom = [0.0] * len(names)
    colors = {"predict": "#4C78A8", "verify": "#F58518",
              "compress": "#54A24B", "other": "#B0B0B0"}
    for p in PHASES:
        vals = [m["fraction"][p] * 100 for _, m in rows]
        ax.bar(list(x), vals, bottom=bottom, label=p, color=colors[p])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("share of decode time (%)")
    ax.set_ylim(0, 100)
    ax.set_title("MTP decode-time breakdown")
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {path}")


def _expand(patterns):
    out = []
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        out.extend(hits if hits else ([p] if os.path.isfile(p) else []))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="result jsonl files/globs")
    ap.add_argument("--group", action="append", default=[], metavar="NAME=GLOB",
                    help="named group of jsonls; repeatable for comparison")
    ap.add_argument("--plot", default=None, help="output png for stacked-bar plot")
    ap.add_argument("--csv", default=None, help="output csv summary")
    ap.add_argument("--json", default=None, help="dump derived metrics as json")
    args = ap.parse_args()

    groups = []
    for g in args.group:
        assert "=" in g, f"--group expects NAME=GLOB, got {g!r}"
        name, pat = g.split("=", 1)
        groups.append((name, _expand([pat])))
    if args.paths:
        groups.append(("all", _expand(args.paths)))
    if not groups:
        ap.error("provide result paths or --group NAME=GLOB")

    rows, all_metrics = [], {}
    for name, paths in groups:
        agg = aggregate(paths)
        if agg is None:
            print(f"[warn] group '{name}': no runtime_breakdown found "
                  f"(run with --profile_breakdown true?) — {len(paths)} files")
            continue
        m = derive(agg)
        print_report(name, m)
        rows.append((name, m))
        all_metrics[name] = m

    if not rows:
        return
    if len(rows) > 1:
        print(f"\n{'='*64}\nSUMMARY (share of decode time)\n{'='*64}")
        print(f"{'group':<12}{'predict':>9}{'verify':>9}{'compress':>10}{'other':>8}")
        for name, m in rows:
            fr = m["fraction"]
            print(f"{name:<12}{fr['predict']*100:>8.1f}%{fr['verify']*100:>8.1f}%"
                  f"{fr['compress']*100:>9.1f}%{fr['other']*100:>7.1f}%")
    if args.csv:
        write_csv(args.csv, rows)
    if args.plot:
        plot(args.plot, rows)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    main()
