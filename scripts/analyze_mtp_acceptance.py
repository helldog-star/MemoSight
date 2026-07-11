#!/usr/bin/env python
"""Analyze MTP self-speculative-decoding acceptance rate.

Consumes the inference output jsonl files produced by LightThinker/inference.py
when run with `--spec_decode true`. Each record carries a per-sample `mtp_stats`
block (see MTPStats.summary in inference.py). This script re-aggregates the raw
counters across all samples (correct pooling: sum counts, then divide) and
reports the metrics that matter for speculative decoding:

  * mean accepted length  tau   = committed_tokens / decode_steps
        (equivalently the average number of tokens emitted per outer step)
  * overall acceptance    alpha = spec_accepted / spec_proposed  in [0, 1]
  * per-position alpha_k   (both unconditional over verify steps and
        conditional given the position was reached)
  * tokens / forward pass         (compute-bound speedup proxy vs 1.0 baseline)
  * measured decode throughput    tokens / s  (from infer_time & output_len)
  * histogram of committed tokens per step

Usage:
  # analyze one or more result files / globs (draft_len read from records)
  python scripts/analyze_mtp_acceptance.py results/dl2/gsm8k/*.jsonl

  # group several draft-length sweeps into one comparison table + plot
  python scripts/analyze_mtp_acceptance.py \
      --group dl1=results/dl1/**/*.jsonl \
      --group dl2=results/dl2/**/*.jsonl \
      --group dl3=results/dl3/**/*.jsonl \
      --plot mtp_acceptance.png --csv mtp_acceptance.csv
"""
import os
import csv
import glob
import json
import argparse
from collections import defaultdict


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
    """Pool raw counters across all samples in `paths`. Returns None if no
    record carries mtp_stats (i.e. the run was not spec_decode)."""
    agg = dict(
        n_samples=0,
        n_spec_samples=0,
        decode_steps=0,
        committed_tokens=0,
        forward_passes=0,
        verify_steps=0,
        control_steps=0,
        spec_proposed=0,
        spec_accepted=0,
        # end-to-end wall clock (from all samples that report them)
        infer_time=0.0,
        output_len=0,
        # correctness, so accuracy isn't silently lost while chasing speed
        n_correct=0,
    )
    pos_reached = defaultdict(int)
    pos_accepted = defaultdict(int)
    accept_hist = defaultdict(int)
    draft_lens = set()

    for rec in _iter_records(paths):
        agg["n_samples"] += 1
        if rec.get("acc_state") is True:
            agg["n_correct"] += 1
        if "infer_time" in rec:
            agg["infer_time"] += float(rec["infer_time"])
        if "output_len" in rec:
            agg["output_len"] += int(rec["output_len"])

        st = rec.get("mtp_stats")
        if not st:
            continue
        agg["n_spec_samples"] += 1
        if "mtp_draft_len" in rec:
            draft_lens.add(int(rec["mtp_draft_len"]))
        for k in ("decode_steps", "committed_tokens", "forward_passes",
                  "verify_steps", "control_steps", "spec_proposed",
                  "spec_accepted"):
            agg[k] += int(st.get(k, 0))
        # exact re-aggregation from raw per-position counts emitted by
        # MTPStats.summary (json keys come back as strings)
        for k_str, cnt in (st.get("pos_reached", {}) or {}).items():
            pos_reached[int(k_str)] += int(cnt)
        for k_str, cnt in (st.get("pos_accepted", {}) or {}).items():
            pos_accepted[int(k_str)] += int(cnt)
        for c_str, freq in (st.get("accept_hist", {}) or {}).items():
            accept_hist[int(c_str)] += int(freq)

    if agg["n_spec_samples"] == 0:
        return None

    agg["pos_reached"] = dict(pos_reached)
    agg["pos_accepted"] = dict(pos_accepted)
    agg["accept_hist"] = dict(accept_hist)
    agg["draft_lens"] = sorted(draft_lens)
    return agg


def derive(agg):
    """Turn pooled counters into the reported metrics."""
    steps = max(agg["decode_steps"], 1)
    vsteps = max(agg["verify_steps"], 1)
    fwd = max(agg["forward_passes"], 1)
    out = dict(
        draft_len=(agg["draft_lens"][0] if len(agg["draft_lens"]) == 1
                   else agg["draft_lens"]),
        n_samples=agg["n_samples"],
        accuracy=(agg["n_correct"] / agg["n_samples"] if agg["n_samples"] else 0.0),
        decode_steps=agg["decode_steps"],
        committed_tokens=agg["committed_tokens"],
        forward_passes=agg["forward_passes"],
        verify_steps=agg["verify_steps"],
        control_steps=agg["control_steps"],
        mean_accept_len=agg["committed_tokens"] / steps,
        overall_accept_rate=(agg["spec_accepted"] / agg["spec_proposed"]
                             if agg["spec_proposed"] > 0 else 0.0),
        tokens_per_forward=agg["committed_tokens"] / fwd,
        # theoretical speedup vs single-token AR (which is 1 tok / forward)
        theoretical_speedup=agg["committed_tokens"] / fwd,
        tokens_per_sec=(agg["output_len"] / agg["infer_time"]
                        if agg["infer_time"] > 0 else 0.0),
    )
    gammas = sorted(agg["pos_reached"].keys())
    out["per_position_accept_uncond"] = {
        k: agg["pos_accepted"].get(k, 0) / vsteps for k in gammas
    }
    out["per_position_accept_cond"] = {
        k: (agg["pos_accepted"].get(k, 0) / agg["pos_reached"][k]
            if agg["pos_reached"][k] > 0 else 0.0)
        for k in gammas
    }
    total_steps = max(sum(agg["accept_hist"].values()), 1)
    out["accept_hist_frac"] = {
        k: agg["accept_hist"][k] / total_steps
        for k in sorted(agg["accept_hist"].keys())
    }
    return out


def print_report(name, m):
    print(f"\n{'='*64}\n[{name}]  draft_len={m['draft_len']}  "
          f"samples={m['n_samples']}  accuracy={m['accuracy']:.4f}\n{'='*64}")
    print(f"  decode steps          : {m['decode_steps']}")
    print(f"  committed tokens      : {m['committed_tokens']}")
    print(f"  forward passes        : {m['forward_passes']}")
    print(f"  verify steps          : {m['verify_steps']}  "
          f"(control-only steps: {m['control_steps']})")
    print(f"  mean accept length tau: {m['mean_accept_len']:.3f} tokens/step")
    print(f"  overall accept rate a : {m['overall_accept_rate']:.4f}")
    print(f"  tokens / forward      : {m['tokens_per_forward']:.3f}  "
          f"(theoretical speedup x{m['theoretical_speedup']:.2f} vs AR)")
    if m["tokens_per_sec"] > 0:
        print(f"  measured throughput   : {m['tokens_per_sec']:.1f} tokens/s")
    print("  per-position accept rate a_k (k = draft position, 1-indexed):")
    for k in sorted(m["per_position_accept_uncond"].keys()):
        print(f"      pos {k+1}: uncond={m['per_position_accept_uncond'][k]:.4f}  "
              f"cond={m['per_position_accept_cond'][k]:.4f}")
    print("  committed-tokens-per-step histogram:")
    for c in sorted(m["accept_hist_frac"].keys()):
        bar = "#" * int(round(m["accept_hist_frac"][c] * 40))
        print(f"      {c:>2} tok: {m['accept_hist_frac'][c]*100:5.1f}%  {bar}")


def write_csv(path, rows):
    fields = ["group", "draft_len", "n_samples", "accuracy",
              "mean_accept_len", "overall_accept_rate", "tokens_per_forward",
              "theoretical_speedup", "tokens_per_sec", "verify_steps",
              "control_steps"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, m in rows:
            w.writerow({
                "group": name,
                "draft_len": m["draft_len"],
                "n_samples": m["n_samples"],
                "accuracy": round(m["accuracy"], 4),
                "mean_accept_len": round(m["mean_accept_len"], 4),
                "overall_accept_rate": round(m["overall_accept_rate"], 4),
                "tokens_per_forward": round(m["tokens_per_forward"], 4),
                "theoretical_speedup": round(m["theoretical_speedup"], 4),
                "tokens_per_sec": round(m["tokens_per_sec"], 2),
                "verify_steps": m["verify_steps"],
                "control_steps": m["control_steps"],
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: per-position acceptance curves
    for name, m in rows:
        ks = sorted(m["per_position_accept_uncond"].keys())
        ax0_x = [k + 1 for k in ks]
        axes[0].plot(ax0_x, [m["per_position_accept_uncond"][k] for k in ks],
                     marker="o", label=name)
    axes[0].set_xlabel("draft position k")
    axes[0].set_ylabel("acceptance rate (unconditional)")
    axes[0].set_title("Per-position MTP acceptance")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    # right: mean accept length & speedup vs draft_len
    names = [n for n, _ in rows]
    tau = [m["mean_accept_len"] for _, m in rows]
    spd = [m["theoretical_speedup"] for _, m in rows]
    x = range(len(names))
    axes[1].bar([i - 0.2 for i in x], tau, width=0.4, label="mean accept len")
    axes[1].bar([i + 0.2 for i in x], spd, width=0.4, label="tokens/forward")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_title("Accept length & speedup proxy")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend()
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
    ap.add_argument("paths", nargs="*",
                    help="result jsonl files/globs (single ungrouped analysis)")
    ap.add_argument("--group", action="append", default=[],
                    metavar="NAME=GLOB",
                    help="named group of jsonls; repeatable for comparison")
    ap.add_argument("--plot", default=None, help="output png for comparison plot")
    ap.add_argument("--csv", default=None, help="output csv summary")
    ap.add_argument("--json", default=None, help="dump full derived metrics as json")
    args = ap.parse_args()

    groups = []
    if args.group:
        for g in args.group:
            assert "=" in g, f"--group expects NAME=GLOB, got {g!r}"
            name, pat = g.split("=", 1)
            groups.append((name, _expand([pat])))
    if args.paths:
        groups.append(("all", _expand(args.paths)))
    if not groups:
        ap.error("provide result paths or --group NAME=GLOB")

    rows = []
    all_metrics = {}
    for name, paths in groups:
        agg = aggregate(paths)
        if agg is None:
            print(f"[warn] group '{name}': no mtp_stats found "
                  f"(was it run with --spec_decode true?) — {len(paths)} files")
            continue
        m = derive(agg)
        print_report(name, m)
        rows.append((name, m))
        all_metrics[name] = m

    if not rows:
        return
    if len(rows) > 1:
        print(f"\n{'='*64}\nSUMMARY\n{'='*64}")
        print(f"{'group':<12}{'draft':>6}{'tau':>8}{'alpha':>8}"
              f"{'tok/fwd':>9}{'acc':>8}")
        for name, m in rows:
            dl = m["draft_len"]
            print(f"{name:<12}{str(dl):>6}{m['mean_accept_len']:>8.3f}"
                  f"{m['overall_accept_rate']:>8.3f}{m['tokens_per_forward']:>9.3f}"
                  f"{m['accuracy']:>8.3f}")
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
