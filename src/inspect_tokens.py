#!/usr/bin/env python3
"""
inspect_tokens.py — Token distribution analysis across all data sources.

Helps decide the optimal max_seq_len for training by showing:
  - Per-source token distributions (percentiles, histogram)
  - Coverage at common context limits (1024 / 2048 / 3000 / 4096 / 8192)
  - Truncation cost: how many tokens are lost at each limit
  - VRAM impact estimate per limit
  - Final recommendation

Usage:
  python3 src/inspect_tokens.py
  python3 src/inspect_tokens.py --limit 4096   # show what fits at a specific limit
"""

import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

import tiktoken

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENCODING = "gpt-4o"
MSG_OVERHEAD = 4
REPLY_PRIMER = 2

SOURCES = [
    # ── Existing adapters ──────────────────────────────────────────────────
    ("processed/exploitdb.jsonl",          "ExploitDB",          "exploit code + reasoning (single-turn)"),
    ("raw/htb_writeups.jsonl",             "HTB",                "pentest methodology (multi-turn, 7 turns avg)"),
    ("raw/vulhub.jsonl",                   "Vulhub",             "CVE exploitation chains (1–7 turns)"),
    ("raw/attack.jsonl",                   "ATT&CK",             "red team techniques (4 turns avg)"),
    ("raw/ad.jsonl",                       "AD",                 "Active Directory attack chains"),
    ("raw/webapp.jsonl",                   "WebApp",             "web application testing chains"),
    ("raw/osint.jsonl",                    "OSINT",              "external recon methodology chains"),
    ("raw/cloud.jsonl",                    "Cloud",              "AWS/Azure/GCP attack chains"),
    # ── New adapter sources (source10 / source11 / source12) ──────────────
    ("raw/executor_correction.jsonl",      "Executor-Correction","command correction: bad cmd + stderr → fix"),
    ("raw/executor_filtering.jsonl",       "Executor-Filtering", "output filtering: noisy output → findings"),
    ("raw/analyst_h1.jsonl",               "Analyst-H1",         "HackerOne reports → structured findings"),
    ("raw/analyst_synth.jsonl",            "Analyst-Synth",      "synthetic tool output → analyst chains"),
    ("raw/planner_decomp.jsonl",           "Planner-Decomp",     "target + phase → step-by-step attack plan"),
    ("raw/planner_replan.jsonl",           "Planner-Replan",     "unexpected result → adaptive re-plan"),
    # ── Researcher adapter (source13) ─────────────────────────────────────
    ("raw/researcher_synth.jsonl",         "Researcher-Synth",   "synthetic: standard tools fail → anomaly reasoning → novel finding"),
    ("raw/researcher_pz.jsonl",            "Researcher-PZ",      "Project Zero reconstructions → hypothesis chains"),
    ("raw/researcher_ctf.jsonl",           "Researcher-CTF",     "novel CTF writeups → hypothesis chains"),
]

LIMITS = [1024, 2048, 3000, 4096, 8192]

# A100 80GB reference: BF16 forward+backward memory per token at different model sizes
# Rough estimate: activation memory ≈ 2 × seq_len × hidden × n_layers × 2 bytes
# With Unsloth gradient checkpointing: ~0.5× reduction
VRAM_PER_TOKEN_GB = {
    "14b": 0.000012,   # ~12 MB per 1000 tokens per batch item (empirical)
    "32b": 0.000028,
}
BATCH_SIZE = 4

enc = tiktoken.encoding_for_model(ENCODING)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_tokens(example: dict) -> int:
    return sum(MSG_OVERHEAD + len(enc.encode(m["content"]))
               for m in example["messages"]) + REPLY_PRIMER


def percentile(data: list[int], p: float) -> int:
    idx = max(0, min(int(len(data) * p / 100), len(data) - 1))
    return sorted(data)[idx]


def histogram(data: list[int], buckets: int = 12, width: int = 40) -> str:
    lo, hi = min(data), max(data)
    step = max(1, (hi - lo) // buckets)
    counts = defaultdict(int)
    for v in data:
        b = ((v - lo) // step) * step + lo
        counts[b] += 1
    max_count = max(counts.values())
    lines = []
    for b in sorted(counts):
        bar = "█" * int(width * counts[b] / max_count)
        lines.append(f"  {b:>5}–{b+step:<5}  {bar:<{width}}  {counts[b]}")
    return "\n".join(lines)


def truncation_loss(data: list[int], limit: int) -> tuple[int, int, float]:
    """Returns (n_truncated, total_tokens_lost, pct_info_lost)."""
    n_trunc = 0
    total_lost = 0
    total_original = sum(data)
    for t in data:
        if t > limit:
            n_trunc += 1
            total_lost += (t - limit)
    pct = 100 * total_lost / max(1, total_original)
    return n_trunc, total_lost, pct


# ---------------------------------------------------------------------------
# Per-source analysis
# ---------------------------------------------------------------------------
def analyze_source(path: str, label: str, description: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None

    with open(p) as f:
        examples = [json.loads(l) for l in f if l.strip()]

    token_counts = sorted(count_tokens(e) for e in examples)
    n = len(token_counts)
    if n == 0:
        return None

    # Per-role breakdown (first example structure)
    role_avg = defaultdict(list)
    for e in examples:
        for m in e["messages"]:
            role_avg[m["role"]].append(
                MSG_OVERHEAD + len(enc.encode(m["content"]))
            )

    return {
        "label":       label,
        "description": description,
        "n":           n,
        "tokens":      token_counts,
        "role_avg":    {r: statistics.mean(v) for r, v in role_avg.items()},
        "min":         token_counts[0],
        "p10":         percentile(token_counts, 10),
        "p25":         percentile(token_counts, 25),
        "p50":         percentile(token_counts, 50),
        "p75":         percentile(token_counts, 75),
        "p90":         percentile(token_counts, 90),
        "p95":         percentile(token_counts, 95),
        "p99":         percentile(token_counts, 99),
        "max":         token_counts[-1],
        "mean":        statistics.mean(token_counts),
        "stdev":       statistics.stdev(token_counts) if n > 1 else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",      type=int, default=None,
                        help="Show detailed breakdown for a specific token limit")
    parser.add_argument("--no-hist",    action="store_true",
                        help="Skip per-source histograms")
    args = parser.parse_args()

    W = 68
    print(f"\n{'═' * W}")
    print(f"  TOKEN DISTRIBUTION ANALYSIS — Mythos-v4 Dataset")
    print(f"{'═' * W}\n")

    analyses = []
    for path, label, desc in SOURCES:
        result = analyze_source(path, label, desc)
        if result is None:
            print(f"  [SKIP] {path} — not found\n")
        else:
            analyses.append(result)

    if not analyses:
        print("No source files found. Run source scripts first.")
        return

    # ── Per-source summary ────────────────────────────────────────────────
    print(f"  {'Source':<12}  {'n':>5}  {'min':>5}  {'p50':>5}  {'p90':>5}  "
          f"{'p95':>5}  {'p99':>5}  {'max':>5}  {'mean':>6}")
    print(f"  {'─'*12}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  "
          f"{'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}")
    for a in analyses:
        print(f"  {a['label']:<12}  {a['n']:>5}  {a['min']:>5}  {a['p50']:>5}  "
              f"{a['p90']:>5}  {a['p95']:>5}  {a['p99']:>5}  {a['max']:>5}  "
              f"{a['mean']:>6.0f}")

    # ── Coverage table ────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  COVERAGE  —  examples fitting within each limit (no truncation needed)")
    print(f"{'─' * W}")
    hdr = f"  {'Source':<12}"
    for lim in LIMITS:
        hdr += f"  {lim:>6}"
    print(hdr)
    print(f"  {'─'*12}" + "  ──────" * len(LIMITS))

    all_tokens = []
    for a in analyses:
        row = f"  {a['label']:<12}"
        for lim in LIMITS:
            fits = sum(1 for t in a["tokens"] if t <= lim)
            pct  = 100 * fits / a["n"]
            row += f"  {pct:>5.1f}%"
        print(row)
        all_tokens.extend(a["tokens"])

    # Combined row
    n_all = len(all_tokens)
    row = f"  {'COMBINED':<12}"
    for lim in LIMITS:
        fits = sum(1 for t in all_tokens if t <= lim)
        pct  = 100 * fits / n_all
        row += f"  {pct:>5.1f}%"
    print(f"  {'─'*12}" + "  ──────" * len(LIMITS))
    print(row)

    # ── Truncation cost ───────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  TRUNCATION COST  —  examples cut + % of total tokens lost")
    print(f"{'─' * W}")
    hdr = f"  {'Source':<12}"
    for lim in LIMITS:
        hdr += f"  {lim:>9}"
    print(hdr)
    print(f"  {'─'*12}" + "  ─────────" * len(LIMITS))

    for a in analyses:
        row = f"  {a['label']:<12}"
        for lim in LIMITS:
            n_trunc, tokens_lost, pct_lost = truncation_loss(a["tokens"], lim)
            row += f"  {n_trunc:>3}({pct_lost:>4.1f}%)"
        print(row)

    # Combined
    row = f"  {'COMBINED':<12}"
    for lim in LIMITS:
        n_trunc, tokens_lost, pct_lost = truncation_loss(all_tokens, lim)
        row += f"  {n_trunc:>3}({pct_lost:>4.1f}%)"
    print(f"  {'─'*12}" + "  ─────────" * len(LIMITS))
    print(row)

    # ── VRAM impact ───────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  VRAM IMPACT  —  estimated extra activation memory (14B / 32B, batch={BATCH_SIZE})")
    print(f"  (baseline 2048 → relative increase at each limit)")
    print(f"{'─' * W}")
    base_tokens_per_batch = 2048 * BATCH_SIZE
    for model in ["14b", "32b"]:
        base_vram = base_tokens_per_batch * VRAM_PER_TOKEN_GB[model]
        row = f"  {model:<12}"
        for lim in LIMITS:
            t_per_batch = lim * BATCH_SIZE
            vram = t_per_batch * VRAM_PER_TOKEN_GB[model]
            mult = vram / base_vram
            row += f"  {mult:>5.1f}×   "
        print(row)

    # ── Histograms ────────────────────────────────────────────────────────
    if not args.no_hist:
        print(f"\n{'─' * W}")
        print("  HISTOGRAMS  —  token distribution per source")
        print(f"{'─' * W}")
        for a in analyses:
            print(f"\n  ┌─ {a['label']} ({a['description']}) ─")
            print(histogram(a["tokens"]))

    # ── Recommendation ────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print("  RECOMMENDATION")
    print(f"{'═' * W}")

    for lim in LIMITS:
        fits = sum(1 for t in all_tokens if t <= lim)
        n_trunc, _, pct_lost = truncation_loss(all_tokens, lim)
        pct_fits = 100 * fits / n_all
        print(f"  {lim:>5} tokens : {pct_fits:>5.1f}% examples fit, "
              f"{n_trunc:>3} need truncation, {pct_lost:.2f}% token-info lost")

    print()
    # Auto-recommendation: find smallest limit where >97% fits
    for lim in LIMITS:
        fits = sum(1 for t in all_tokens if t <= lim)
        pct_fits = 100 * fits / n_all
        if pct_fits >= 97:
            _, _, pct_lost = truncation_loss(all_tokens, lim)
            print(f"  ► Recommended: {lim} tokens")
            print(f"    → {pct_fits:.1f}% of all examples fit without truncation")
            print(f"    → Only {100-pct_fits:.1f}% need truncation, losing {pct_lost:.2f}% of token content")
            print(f"    → 2× VRAM vs 1024, same as current training baseline")
            break

    if args.limit:
        lim = args.limit
        print(f"\n{'─' * W}")
        print(f"  DETAIL at --limit {lim}")
        print(f"{'─' * W}")
        for a in analyses:
            over = [t for t in a["tokens"] if t > lim]
            n_trunc, tok_lost, pct_lost = truncation_loss(a["tokens"], lim)
            print(f"\n  {a['label']}:")
            print(f"    Fits     : {a['n'] - n_trunc}/{a['n']} ({100*(a['n']-n_trunc)/a['n']:.1f}%)")
            print(f"    Truncated: {n_trunc} examples")
            if over:
                print(f"    Lengths of truncated examples: "
                      f"min={min(over)}  median={sorted(over)[len(over)//2]}  max={max(over)}")
            print(f"    Token info lost: {tok_lost:,} tokens ({pct_lost:.2f}% of {a['label']} total)")

    print()


if __name__ == "__main__":
    main()
