#!/usr/bin/env python3
"""
prepare_data.py — Merge and prepare all raw datasets for training.

Reads (in order):
  processed/exploitdb.jsonl   Source 1 — exploit code + reasoning (already truncated)
  raw/htb_writeups.jsonl      Source 2 — HTB multi-turn pentest methodology
  raw/vulhub.jsonl            Source 3 — CVE exploitation chains (Vulhub)
  raw/attack.jsonl            Source 4 — MITRE ATT&CK red team technique chains
  raw/ad.jsonl                Source 5 — Active Directory attack chains
  raw/webapp.jsonl            Source 6 — Web application exploitation
  raw/osint.jsonl             Source 7 — OSINT and external recon
  raw/cloud.jsonl             Source 8 — Cloud security (AWS/Azure/GCP)

Writes:
  processed/htb_writeups.jsonl   Processed HTB examples
  processed/vulhub.jsonl         Processed Vulhub examples
  processed/attack.jsonl         Processed ATT&CK examples
  processed/ad.jsonl             Processed AD examples
  processed/webapp.jsonl         Processed webapp examples
  processed/osint.jsonl          Processed OSINT examples
  processed/cloud.jsonl          Processed cloud examples
  processed/combined.jsonl       All sources merged + shuffled

Per-source max_tokens (tight-fit to P99.5, rounded to nearest 256):
  exploitdb  → 2048  (max 2043 — perfect fit, no waste)
  htb        → 2560  (P97 coverage; tail turns truncated for long chains, saves 37% vs 4096)
  vulhub     → 2048  (P99.5=1913 — was 3000, saves 32%)
  attack     → 1280  (P99.5=1053 — was 2048, saves 37%)
  ad         → 1024  (max=940 — was 4096, saves 75%!)
  webapp     → 1792  (P99.5=1653 — was 3000, saves 40%)
  osint      → 1024  (P99.5=848  — was 2048, saves 50%)
  cloud      → 1280  (P99.5=1052 — was 2048, saves 37%)

Multi-turn truncation strategy (Sources 2–4):
  1. Total ≤ MAX_TOKENS → keep as-is
  2. Total > MAX_TOKENS → drop full tail turn-pairs until fits
     (minimum 2 turn pairs preserved)
  3. Still over → truncate last assistant message at token boundary

Usage:
  python3 prepare_data.py                        # process all sources + write combined
  python3 prepare_data.py --dry-run              # stats only, no write
  python3 prepare_data.py --source htb           # process only HTB → processed/htb_writeups.jsonl
"""

import json
import random
import argparse
import statistics
from pathlib import Path

import tiktoken

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MSG_OVERHEAD  = 4     # role + separators per message
REPLY_PRIMER  = 2     # assistant reply primer
ENCODING      = "gpt-4o"

# (raw_input, source_label, max_tokens, processed_output)
# max_tokens tuned to each source's actual distribution (see inspect_tokens.py)
SOURCES = [
    ("processed/exploitdb.jsonl", "exploitdb",   2048, "processed/exploitdb.jsonl"),
    ("raw/htb_writeups.jsonl",    "htb_writeup", 2560, "processed/htb_writeups.jsonl"),
    ("raw/vulhub.jsonl",          "vulhub",      2048, "processed/vulhub.jsonl"),
    ("raw/attack.jsonl",          "mitre_attack",1280, "processed/attack.jsonl"),
    ("raw/ad.jsonl",              "ad",          1024, "processed/ad.jsonl"),
    ("raw/webapp.jsonl",          "webapp",      1792, "processed/webapp.jsonl"),
    ("raw/osint.jsonl",           "osint",       1024, "processed/osint.jsonl"),
    ("raw/cloud.jsonl",           "cloud",       1280, "processed/cloud.jsonl"),
]

SOURCE_FILTER = {
    "exploitdb": 0,
    "htb":       1,
    "vulhub":    2,
    "attack":    3,
    "ad":        4,
    "webapp":    5,
    "osint":     6,
    "cloud":     7,
}

OUTPUT_PATH = "processed/combined.jsonl"

enc = tiktoken.encoding_for_model(ENCODING)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
def _count_msg(msg: dict) -> int:
    return MSG_OVERHEAD + len(enc.encode(msg["content"]))


def _count_example(example: dict) -> int:
    return sum(_count_msg(m) for m in example["messages"]) + REPLY_PRIMER


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------
def _truncate_content(content: str, token_budget: int) -> str:
    tokens = enc.encode(content)
    if len(tokens) <= token_budget:
        return content
    return enc.decode(tokens[:token_budget]).rstrip() + "\n[truncated]"


def truncate_example(example: dict, max_tokens: int) -> dict | None:
    """
    Return a (possibly truncated) copy of example that fits within max_tokens.
    Returns None if the example cannot be salvaged.

    Single-turn (ExploitDB):  truncate the last assistant message.
    Multi-turn (HTB/Vulhub/ATT&CK): drop whole tail turn-pairs first,
                                     then truncate the final assistant message.
    """
    msgs = example["messages"]

    if _count_example(example) <= max_tokens:
        return example

    system  = [m for m in msgs if m["role"] == "system"]
    others  = [m for m in msgs if m["role"] != "system"]

    system_tokens = sum(_count_msg(m) for m in system) + REPLY_PRIMER
    if system_tokens >= max_tokens:
        return None

    # Pair up user+assistant exchanges
    pairs: list[tuple[dict, dict]] = []
    i = 0
    while i < len(others) - 1:
        if others[i]["role"] == "user" and others[i + 1]["role"] == "assistant":
            pairs.append((others[i], others[i + 1]))
            i += 2
        else:
            i += 1

    if len(pairs) > 1:
        # Drop tail pairs until fits (keep ≥ 2 for multi-turn signal)
        while len(pairs) > 2:
            candidate = system + [m for p in pairs for m in p]
            if sum(_count_msg(m) for m in candidate) + REPLY_PRIMER <= max_tokens:
                break
            pairs.pop()
        msgs = system + [m for p in pairs for m in p]

    # Final pass: truncate last assistant message if still over
    if _count_example({"messages": msgs}) > max_tokens:
        prefix_tokens = (
            sum(_count_msg(m) for m in msgs[:-1]) + REPLY_PRIMER + MSG_OVERHEAD
        )
        budget = max_tokens - prefix_tokens
        if budget < 32:
            return None

        last = msgs[-1].copy()
        last["content"] = _truncate_content(last["content"], budget)
        msgs = msgs[:-1] + [last]

    result = example.copy()
    result["messages"] = msgs
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_source(
    path_str: str,
    source_label: str,
    max_tokens: int,
    out_path: str | None,
    dry_run: bool,
) -> list[dict]:
    """Load, truncate, and optionally write one source. Returns kept examples."""
    path = Path(path_str)
    if not path.exists():
        print(f"  [SKIP] {path_str} — not found (run source script first)\n")
        return []

    with open(path) as f:
        raw = [json.loads(l) for l in f if l.strip()]

    kept_examples = []
    skipped = truncated = 0
    tokens_before: list[int] = []
    tokens_after:  list[int] = []

    for ex in raw:
        tb = _count_example(ex)
        tokens_before.append(tb)

        result = truncate_example(ex, max_tokens)
        if result is None:
            skipped += 1
            continue

        ta = _count_example(result)
        tokens_after.append(ta)
        if ta < tb:
            truncated += 1
        kept_examples.append(result)

    print(f"  Source : {source_label}  ({path_str})  [limit={max_tokens}]")
    print(f"    Raw      : {len(raw)}")
    print(f"    Kept     : {len(kept_examples)}")
    print(f"    Truncated: {truncated}  ({100*truncated/max(1,len(kept_examples)):.0f}%)")
    print(f"    Skipped  : {skipped}")
    if tokens_before:
        print(f"    Before   : min={min(tokens_before)}  "
              f"median={statistics.median(tokens_before):.0f}  "
              f"max={max(tokens_before)}")
    if tokens_after:
        print(f"    After    : min={min(tokens_after)}  "
              f"median={statistics.median(tokens_after):.0f}  "
              f"max={max(tokens_after)}")

    # Write individual processed file (skip for exploitdb — already in processed/)
    if out_path and out_path != path_str and not dry_run and kept_examples:
        Path("processed").mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            for ex in kept_examples:
                f.write(json.dumps(ex) + "\n")
        print(f"    → Written : {out_path}  ({len(kept_examples)} examples)")
    print()
    return kept_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats only — do not write output")
    parser.add_argument("--source",  choices=list(SOURCE_FILTER.keys()), default=None,
                        help="Process a single source only (writes processed/<source>.jsonl)")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\n{'═' * 60}")
    print(f"  Mythos-v4 — prepare_data.py")
    print(f"  Output : {OUTPUT_PATH}")
    print(f"{'═' * 60}\n")

    sources_to_run = (
        [SOURCES[SOURCE_FILTER[args.source]]] if args.source else SOURCES
    )

    all_examples: list[dict] = []
    for raw_in, label, max_tok, proc_out in sources_to_run:
        examples = process_source(raw_in, label, max_tok, proc_out, args.dry_run)
        all_examples.extend(examples)

    if not all_examples:
        print("No examples loaded — nothing to write.")
        return

    # If only one source was processed, skip combined output
    if args.source:
        print(f"  Single-source mode — skipping combined output.\n")
        return

    random.shuffle(all_examples)

    all_tokens = [_count_example(e) for e in all_examples]

    from collections import Counter
    src_counts = Counter(e["metadata"].get("source", "unknown") for e in all_examples)
    cat_counts = Counter(e["metadata"].get("category", "?")     for e in all_examples)

    print(f"{'─' * 60}")
    print(f"  Combined dataset")
    print(f"    Total examples : {len(all_examples)}")
    print(f"    Tokens — min={min(all_tokens)}  "
          f"median={statistics.median(all_tokens):.0f}  "
          f"max={max(all_tokens)}")

    print(f"\n  Source breakdown:")
    for src, n in src_counts.most_common():
        bar = "█" * int(30 * n / len(all_examples))
        print(f"    {src:<22} {n:>5}  {bar}")

    print(f"\n  Top categories:")
    for cat, n in cat_counts.most_common(12):
        bar = "█" * int(20 * n / max(cat_counts.values()))
        print(f"    {cat:<35} {n:>5}  {bar}")
    print(f"{'─' * 60}\n")

    if args.dry_run:
        print("  [dry-run] No file written.")
        return

    Path("processed").mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"  Written → {OUTPUT_PATH}  ({len(all_examples)} examples)\n")


if __name__ == "__main__":
    main()
