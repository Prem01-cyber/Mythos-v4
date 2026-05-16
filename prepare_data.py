#!/usr/bin/env python3
"""
prepare_data.py — Merge and prepare all raw datasets for training.

Reads (in order):
  processed/exploitdb.jsonl   Source 1 — exploit code + reasoning (already truncated)
  raw/htb_writeups.jsonl      Source 2 — HTB multi-turn pentest methodology
  raw/vulhub.jsonl            Source 3 — CVE exploitation chains (Vulhub)
  raw/attack.jsonl            Source 4 — MITRE ATT&CK red team technique chains

Writes:
  processed/combined.jsonl    Merged, shuffled, token-truncated training set

Multi-turn truncation strategy (Sources 2–4):
  1. Total ≤ MAX_TOKENS → keep as-is
  2. Total > MAX_TOKENS → drop full tail turn-pairs until fits
     (minimum 2 turn pairs preserved)
  3. Still over → truncate last assistant message at token boundary

Usage:
  python3 prepare_data.py                        # merge all available sources
  python3 prepare_data.py --max-tokens 2048
  python3 prepare_data.py --dry-run              # stats only, no write
  python3 prepare_data.py --source exploitdb     # single-source processed file
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

SOURCES = [
    ("processed/exploitdb.jsonl", "exploitdb"),       # Source 1 — exploit code + reasoning
    ("raw/htb_writeups.jsonl",    "htb_writeup"),     # Source 2 — HTB pentest methodology
    ("raw/vulhub.jsonl",          "vulhub"),           # Source 3 — Vulhub CVE exploitation
    ("raw/attack.jsonl",          "mitre_attack"),     # Source 4 — ATT&CK red team techniques
]

SOURCE_FILTER = {
    "exploitdb":    "processed/exploitdb.jsonl",
    "htb":         "raw/htb_writeups.jsonl",
    "vulhub":      "raw/vulhub.jsonl",
    "attack":      "raw/attack.jsonl",
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
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Context limit per example (default: 2048)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print stats only — do not write output")
    parser.add_argument("--source",     choices=list(SOURCE_FILTER.keys()), default=None,
                        help="Process a single source only (optional)")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\n{'═' * 60}")
    print(f"  Mythos-v4 — prepare_data.py")
    print(f"  Max tokens : {args.max_tokens}")
    print(f"  Output     : {OUTPUT_PATH}")
    print(f"{'═' * 60}\n")

    sources_to_use = SOURCES
    if args.source:
        path = SOURCE_FILTER[args.source]
        label = args.source
        sources_to_use = [(path, label)]

    all_examples: list[dict] = []

    for path_str, source_label in sources_to_use:
        path = Path(path_str)
        if not path.exists():
            print(f"  [SKIP] {path_str} — not found (run source script first)")
            continue

        with open(path) as f:
            raw = [json.loads(l) for l in f if l.strip()]

        kept = skipped = truncated = 0
        tokens_before: list[int] = []
        tokens_after:  list[int] = []

        for ex in raw:
            tb = _count_example(ex)
            tokens_before.append(tb)

            result = truncate_example(ex, args.max_tokens)
            if result is None:
                skipped += 1
                continue

            ta = _count_example(result)
            tokens_after.append(ta)
            if ta < tb:
                truncated += 1

            all_examples.append(result)
            kept += 1

        print(f"  Source : {source_label}  ({path_str})")
        print(f"    Raw examples : {len(raw)}")
        print(f"    Kept         : {kept}")
        print(f"    Truncated    : {truncated}  ({100*truncated/max(1,kept):.0f}%)")
        print(f"    Skipped      : {skipped}")
        if tokens_before:
            print(f"    Tokens before: min={min(tokens_before)}  "
                  f"median={statistics.median(tokens_before):.0f}  "
                  f"max={max(tokens_before)}")
        if tokens_after:
            print(f"    Tokens after : min={min(tokens_after)}  "
                  f"median={statistics.median(tokens_after):.0f}  "
                  f"max={max(tokens_after)}")
        print()

    if not all_examples:
        print("No examples loaded — nothing to write.")
        return

    random.shuffle(all_examples)

    all_tokens = [_count_example(e) for e in all_examples]
    over_limit = sum(1 for t in all_tokens if t > args.max_tokens)

    from collections import Counter
    src_counts = Counter(e["metadata"].get("source", "unknown") for e in all_examples)
    cat_counts = Counter(e["metadata"].get("category", "?")     for e in all_examples)

    print(f"{'─' * 60}")
    print(f"  Combined dataset")
    print(f"    Total examples : {len(all_examples)}")
    print(f"    Over limit     : {over_limit}  (should be 0)")
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
