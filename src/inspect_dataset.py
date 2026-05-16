#!/usr/bin/env python3
"""
Dataset inspector and token-aware truncation tool.

Usage:
  # Analyze only (no writes):
  python3 src/inspect_dataset.py

  # Analyze a specific file:
  python3 src/inspect_dataset.py --file raw/exploitdb.jsonl

  # Truncate examples over a limit and write to processed/:
  python3 src/inspect_dataset.py --truncate --max-tokens 4096

  # Truncate all raw/*.jsonl files and merge into a single output:
  python3 src/inspect_dataset.py --truncate --max-tokens 4096 --merge

Run from: /home/premjampuram/Projects/Mythos-v4/
"""

import os
import re
import json
import argparse
import statistics
from glob import glob
from dataclasses import dataclass, field

import tiktoken
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENCODING_MODEL = "gpt-4o"      # tokenizer to use for counting
RAW_DIR = "raw"
PROCESSED_DIR = "processed"

# OpenAI chat fine-tune token overhead per message
MSG_OVERHEAD = 4    # role + separators
REPLY_PRIMER = 2    # assistant reply primer tokens

# Common training context-window limits to report against
CONTEXT_LIMITS = [2048, 4096, 8192, 16384, 32768]

enc = tiktoken.encoding_for_model(ENCODING_MODEL)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens_msg(msg: dict) -> int:
    return MSG_OVERHEAD + len(enc.encode(msg["content"]))


def count_tokens_example(example: dict) -> int:
    return sum(count_tokens_msg(m) for m in example["messages"]) + REPLY_PRIMER


def count_tokens_str(s: str) -> int:
    return len(enc.encode(s))


# ---------------------------------------------------------------------------
# Per-example analysis
# ---------------------------------------------------------------------------

REASONING_SECTIONS = [
    "VULNERABILITY PRIMITIVE",
    "REQUIRED CONDITIONS",
    "ATTACK CHAIN",
    "WHY THIS WORKS",
    "DETECTION AND EVASION",
]


@dataclass
class ExampleStats:
    idx: int
    source_file: str
    exploit_id: int | None
    total_tokens: int
    system_tokens: int
    user_tokens: int
    assistant_tokens: int
    reasoning_tokens: int
    code_tokens: int
    was_ruby: bool
    verified: bool
    source: str
    # category fields
    category: str        # e.g. "webapps:sqli"
    vuln_class: str      # e.g. "sqli"
    exploit_type: str    # e.g. "webapps"
    # quality signals
    has_reasoning: bool
    has_code: bool
    reasoning_sections: int  # 0–5 sections present


def analyze_example(example: dict, idx: int, source_file: str) -> ExampleStats:
    msgs = example["messages"]
    meta = example.get("metadata", {})

    role_tokens = {m["role"]: count_tokens_str(m["content"]) for m in msgs}
    assistant_content = msgs[-1]["content"]

    reasoning_m = re.search(r"<reasoning>(.*?)</reasoning>", assistant_content, re.DOTALL)
    code_m      = re.search(r"```(?:python|bash|sh|ruby)?\n(.*?)```", assistant_content, re.DOTALL)

    reasoning_body   = reasoning_m.group(1) if reasoning_m else ""
    reasoning_tokens = count_tokens_str(reasoning_body)
    code_tokens      = count_tokens_str(code_m.group(1)) if code_m else 0

    sections_found = sum(
        1 for s in REASONING_SECTIONS if s in reasoning_body
    )

    category     = meta.get("category", "")
    vuln_class   = meta.get("vuln_class", "")
    exploit_type = meta.get("type", "")
    if not category and exploit_type:
        category = f"{exploit_type}:?"

    return ExampleStats(
        idx=idx,
        source_file=source_file,
        exploit_id=meta.get("id"),
        total_tokens=count_tokens_example(example),
        system_tokens=role_tokens.get("system", 0),
        user_tokens=role_tokens.get("user", 0),
        assistant_tokens=role_tokens.get("assistant", 0),
        reasoning_tokens=reasoning_tokens,
        code_tokens=code_tokens,
        was_ruby=meta.get("was_ruby", False),
        verified=meta.get("verified", False),
        source=meta.get("source", "unknown"),
        category=category,
        vuln_class=vuln_class,
        exploit_type=exploit_type,
        has_reasoning=bool(reasoning_m),
        has_code=bool(code_m),
        reasoning_sections=sections_found,
    )


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "0%"


def percentile(data: list, p: float) -> int:
    idx = int(len(data) * p / 100)
    return sorted(data)[min(idx, len(data) - 1)]


def print_distribution(label: str, values: list[int]) -> None:
    s = sorted(values)
    n = len(s)
    print(f"\n{'─' * 56}")
    print(f"  {label}  (n={n})")
    print(f"{'─' * 56}")
    rows = [
        ("min",    s[0]),
        ("p10",    percentile(s, 10)),
        ("p25",    percentile(s, 25)),
        ("median", int(statistics.median(s))),
        ("p75",    percentile(s, 75)),
        ("p90",    percentile(s, 90)),
        ("p95",    percentile(s, 95)),
        ("p99",    percentile(s, 99)),
        ("max",    s[-1]),
    ]
    for name, val in rows:
        bar = "█" * (val // 50)
        print(f"  {name:>7}  {val:>6}  {bar}")


def print_limit_coverage(total_tokens: list[int]) -> None:
    n = len(total_tokens)
    print(f"\n{'─' * 56}")
    print("  Context-window coverage")
    print(f"{'─' * 56}")
    print(f"  {'Limit':>8}   {'Fits':>6}   {'%':>6}   {'Dropped':>7}")
    for limit in CONTEXT_LIMITS:
        fits = sum(1 for t in total_tokens if t <= limit)
        dropped = n - fits
        bar = "▓" * int(20 * fits / n)
        print(f"  {limit:>8}   {fits:>6}   {pct(fits, n):>6}   {dropped:>7}   {bar}")


def print_source_breakdown(stats: list[ExampleStats]) -> None:
    from collections import Counter
    by_source = Counter(s.source for s in stats)
    print(f"\n{'─' * 56}")
    print("  Source breakdown")
    print(f"{'─' * 56}")
    for src, count in by_source.most_common():
        print(f"  {src:<30} {count:>5}  {pct(count, len(stats)):>6}")


def print_flag_breakdown(stats: list[ExampleStats]) -> None:
    n = len(stats)
    ruby = sum(1 for s in stats if s.was_ruby)
    verified = sum(1 for s in stats if s.verified)
    print(f"\n{'─' * 56}")
    print("  Flags")
    print(f"{'─' * 56}")
    print(f"  was_ruby   {ruby:>5}  {pct(ruby, n):>6}")
    print(f"  verified   {verified:>5}  {pct(verified, n):>6}")


def print_category_coverage(stats: list[ExampleStats]) -> None:
    from collections import defaultdict

    # Group by category
    by_cat: dict[str, list[ExampleStats]] = defaultdict(list)
    for s in stats:
        by_cat[s.category or "(none)"].append(s)

    # Group by type for the summary header
    by_type: dict[str, int] = defaultdict(int)
    for s in stats:
        by_type[s.exploit_type or "(none)"] += 1

    n_total = len(stats)
    max_count = max(len(v) for v in by_cat.values()) if by_cat else 1
    BAR_WIDTH = 20

    # ── type summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  CATEGORY COVERAGE")
    print(f"{'═' * 72}")
    print("  Type summary:")
    for t, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        bar = "█" * int(BAR_WIDTH * cnt / n_total)
        print(f"    {t:<15} {cnt:>5}  {pct(cnt, n_total):>6}  {bar}")

    # ── per-category table ────────────────────────────────────────────────────
    header = (
        f"\n  {'CATEGORY':<32} {'N':>5}  {'%':>5}  "
        f"{'MED_TOK':>7}  {'RSN_TOK':>7}  {'COD_TOK':>7}  "
        f"{'§/5':>4}  {'PYTHON':>6}  BAR"
    )
    print(header)
    print(f"  {'─' * 88}")

    prev_type = None
    for cat in sorted(by_cat.keys()):
        grp = by_cat[cat]
        exploit_type = grp[0].exploit_type or cat.split(":")[0]

        # blank line between type groups
        if exploit_type != prev_type:
            if prev_type is not None:
                print()
            prev_type = exploit_type

        cnt          = len(grp)
        med_tok      = int(statistics.median(s.total_tokens for s in grp))
        med_rsn      = int(statistics.median(s.reasoning_tokens for s in grp))
        med_cod      = int(statistics.median(s.code_tokens for s in grp))
        avg_sections = sum(s.reasoning_sections for s in grp) / cnt
        pct_python   = pct(sum(1 for s in grp if s.has_code), cnt)
        bar          = "▓" * int(BAR_WIDTH * cnt / max_count)

        print(
            f"  {cat:<32} {cnt:>5}  {pct(cnt, n_total):>5}  "
            f"{med_tok:>7}  {med_rsn:>7}  {med_cod:>7}  "
            f"{avg_sections:>4.1f}  {pct_python:>6}  {bar}"
        )

    # ── quality summary across all categories ─────────────────────────────────
    has_rsn   = sum(1 for s in stats if s.has_reasoning)
    has_code  = sum(1 for s in stats if s.has_code)
    full_rsn  = sum(1 for s in stats if s.reasoning_sections == 5)

    print(f"\n  {'─' * 72}")
    print(f"  Quality signals (all {n_total} examples):")
    print(f"    has <reasoning>     {has_rsn:>5}  {pct(has_rsn, n_total):>6}")
    print(f"    has ```python       {has_code:>5}  {pct(has_code, n_total):>6}")
    print(f"    all 5 RSN sections  {full_rsn:>5}  {pct(full_rsn, n_total):>6}")
    print(f"{'═' * 72}")


def print_outliers(stats: list[ExampleStats], max_tokens: int, top_n: int = 10) -> None:
    over = [s for s in stats if s.total_tokens > max_tokens]
    if not over:
        print(f"\n  ✓ All {len(stats)} examples fit within {max_tokens} tokens.")
        return
    over.sort(key=lambda s: s.total_tokens, reverse=True)
    print(f"\n{'─' * 56}")
    print(f"  Over {max_tokens} tokens  ({len(over)} examples)")
    print(f"{'─' * 56}")
    print(f"  {'id':>8}  {'total':>6}  {'reasoning':>10}  {'code':>6}  {'file'}")
    for s in over[:top_n]:
        print(
            f"  {str(s.exploit_id):>8}  {s.total_tokens:>6}  "
            f"{s.reasoning_tokens:>10}  {s.code_tokens:>6}  "
            f"{os.path.basename(s.source_file)}"
        )
    if len(over) > top_n:
        print(f"  ... and {len(over) - top_n} more")


# ---------------------------------------------------------------------------
# Smart truncation
# ---------------------------------------------------------------------------

def truncate_example(example: dict, max_tokens: int) -> tuple[dict, bool]:
    """
    Truncate an example to fit within max_tokens.

    Strategy (in order):
      1. Shorten the code block — the reasoning is the expensive GPT output,
         never touch it.
      2. If even an empty code block doesn't fit, drop the example entirely
         (return None).

    Returns (truncated_example, was_truncated).
    """
    if count_tokens_example(example) <= max_tokens:
        return example, False

    assistant = example["messages"][-1]["content"]

    code_m = re.search(r"(```(?:python|bash|sh|ruby)?\n)(.*?)(```)", assistant, re.DOTALL)
    if not code_m:
        # No code block found — can't truncate, keep as-is and flag
        return example, False

    fence_open = code_m.group(1)
    code_body = code_m.group(2)
    fence_close = code_m.group(3)

    # Calculate how many tokens the non-code parts cost
    placeholder = fence_open + "{CODE}" + fence_close
    shell = example.copy()
    shell["messages"] = [m.copy() for m in example["messages"]]
    shell["messages"][-1] = {
        "role": "assistant",
        "content": assistant[:code_m.start()] + placeholder + assistant[code_m.end():],
    }
    shell_tokens = count_tokens_example(shell) - count_tokens_str("{CODE}")

    budget = max_tokens - shell_tokens

    TRAILER = "\n# ... [truncated to fit context window]"
    trailer_tokens = count_tokens_str(TRAILER)
    # Leave a small safety margin for tokeniser boundary effects
    MARGIN = 5
    budget = budget - trailer_tokens - MARGIN

    if budget < 50:
        # Even without any code there's no room — drop
        return None, True

    # Take longest token prefix of code_body that fits in budget
    code_tokens = enc.encode(code_body)
    if len(code_tokens) <= budget:
        return example, False  # shouldn't happen but guard

    truncated_code = enc.decode(code_tokens[:budget]).rstrip()
    truncated_code += TRAILER

    new_assistant = (
        assistant[:code_m.start()]
        + fence_open
        + truncated_code
        + "\n"
        + fence_close
        + assistant[code_m.end():]
    )

    result = {
        "messages": [
            *example["messages"][:-1],
            {"role": "assistant", "content": new_assistant},
        ],
        "metadata": {
            **example.get("metadata", {}),
            "truncated": True,
            "original_tokens": count_tokens_example(example),
        },
    }
    return result, True


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_files(paths: list[str], max_tokens_report: int = 4096) -> list[ExampleStats]:
    all_stats: list[ExampleStats] = []
    raw_examples: list[tuple[str, dict]] = []

    for path in paths:
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"  {path}: {len(lines)} lines")
        for i, line in enumerate(lines):
            try:
                ex = json.loads(line)
                raw_examples.append((path, ex))
                all_stats.append(analyze_example(ex, i, path))
            except Exception as e:
                print(f"    [WARN] line {i}: {e}")

    if not all_stats:
        print("No examples found.")
        return []

    total_tokens = [s.total_tokens for s in all_stats]
    reasoning_tokens = [s.reasoning_tokens for s in all_stats]
    code_tokens = [s.code_tokens for s in all_stats]

    print_distribution("Total tokens per example", total_tokens)
    print_distribution("Reasoning tokens (inside <reasoning>)", reasoning_tokens)
    print_distribution("Code tokens (inside ```python block)", code_tokens)
    print_limit_coverage(total_tokens)
    print_source_breakdown(all_stats)
    print_flag_breakdown(all_stats)
    print_category_coverage(all_stats)
    print_outliers(all_stats, max_tokens=max_tokens_report)

    print(f"\n{'═' * 56}")
    print(f"  RECOMMENDATION")
    print(f"{'═' * 56}")
    max_tok = max(total_tokens)
    for limit in CONTEXT_LIMITS:
        if limit >= max_tok:
            fits_pct = pct(sum(1 for t in total_tokens if t <= limit), len(total_tokens))
            print(f"  Safe limit: {limit} tokens  ({fits_pct} of examples fit without truncation)")
            print(f"  Smallest safe limit: {limit}")
            break
    print(f"  Absolute max in dataset: {max_tok} tokens")
    print(f"  Truncation target (if needed): shorten code block only, keep reasoning intact")

    return all_stats


# ---------------------------------------------------------------------------
# Truncation pipeline
# ---------------------------------------------------------------------------

def run_truncation(paths: list[str], max_tokens: int, merge: bool) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    total_in = total_out = total_truncated = total_dropped = 0

    all_output: list[dict] = []

    for path in paths:
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]

        file_out: list[dict] = []
        file_truncated = file_dropped = 0

        for line in lines:
            try:
                ex = json.loads(line)
                total_in += 1

                result, was_truncated = truncate_example(ex, max_tokens)

                if result is None:
                    file_dropped += 1
                    total_dropped += 1
                    continue

                if was_truncated:
                    file_truncated += 1
                    total_truncated += 1

                file_out.append(result)
                total_out += 1

            except Exception as e:
                print(f"  [WARN] {path}: {e}")

        fname = os.path.basename(path)
        if not merge:
            out_path = os.path.join(PROCESSED_DIR, fname)
            with open(out_path, "w") as f:
                for ex in file_out:
                    f.write(json.dumps(ex) + "\n")
            print(
                f"  {fname}: {len(lines)} in → {len(file_out)} out "
                f"(truncated={file_truncated}, dropped={file_dropped})"
            )
            print(f"  → {out_path}")
        else:
            all_output.extend(file_out)

    if merge:
        out_path = os.path.join(PROCESSED_DIR, "merged.jsonl")
        with open(out_path, "w") as f:
            for ex in all_output:
                f.write(json.dumps(ex) + "\n")
        print(f"\n  Merged output: {out_path}")

    print(f"\n  Total: {total_in} in → {total_out} out")
    print(f"  Truncated: {total_truncated}  |  Dropped: {total_dropped}")

    # Verify nothing exceeds limit in output
    over = 0
    for ex in all_output if merge else []:
        if count_tokens_example(ex) > max_tokens:
            over += 1
    if not merge:
        # recheck from files
        for path in paths:
            fname = os.path.basename(path)
            out_path = os.path.join(PROCESSED_DIR, fname)
            if os.path.exists(out_path):
                with open(out_path) as f:
                    for line in f:
                        ex = json.loads(line.strip())
                        if count_tokens_example(ex) > max_tokens:
                            over += 1

    if over:
        print(f"\n  [WARN] {over} examples still over {max_tokens} tokens after truncation")
    else:
        print(f"\n  ✓ All output examples are within {max_tokens} tokens")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect dataset token counts and optionally truncate."
    )
    parser.add_argument(
        "--file", default=None,
        help="Single JSONL file to analyze (default: all raw/*.jsonl)"
    )
    parser.add_argument(
        "--truncate", action="store_true",
        help="Truncate examples over --max-tokens and write to processed/"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Token limit for truncation and outlier reporting (default: 4096)"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="When truncating multiple files, merge all into processed/merged.jsonl"
    )
    args = parser.parse_args()

    # Resolve file paths
    if args.file:
        paths = [args.file]
    else:
        paths = sorted(glob(os.path.join(RAW_DIR, "*.jsonl")))

    if not paths:
        print(f"No JSONL files found in {RAW_DIR}/")
        return

    print(f"\nFiles to analyze: {len(paths)}")
    for p in paths:
        print(f"  {p}")

    print(f"\nTokenizer: {ENCODING_MODEL}")
    print(f"Reporting outliers vs limit: {args.max_tokens}\n")

    analyze_files(paths, max_tokens_report=args.max_tokens)

    if args.truncate:
        print(f"\n{'═' * 56}")
        print(f"  TRUNCATION PASS  (max_tokens={args.max_tokens})")
        print(f"{'═' * 56}\n")
        run_truncation(paths, max_tokens=args.max_tokens, merge=args.merge)


if __name__ == "__main__":
    main()
