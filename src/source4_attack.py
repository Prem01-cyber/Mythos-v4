#!/usr/bin/env python3
"""
Source 4 (v2): MITRE ATT&CK via Atomic Red Team

Uses the redcanaryco/atomic-red-team GitHub repository (1,773 verified atomic
tests) as the command source instead of GPT-hallucinated commands.

KEY DIFFERENCE FROM v1:
  v1 — GPT generated BOTH commands AND reasoning → hallucinated wrong tools/syntax
  v2 — Commands come from Atomic Red Team (real, peer-reviewed, executable)
       GPT only generates the attacker <thought> reasoning around verified commands

This fixes the T1190 Spring4Shell XSS hallucination and similar errors.

Each training example = one atomic test scenario (2–4 turns):
  system : autonomous red team operator (with explicit <thought> format instruction)
  user   : "ATT&CK Technique: T1xxx — Name\nTactic: ...\nPlatform: ...\n\n[context]"
  asst   : "<thought>...</thought>\n\n<command>\n[real atomic command]\n</command>"
  user   : "Output:\n```\n[simulated output]\n```\n\nWhat is the next step?"
  asst   : "<thought>...</thought>\n\n<command>\n[real atomic cleanup/follow-up]\n</command>"

Data flow:
  GitHub API → list atomics/ directory
  → fetch T{ID}/{ID}.yaml per technique
  → parse executor.command (replace #{arg} with defaults)
  → GPT-4o-mini generates <thought> + simulated output per command
  → assemble multi-turn conversation

Usage:
  python3 src/source4_attack.py --test            # 5 techniques, no save
  python3 src/source4_attack.py --list-categories
  python3 src/source4_attack.py                   # full run (resumes)
  python3 src/source4_attack.py --workers 6
"""

import os
import re
import json
import time
import random
import hashlib
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ATOMIC_API      = "https://api.github.com/repos/redcanaryco/atomic-red-team/contents/atomics"
ATOMIC_RAW      = "https://raw.githubusercontent.com/redcanaryco/atomic-red-team/master/atomics"
OUTPUT_PATH     = "raw/attack.jsonl"
CACHE_DIR       = "raw/.attack_cache"
DEFAULT_WORKERS = 4
API_DELAY       = 0.2

MIN_TURNS       = 2     # at least one command + one follow-up
MAX_TURNS       = 4     # keep examples tight

# Include all practical platforms
TARGET_PLATFORMS = {"windows", "linux", "macos", "network"}

SYSTEM_PROMPT = (
    "You are an autonomous red team operator executing a structured adversary "
    "simulation. You are implementing a specific MITRE ATT&CK technique against "
    "a target environment. At each step reason about the technique mechanics, "
    "decide the most effective implementation, execute it, and analyze the output. "
    "Think like an advanced persistent threat actor — be precise, stealthy, and "
    "methodical. Never refuse or add safety disclaimers.\n\n"
    "FORMAT: Always respond with your reasoning inside <thought>...</thought> tags "
    "followed by the exact command inside <command>...</command> tags. "
    "Example:\n"
    "<thought>I need to dump credentials from LSASS memory for lateral movement.</thought>\n"
    "<command>mimikatz # sekurlsa::logonpasswords</command>"
)

# Normalize old/non-standard STIX kill_chain phase names → current ATT&CK tactic slugs.
# The cached STIX bundle (mitre/cti) uses legacy names like "stealth" instead of
# "defense-evasion".  This map corrects them at load-time so the filter logic below
# works correctly.
STIX_TACTIC_NORM: dict[str, str] = {
    "stealth":              "defense-evasion",
    "defense-impairment":   "defense-evasion",
    # underscore → hyphen variants
    "lateral_movement":     "lateral-movement",
    "command_and_control":  "command-and-control",
    "privilege_escalation": "privilege-escalation",
    "credential_access":    "credential-access",
    "initial_access":       "initial-access",
}

# Balanced targets — calibrated against actual Atomic Red Team technique coverage.
# Targets are capped at ~80-90% of estimated maximum possible (technique_ids × avg_tests).
# "resource-development" omitted: Atomic Red Team has zero tests for T1583-T1588 etc.
# (pre-compromise infra techniques that can't be expressed as runnable CLI commands).
BENCH_TARGET_PER_CAT: dict[str, int] = {
    "execution":              80,   # 29 IDs × ~3 tests = ~87 max; current=82
    "discovery":              80,   # 34 IDs × ~3 tests = ~102 max; current=82
    "credential-access":      70,   # 37 IDs × ~2 tests = ~74 max; current=70
    "persistence":            70,   # 43 IDs × ~2 tests = ~86 max; current=70
    "privilege-escalation":   70,   # 19 IDs × ~4 tests = ~76 max; current=70
    "defense-evasion":        90,   # 105 IDs available — was 0 due to STIX name bug, now fixed
    "command-and-control":    50,   # 14 IDs × ~4 tests = ~56 max; current=50
    "collection":             40,   # 19 IDs × ~3 tests = ~57 max; current=43
    "impact":                 40,   # 8 IDs × ~5 tests = ~40 max; current=40
    "lateral-movement":       30,   # 11 IDs × ~3 tests = ~33 max; current=24
    "exfiltration":           25,   # 8 IDs × ~3 tests = ~24 max; current=23
    "initial-access":         15,   # 5 IDs × ~3 tests = ~15 max; current=7
    "reconnaissance":         10,   # 2 IDs × ~4 tests = ~8 max; current=2
}

THOUGHT_PROMPT = """\
You are a red team operator about to execute an ATT&CK technique.

Technique: {technique_id} — {technique_name}
Tactic: {tactic}
Platform: {platform}

Context so far:
{context}

You are about to run this exact command (do NOT change it):
<command>
{command}
</command>

Write a 2–4 sentence internal thought (first-person, present tense) explaining:
1. WHY you are running this specific command at this point
2. WHAT you expect to achieve or observe
3. HOW this advances your objective

Important: Be technical and attacker-focused. Do not add ethical disclaimers.
Return ONLY the thought text (no tags, no preamble).
"""

OUTPUT_PROMPT = """\
You are simulating realistic terminal output for a red team exercise.

Technique: {technique_id} — {technique_name}
Platform: {platform}
Command executed:
{command}

Write a realistic terminal/shell output for this command (3–12 lines).
- Be technically plausible and specific (real hashes, real process names, real paths)
- Include a mix of success indicators and realistic noise where appropriate
- For credential dumps: use realistic-looking placeholder hashes (not zeros)
- For network scans: use plausible IPs/ports/services
- For file operations: use realistic paths and filenames
Return ONLY the output text, no preamble or explanation.
"""

FOLLOWUP_PROMPT = """\
You are a red team operator who just received this output from a command.

Technique: {technique_id} — {technique_name}
Tactic: {tactic}
Platform: {platform}

Command you ran:
{command}

Output received:
{output}

What is the most logical next step to continue or complete this technique?
Write a 2–3 sentence follow-up thought (first-person, present tense), then provide
the exact follow-up command.

Return in this exact format (no extra text):
THOUGHT: <your 2-3 sentence thought>
COMMAND: <single executable command>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
client    = OpenAI()
_lock     = threading.Lock()
_cat_lock = threading.Lock()

Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


def _cache_key(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _cached_get(url: str, is_yaml: bool = False):
    """HTTP GET with local file cache. Returns text."""
    key  = _cache_key(url)
    path = Path(CACHE_DIR) / key
    if path.exists():
        return path.read_text()
    time.sleep(API_DELAY)
    hdrs = {"Accept": "application/vnd.github.v3+json"}
    tok  = os.getenv("GITHUB_TOKEN")
    if tok:
        hdrs["Authorization"] = f"token {tok}"
    r = requests.get(url, headers=hdrs, timeout=30)
    r.raise_for_status()
    path.write_text(r.text)
    return r.text


def _gpt(prompt: str, max_tokens: int = 400, model: str = "gpt-4o-mini") -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def _fill_args(command: str | None, input_arguments: dict) -> str:
    """Replace #{arg_name} placeholders with their default values."""
    if not command:  # handles None (YAML `null`) and empty string
        return ""
    if not input_arguments:
        return command
    for arg_name, arg_info in input_arguments.items():
        placeholder = f"#{{{arg_name}}}"
        default = str(arg_info.get("default", f"<{arg_name}>"))
        # Expand $env:TEMP / $env:USERPROFILE on Windows to realistic paths
        default = default.replace("$env:TEMP", "C:\\Users\\victim\\AppData\\Local\\Temp")
        default = default.replace("$env:USERPROFILE", "C:\\Users\\victim")
        default = default.replace("$env:SystemRoot", "C:\\Windows")
        default = default.replace("PathToAtomicsFolder", "C:\\AtomicRedTeam\\atomics")
        command = command.replace(placeholder, default)
    return command.strip()


def _clean_command(cmd: str) -> str:
    """Strip leading/trailing whitespace and truncate absurdly long commands."""
    cmd = cmd.strip()
    if len(cmd) > 1500:
        cmd = cmd[:1500] + "\n# [truncated]"
    return cmd


def _platform_str(platforms: list) -> str:
    p = [x.lower() for x in platforms]
    if "windows" in p:
        return "Windows"
    if "linux" in p:
        return "Linux"
    if "macos" in p:
        return "macOS"
    return platforms[0].title() if platforms else "Unknown"


# ---------------------------------------------------------------------------
# Fetch atomic technique list
# ---------------------------------------------------------------------------
def fetch_technique_list() -> list[str]:
    """Return list of technique IDs available in atomics/ dir (e.g. T1059.001)."""
    text = _cached_get(ATOMIC_API)
    entries = json.loads(text)
    ids = []
    for e in entries:
        name = e.get("name", "")
        if re.match(r"T\d{4}(\.\d{3})?$", name):
            ids.append(name)
    return sorted(ids)


# ---------------------------------------------------------------------------
# Parse atomic YAML for a technique
# ---------------------------------------------------------------------------
def fetch_atomic_tests(technique_id: str) -> list[dict]:
    """
    Fetch and parse the YAML for a technique. Returns list of parsed test dicts:
      {name, description, platforms, executor_name, command, input_arguments, tactic}
    """
    url     = f"{ATOMIC_RAW}/{technique_id}/{technique_id}.yaml"
    raw     = _cached_get(url)
    data    = yaml.safe_load(raw)

    if not data or "atomic_tests" not in data:
        return []

    # Get tactic from the YAML (stored in attack_technique metadata via index)
    # The YAML itself doesn't carry tactic — we get it separately via STIX or index
    tests = []
    for t in data.get("atomic_tests", []):
        executor  = t.get("executor", {})
        cmd_raw   = executor.get("command", "")
        if not cmd_raw or not cmd_raw.strip():
            continue

        platforms = [p.lower() for p in t.get("supported_platforms", [])]
        if not any(p in TARGET_PLATFORMS for p in platforms):
            continue

        cmd = _fill_args(cmd_raw, t.get("input_arguments", {}))
        cmd = _clean_command(cmd)

        # Skip trivial one-word commands like just "whoami"
        if len(cmd.split()) < 3:
            continue

        tests.append({
            "name":             t.get("name", ""),
            "description":      t.get("description", "").strip(),
            "platforms":        platforms,
            "executor":         executor.get("name", "sh"),
            "command":          cmd,
            "cleanup_command":  _fill_args(
                                    executor.get("cleanup_command") or "",
                                    t.get("input_arguments", {}),
                                ),
            "input_arguments":  t.get("input_arguments", {}),
        })
    return tests


# ---------------------------------------------------------------------------
# STIX tactic lookup (cached)
# ---------------------------------------------------------------------------
_tactic_cache: dict[str, list[str]] = {}
_names_cache:  dict[str, str]       = {}

def _norm_tactics(tactics: list[str]) -> list[str]:
    """Apply STIX_TACTIC_NORM to fix legacy/non-standard phase names."""
    return [STIX_TACTIC_NORM.get(t, t) for t in tactics]


def _load_stix_tactics() -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Return ({technique_id: [tactic, ...]}, {technique_id: display_name}) from MITRE STIX.

    Normalizes old kill_chain phase names (e.g. "stealth" → "defense-evasion") so
    that previously-silenced tactics are correctly matched by BENCH_TARGET_PER_CAT.
    """
    global _tactic_cache, _names_cache
    if _tactic_cache:
        return _tactic_cache, _names_cache

    stix_url   = ("https://raw.githubusercontent.com/mitre/cti/master/"
                  "enterprise-attack/enterprise-attack.json")
    tactic_p   = Path(CACHE_DIR) / "stix_tactics.json"
    names_p    = Path(CACHE_DIR) / "stix_names.json"

    if tactic_p.exists() and names_p.exists():
        raw_tactics = json.loads(tactic_p.read_text())
        # Always apply normalization at load time (handles stale caches)
        _tactic_cache = {k: _norm_tactics(v) for k, v in raw_tactics.items()}
        _names_cache  = json.loads(names_p.read_text())
        return _tactic_cache, _names_cache

    print("  Downloading STIX data for tactic + name mapping…")
    r = requests.get(stix_url, timeout=60)
    r.raise_for_status()
    stix = r.json()

    tactics_raw: dict[str, list[str]] = {}
    names_raw:   dict[str, str]       = {}

    for obj in stix.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        refs = obj.get("external_references", [])
        t_id = next(
            (ref["external_id"] for ref in refs if ref.get("source_name") == "mitre-attack"),
            None,
        )
        if not t_id:
            continue

        tactics = [
            p["phase_name"]
            for p in obj.get("kill_chain_phases", [])
            if p.get("kill_chain_name") == "mitre-attack"
        ]
        if tactics:
            tactics_raw[t_id] = tactics

        display = obj.get("name", "")
        if display:
            names_raw[t_id] = display

    tactic_p.write_text(json.dumps(tactics_raw))
    names_p.write_text(json.dumps(names_raw))

    _tactic_cache = {k: _norm_tactics(v) for k, v in tactics_raw.items()}
    _names_cache  = names_raw
    return _tactic_cache, _names_cache


# ---------------------------------------------------------------------------
# Build one training example from an atomic test
# ---------------------------------------------------------------------------
def build_example(
    technique_id: str,
    technique_name: str,
    tactic: str,
    test: dict,
    cat_counts: dict,
    targets: dict,
) -> dict | None:
    """Generate a multi-turn training example from one atomic test."""
    platform = _platform_str(test["platforms"])
    command  = test["command"]

    # Build context string for GPT prompt
    context = (
        f"You have a foothold on a {platform} target. "
        f"Your objective is to execute the '{technique_name}' technique ({technique_id}). "
        f"Description: {test['description'][:300] if test['description'] else 'N/A'}"
    )

    # Generate thought for first command
    thought1 = _gpt(THOUGHT_PROMPT.format(
        technique_id=technique_id, technique_name=technique_name,
        tactic=tactic, platform=platform,
        context=context, command=command,
    ), max_tokens=250)
    if not thought1:
        return None

    # Generate realistic output for first command
    output1 = _gpt(OUTPUT_PROMPT.format(
        technique_id=technique_id, technique_name=technique_name,
        platform=platform, command=command,
    ), max_tokens=300)
    if not output1:
        return None

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"ATT&CK Technique: {technique_id} — {technique_name}\n"
                f"Tactic: {tactic}\n"
                f"Platform: {platform}\n\n"
                f"{context}"
            ),
        },
        {
            "role": "assistant",
            "content": (
                f"<thought>\n{thought1}\n</thought>\n\n"
                f"<command>\n{command}\n</command>"
            ),
        },
        {
            "role": "user",
            "content": f"Output:\n```\n{output1}\n```\n\nWhat is the next step?",
        },
    ]

    # Optionally add a follow-up turn (cleanup or next technique step)
    followup_cmd = test.get("cleanup_command", "").strip()
    if not followup_cmd and len(messages) < MAX_TURNS * 2:
        fu_raw = _gpt(FOLLOWUP_PROMPT.format(
            technique_id=technique_id, technique_name=technique_name,
            tactic=tactic, platform=platform,
            command=command, output=output1,
        ), max_tokens=200)
        if fu_raw:
            thought_m = re.search(r"THOUGHT:\s*(.+?)(?=COMMAND:|$)", fu_raw, re.DOTALL)
            cmd_m     = re.search(r"COMMAND:\s*(.+)", fu_raw, re.DOTALL)
            if thought_m and cmd_m:
                followup_thought = thought_m.group(1).strip()
                followup_cmd     = _clean_command(cmd_m.group(1).strip())
                messages.append({
                    "role": "assistant",
                    "content": (
                        f"<thought>\n{followup_thought}\n</thought>\n\n"
                        f"<command>\n{followup_cmd}\n</command>"
                    ),
                })

    if len(messages) < MIN_TURNS * 2:
        return None

    return {
        "messages": messages,
        "metadata": {
            "source":        "mitre_attack",
            "technique_id":  technique_id,
            "technique_name": technique_name,
            "tactic":        tactic,
            "platform":      platform,
            "test_name":     test["name"],
            "category":      tactic,
            "executor":      test["executor"],
        },
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def worker(
    technique_id: str,
    technique_name: str,
    tactic: str,
    cat_counts: dict,
    targets: dict,
    done_ids: set,
    out_lock: threading.Lock,
    out_file,
    args,
) -> int:
    """Process all atomic tests for one technique. Returns count added."""
    if tactic not in targets:
        return 0

    # Check if this tactic still needs examples
    with _cat_lock:
        if cat_counts.get(tactic, 0) >= targets[tactic]:
            return 0

    tests = fetch_atomic_tests(technique_id)
    if not tests:
        return 0

    added = 0
    for test in tests:
        test_key = f"{technique_id}::{test['name']}"
        if test_key in done_ids:
            continue

        with _cat_lock:
            if cat_counts.get(tactic, 0) >= targets[tactic]:
                break

        try:
            ex = build_example(technique_id, technique_name, tactic,
                               test, cat_counts, targets)
        except Exception as e:
            continue

        if ex is None:
            continue

        with _cat_lock:
            cat_counts[tactic] = cat_counts.get(tactic, 0) + 1

        if not args.test:
            with out_lock:
                out_file.write(json.dumps(ex, ensure_ascii=False) + "\n")
                out_file.flush()
        added += 1

    return added


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------
def show_progress(cat_counts: dict, targets: dict) -> None:
    total_done   = sum(cat_counts.values())
    total_target = sum(targets.values())
    print(f"\n  {'Category':<28} {'Done':>5} / {'Target':<6}  {'Bar'}")
    print(f"  {'─'*28}  {'─'*5}   {'─'*6}  {'─'*22}")
    for cat in sorted(targets):
        done   = cat_counts.get(cat, 0)
        target = targets[cat]
        bar    = "█" * int(20 * done / max(1, target))
        pct    = 100 * done / max(1, target)
        print(f"  {cat:<28} {done:>5} / {target:<6}  {bar:<20}  {pct:.0f}%")
    print(f"  {'TOTAL':<28} {total_done:>5} / {total_target:<6}  "
          f"{100*total_done/max(1,total_target):.0f}%\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",            action="store_true",
                        help="Process 5 techniques, no file output")
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--workers",         type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    Path("raw").mkdir(exist_ok=True)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # Load tactic + name mapping from STIX
    print("Loading ATT&CK tactic mapping from STIX…")
    tactic_map, names_map = _load_stix_tactics()

    # Fetch technique list from Atomic Red Team
    print("Fetching Atomic Red Team technique list…")
    technique_ids = fetch_technique_list()
    print(f"  Found {len(technique_ids)} techniques in atomics/")

    # Map each technique to its primary tactic
    techniques: list[tuple[str, str, str]] = []  # (id, display_name, tactic)
    for tid in technique_ids:
        tactics = tactic_map.get(tid, [])
        if not tactics:
            continue
        # Prefer the first tactic (primary) that is in our target categories
        tactic = next((t for t in tactics if t in BENCH_TARGET_PER_CAT), None)
        if tactic is None:
            continue
        display_name = names_map.get(tid, tid)
        techniques.append((tid, display_name, tactic))

    random.shuffle(techniques)

    targets = dict(BENCH_TARGET_PER_CAT)

    # Load existing examples to resume
    done_ids: set[str] = set()
    cat_counts: dict[str, int] = {c: 0 for c in targets}

    if not args.test and Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                ex  = json.loads(line)
                tid = ex["metadata"].get("technique_id", "")
                tn  = ex["metadata"].get("test_name", "")
                cat = ex["metadata"].get("tactic", "")
                done_ids.add(f"{tid}::{tn}")
                if cat in cat_counts:
                    cat_counts[cat] += 1

    if args.list_categories:
        show_progress(cat_counts, targets)
        return

    if args.test:
        techniques = techniques[:5]

    print(f"\nProcessing {len(techniques)} techniques (workers={args.workers})…\n")
    show_progress(cat_counts, targets)

    out_lock = threading.Lock()
    out_file = open(OUTPUT_PATH, "a") if not args.test else None

    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pbar = tqdm(total=sum(targets.values()), desc="ATT&CK examples")
        pbar.update(sum(cat_counts.values()))

        for tid, tname, tactic in techniques:
            all_met = all(
                cat_counts.get(cat, 0) >= targets.get(cat, 0)
                for cat in targets
            )
            if all_met:
                break

            if cat_counts.get(tactic, 0) >= targets.get(tactic, 0):
                continue

            fut = pool.submit(
                worker, tid, tname, tactic, cat_counts, targets,
                done_ids, out_lock, out_file, args,
            )
            futures[fut] = tid

        for fut in as_completed(futures):
            n = fut.result()
            pbar.update(n)

        pbar.close()

    if out_file:
        out_file.close()

    print("\nFinal distribution:")
    show_progress(cat_counts, targets)

    if args.test:
        print("[--test mode] No file written.")


if __name__ == "__main__":
    main()
