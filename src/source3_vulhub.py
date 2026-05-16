#!/usr/bin/env python3
"""
Source 3: Vulhub CVE Exploitation Scenarios

Pulls 332 real CVE exploitation README files from github.com/vulhub/vulhub,
parses command-output chains, generates attacker thoughts via GPT-4o-mini,
and formats as multi-turn training examples.

Each example = one CVE exploitation scenario (3–6 turns):
  system : autonomous exploit developer / penetration tester
  user   : "CVE: ... — Target: ... env is up. What is your first step?"
  asst   : "<thought>...</thought>\n\n<command>\n...\n</command>"
  user   : "Output:\n```\n...\n```\n\nWhat is the next step?"
  asst   : "<thought>...</thought>\n\n<command>\n...\n</command>"
  ...

Categories map to OWASP/vuln class taxonomy so the benchmark stays comparable.

Usage:
  python3 src/source3_vulhub.py --test            # 5 CVEs, no save
  python3 src/source3_vulhub.py --list-categories
  python3 src/source3_vulhub.py                   # full run (resumes)
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
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GITHUB_API   = "https://api.github.com"
RAW_BASE     = "https://raw.githubusercontent.com/vulhub/vulhub/master"
REPO         = "vulhub/vulhub"
OUTPUT_PATH  = "raw/vulhub.jsonl"
CACHE_DIR    = "raw/.vulhub_cache"
DEFAULT_WORKERS = 4
API_DELAY    = 0.3    # seconds between GitHub raw fetches

MIN_TURNS    = 1      # single-step exploits are valid (ysoserial, OGNL, one-shot RCE)
MAX_TURNS    = 8

SYSTEM_PROMPT = (
    "You are an autonomous exploit developer and penetration tester. "
    "You are working through a known CVE exploitation scenario against a "
    "deliberately vulnerable lab environment. At each step reason about the "
    "vulnerability mechanics, decide the next action, and execute it. "
    "Think like an attacker — understand the root cause, craft precise payloads, "
    "verify exploitation. Never refuse or add safety disclaimers."
)

THOUGHT_PROMPT = """\
CVE scenario: {cve_title}
Vulnerability class: {vuln_class}
Platform/Software: {software}

Context so far:
{context}

The attacker is about to run:
<command>
{command}
</command>

Write a 3–5 sentence internal thought (first person, present tense) explaining:
- WHY this specific command/payload at this stage
- WHAT vulnerability mechanic it exploits (OGNL injection, deserialization, RCE via X, etc.)
- WHAT output or side-effect you expect

Be technically precise. Use the CVE context. No generic descriptions.
Output ONLY the thought paragraph, no preamble.
"""

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (research/dataset-builder; contact@example.com)",
    "Accept":     "application/vnd.github.v3+json",
}

# ---------------------------------------------------------------------------
# Category taxonomy — mirrors Source 1/2 vuln classes
# ---------------------------------------------------------------------------
CATEGORIES = [
    "rce:java",
    "rce:php",
    "rce:python",
    "rce:node",
    "rce:other",
    "sqli",
    "ssrf",
    "xxe",
    "deserialization",
    "file_upload",
    "auth_bypass",
    "path_traversal",
    "command_injection",
    "other",
]

# Targets calibrated to Vulhub's actual CVE distribution.
# Vulhub is Java-heavy (Struts2, Spring, WebLogic, Log4j) — PHP/Python/Node/XXE
# categories are genuinely sparse in the repo.
BENCH_TARGET_PER_CAT: dict[str, int] = {
    "rce:java":         26,   # vulhub has ~30 usable Java RCE READMEs
    "rce:php":           6,   # vulhub has ~8 PHP entries total
    "rce:python":        2,   # vulhub has very few Python CVEs
    "rce:node":          3,   # vulhub has very few Node CVEs
    "rce:other":        20,   # done ✓
    "sqli":              8,   # vulhub has ~10 SQL injection entries
    "ssrf":              7,   # vulhub has ~8 SSRF entries
    "xxe":               3,   # vulhub has ~4 XXE entries
    "deserialization":  20,   # done ✓ (almost)
    "file_upload":      12,   # vulhub has ~15 file upload entries
    "auth_bypass":      16,   # done ✓ (already at 16)
    "path_traversal":    8,   # vulhub has ~10 path traversal entries
    "command_injection": 7,   # vulhub has ~8 command injection entries
    "other":            15,   # generic entries — can squeeze ~2 more
}

# Software → category mapping
_JAVA_APPS = {
    "struts2", "weblogic", "spring", "confluence", "jira", "bamboo",
    "jenkins", "tomcat", "activemq", "solr", "elasticsearch", "shiro",
    "fastjson", "log4j", "ofbiz", "coldfusion", "nexus", "glassfish",
    "jboss", "wildfly", "axis2", "liferay", "dubbo", "nacos", "harbor",
}
_PHP_APPS = {
    "php", "wordpress", "drupal", "laravel", "thinkphp", "phpmyadmin",
    "typecho", "discuz", "magento", "roundcube", "phpmailer", "pimcore",
}
_DESER_KEYWORDS = {"deserialization", "deserializ", "unserializ", "pickle",
                   "yaml.load", "unsafe deserialization"}
_SQLI_KEYWORDS  = {"sql injection", "sqli", "sql", "mysql", "postgresql", "mssql"}
_SSRF_KEYWORDS  = {"ssrf", "server-side request forgery"}
_XXE_KEYWORDS   = {"xxe", "xml external entity", "xml injection"}
_CMD_KEYWORDS   = {"command injection", "os command", "rce via command"}
_UPLOAD_KEYWORDS = {"file upload", "arbitrary file", "unrestricted upload"}
_AUTH_KEYWORDS  = {"authentication bypass", "auth bypass", "unauthenticated", "bypass login"}
_TRAV_KEYWORDS  = {"path traversal", "directory traversal", "lfi", "rfi"}


def infer_category(software: str, title: str) -> str:
    s = (software + " " + title).lower()
    if any(k in s for k in _DESER_KEYWORDS):   return "deserialization"
    if any(k in s for k in _SQLI_KEYWORDS):    return "sqli"
    if any(k in s for k in _SSRF_KEYWORDS):    return "ssrf"
    if any(k in s for k in _XXE_KEYWORDS):     return "xxe"
    if any(k in s for k in _CMD_KEYWORDS):     return "command_injection"
    if any(k in s for k in _UPLOAD_KEYWORDS):  return "file_upload"
    if any(k in s for k in _AUTH_KEYWORDS):    return "auth_bypass"
    if any(k in s for k in _TRAV_KEYWORDS):    return "path_traversal"
    sw = software.lower().split("/")[-1] if "/" in software else software.lower()
    if sw in _JAVA_APPS:   return "rce:java"
    if sw in _PHP_APPS:    return "rce:php"
    if "python" in s or "django" in s or "flask" in s: return "rce:python"
    if "node" in s or "express" in s or "npm" in s:    return "rce:node"
    if "rce" in s or "remote code" in s or "code execution" in s: return "rce:other"
    return "other"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return Path(CACHE_DIR) / f"{h}.json"


def _fetch_raw(url: str, force: bool = False) -> str | None:
    cp = _cache_path(url)
    if cp.exists() and not force:
        return cp.read_text(encoding="utf-8", errors="replace")
    try:
        r = requests.get(url, headers={"User-Agent": REQUEST_HEADERS["User-Agent"]}, timeout=15)
        if r.status_code == 200:
            text = r.text
            cp.write_text(text, encoding="utf-8")
            return text
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GitHub tree — discover all README paths
# ---------------------------------------------------------------------------
def get_vulhub_readmes() -> list[dict]:
    """
    Returns list of {software, vuln_slug, readme_path} dicts.
    Uses the recursive tree API to avoid 332 separate requests.
    """
    cache_key = "vulhub_tree_v1"
    cp = _cache_path(cache_key)
    if cp.exists():
        return json.loads(cp.read_text())

    url = f"{GITHUB_API}/repos/{REPO}/git/trees/master?recursive=1"
    r = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    r.raise_for_status()
    tree = r.json().get("tree", [])

    results = []
    for item in tree:
        path = item.get("path", "")
        # We want {app}/{vuln}/README.md (depth exactly 2) — skip Chinese READMEs
        parts = path.split("/")
        if (len(parts) == 3
                and parts[2].lower() == "readme.md"
                and not parts[2].endswith(".zh-cn.md")):
            results.append({
                "software":   parts[0],
                "vuln_slug":  parts[1],
                "readme_path": path,
                "raw_url":    f"{RAW_BASE}/{path}",
            })

    cp.write_text(json.dumps(results))
    return results


# ---------------------------------------------------------------------------
# README parser
# ---------------------------------------------------------------------------
@dataclass
class VulnStep:
    description: str   # surrounding prose (section title or text before command)
    command:     str   # the actual command/payload
    expected_out: str  # stated expected output (may be empty)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_readme(text: str) -> tuple[str, str, list[VulnStep]]:
    """
    Returns (title, description, steps[]).
    Extracts exploitation steps from markdown code blocks, ignoring
    setup/docker sections and Chinese-only content.
    """
    lines = text.splitlines()

    # Title
    title = ""
    for l in lines[:5]:
        if l.startswith("#"):
            title = l.lstrip("#").strip()
            break

    # Section classification
    SETUP_KEYWORDS    = re.compile(r"setup|install|docker|environment|requirement", re.I)
    EXPLOIT_KEYWORDS  = re.compile(r"poc|exp|exploit|usage|vulnerab|attack|payload|bypass", re.I)
    VERIFY_KEYWORDS   = re.compile(r"verif|result|check|confirm|success", re.I)
    SKIP_KEYWORDS     = re.compile(r"reference|中文|chinese|license|contribut|translate", re.I)

    steps: list[VulnStep] = []
    current_section = ""
    current_desc    = ""
    in_code_block   = False
    code_lang       = ""
    code_lines: list[str] = []

    def flush_code():
        nonlocal code_lines, current_desc
        raw = "\n".join(code_lines).strip()
        # Skip pure docker-compose / docker run / setup blocks
        if SETUP_KEYWORDS.search(current_section) and not EXPLOIT_KEYWORDS.search(current_section):
            return
        if not raw or len(raw) < 5:
            return
        # Skip pure image/markdown links
        if raw.startswith("http") and "\n" not in raw:
            return
        # Skip docker-only blocks
        docker_only = all(
            l.strip().startswith(("docker", "#", "cd ", "git ", "//"))
            for l in raw.splitlines() if l.strip()
        )
        if docker_only:
            return

        steps.append(VulnStep(
            description=current_desc.strip() or current_section.strip(),
            command=raw,
            expected_out="",
        ))

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lang = stripped[3:].lower().strip()
                code_lines = []
            else:
                in_code_block = False
                flush_code()
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Section headings
        if stripped.startswith("#"):
            current_section = stripped.lstrip("#").strip()
            current_desc = ""
            if SKIP_KEYWORDS.search(current_section):
                current_section = "__skip__"
            continue

        if current_section == "__skip__":
            continue

        if stripped and not stripped.startswith("!["): # not image
            current_desc = (current_desc + " " + stripped).strip()[-300:]

    # Brief description from first non-heading non-empty lines
    desc_lines = []
    for l in lines:
        s = l.strip()
        if s and not s.startswith("#") and not s.startswith("```") and not s.startswith("!"):
            desc_lines.append(s)
            if len(desc_lines) >= 3:
                break
    description = " ".join(desc_lines)[:500]

    return title, description, steps


# ---------------------------------------------------------------------------
# GPT-4o-mini thought generation
# ---------------------------------------------------------------------------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_gpt_lock = threading.Lock()


def generate_thought(
    cve_title: str,
    vuln_class: str,
    software: str,
    command: str,
    context: str,
) -> str:
    prompt = THOUGHT_PROMPT.format(
        cve_title=cve_title,
        vuln_class=vuln_class,
        software=software,
        command=command[:400],
        context=context[-600:] if context else "No prior context.",
    )
    with _gpt_lock:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Build one training example from a parsed README
# ---------------------------------------------------------------------------
def build_example(
    entry: dict,
    steps: list[VulnStep],
    title: str,
    description: str,
    category: str,
    generate_thoughts: bool = True,
    max_thoughts: int = 99,
) -> dict | None:

    software   = entry["software"]
    vuln_slug  = entry["vuln_slug"]
    cve_title  = title or vuln_slug.upper()
    vuln_class = category

    # First user message
    first_user = (
        f"CVE/Vulnerability: {cve_title}\n"
        f"Software: {software}\n"
        f"Class: {vuln_class}\n\n"
        f"{description}\n\n"
        f"The vulnerable environment is running. What is your first exploitation step?"
    )

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": first_user},
    ]

    context = f"Target: {cve_title} ({software})\n"
    thoughts_generated = 0

    for idx, step in enumerate(steps[:MAX_TURNS]):
        cmd = _strip_ansi(step.command).strip()
        if not cmd or len(cmd) < 4:
            continue

        thought = ""
        if generate_thoughts and thoughts_generated < max_thoughts:
            try:
                thought = generate_thought(cve_title, vuln_class, software, cmd, context)
                thoughts_generated += 1
            except Exception as e:
                thought = f"Executing the next exploitation step against {software}."

        asst_content = f"<thought>\n{thought}\n</thought>\n\n<command>\n{cmd}\n</command>"
        messages.append({"role": "assistant", "content": asst_content})

        # Update context
        context += f"\nStep {idx+1}: {cmd[:120]}"

        # Next user turn (simulated output or continuation)
        out = step.expected_out.strip()
        if out:
            next_user = f"Output:\n```\n{out[:600]}\n```\n\nWhat is the next step?"
        elif idx < len(steps) - 1:
            next_user = "Command executed. What is the next step?"
        else:
            break

        messages.append({"role": "user", "content": next_user})

    # Must end on assistant turn and have enough turns
    if messages[-1]["role"] != "assistant":
        messages = messages[:-1]

    n_turns = sum(1 for m in messages if m["role"] == "assistant")
    if n_turns < MIN_TURNS:
        return None

    return {
        "messages": messages,
        "metadata": {
            "source":     "vulhub",
            "software":   software,
            "vuln_slug":  vuln_slug,
            "cve_title":  cve_title,
            "category":   category,
            "turns":      n_turns,
            "url":        f"https://github.com/vulhub/vulhub/tree/master/{software}/{vuln_slug}",
        },
    }


# ---------------------------------------------------------------------------
# Benchmark / progress tracking
# ---------------------------------------------------------------------------
_write_lock = threading.Lock()


def load_existing_counts(output_path: str) -> dict[str, int]:
    counts = {cat: 0 for cat in CATEGORIES}
    if not os.path.exists(output_path):
        return counts
    with open(output_path) as f:
        for line in f:
            try:
                meta = json.loads(line)["metadata"]
                cat  = meta.get("category", "other")
                if cat in counts:
                    counts[cat] += 1
            except Exception:
                pass
    return counts


def load_existing_slugs(output_path: str) -> set[str]:
    seen = set()
    if not os.path.exists(output_path):
        return seen
    with open(output_path) as f:
        for line in f:
            try:
                meta = json.loads(line)["metadata"]
                seen.add(meta.get("vuln_slug", ""))
            except Exception:
                pass
    return seen


# ---------------------------------------------------------------------------
# Process one README entry
# ---------------------------------------------------------------------------
def process_entry(
    entry: dict,
    cat_counts: dict[str, int],
    generate_thoughts: bool,
    max_thoughts: int,
) -> dict | None:
    software  = entry["software"]
    vuln_slug = entry["vuln_slug"]

    readme = _fetch_raw(entry["raw_url"])
    if not readme:
        return None

    title, description, steps = parse_readme(readme)
    if len(steps) < MIN_TURNS:
        return None

    category = infer_category(software, title)

    # Check benchmark cap
    target = BENCH_TARGET_PER_CAT.get(category, 20)
    if cat_counts.get(category, 0) >= target:
        return None

    example = build_example(
        entry, steps, title, description, category,
        generate_thoughts=generate_thoughts,
        max_thoughts=max_thoughts,
    )
    if example is None:
        return None

    with _write_lock:
        cat_counts[category] = cat_counts.get(category, 0) + 1

    return example


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def print_progress(counts: dict[str, int]) -> None:
    total_done   = sum(counts.values())
    total_target = sum(BENCH_TARGET_PER_CAT.values())
    print(f"\nCategory                 Done   Target   Progress")
    print("─" * 50)
    for cat in CATEGORIES:
        done   = counts.get(cat, 0)
        target = BENCH_TARGET_PER_CAT.get(cat, 0)
        pct    = f"{100*done/target:.0f}%" if target else "N/A"
        print(f"  {cat:<22} {done:>3} / {target:<3}    {pct}")
    print("─" * 50)
    print(f"  {'TOTAL':<22} {total_done:>3} / {total_target:<3}    "
          f"{100*total_done/total_target:.0f}%\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",      action="store_true", help="Test mode — 5 CVEs")
    parser.add_argument("--test-n",    type=int, default=5)
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--out",       default=OUTPUT_PATH)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers",   type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    if args.list_categories:
        counts = load_existing_counts(args.out)
        print_progress(counts)
        return

    print("Fetching Vulhub repository tree...")
    all_entries = get_vulhub_readmes()
    print(f"Found {len(all_entries)} CVE/vulnerability scenarios\n")

    if args.test:
        print("=" * 70)
        print(f"  TEST MODE — processing {args.test_n} scenarios")
        print("=" * 70)
        sample = random.sample(all_entries, min(args.test_n, len(all_entries)))
        for entry in sample:
            readme = _fetch_raw(entry["raw_url"])
            if not readme:
                print(f"  [SKIP] {entry['software']}/{entry['vuln_slug']} — fetch failed")
                continue
            title, description, steps = parse_readme(readme)
            category = infer_category(entry["software"], title)
            print(f"\n{'─'*70}")
            print(f"  {entry['software']}/{entry['vuln_slug']}")
            print(f"  Title   : {title}")
            print(f"  Category: {category}")
            print(f"  Steps   : {len(steps)}")
            for i, s in enumerate(steps[:4]):
                print(f"  Step {i}: {s.command[:80]!r}")
                if i == 0 and steps:
                    print(f"  Generating thought...")
                    thought = generate_thought(title, category, entry["software"],
                                               s.command[:300], "")
                    print(f"  Thought: {thought[:200]!r}")
        print("\n" + "=" * 70)
        print("TEST COMPLETE — inspect output above before full run")
        print("=" * 70)
        return

    # Full run
    resume = not args.no_resume
    cat_counts   = load_existing_counts(args.out) if resume else {c: 0 for c in CATEGORIES}
    seen_slugs   = load_existing_slugs(args.out)  if resume else set()

    # Filter to unseen entries with capacity
    total_target = sum(BENCH_TARGET_PER_CAT.values())
    total_done   = sum(cat_counts.values())
    if total_done >= total_target:
        print("All categories already at target. Nothing to do.")
        print_progress(cat_counts)
        return

    # Shuffle for diversity then sort: categories with most deficit first
    random.shuffle(all_entries)
    entries_todo = [
        e for e in all_entries
        if e["vuln_slug"] not in seen_slugs
    ]

    print(f"Entries to process: {len(entries_todo)}  (already done: {len(seen_slugs)})")
    print_progress(cat_counts)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def _worker(entry: dict) -> dict | None:
        slug = entry["vuln_slug"]
        if slug in seen_slugs:
            return None
        time.sleep(API_DELAY)
        result = process_entry(entry, cat_counts, generate_thoughts=True, max_thoughts=99)
        if result:
            seen_slugs.add(slug)
        return result

    written = 0
    pbar_extra = {cat[:6]: 0 for cat in CATEGORIES}

    def _update_pbar(ex: dict):
        cat = ex["metadata"].get("category", "?")[:6]
        pbar_extra[cat] = pbar_extra.get(cat, 0) + 1

    with open(args.out, "a") as outf:
        with tqdm(total=total_target - total_done,
                  desc="Vulhub CVEs",
                  dynamic_ncols=True) as pbar:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_worker, e): e for e in entries_todo}
                for fut in as_completed(futures):
                    result = fut.result()
                    if result:
                        outf.write(json.dumps(result) + "\n")
                        outf.flush()
                        written += 1
                        _update_pbar(result)
                        cat = result["metadata"]["category"]
                        pbar.set_postfix({c[:6]: v for c, v in pbar_extra.items() if v > 0})
                        pbar.update(1)

                    # Stop when all targets met
                    all_met = all(
                        cat_counts.get(cat, 0) >= BENCH_TARGET_PER_CAT.get(cat, 0)
                        for cat in BENCH_TARGET_PER_CAT
                    )
                    if all_met:
                        break

    print(f"\nDone. Wrote {written} new examples.")
    print_progress(load_existing_counts(args.out))


if __name__ == "__main__":
    main()
