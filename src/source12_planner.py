#!/usr/bin/env python3
"""
Source 12: Planner Adapter Training Data

Two data types:
  A) Goal-decomposition  — target + mode + phase  →  step-by-step attack plan with tool choices
  B) Adaptive re-planning  — initial plan + unexpected tool output  →  updated plan / pivot

Data sources:
  1. HackTricks GitHub methodology pages (same source as source7_osint)
  2. Synthetic (GPT) planning chains for bug bounty and internal pentest scenarios

Target: 2000 examples (1000 goal-decomp + 1000 re-plan)

Usage:
  python3 src/source12_planner.py --test               # 3 examples, no file writes
  python3 src/source12_planner.py --test --type decomp # only goal-decomp examples
  python3 src/source12_planner.py --test --type replan # only re-plan examples
  python3 src/source12_planner.py --list-categories
  python3 src/source12_planner.py                      # full run
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
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from prompts import PLANNER as SYSTEM_PLANNER, PLANNER as SYSTEM_REPLANNER


load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DECOMP   = "raw/planner_decomp.jsonl"
OUTPUT_REPLAN   = "raw/planner_replan.jsonl"
CACHE_DIR       = "raw/.planner_cache"
DEFAULT_WORKERS = 4
API_DELAY       = 0.3

TARGET_DECOMP   = 1000
TARGET_REPLAN   = 1000

GITHUB_API  = "https://api.github.com"
RAW_GITHUB  = "https://raw.githubusercontent.com"

# HackTricks methodology files to pull
HACKTRICKS_PENTESTING_PATHS = [
    "generic-methodologies-and-resources/pentesting-methodology.md",
    "network-services-pentesting/pentesting-web/README.md",
    "network-services-pentesting/pentesting-web/web-vulnerabilities-methodology.md",
    "network-services-pentesting/pentesting-web/api-methodology.md",
    "network-services-pentesting/pentesting-ftp.md",
    "network-services-pentesting/pentesting-smtp-smtps.md",
    "network-services-pentesting/pentesting-rdp.md",
    "network-services-pentesting/pentesting-smb.md",
    "pentesting-web/ssrf-server-side-request-forgery/README.md",
    "pentesting-web/sql-injection/README.md",
    "pentesting-web/xss-cross-site-scripting/README.md",
    "pentesting-web/idor.md",
    "pentesting-web/file-upload/README.md",
    "pentesting-web/subdomain-takeover.md",
    "pentesting-web/oauth-to-account-takeover.md",
    "pentesting-web/jwt-vulnerabilities.md",
    "pentesting-web/cors-bypass.md",
    "pentesting-web/graphql.md",
]



DECOMP_PROMPT = """\
You are generating training data for a penetration testing planner AI.

Engagement details:
  Target:    {target}
  Mode:      {mode}
  Phase:     {phase}
  Objective: {objective}
  Scope:     {scope}
  Constraints: {constraints}

Generate a complete attack plan as a JSON object:
{{
  "assessment": "<2-3 sentence assessment of the target and attack surface>",
  "plan": [
    {{
      "step": <int>,
      "objective": "<what this step achieves>",
      "tool": "<primary tool>",
      "command": "<exact command with real flags>",
      "alternatives": ["<alt tool/command if primary fails>"],
      "success_criteria": "<what output confirms success>",
      "on_success": "<what to do next>",
      "on_failure": "<fallback if this fails>"
    }}
  ],
  "total_steps": <int>,
  "estimated_time": "<realistic time estimate>",
  "key_risks": ["<risk1>", "<risk2>"]
}}

Rules:
- 4-8 steps total, ordered by execution sequence
- Use REAL tool names and REAL flag syntax
- Commands must be copy-pasteable
- Adapt the plan to the mode (bug bounty = no DoS, no destructive tests; internal = full TTPs)
- Return valid JSON only.
"""

REPLAN_PROMPT = """\
You are generating training data for an adaptive re-planning AI.

Original plan step:
  Step {step_num}: {original_objective}
  Command: {command}

Unexpected result:
  Exit code: {exit_code}
  Output:
{output}

Context so far:
{context}

Generate an adaptive re-plan as a JSON object:
{{
  "analysis": "<what happened and why — interpret the unexpected result>",
  "model_update": "<how this changes your understanding of the target>",
  "pivot": "<new hypothesis or angle to pursue>",
  "revised_steps": [
    {{
      "step": <int>,
      "objective": "<updated objective>",
      "tool": "<tool>",
      "command": "<command>",
      "rationale": "<why this approach given what we now know>"
    }}
  ]
}}

Return valid JSON only.
"""

ENGAGEMENT_SCENARIOS = [
    {
        "target": "api.fintech.com",
        "mode": "bug-bounty",
        "phase": "initial-recon",
        "objective": "Map the full attack surface of the public API",
        "scope": "*.fintech.com, api.fintech.com",
        "constraints": "No DoS, no automated brute-force of auth endpoints, no credential stuffing",
    },
    {
        "target": "10.10.10.0/24",
        "mode": "internal-pentest",
        "phase": "discovery",
        "objective": "Identify all live hosts and services on the internal subnet",
        "scope": "10.10.10.0/24",
        "constraints": "Avoid disrupting production services, limited to business hours",
    },
    {
        "target": "corp.example.com",
        "mode": "bug-bounty",
        "phase": "webapp-testing",
        "objective": "Find authentication vulnerabilities and access control issues",
        "scope": "corp.example.com and all subdomains",
        "constraints": "Only use test accounts provided by program",
    },
    {
        "target": "shop.retailer.io",
        "mode": "bug-bounty",
        "phase": "payment-flow",
        "objective": "Test payment and checkout flow for business logic vulnerabilities",
        "scope": "shop.retailer.io/checkout, shop.retailer.io/api/payment",
        "constraints": "Use only test payment tokens, no real transactions",
    },
    {
        "target": "192.168.1.50",
        "mode": "internal-pentest",
        "phase": "post-exploitation",
        "objective": "Escalate privileges from low-privileged shell",
        "scope": "Single host 192.168.1.50",
        "constraints": "Do not delete any files, document all changes",
    },
    {
        "target": "cdn.media.com",
        "mode": "bug-bounty",
        "phase": "file-upload",
        "objective": "Test file upload functionality for arbitrary code execution",
        "scope": "cdn.media.com/upload endpoint",
        "constraints": "Upload only test files, clean up after testing",
    },
    {
        "target": "oauth.provider.com",
        "mode": "bug-bounty",
        "phase": "auth-testing",
        "objective": "Test OAuth2 implementation for token hijacking and account takeover",
        "scope": "oauth.provider.com/authorize, /token, /callback",
        "constraints": "Only test with provided test accounts",
    },
    {
        "target": "intranet.company.local",
        "mode": "internal-pentest",
        "phase": "lateral-movement",
        "objective": "Move laterally from compromised workstation to domain controller",
        "scope": "company.local domain",
        "constraints": "Authorized engagement, full AD access permitted",
    },
    {
        "target": "*.saas.io",
        "mode": "bug-bounty",
        "phase": "subdomain-recon",
        "objective": "Enumerate all subdomains and find forgotten/staging environments",
        "scope": "*.saas.io",
        "constraints": "Passive recon preferred, active scanning allowed",
    },
    {
        "target": "api.healthapp.com",
        "mode": "bug-bounty",
        "phase": "api-testing",
        "objective": "Test REST API for IDOR and broken access control vulnerabilities",
        "scope": "api.healthapp.com/v1/, api.healthapp.com/v2/",
        "constraints": "HIPAA-adjacent data, extra care with PII, only test accounts",
    },
]

def _fmt(template: str, target: str) -> str:
    """Safe format — only substitutes TARGET placeholder, leaves other braces alone."""
    return template.replace("TARGET", target)


UNEXPECTED_RESULTS = [
    {
        "objective": "Port scan the target",
        "command": "nmap -sV -p- -T4 TARGET",
        "exit_code": 0,
        "output": "All 65535 scanned ports on TARGET are in ignored states.\nNot shown: 65535 filtered tcp ports (no-response)\n\nNmap done: 1 IP address (1 host up) scanned in 847.32 seconds",
        "context": "Target responded to ping. We know the host is up but all ports appear filtered.",
    },
    {
        "objective": "Directory fuzzing",
        "command": "ffuf -u https://TARGET/FUZZ -w /usr/share/wordlists/dirb/common.txt",
        "exit_code": 0,
        "output": "[Status: 200, Size: 1337, Words: 45, Lines: 10] :: /index.php\n[Status: 429, Size: 300] :: /admin\n[Status: 429, Size: 300] :: /api\n... rate limited after 50 requests ...",
        "context": "Fuzzing was rate-limited at 50 requests. We found /index.php is accessible.",
    },
    {
        "objective": "Subdomain enumeration",
        "command": "subfinder -d TARGET -all -silent",
        "exit_code": 0,
        "output": "dev.TARGET\nstaging.TARGET\napi.TARGET\nmail.TARGET",
        "context": "Found 4 subdomains. dev and staging are particularly interesting.",
    },
    {
        "objective": "SQL injection test",
        "command": "sqlmap -u 'https://TARGET/login' --data='username=test&password=test' --level=3",
        "exit_code": 0,
        "output": "[WARNING] POST parameter 'username' does not seem to be injectable\n[WARNING] POST parameter 'password' does not seem to be injectable\n[WARNING] it seems that the target URL content is not stable\n[INFO] target URL seems to have anti-CSRF token",
        "context": "Login form has anti-CSRF protection. Direct injection didn't work.",
    },
    {
        "objective": "Authentication endpoint probe",
        "command": "curl -s -X POST https://TARGET/api/login -H 'Content-Type: application/json' -d '{\"username\":\"admin\",\"password\":\"admin\"}'",
        "exit_code": 0,
        "output": "{\"error\":\"Too many attempts. Please try again in 15 minutes.\",\"lockout\":true,\"remaining\":0}",
        "context": "Account lockout triggered. We now know the API returns lockout state in JSON.",
    },
]

_gpt_lock = threading.Lock()
client = OpenAI()


def _gpt(prompt: str, max_tokens: int = 2000, model: str = "gpt-4o-mini") -> str:
    with _gpt_lock:
        time.sleep(API_DELAY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.85,
    )
    return resp.choices[0].message.content.strip()


def _parse_json_obj(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# HackTricks fetch (reuses pattern from source7_osint)
# ---------------------------------------------------------------------------
def _cached_get(url: str) -> str:
    slug = hashlib.md5(url.encode()).hexdigest()
    cache_path = Path(CACHE_DIR) / f"{slug}.txt"
    if cache_path.exists():
        return cache_path.read_text()
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "MythosEngine/1.0"})
        r.raise_for_status()
        text = r.text
        cache_path.write_text(text)
        return text
    except Exception:
        return ""


def fetch_hacktricks_pages() -> list[dict]:
    """Fetch HackTricks methodology markdown pages."""
    results = []
    base = f"{RAW_GITHUB}/HackTricks-Cloud/hacktricks/master"
    for path in HACKTRICKS_PENTESTING_PATHS:
        url  = f"{base}/{path}"
        text = _cached_get(url)
        if len(text) > 500:
            results.append({"path": path, "text": text[:8000]})
        time.sleep(0.1)
    return results


def extract_commands_from_md(text: str) -> list[str]:
    """Pull code blocks that look like shell commands."""
    blocks = re.findall(r"```(?:bash|sh|shell)?\n(.*?)```", text, re.DOTALL)
    cmds = []
    for block in blocks:
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and len(line) > 5:
                cmds.append(line)
    return cmds[:20]


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def generate_decomp_example(scenario: dict) -> dict | None:
    prompt = DECOMP_PROMPT.format(**scenario)
    try:
        raw  = _gpt(prompt, max_tokens=2000)
        data = _parse_json_obj(raw)
    except Exception:
        return None

    if not data or "plan" not in data:
        return None

    plan_text = data.get("assessment", "") + "\n\nPLAN:\n"
    for step in data.get("plan", []):
        plan_text += (
            f"\nStep {step.get('step')}: {step.get('objective')}\n"
            f"  Tool   : {step.get('tool')}\n"
            f"  Command: {step.get('command')}\n"
            f"  Success: {step.get('success_criteria')}\n"
        )
    plan_text += f"\nEstimated time: {data.get('estimated_time', 'unknown')}"
    if data.get("key_risks"):
        plan_text += f"\nKey risks: {', '.join(data['key_risks'])}"

    user_content = (
        f"Target: {scenario['target']}\n"
        f"Mode: {scenario['mode']}\n"
        f"Phase: {scenario['phase']}\n"
        f"Objective: {scenario['objective']}\n"
        f"Scope: {scenario['scope']}\n"
        f"Constraints: {scenario['constraints']}\n\n"
        f"Create a step-by-step attack plan."
    )
    return {
        "type": "decomp",
        "mode": scenario["mode"],
        "phase": scenario["phase"],
        "n_steps": data.get("total_steps", len(data.get("plan", []))),
        "messages": [
            {"role": "system",    "content": SYSTEM_PLANNER},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": plan_text},
        ],
    }


def generate_replan_example(scenario: dict, unexpected: dict) -> dict | None:
    raw_target = scenario["target"].replace("*.", "api.")
    # For URL contexts strip CIDR suffix — use first IP for internal ranges
    url_target = raw_target.split("/")[0] if "/" in raw_target else raw_target
    target = url_target
    step_num = random.randint(2, 5)
    cmd = _fmt(unexpected["command"], target)
    out = _fmt(unexpected["output"], target)
    ctx = _fmt(unexpected["context"], target)

    prompt = REPLAN_PROMPT.format(
        step_num=step_num,
        original_objective=unexpected["objective"],
        command=cmd,
        exit_code=unexpected["exit_code"],
        output=out,
        context=ctx,
    )
    try:
        raw  = _gpt(prompt, max_tokens=2000)
        data = _parse_json_obj(raw)
    except Exception:
        return None

    if not data or "revised_steps" not in data:
        return None

    replan_text = (
        f"ANALYSIS: {data.get('analysis', '')}\n\n"
        f"MODEL UPDATE: {data.get('model_update', '')}\n\n"
        f"PIVOT: {data.get('pivot', '')}\n\n"
        f"REVISED PLAN:\n"
    )
    for step in data.get("revised_steps", []):
        replan_text += (
            f"\nStep {step.get('step')}: {step.get('objective')}\n"
            f"  Tool   : {step.get('tool')}\n"
            f"  Command: {step.get('command')}\n"
            f"  Why    : {step.get('rationale')}\n"
        )

    user_content = (
        f"Target: {target}\n"
        f"Mode: {scenario['mode']}\n\n"
        f"We executed step {step_num} ({unexpected['objective']}):\n"
        f"Command: {cmd}\n\n"
        f"Result (exit {unexpected['exit_code']}):\n```\n{out}\n```\n\n"
        f"Context: {ctx}\n\n"
        f"The result was unexpected. Analyse it and revise the remaining plan."
    )
    return {
        "type": "replan",
        "mode": scenario["mode"],
        "messages": [
            {"role": "system",    "content": SYSTEM_REPLANNER},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": replan_text},
        ],
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
write_lock = threading.Lock()


def append_jsonl(path: str, examples: list[dict]) -> int:
    if not examples:
        return 0
    with write_lock:
        with open(path, "a") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
    return len(examples)


def count_written(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for _ in p.read_text().splitlines() if _.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",            action="store_true")
    parser.add_argument("--test-n",          type=int, default=2)
    parser.add_argument("--type",            choices=["decomp", "replan", "both"], default="both")
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--workers",         type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--out-decomp",      default=OUTPUT_DECOMP)
    parser.add_argument("--out-replan",      default=OUTPUT_REPLAN)
    args = parser.parse_args()

    if args.list_categories:
        print(f"Decomp  : {count_written(args.out_decomp)}/{TARGET_DECOMP}")
        print(f"Replan  : {count_written(args.out_replan)}/{TARGET_REPLAN}")
        print(f"\nEngagement scenarios : {len(ENGAGEMENT_SCENARIOS)}")
        print(f"Unexpected result templates : {len(UNEXPECTED_RESULTS)}")
        print(f"HackTricks pages: {len(HACKTRICKS_PENTESTING_PATHS)}")
        return

    # ── TEST MODE ────────────────────────────────────────────────────────────
    if args.test:
        print("=" * 70)
        print("TEST MODE — source12_planner  (no file writes)")
        print("=" * 70)

        if args.type in ("decomp", "both"):
            print("\n[A] Testing goal-decomposition (HackTricks fetch + GPT plan)...")
            pages = fetch_hacktricks_pages()
            print(f"  Fetched {len(pages)} HackTricks pages")
            for p in pages[:3]:
                cmds = extract_commands_from_md(p["text"])
                print(f"    {p['path'][:60]}  →  {len(cmds)} commands extracted")

            print(f"\n  Generating {args.test_n} decomp example(s)...")
            for _ in range(args.test_n):
                scenario = random.choice(ENGAGEMENT_SCENARIOS)
                print(f"  Scenario: {scenario['target']} / {scenario['mode']} / {scenario['phase']}")
                ex = generate_decomp_example(scenario)
                if ex:
                    print(f"  Steps : {ex['n_steps']}")
                    print(f"  User  : {ex['messages'][1]['content'][:200]!r}")
                    print(f"  Asst  : {ex['messages'][2]['content'][:300]!r}")
                else:
                    print("  FAILED")

        if args.type in ("replan", "both"):
            print(f"\n[B] Testing re-planning generation...")
            for _ in range(args.test_n):
                scenario   = random.choice(ENGAGEMENT_SCENARIOS)
                unexpected = random.choice(UNEXPECTED_RESULTS)
                print(f"  Scenario  : {scenario['target']} / {scenario['mode']}")
                print(f"  Unexpected: {unexpected['objective']}")
                ex = generate_replan_example(scenario, unexpected)
                if ex:
                    print(f"  User  : {ex['messages'][1]['content'][:200]!r}")
                    print(f"  Asst  : {ex['messages'][2]['content'][:300]!r}")
                else:
                    print("  FAILED")

        print("\n" + "=" * 70)
        print("TEST COMPLETE — run without --test to generate full dataset")
        return

    # ── FULL RUN ─────────────────────────────────────────────────────────────
    Path(args.out_decomp).parent.mkdir(parents=True, exist_ok=True)
    total_written = 0

    if args.type in ("decomp", "both"):
        existing = count_written(args.out_decomp)
        needed   = TARGET_DECOMP - existing
        print(f"\n[A] Decomp: {existing}/{TARGET_DECOMP}")

        def _decomp_worker(_: int) -> int:
            scenario = random.choice(ENGAGEMENT_SCENARIOS)
            ex = generate_decomp_example(scenario)
            return append_jsonl(args.out_decomp, [ex] if ex else [])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_decomp_worker, i): i for i in range(needed)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="decomp"):
                total_written += fut.result()

    if args.type in ("replan", "both"):
        existing = count_written(args.out_replan)
        needed   = TARGET_REPLAN - existing
        print(f"\n[B] Replan: {existing}/{TARGET_REPLAN}")

        def _replan_worker(_: int) -> int:
            scenario   = random.choice(ENGAGEMENT_SCENARIOS)
            unexpected = random.choice(UNEXPECTED_RESULTS)
            ex = generate_replan_example(scenario, unexpected)
            return append_jsonl(args.out_replan, [ex] if ex else [])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_replan_worker, i): i for i in range(needed)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="replan"):
                total_written += fut.result()

    print(f"\nWrote {total_written} new examples total")
    print(f"  Decomp: {args.out_decomp}")
    print(f"  Replan: {args.out_replan}")


if __name__ == "__main__":
    main()
