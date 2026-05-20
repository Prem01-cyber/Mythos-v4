#!/usr/bin/env python3
"""
Source 11: Analyst Adapter Training Data

Two data types:
  A) HackerOne public disclosed reports  →  structured finding + severity + remediation
  B) Synthetic tool output interpretation  →  full analysis chains (recon → finding → context)

Data flow for type-A:
  1. Fetch list of disclosed reports from HackerOne hacktivity feed
  2. For each report: fetch title, vulnerability type, severity, PoC, weakness, fix
  3. Convert to: "you received this evidence, what is the finding?" training pair

Data flow for type-B:
  1. GPT generates realistic multi-step recon outputs (nmap → httpx → ffuf → nuclei)
  2. GPT generates the analyst's interpretation at each step

Target: 3000 examples (1500 H1 + 1500 synthetic)

Usage:
  python3 src/source11_analyst.py --test              # 3 examples, no file writes
  python3 src/source11_analyst.py --test --type h1    # only HackerOne examples
  python3 src/source11_analyst.py --test --type synth # only synthetic examples
  python3 src/source11_analyst.py --list-categories
  python3 src/source11_analyst.py                     # full run
"""

import os
import re
import json
import time
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_H1     = "raw/analyst_h1.jsonl"
OUTPUT_SYNTH  = "raw/analyst_synth.jsonl"
CACHE_DIR     = "raw/.analyst_cache"
DEFAULT_WORKERS = 4

TARGET_H1     = 1500
TARGET_SYNTH  = 1500

# HackerOne endpoints (attempted as bonus; fully synthetic fallback if blocked)
H1_GRAPHQL    = "https://hackerone.com/graphql"
H1_HACKTIVITY = "https://hackerone.com/hacktivity.json"
H1_REPORT_URL = "https://hackerone.com/reports/{id}.json"
H1_HEADERS    = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Research/1.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://hackerone.com/hacktivity",
}
H1_PAGE_SIZE  = 25
H1_MAX_PAGES  = 60

SYSTEM_ANALYST = (
    "You are a senior security analyst embedded in an autonomous penetration testing system. "
    "Your job is to interpret raw evidence collected by automated scanners and recon tools, "
    "then identify, classify, and prioritise security findings. "
    "For each finding, state: what it is, why it is exploitable, the CVSS severity, "
    "and the immediate recommended fix. Be concise and technical. "
    "Format your output as structured FINDINGS blocks."
)

SYSTEM_INTERP = (
    "You are an autonomous security analyst. "
    "Given a sequence of tool outputs from an ongoing penetration test, "
    "reason about what each finding means, how findings chain together, "
    "and what actions should be taken next. "
    "Think like a red teamer who correlates data across multiple tools."
)

H1_TO_FINDING_PROMPT = """\
You are generating training data for a security analyst AI.

HackerOne Report Details:
  Title: {title}
  Severity: {severity}
  Vulnerability Type: {vuln_type}
  Weakness: {weakness}
  Summary / PoC: {summary}

Generate a training example in this format:
{{
  "evidence": "<2-4 paragraphs describing the raw evidence an attacker would have collected before filing this report — HTTP requests, responses, error messages, screenshots described in text, command outputs>",
  "finding": "<structured finding: TITLE / SEVERITY / DESCRIPTION / IMPACT / REPRODUCTION STEPS (3-5) / RECOMMENDED FIX>",
  "thought": "<3-4 sentence analyst reasoning: what in the evidence triggered the finding, why severity is rated as it is, what makes it exploitable>"
}}

Return valid JSON only.
"""

SYNTH_CHAIN_PROMPT = """\
You are building a training dataset for a security analyst AI.

Scenario:
  Target:    {target}
  Mode:      {mode}
  Phase:     {phase}
  Tool chain used: {tools}

Generate a realistic {n_steps}-step analysis chain where each step is:
  1. An automated tool output (realistic, noisy, 10-25 lines)
  2. The analyst's interpretation of that output

Return a JSON array of {n_steps} objects:
{{
  "step": <int>,
  "tool": "<tool name>",
  "command": "<exact command>",
  "output": "<realistic tool output — include noise, progress, and 1-3 actual findings>",
  "analyst_interpretation": "<what this output means: what was found, severity, next pivot>"
}}

Rules:
- Tool outputs must be realistic (real flag formats, real output styles)
- Analyst interpretations must chain: refer to findings from prior steps
- Include at least one real finding in the chain (open port, exposed endpoint, vuln, misconfiguration)
- Return JSON array only.
"""

VULN_TYPES = [
    "Cross-Site Scripting (XSS)",
    "SQL Injection",
    "SSRF",
    "IDOR",
    "Authentication Bypass",
    "Open Redirect",
    "Information Disclosure",
    "Race Condition",
    "Business Logic Error",
    "Broken Access Control",
    "Command Injection",
    "Path Traversal",
    "XXE",
    "CSRF",
    "Subdomain Takeover",
    "Exposed API Keys",
    "Insecure Direct Object Reference",
    "Mass Assignment",
    "Privilege Escalation",
    "Weak Cryptography",
    "GraphQL Introspection",
    "JWT Algorithm Confusion",
    "OAuth Account Takeover",
    "Host Header Injection",
    "Cache Poisoning",
    "HTTP Request Smuggling",
    "Prototype Pollution",
    "Regex Denial of Service (ReDoS)",
    "Dependency Confusion",
    "CORS Misconfiguration",
]

SEVERITIES = ["critical", "high", "medium", "low"]

SEVERITY_WEIGHTS = [0.15, 0.40, 0.35, 0.10]  # realistic H1 distribution

TARGET_TYPES = [
    "SaaS API (REST)",
    "e-commerce web app",
    "financial services portal",
    "OAuth/SSO provider",
    "mobile app backend",
    "content delivery platform",
    "developer API",
    "admin dashboard",
    "healthcare portal",
    "social media platform",
]

# Synthetic H1-style report generation — used when real H1 API is blocked
SYNTH_H1_REPORT_PROMPT = """\
Generate a realistic HackerOne-style bug bounty vulnerability report.

Vulnerability type: {vuln_type}
Severity: {severity}
Target type: {target_type}

Return a JSON object:
{{
  "title": "<specific, actionable report title — not generic, name the endpoint or parameter>",
  "severity": "{severity}",
  "vuln_type": "{vuln_type}",
  "weakness": "<exact CWE name, e.g. CWE-79 Improper Neutralization of Input...>",
  "summary": "<4-6 paragraph vulnerability report including:\\n1. Vulnerability description\\n2. Affected endpoint(s) with realistic URLs\\n3. Step-by-step reproduction (4-6 numbered steps with exact HTTP requests/payloads)\\n4. Example HTTP request and response showing the vulnerability\\n5. Impact and exploitability\\n6. Recommended fix>"
}}

Rules:
- Use realistic domain names (e.g. api.acme.com, app.corp.io)
- HTTP examples must be accurate (real headers, realistic parameter names)
- Reproduction steps must be copy-pasteable
- The summary must be detailed enough that an AI reading only the evidence could identify and classify this vulnerability
- Return valid JSON only.
"""

SYNTH_SCENARIOS = [
    {"target": "api.acme.com",  "mode": "bug-bounty",  "phase": "recon",     "tools": ["subfinder", "httpx", "nuclei"]},
    {"target": "shop.corp.com", "mode": "bug-bounty",  "phase": "webapp",    "tools": ["ffuf", "nikto", "dalfox"]},
    {"target": "10.10.10.5",    "mode": "internal",    "phase": "initial",   "tools": ["nmap", "naabu", "httpx"]},
    {"target": "admin.target.io","mode": "bug-bounty", "phase": "auth",      "tools": ["curl", "ffuf", "arjun"]},
    {"target": "*.example.com", "mode": "bug-bounty",  "phase": "recon",     "tools": ["amass", "subfinder", "dnsx", "httpx"]},
    {"target": "192.168.1.0/24","mode": "internal",    "phase": "discovery", "tools": ["nmap", "masscan", "enum4linux"]},
    {"target": "login.target.com","mode": "bug-bounty","phase": "auth",      "tools": ["curl", "sqlmap", "burp-passive"]},
    {"target": "cdn.service.io", "mode": "bug-bounty", "phase": "recon",     "tools": ["gau", "waybackurls", "httpx", "nuclei"]},
    {"target": "vpn.company.com","mode": "internal",   "phase": "network",   "tools": ["nmap", "sslyze", "testssl"]},
    {"target": "dev.startup.io", "mode": "bug-bounty", "phase": "webapp",    "tools": ["ffuf", "nikto", "whatweb", "nuclei"]},
]

_gpt_lock = threading.Lock()
client = OpenAI()

# Batch processing config
BATCH_SIZE = 20  # process 20 prompts at once
API_DELAY  = 0.05  # minimal delay between batch requests


def _gpt(prompt: str, max_tokens: int = 2000, model: str = "gpt-4o-mini") -> str:
    """Single GPT call with rate limiting."""
    with _gpt_lock:
        time.sleep(API_DELAY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.85,
    )
    return resp.choices[0].message.content.strip()


def _gpt_batch(prompts: list[str], max_tokens: int = 2000, model: str = "gpt-4o-mini") -> list[str]:
    """Batch GPT calls for efficiency."""
    results = []
    for prompt in prompts:
        try:
            with _gpt_lock:
                time.sleep(API_DELAY)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.85,
            )
            results.append(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"    [batch error: {e}]")
            results.append("")
    return results


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


def _parse_json_array(raw: str) -> list[dict]:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return []


# ---------------------------------------------------------------------------
# Synthetic H1-style report generation (primary when real API is blocked)
# ---------------------------------------------------------------------------
def generate_synth_h1_report() -> dict | None:
    """Generate a realistic H1-style vulnerability report entirely via GPT."""
    vuln_type   = random.choice(VULN_TYPES)
    severity    = random.choices(SEVERITIES, weights=SEVERITY_WEIGHTS)[0]
    target_type = random.choice(TARGET_TYPES)
    prompt = SYNTH_H1_REPORT_PROMPT.format(
        vuln_type=vuln_type, severity=severity, target_type=target_type
    )
    try:
        raw  = _gpt(prompt, max_tokens=1800)
        data = _parse_json_obj(raw)
    except Exception:
        return None
    if not all(k in data for k in ("title", "summary")):
        return None
    # Normalise to the same shape extract_report_fields expects
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary) if summary else ""
    return {
        "title":     data.get("title", ""),
        "severity":  data.get("severity", severity),
        "vuln_type": data.get("vuln_type", vuln_type),
        "weakness":  data.get("weakness", ""),
        "summary":   summary[:2000],
    }


# ---------------------------------------------------------------------------
# HackerOne data fetching (bonus — falls back to synthetic if blocked)
# ---------------------------------------------------------------------------
H1_GRAPHQL_QUERY = """
query HacktivityFeed($page: Int!, $count: Int!) {
  hacktivity(
    order_direction: DESC
    order_field: popular
    followed_only: false
    page: $page
    count: $count
  ) {
    nodes {
      ... on Report {
        id
        title
        disclosed_at
        severity { rating }
        vulnerability_types { name }
        weakness { name }
      }
    }
  }
}
"""


def _fetch_h1_graphql(page: int, count: int) -> list[dict]:
    """Try the GraphQL endpoint (no auth needed for public data in some regions)."""
    resp = requests.post(
        H1_GRAPHQL,
        json={"query": H1_GRAPHQL_QUERY, "variables": {"page": page, "count": count}},
        headers={**H1_HEADERS, "Content-Type": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data and not data.get("data"):
        return []
    nodes = data.get("data", {}).get("hacktivity", {}).get("nodes", [])
    return [n for n in nodes if n]


def _fetch_h1_json(page: int, count: int) -> list[dict]:
    """Try the public hacktivity JSON endpoint (no auth, works without cookies)."""
    params = {
        "order_direction": "DESC",
        "order_field": "popular",
        "followed_only": "false",
        "collaboration": "false",
        "page": page,
        "count": count,
    }
    resp = requests.get(H1_HACKTIVITY, params=params, headers=H1_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # Shape: {"reports": [...]} or {"hacktivity": [...]}
    records = (
        data.get("reports")
        or data.get("hacktivity")
        or data.get("data", {}).get("hacktivity", {}).get("nodes", [])
        or []
    )
    results = []
    for r in records:
        # Normalise to the GraphQL shape our downstream code expects
        entry = r if isinstance(r, dict) else {}
        if "id" not in entry and "report" in entry:
            entry = entry["report"]
        if entry.get("id"):
            results.append(entry)
    return results


def fetch_h1_report_list(page: int = 1, count: int = H1_PAGE_SIZE) -> list[dict]:
    """Fetch a page of disclosed reports — tries GraphQL, then JSON endpoint."""
    for fetcher in (_fetch_h1_graphql, _fetch_h1_json):
        try:
            results = fetcher(page, count)
            if results:
                return results
        except Exception:
            continue
    return []


def fetch_h1_report_detail(report_id: str) -> dict:
    """Fetch full report JSON from HackerOne (public, disclosed reports only)."""
    cache_path = Path(CACHE_DIR) / f"h1_{report_id}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(
            H1_REPORT_URL.format(id=report_id),
            headers=H1_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        cache_path.write_text(json.dumps(data))
        return data
    except Exception:
        return {}


def extract_report_fields(report: dict) -> dict | None:
    """Pull key fields from a HackerOne report — handles GraphQL, REST, and cached JSON shapes."""
    try:
        # REST shape: {"data": {"attributes": {...}}}
        attrs = (
            report.get("data", {}).get("attributes")
            or report.get("attributes")
            or report  # GraphQL hacktivity nodes are flat
        )
        if not attrs:
            return None

        title = attrs.get("title", "")

        # severity: {"rating": "high"} or nested {"data": {"attributes": {"rating": ...}}}
        sev_raw = attrs.get("severity") or {}
        if isinstance(sev_raw, dict):
            severity = (
                sev_raw.get("rating")
                or sev_raw.get("data", {}).get("attributes", {}).get("rating", "medium")
            )
        else:
            severity = "medium"

        # vulnerability_types: [{"name": "XSS"}] or [{"attributes": {"name": "XSS"}}]
        raw_types = attrs.get("vulnerability_types") or []
        vuln_types = []
        for v in raw_types:
            if isinstance(v, dict):
                vuln_types.append(v.get("name") or v.get("attributes", {}).get("name", ""))
        vuln_types = [x for x in vuln_types if x]

        weakness_raw = attrs.get("weakness") or {}
        weakness = (
            weakness_raw.get("name")
            or weakness_raw.get("data", {}).get("attributes", {}).get("name", "")
            if isinstance(weakness_raw, dict) else ""
        )

        summary = (
            attrs.get("vulnerability_information")
            or attrs.get("vulnerability_information_html")
            or attrs.get("summary")
            or ""
        )
        # Strip HTML tags if present
        summary = re.sub(r"<[^>]+>", " ", summary).strip()

        if not title or not summary:
            return None
        return {
            "title":     title,
            "severity":  severity or "medium",
            "vuln_type": ", ".join(vuln_types) or "General",
            "weakness":  weakness,
            "summary":   summary[:2000],
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def generate_one_h1_example() -> dict | None:
    """Try real H1 report first; fall back to fully synthetic."""
    # Attempt real H1 only if we have cached reports
    cached = list(Path(CACHE_DIR).glob("h1_*.json")) if Path(CACHE_DIR).exists() else []
    if cached:
        try:
            raw_report = json.loads(random.choice(cached).read_text())
            fields = extract_report_fields(raw_report)
            if fields:
                return generate_h1_example(fields)
        except Exception:
            pass
    # Fully synthetic
    fields = generate_synth_h1_report()
    if not fields:
        return None
    return generate_h1_example(fields)


def generate_h1_example(fields: dict) -> dict | None:
    prompt = H1_TO_FINDING_PROMPT.format(**fields)
    try:
        raw  = _gpt(prompt, max_tokens=1500)
        data = _parse_json_obj(raw)
    except Exception:
        return None

    if not all(k in data for k in ("evidence", "finding", "thought")):
        return None

    user_content = (
        f"Target context: {fields['title']}\n\n"
        f"Evidence collected:\n{data['evidence']}\n\n"
        f"Analyse this evidence. Identify and classify any security findings."
    )
    asst_content = (
        f"<thought>{data['thought']}</thought>\n\n"
        f"{data['finding']}"
    )
    return {
        "type": "h1",
        "vuln_type": fields["vuln_type"],
        "severity": fields["severity"],
        "messages": [
            {"role": "system",    "content": SYSTEM_ANALYST},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst_content},
        ],
    }


def generate_synth_chain(scenario: dict) -> dict | None:
    n_steps = random.randint(2, len(scenario["tools"]))
    tools   = scenario["tools"][:n_steps]
    prompt  = SYNTH_CHAIN_PROMPT.format(
        target=scenario["target"], mode=scenario["mode"],
        phase=scenario["phase"], tools=", ".join(tools), n_steps=n_steps,
    )
    try:
        raw   = _gpt(prompt, max_tokens=3000)
        steps = _parse_json_array(raw)
    except Exception:
        return None

    if not steps:
        return None

    messages: list[dict] = [{"role": "system", "content": SYSTEM_INTERP}]
    messages.append({
        "role": "user",
        "content": (
            f"Target: {scenario['target']}\n"
            f"Mode: {scenario['mode']}\n"
            f"Phase: {scenario['phase']}\n\n"
            f"Tool: {steps[0]['tool']}\n"
            f"Command: {steps[0].get('command', '')}\n\n"
            f"Output:\n```\n{steps[0]['output']}\n```\n\n"
            f"Interpret this output."
        ),
    })
    messages.append({
        "role": "assistant",
        "content": steps[0]["analyst_interpretation"],
    })
    for step in steps[1:]:
        messages.append({
            "role": "user",
            "content": (
                f"Tool: {step['tool']}\n"
                f"Command: {step.get('command', '')}\n\n"
                f"Output:\n```\n{step['output']}\n```\n\n"
                f"What does this tell us? What are the next steps?"
            ),
        })
        messages.append({
            "role": "assistant",
            "content": step["analyst_interpretation"],
        })

    return {
        "type": "synth",
        "phase": scenario["phase"],
        "mode": scenario["mode"],
        "n_steps": n_steps,
        "messages": messages,
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
    parser.add_argument("--type",            choices=["h1", "synth", "both"], default="both")
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--workers",         type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--out-h1",          default=OUTPUT_H1)
    parser.add_argument("--out-synth",       default=OUTPUT_SYNTH)
    args = parser.parse_args()

    if args.list_categories:
        print(f"H1    : {count_written(args.out_h1)}/{TARGET_H1}")
        print(f"Synth : {count_written(args.out_synth)}/{TARGET_SYNTH}")
        print(f"\nSynth scenarios: {len(SYNTH_SCENARIOS)}")
        print(f"Vuln types     : {len(VULN_TYPES)}")
        return

    # ── TEST MODE ────────────────────────────────────────────────────────────
    if args.test:
        print("=" * 70)
        print("TEST MODE — source11_analyst  (no file writes)")
        print("=" * 70)

        if args.type in ("h1", "both"):
            print("\n[A] Testing HackerOne fetch...")

            # Probe each fetcher individually to show which works
            for name, fetcher in [("GraphQL", _fetch_h1_graphql), ("JSON-REST", _fetch_h1_json)]:
                try:
                    results = fetcher(page=1, count=3)
                    print(f"  [{name}] returned {len(results)} results")
                    if results:
                        print(f"    first keys: {list(results[0].keys())[:8]}")
                        print(f"    sample    : {json.dumps(results[0])[:300]!r}")
                except Exception as e:
                    print(f"  [{name}] FAILED: {e}")

            reports = fetch_h1_report_list(page=1, count=5)
            print(f"\n  Combined fetch: {len(reports)} real reports")

            if reports:
                for r in reports[:3]:
                    sev = r.get("severity") or {}
                    rating = sev.get("rating", "?") if isinstance(sev, dict) else "?"
                    print(f"    id={r.get('id')}  title={str(r.get('title','?'))[:60]}  severity={rating}")
                detail = fetch_h1_report_detail(str(reports[0]["id"]))
                fields = extract_report_fields(detail)
            else:
                print("  Real H1 API blocked — testing synthetic report generation...")
                fields = generate_synth_h1_report()

            if fields:
                print(f"\n  Fields obtained:")
                print(f"    Title   : {fields['title'][:70]}")
                print(f"    Severity: {fields['severity']}")
                print(f"    VulnType: {fields['vuln_type']}")
                print(f"    Summary : {fields['summary'][:250]!r}")
                print(f"\n  Generating training example from fields...")
                ex = generate_h1_example(fields)
                if ex:
                    print(f"  User msg : {ex['messages'][1]['content'][:300]!r}")
                    print(f"  Asst msg : {ex['messages'][2]['content'][:300]!r}")
                else:
                    print("  FAILED to generate training example")
            else:
                print("  FAILED to obtain any report fields")

        if args.type in ("synth", "both"):
            print("\n[B] Testing synthetic chain generation...")
            scenario = random.choice(SYNTH_SCENARIOS)
            print(f"  Scenario: {scenario}")
            ex = generate_synth_chain(scenario)
            if ex:
                print(f"  Steps   : {ex['n_steps']}")
                print(f"  Turns   : {len(ex['messages'])}")
                print(f"  User[0] : {ex['messages'][1]['content'][:300]!r}")
                print(f"  Asst[0] : {ex['messages'][2]['content'][:300]!r}")
            else:
                print("  FAILED")

        print("\n" + "=" * 70)
        print("TEST COMPLETE — run without --test to generate full dataset")
        return

    # ── FULL RUN ─────────────────────────────────────────────────────────────
    Path(args.out_h1).parent.mkdir(parents=True, exist_ok=True)
    total_written = 0

    if args.type in ("h1", "both"):
        existing = count_written(args.out_h1)
        needed   = TARGET_H1 - existing
        print(f"\n[A] H1 examples: {existing}/{TARGET_H1}")

        # Best-effort: seed the cache with real H1 report details if accessible
        report_meta: list[dict] = []
        for page in range(1, H1_MAX_PAGES + 1):
            if len(report_meta) >= needed:
                break
            batch = fetch_h1_report_list(page=page, count=H1_PAGE_SIZE)
            if not batch:
                break
            report_meta.extend(batch)
            time.sleep(0.3)

        if report_meta:
            print(f"  Seeding cache with {len(report_meta)} real H1 report stubs...")
            def _seed_cache(meta: dict) -> None:
                fetch_h1_report_detail(str(meta["id"]))

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(tqdm(ex.map(_seed_cache, report_meta[:needed]),
                          total=min(len(report_meta), needed), desc="h1-cache"))
        else:
            print("  H1 API not accessible — generating all examples synthetically")

        # Batch generate H1 examples (much faster)
        print(f"  Generating {needed} examples in batches of {BATCH_SIZE}...")
        for batch_start in tqdm(range(0, needed, BATCH_SIZE), desc="h1-batches"):
            batch_size = min(BATCH_SIZE, needed - batch_start)
            
            # Generate batch of prompts
            prompts = []
            contexts = []
            for _ in range(batch_size):
                vuln_type   = random.choice(VULN_TYPES)
                severity    = random.choices(SEVERITIES, weights=SEVERITY_WEIGHTS)[0]
                target_type = random.choice(TARGET_TYPES)
                prompt = SYNTH_H1_REPORT_PROMPT.format(
                    vuln_type=vuln_type, severity=severity, target_type=target_type
                )
                prompts.append(prompt)
                contexts.append((vuln_type, severity))
            
            # Batch GPT call
            responses = _gpt_batch(prompts, max_tokens=1800)
            
            # Process batch results
            batch_examples = []
            for raw, (vuln_type, severity) in zip(responses, contexts):
                if not raw:
                    continue
                try:
                    data = _parse_json_obj(raw)
                    if not all(k in data for k in ("title", "summary")):
                        continue
                    summary = data.get("summary", "")
                    if not isinstance(summary, str):
                        summary = str(summary) if summary else ""
                    fields = {
                        "title":     data.get("title", ""),
                        "severity":  data.get("severity", severity),
                        "vuln_type": data.get("vuln_type", vuln_type),
                        "weakness":  data.get("weakness", ""),
                        "summary":   summary[:2000],
                    }
                    ex = build_h1_training_example(fields)
                    if ex:
                        batch_examples.append(ex)
                except Exception:
                    continue
            
            # Write batch
            total_written += append_jsonl(args.out_h1, batch_examples)

    if args.type in ("synth", "both"):
        existing = count_written(args.out_synth)
        needed   = TARGET_SYNTH - existing
        print(f"\n[B] Synth examples: {existing}/{TARGET_SYNTH}")

        # Batch generate synth examples  
        print(f"  Generating {needed} examples in batches of {BATCH_SIZE}...")
        for batch_start in tqdm(range(0, needed, BATCH_SIZE), desc="synth-batches"):
            batch_size = min(BATCH_SIZE, needed - batch_start)
            
            # Process batch
            batch_examples = []
            for _ in range(batch_size):
                scenario = random.choice(SYNTH_SCENARIOS)
                ex = generate_synth_chain(scenario)
                if ex:
                    batch_examples.append(ex)
            
            # Write batch
            total_written += append_jsonl(args.out_synth, batch_examples)

    print(f"\nWrote {total_written} new examples total")
    print(f"  H1    : {args.out_h1}")
    print(f"  Synth : {args.out_synth}")


if __name__ == "__main__":
    main()
