#!/usr/bin/env python3
"""
Source 13: Researcher Adapter Training Data

This adapter is trained on the REASONING PROCESS of discovering unknown vulnerabilities,
not on known techniques. The training format is fundamentally different from all other adapters:

  NOT: "here is the technique"
  BUT: "standard approaches returned nothing — I reasoned from anomalies to a novel finding"

Three data subtypes:
  A) Synthetic exhaustion chains (1200 examples):
     GPT generates chains where 3-4 standard tools fail, then anomaly reasoning discovers something novel.
     The model learns to reason from behavioral inconsistencies, not pattern-match on vuln types.

  B) Project Zero reconstruction (400 examples):
     Fetch google/security-research pocs/*/README.md files.
     GPT reformats each into the hypothesis chain format.
     These are real novel bugs discovered by expert researchers.

  C) Novel CTF writeups (400 examples):
     Fetch CTF writeups where standard tools didn't work (custom scripts, timing attacks, etc.).
     GPT reformats into hypothesis chain format.

Training format (all subtypes):
  CONTEXT: <target description + what standard tools returned (nothing actionable)>
  OBSERVATIONS:
    - <anomaly 1: timing, error leak, size deviation, behavioral diff, tech fingerprint>
    - <anomaly 2>
    ...
  HYPOTHESIS: <what the anomalies suggest — specific, testable>
  PROBE: <minimal command/request to test the hypothesis — not a full tool scan>
  EXPECTED_INDICATOR: <what in the response would confirm or refute the hypothesis>
  RESULT: <what the probe returned>
  TECHNIQUE: <what was discovered and why no existing tool covers it>
  PIVOT_IF_NEGATIVE: <what to try if the hypothesis is wrong>

Usage:
  python3 src/source13_researcher.py --test              # 2 examples, no writes
  python3 src/source13_researcher.py --test --type synth # only synthetic chains
  python3 src/source13_researcher.py --test --type pz    # only Project Zero
  python3 src/source13_researcher.py --test --type ctf   # only CTF writeups
  python3 src/source13_researcher.py --list-categories
  python3 src/source13_researcher.py                     # full run
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

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_SYNTH = "raw/researcher_synth.jsonl"
OUTPUT_PZ    = "raw/researcher_pz.jsonl"
OUTPUT_CTF   = "raw/researcher_ctf.jsonl"
CACHE_DIR    = "raw/.researcher_cache"
DEFAULT_WORKERS = 4
API_DELAY    = 0.3

TARGET_SYNTH = 1200
TARGET_PZ    = 400
TARGET_CTF   = 400

GITHUB_API  = "https://api.github.com"
RAW_GITHUB  = "https://raw.githubusercontent.com"
GITHUB_HEADERS = {
    "User-Agent": "MythosEngine-Research/1.0",
    "Accept": "application/vnd.github+json",
}
if os.getenv("GITHUB_TOKEN"):
    GITHUB_HEADERS["Authorization"] = f"Bearer {os.getenv('GITHUB_TOKEN')}"

# ---------------------------------------------------------------------------
# System prompt — what the researcher adapter is trained to be
# ---------------------------------------------------------------------------
SYSTEM_RESEARCHER = (
    "You are an autonomous security researcher. You reason from anomalies and behavioral "
    "inconsistencies — not from known vulnerability patterns. When standard tools return "
    "nothing actionable, you form hypotheses from unexplained signals (timing differences, "
    "error message leaks, response size deviations, behavioral inconsistencies for similar inputs), "
    "design minimal probes to test them, and iterate based on results. "
    "You never assume a target is secure just because known scanners found nothing. "
    "The most interesting bugs are found in the gaps between what tools know to look for.\n\n"
    "FORMAT: Always respond with:\n"
    "OBSERVATIONS: <list of unexplained behavioral signals>\n"
    "HYPOTHESIS: <what this suggests — specific and testable>\n"
    "PROBE: <minimal request/command to test the hypothesis>\n"
    "EXPECTED_INDICATOR: <what in the response confirms or refutes it>\n"
    "PIVOT_IF_NEGATIVE: <next hypothesis if this one fails>"
)

# ---------------------------------------------------------------------------
# Synthetic exhaustion chain generation
# ---------------------------------------------------------------------------

# Standard tools and their "nothing found" outputs — the exhaustion context
STANDARD_TOOL_FAILURES = [
    {
        "tool": "nmap -sV -p- -T4",
        "output": "Not shown: 65528 filtered tcp ports (no-response)\nPORT   STATE SERVICE VERSION\n80/tcp open  http    nginx 1.18.0\n443/tcp open  ssl/https nginx 1.18.0",
        "verdict": "Only standard web ports open. Service versions reveal nothing critical.",
    },
    {
        "tool": "nuclei -u TARGET -t cves/ -t exposures/",
        "output": "[INF] No results found.\n[INF] Scanning with 2847 templates...\n[INF] Templates executed: 2847, Matched: 0",
        "verdict": "No known CVEs or exposures match.",
    },
    {
        "tool": "sqlmap -u 'TARGET/api/search?q=test' --level=5 --risk=3",
        "output": "[WARNING] GET parameter 'q' does not seem to be injectable\n[WARNING] it seems that the target URL content is not stable\n[CRITICAL] all tested parameters do not appear to be injectable",
        "verdict": "No SQL injection found in search parameter.",
    },
    {
        "tool": "ffuf -u TARGET/FUZZ -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt",
        "output": ":: Progress: [220560/220560] :: Job [1/1] :: 847 req/sec :: Duration: [0:04:20] :: Errors: 0 ::\n[Status: 200, Size: 1337, Words: 45, Lines: 10] :: /",
        "verdict": "Only root path returns 200. No additional directories found.",
    },
    {
        "tool": "dalfox url TARGET/search?q=test",
        "output": "[W] Not Found XSS\n[I] Finish :P",
        "verdict": "No reflected XSS in search parameter.",
    },
    {
        "tool": "nikto -h TARGET",
        "output": "- No web server found on TARGET:80\n+ Server: nginx/1.18.0\n+ No CGI Directories found\n+ 0 item(s) reported on remote host",
        "verdict": "No common web vulnerabilities detected by Nikto.",
    },
    {
        "tool": "wfuzz -c -z range,1-1000 -u TARGET/api/user/FUZZ --hc 404",
        "output": "Total time: 120.00 s\nProcessed Requests: 1000\nFiltered Requests: 1000\nRequests/sec: 8.33",
        "verdict": "All user ID probes returned 404 — no IDOR surface found.",
    },
]

# Anomaly types — the interesting signals that remain after standard tools fail
ANOMALY_TEMPLATES = [
    {
        "type": "timing_oracle",
        "scenarios": [
            {
                "observation": "/api/v1/export?format=csv responds in 45ms. Same endpoint with format=pdf responds in 3200ms.",
                "hypothesis": "The server is generating the PDF server-side using a template engine. The slow response suggests actual rendering, not just format rejection. Template injection may be possible in the format parameter.",
                "probe": 'curl -s -w "\\nTime: %{time_total}s\\n" -X GET "TARGET/api/v1/export?format={{7*7}}"',
                "expected_indicator": "Response time > 2s AND response body contains \'49\' or modified content",
                "technique": "Timing-based SSTI detection — response time reveals server-side template processing",
                "pivot_if_negative": "Try format=../etc/passwd for path traversal, or format=x%0a%0d for CRLF injection",
            },
            {
                "observation": "POST /api/search responds in 20ms for random strings. POST /api/search with q=admin@company.com responds in 850ms.",
                "hypothesis": "The extra latency suggests the server is performing a database lookup when a valid email format is detected. This is a timing oracle for email enumeration — valid accounts trigger a DB query, invalid ones fail fast.",
                "probe": 'for email in admin@target.com test@target.com nonexistent12345@target.com; do echo -n "$email: "; time curl -s -X POST TARGET/api/search -d "q=$email" > /dev/null; done',
                "expected_indicator": "Valid corporate email addresses consistently take > 500ms; random strings take < 50ms",
                "technique": "Timing oracle for user enumeration via search — not a login endpoint, bypasses lockout protections",
                "pivot_if_negative": "Test /api/forgot-password and /api/check-email endpoints for same timing pattern",
            },
        ],
    },
    {
        "type": "error_leak",
        "scenarios": [
            {
                "observation": "POST /api/render with body {\"template\": \"hello\"} returns 200. With body {\"template\": \"{{\"} returns 500 with body: 'template: render.tmpl:14: unexpected EOF'",
                "hypothesis": "The error message reveals: (a) Go template engine, (b) template file named render.tmpl, (c) the user input is evaluated as a Go template. Go templates allow calling arbitrary methods on objects in scope.",
                "probe": 'curl -s -X POST TARGET/api/render -H \'Content-Type: application/json\' -d \'{"template": "{{.}}"}\'',
                "expected_indicator": "Response reveals the template data context object — may expose internal struct fields, credentials, or config",
                "technique": "Go SSTI via template parameter — error message reveals engine and confirms eval. No scanner tests this because the endpoint doesn\'t look like a traditional template.",
                "pivot_if_negative": 'Try {"template": "{{range $k,$v := .}}{{$k}}={{$v}} {{end}}"} to enumerate all context variables',
            },
            {
                "observation": "GET /api/report?id=1 returns a PDF. GET /api/report?id=abc returns 500 with: 'java.sql.SQLException: Conversion failed when converting the nvarchar value \'abc\' to data type int.'",
                "hypothesis": "The error leaks: (a) Java backend, (b) MSSQL database, (c) the id parameter is interpolated unsafely into a query. The nvarchar→int conversion error suggests the raw value is passed to the DB. MSSQL-specific injection techniques apply.",
                "probe": "curl -s 'TARGET/api/report?id=1;WAITFOR+DELAY+%270:0:5%27--'",
                "expected_indicator": "Response delayed by 5+ seconds confirms blind MSSQL injection via time-based technique",
                "technique": "MSSQL blind time-based injection — discovered via error message that revealed DB type and parameter handling",
                "pivot_if_negative": "Try id=1' and id=1-- to test for string-based injection, or id=1 UNION SELECT NULL-- for union-based",
            },
        ],
    },
    {
        "type": "size_deviation",
        "scenarios": [
            {
                "observation": "GET /api/users returns 2341 bytes for user_id=1 through user_id=50. user_id=51 returns 89 bytes. user_id=52 returns 2341 bytes again.",
                "hypothesis": "user_id=51 is anomalous — significantly smaller response suggests either a different code path, a deleted/suspended account, or an admin/service account that the API handles differently. Accounts at boundaries often have different permissions.",
                "probe": 'for id in 49 50 51 52 53; do echo -n "id=$id size="; curl -s -o /dev/null -w "%{size_download}" "TARGET/api/users?user_id=$id"; echo; done',
                "expected_indicator": "Consistent pattern of one ID returning drastically different size — investigate that specific ID manually",
                "technique": "Response size oracle for account enumeration — finds special accounts (admin, service, deleted) not returned by standard enumeration",
                "pivot_if_negative": "Apply same technique to other sequential ID ranges: orders, invoices, reports",
            },
        ],
    },
    {
        "type": "behavioral_diff",
        "scenarios": [
            {
                "observation": "DELETE /api/post/123 returns 403 for my user. DELETE /api/post/123 with added header X-Forwarded-For: 127.0.0.1 returns 204 No Content.",
                "hypothesis": "The access control check is conditional on the source IP. The server trusts X-Forwarded-For and treats localhost requests as admin/internal. This is an authentication bypass via IP spoofing header.",
                "probe": "curl -s -X DELETE TARGET/api/post/123 -H 'X-Forwarded-For: 127.0.0.1' -H 'X-Real-IP: 127.0.0.1' -H 'X-Original-IP: 127.0.0.1' -w '%{http_code}'",
                "expected_indicator": "204 or 200 response — any non-403 confirms IP-based access control bypass",
                "technique": "Header injection for IP-based auth bypass — not tested by standard scanners because they don\'t combine HTTP method + auth header + IP spoofing",
                "pivot_if_negative": "Try X-Forwarded-For: ::1 (IPv6 loopback), X-Forwarded-For: 10.0.0.1 (internal range), or X-Cluster-Client-IP",
            },
            {
                "observation": "POST /api/checkout with amount=100.00 succeeds. POST /api/checkout with amount=-100.00 also returns 200 with order_id.",
                "hypothesis": "The backend accepts negative amounts without validation. This may credit the account or create a negative-balance order that can be exploited to purchase items for free or extract balance.",
                "probe": "curl -s -X POST TARGET/api/checkout -H 'Content-Type: application/json' -d '{\"items\":[{\"id\":1,\"quantity\":1}],\"amount\":-0.01}' -b 'session=SESSION_TOKEN'",
                "expected_indicator": "200 response with order_id AND account balance increases OR item ships without payment",
                "technique": "Business logic: negative amount in payment flow — no scanner tests negative numeric values in payment parameters",
                "pivot_if_negative": "Try amount=0.00, amount=0.001 (below minimum), amount=99999999.99 (overflow), or amount=null",
            },
        ],
    },
    {
        "type": "tech_fingerprint",
        "scenarios": [
            {
                "observation": "Response headers include: X-Powered-By: Express, X-Request-Id: uuid. Error page for /api/nonexistent returns: 'Cannot GET /api/nonexistent' — characteristic Express.js 404.",
                "hypothesis": "The API is Node.js/Express. Express apps commonly use prototype pollution-vulnerable dependencies (lodash, jquery, merge). If the API accepts JSON body with nested objects, prototype pollution may be possible.",
                "probe": "curl -s -X POST TARGET/api/merge -H 'Content-Type: application/json' -d '{\"__proto__\":{\"isAdmin\":true}}' && curl -s TARGET/api/whoami",
                "expected_indicator": "If /api/whoami returns isAdmin:true or elevated privileges after the merge request, prototype pollution is confirmed",
                "technique": "Node.js prototype pollution — fingerprinted via Express error format, exploited via JSON body merge endpoint",
                "pivot_if_negative": "Try constructor.prototype injection: {\"constructor\":{\"prototype\":{\"isAdmin\":true}}}. Also check /api/clone, /api/update, /api/extend endpoints.",
            },
        ],
    },
]

# Diverse targets for synthetic chain generation
SYNTH_TARGETS = [
    {"target": "api.fintech-corp.com", "type": "financial API", "stack_hint": "Java/Spring, PostgreSQL"},
    {"target": "app.saas-startup.io", "type": "SaaS platform", "stack_hint": "Node.js/Express, MongoDB"},
    {"target": "portal.healthcare.com", "type": "healthcare portal", "stack_hint": "Python/Django, MySQL"},
    {"target": "api.ecommerce.io", "type": "e-commerce backend", "stack_hint": "Go microservices, Redis"},
    {"target": "platform.devtools.com", "type": "developer platform", "stack_hint": "Ruby/Rails, PostgreSQL"},
    {"target": "auth.enterprise.corp", "type": "SSO/auth service", "stack_hint": ".NET Core, MSSQL"},
    {"target": "cdn.mediahost.io", "type": "media delivery platform", "stack_hint": "Python/FastAPI, S3"},
    {"target": "rpc.blockchain.io", "type": "blockchain/crypto API", "stack_hint": "Rust/Actix, RocksDB"},
    {"target": "api.logistics.com", "type": "supply chain API", "stack_hint": "PHP/Laravel, MySQL"},
    {"target": "ml.ai-platform.io", "type": "ML inference platform", "stack_hint": "Python/Flask, Redis"},
]

SYNTH_CHAIN_PROMPT = """\
You are generating training data for a security researcher AI that finds novel vulnerabilities.

Target: {target} ({target_type}, stack: {stack_hint})

Standard tools were run and found NOTHING actionable:
{tool_failures}

However, during manual exploration, these anomalies were noticed:
{observations}

Generate a complete hypothesis-test-refine chain as a JSON object:
{{
  "context_summary": "<1-2 sentence summary of what standard tools found (nothing) and why the target is interesting>",
  "observations": ["<anomaly 1>", "<anomaly 2>", "<anomaly 3>"],
  "hypothesis": "<specific, testable hypothesis about what the anomalies indicate>",
  "probe": "<exact minimal command or HTTP request to test the hypothesis — not a full scan>",
  "expected_indicator": "<what in the probe response confirms OR refutes the hypothesis>",
  "result": "<what the probe actually returned — make it realistic, confirm the hypothesis>",
  "technique": "<name + 1-2 sentence explanation of the novel technique discovered>",
  "why_tools_miss_it": "<why no standard scanner catches this>",
  "pivot_if_negative": "<next hypothesis to test if the probe is negative>"
}}

Rules:
- The hypothesis must follow LOGICALLY from the observations — not be a generic vuln type
- The probe must be minimal — one curl command or one specific request, not a full scan
- The result must be realistic and confirm the hypothesis
- The technique name must be specific (e.g. "Go SSTI via export format parameter") not generic ("SSTI")
- why_tools_miss_it must explain the exact gap in scanner coverage
- Return valid JSON only.
"""

# ---------------------------------------------------------------------------
# Project Zero / GitHub source fetching
# ---------------------------------------------------------------------------
PZ_REPOS = [
    ("google", "security-research", "pocs"),
]

CTF_REPOS = [
    ("ctfs", "ctfs", ""),
    ("ctf-writeups", "ctf-writeups", ""),
    ("sajjadium", "ctf-writeups", ""),
]

CTF_NOVEL_KEYWORDS = [
    "no tool", "custom script", "timing", "manual exploit", "wrote a script",
    "bruteforce", "race condition", "prototype pollution", "deserialization",
    "ssti", "template injection", "ssrf", "request smuggling", "cache poison",
]


def _gpt_lock_instance() -> threading.Lock:
    return threading.Lock()


_gpt_lock = _gpt_lock_instance()
client = OpenAI()


def _gpt(prompt: str, max_tokens: int = 2000, model: str = "gpt-4o-mini") -> str:
    with _gpt_lock:
        time.sleep(API_DELAY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.9,
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


def _cached_get(url: str, headers: dict | None = None) -> str:
    slug = hashlib.md5(url.encode()).hexdigest()
    cache_path = Path(CACHE_DIR) / f"{slug}.txt"
    if cache_path.exists():
        return cache_path.read_text()
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, headers=headers or {}, timeout=15)
        r.raise_for_status()
        text = r.text
        cache_path.write_text(text)
        return text
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Synthetic chain generation (Type A)
# ---------------------------------------------------------------------------
def _pick_tool_failures(n: int = 3) -> list[dict]:
    return random.sample(STANDARD_TOOL_FAILURES, min(n, len(STANDARD_TOOL_FAILURES)))


def _pick_anomaly_scenario() -> tuple[str, dict]:
    """Pick a random anomaly type and one of its scenario templates."""
    template = random.choice(ANOMALY_TEMPLATES)
    scenario = random.choice(template["scenarios"])
    return template["type"], scenario


def generate_synth_chain() -> dict | None:
    target_info = random.choice(SYNTH_TARGETS)
    failures    = _pick_tool_failures(random.randint(2, 4))
    atype, anom = _pick_anomaly_scenario()

    failure_text = "\n".join(
        f"  {f['tool'].replace('TARGET', target_info['target'])}:\n    {f['verdict']}"
        for f in failures
    )
    obs_text = anom["observation"].replace("TARGET", target_info["target"])

    prompt = SYNTH_CHAIN_PROMPT.format(
        target=target_info["target"],
        target_type=target_info["type"],
        stack_hint=target_info["stack_hint"],
        tool_failures=failure_text,
        observations=obs_text,
    )
    try:
        raw  = _gpt(prompt, max_tokens=1500)
        data = _parse_json_obj(raw)
    except Exception:
        return None

    if not all(k in data for k in ("hypothesis", "probe", "result", "technique")):
        return None

    observations = data.get("observations") or [obs_text]
    obs_block    = "\n".join(f"  - {o}" for o in observations)
    failure_lines = "\n".join(f"  [{f['tool'][:30]}] → {f['verdict']}" for f in failures)

    user_content = (
        f"Target: {target_info['target']} ({target_info['type']})\n\n"
        f"Standard tools returned nothing actionable:\n{failure_lines}\n\n"
        f"During manual exploration, I noticed:\n{obs_block}\n\n"
        f"Standard approaches are exhausted. Reason from these anomalies."
    )
    asst_content = (
        f"OBSERVATIONS:\n{obs_block}\n\n"
        f"HYPOTHESIS: {data['hypothesis']}\n\n"
        f"PROBE: {data['probe']}\n\n"
        f"EXPECTED_INDICATOR: {data.get('expected_indicator', '')}\n\n"
        f"RESULT: {data['result']}\n\n"
        f"TECHNIQUE: {data['technique']}\n"
        f"WHY TOOLS MISS IT: {data.get('why_tools_miss_it', '')}\n\n"
        f"PIVOT_IF_NEGATIVE: {data.get('pivot_if_negative', 'Try adjacent parameters and endpoints.')}"
    )
    return {
        "type":       "synth",
        "anomaly_type": atype,
        "target_type": target_info["type"],
        "messages": [
            {"role": "system",    "content": SYSTEM_RESEARCHER},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst_content},
        ],
    }


# ---------------------------------------------------------------------------
# Project Zero reconstruction (Type B)
# ---------------------------------------------------------------------------
PZ_REFORMAT_PROMPT = """\
You are generating training data for a security researcher AI.

Below is a Project Zero / security research write-up describing how a vulnerability was found:
---
{writeup}
---

Reformat this into the hypothesis-chain training format. Extract:
1. What the researcher observed that was anomalous or unexpected
2. What hypothesis they formed
3. What minimal probe they used to test it
4. What the probe confirmed
5. Why existing tools would have missed it

Return a JSON object:
{{
  "context_summary": "<1-2 sentences: target system and why standard audits miss this>",
  "observations": ["<key anomaly or unexpected behavior 1>", "<anomaly 2>"],
  "hypothesis": "<what the researcher suspected and why>",
  "probe": "<the minimal test or command used to confirm>",
  "expected_indicator": "<what would confirm the hypothesis>",
  "result": "<what was actually observed>",
  "technique": "<specific name of the technique>",
  "why_tools_miss_it": "<exact gap in automated scanner coverage>",
  "pivot_if_negative": "<what the researcher would have tried next if wrong>"
}}

If the writeup doesn't contain enough information to fill all fields, infer plausible values
that are consistent with the described vulnerability class.
Return valid JSON only.
"""


def fetch_pz_writeups() -> list[dict]:
    """Fetch Project Zero PoC README files from google/security-research."""
    results = []
    pocs_api = f"{GITHUB_API}/repos/google/security-research/contents/pocs"
    pocs_raw = _cached_get(pocs_api, headers=GITHUB_HEADERS)
    if not pocs_raw:
        return results

    try:
        items = json.loads(pocs_raw)
    except Exception:
        return results

    for item in items:
        if item.get("type") != "dir":
            continue
        readme_url = f"{RAW_GITHUB}/google/security-research/master/pocs/{item['name']}/README.md"
        text = _cached_get(readme_url, headers={"User-Agent": "MythosEngine/1.0"})
        if len(text) > 300:
            results.append({"name": item["name"], "text": text[:6000]})
        time.sleep(0.1)
    return results


def generate_pz_example(writeup: dict) -> dict | None:
    prompt = PZ_REFORMAT_PROMPT.format(writeup=writeup["text"][:4000])
    try:
        raw  = _gpt(prompt, max_tokens=1200)
        data = _parse_json_obj(raw)
    except Exception:
        return None

    if not all(k in data for k in ("hypothesis", "probe", "technique")):
        return None

    obs_block = "\n".join(f"  - {o}" for o in data.get("observations", []))
    user_content = (
        f"Research target: {writeup['name'].replace('-', ' ')}\n\n"
        f"Standard security scanners found nothing. During manual review:\n{obs_block}\n\n"
        f"Reason from these observations."
    )
    asst_content = (
        f"OBSERVATIONS:\n{obs_block}\n\n"
        f"HYPOTHESIS: {data['hypothesis']}\n\n"
        f"PROBE: {data['probe']}\n\n"
        f"EXPECTED_INDICATOR: {data.get('expected_indicator', '')}\n\n"
        f"RESULT: {data.get('result', 'Hypothesis confirmed.')}\n\n"
        f"TECHNIQUE: {data['technique']}\n"
        f"WHY TOOLS MISS IT: {data.get('why_tools_miss_it', '')}\n\n"
        f"PIVOT_IF_NEGATIVE: {data.get('pivot_if_negative', 'Escalate to adjacent attack surface.')}"
    )
    return {
        "type":   "pz",
        "source": writeup["name"],
        "messages": [
            {"role": "system",    "content": SYSTEM_RESEARCHER},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst_content},
        ],
    }


# ---------------------------------------------------------------------------
# CTF novel writeups (Type C)
# ---------------------------------------------------------------------------
CTF_REFORMAT_PROMPT = """\
You are generating training data for a security researcher AI.

Below is a CTF challenge writeup:
---
{writeup}
---

This writeup is interesting because the solver had to reason beyond what standard tools provide.
Reformat it into the hypothesis-chain format, focusing on:
- What the solver observed that was unexpected
- What hypothesis they formed
- The minimal test that confirmed it

Return a JSON object:
{{
  "context_summary": "<1-2 sentences: challenge category and why tools weren't enough>",
  "observations": ["<key observation 1>", "<key observation 2>"],
  "hypothesis": "<what the solver suspected and the reasoning behind it>",
  "probe": "<the specific payload, command, or request used to verify>",
  "expected_indicator": "<what success looks like in the response>",
  "result": "<what they observed that confirmed the hypothesis>",
  "technique": "<specific technique name>",
  "why_tools_miss_it": "<why automated scanners wouldn't find this>",
  "pivot_if_negative": "<what they would have tried next>"
}}

Return valid JSON only. If the writeup is about a trivial technique (basic SQLi, easy XSS),
make the reasoning chain more nuanced to show HOW to reason, not just what the answer is.
"""

CTF_REPO_ENTRIES = [
    # GitHub repos with public CTF writeups
    ("CTFWriteups/ctf-writeups", "web"),
    ("reznok/CTF-writeups", ""),
    ("p4-team/ctf-writeups", ""),
    ("mrT4ntr4/CTF-writeups", "web"),
]


def fetch_ctf_writeups() -> list[dict]:
    """Fetch CTF writeup markdown files from GitHub."""
    results = []
    for repo_path, subdir in CTF_REPO_ENTRIES[:2]:
        owner, repo = repo_path.split("/")
        url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{subdir}"
        raw = _cached_get(url, headers=GITHUB_HEADERS)
        if not raw:
            continue
        try:
            items = json.loads(raw)
        except Exception:
            continue

        for item in items[:30]:
            if item.get("type") == "dir":
                # Recurse one level
                suburl = item.get("url", "")
                subraw = _cached_get(suburl, headers=GITHUB_HEADERS) if suburl else ""
                if subraw:
                    try:
                        subitems = json.loads(subraw)
                        for si in subitems[:5]:
                            if si.get("name", "").lower().endswith(".md"):
                                text = _cached_get(si["download_url"])
                                if len(text) > 300 and any(kw in text.lower() for kw in CTF_NOVEL_KEYWORDS):
                                    results.append({"name": si["name"], "repo": repo_path, "text": text[:5000]})
                    except Exception:
                        pass
            elif item.get("name", "").lower().endswith(".md"):
                text = _cached_get(item.get("download_url", ""))
                if len(text) > 300 and any(kw in text.lower() for kw in CTF_NOVEL_KEYWORDS):
                    results.append({"name": item["name"], "repo": repo_path, "text": text[:5000]})
            time.sleep(0.05)

    return results


def generate_ctf_example(writeup: dict) -> dict | None:
    prompt = CTF_REFORMAT_PROMPT.format(writeup=writeup["text"][:4000])
    try:
        raw  = _gpt(prompt, max_tokens=1200)
        data = _parse_json_obj(raw)
    except Exception:
        return None

    if not all(k in data for k in ("hypothesis", "probe", "technique")):
        return None

    obs_block = "\n".join(f"  - {o}" for o in data.get("observations", []))
    user_content = (
        f"Challenge: {writeup['name'].replace('-', ' ').replace('.md', '')}\n\n"
        f"Standard tools returned nothing actionable:\n  [scanner] → No known vulns found.\n\n"
        f"Observations from manual exploration:\n{obs_block}\n\n"
        f"Reason from these observations to find the vulnerability."
    )
    asst_content = (
        f"OBSERVATIONS:\n{obs_block}\n\n"
        f"HYPOTHESIS: {data['hypothesis']}\n\n"
        f"PROBE: {data['probe']}\n\n"
        f"EXPECTED_INDICATOR: {data.get('expected_indicator', '')}\n\n"
        f"RESULT: {data.get('result', 'Hypothesis confirmed — flag captured.')}\n\n"
        f"TECHNIQUE: {data['technique']}\n"
        f"WHY TOOLS MISS IT: {data.get('why_tools_miss_it', '')}\n\n"
        f"PIVOT_IF_NEGATIVE: {data.get('pivot_if_negative', 'Re-examine source code if available.')}"
    )
    return {
        "type":   "ctf",
        "source": writeup["name"],
        "messages": [
            {"role": "system",    "content": SYSTEM_RESEARCHER},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst_content},
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
    parser.add_argument("--type",            choices=["synth", "pz", "ctf", "all"], default="all")
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--workers",         type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--out-synth",       default=OUTPUT_SYNTH)
    parser.add_argument("--out-pz",          default=OUTPUT_PZ)
    parser.add_argument("--out-ctf",         default=OUTPUT_CTF)
    args = parser.parse_args()

    if args.list_categories:
        print(f"Synth  : {count_written(args.out_synth)}/{TARGET_SYNTH}")
        print(f"PZ     : {count_written(args.out_pz)}/{TARGET_PZ}")
        print(f"CTF    : {count_written(args.out_ctf)}/{TARGET_CTF}")
        print(f"\nAnomaly templates : {len(ANOMALY_TEMPLATES)} types")
        print(f"Synthetic targets : {len(SYNTH_TARGETS)}")
        print(f"Tool failure pool : {len(STANDARD_TOOL_FAILURES)}")
        return

    # ── TEST MODE ────────────────────────────────────────────────────────────
    if args.test:
        print("=" * 70)
        print("TEST MODE — source13_researcher  (no file writes)")
        print("=" * 70)

        if args.type in ("synth", "all"):
            print(f"\n[A] Synthetic exhaustion chains ({args.test_n} examples)...")
            for i in range(args.test_n):
                ex = generate_synth_chain()
                if ex:
                    print(f"\n  Example {i+1}:")
                    print(f"  Anomaly type  : {ex['anomaly_type']}")
                    print(f"  Target type   : {ex['target_type']}")
                    print(f"  User msg      : {ex['messages'][1]['content'][:250]!r}")
                    print(f"  Asst msg      : {ex['messages'][2]['content'][:400]!r}")
                else:
                    print(f"  Example {i+1}: FAILED")

        if args.type in ("pz", "all"):
            print(f"\n[B] Project Zero fetch test...")
            writeups = fetch_pz_writeups()
            print(f"  Fetched {len(writeups)} Project Zero PoC writeups")
            for w in writeups[:3]:
                print(f"    {w['name'][:50]}  ({len(w['text'])} chars)")
            if writeups:
                print(f"\n  Generating example from: {writeups[0]['name']}")
                ex = generate_pz_example(writeups[0])
                if ex:
                    print(f"  User msg : {ex['messages'][1]['content'][:250]!r}")
                    print(f"  Asst msg : {ex['messages'][2]['content'][:400]!r}")
                else:
                    print("  FAILED")

        if args.type in ("ctf", "all"):
            print(f"\n[C] CTF writeups fetch test...")
            writeups = fetch_ctf_writeups()
            print(f"  Fetched {len(writeups)} novel CTF writeups")
            for w in writeups[:3]:
                print(f"    {w['name'][:50]}  (repo: {w['repo']}, {len(w['text'])} chars)")
            if writeups:
                print(f"\n  Generating example from: {writeups[0]['name']}")
                ex = generate_ctf_example(writeups[0])
                if ex:
                    print(f"  User msg : {ex['messages'][1]['content'][:250]!r}")
                    print(f"  Asst msg : {ex['messages'][2]['content'][:400]!r}")
                else:
                    print("  FAILED")

        print("\n" + "=" * 70)
        print("TEST COMPLETE — run without --test to generate full dataset")
        return

    # ── FULL RUN ─────────────────────────────────────────────────────────────
    Path(args.out_synth).parent.mkdir(parents=True, exist_ok=True)
    total_written = 0

    if args.type in ("synth", "all"):
        existing = count_written(args.out_synth)
        needed   = TARGET_SYNTH - existing
        print(f"\n[A] Synth: {existing}/{TARGET_SYNTH}")

        def _synth_worker(_: int) -> int:
            ex = generate_synth_chain()
            return append_jsonl(args.out_synth, [ex] if ex else [])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_synth_worker, i): i for i in range(needed)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="synth"):
                total_written += fut.result()

    if args.type in ("pz", "all"):
        existing = count_written(args.out_pz)
        needed   = TARGET_PZ - existing
        print(f"\n[B] Project Zero: {existing}/{TARGET_PZ}")
        writeups = fetch_pz_writeups()
        print(f"  Fetched {len(writeups)} PZ writeups")

        if writeups:
            pool = writeups * (needed // max(len(writeups), 1) + 1)

            def _pz_worker(w: dict) -> int:
                ex = generate_pz_example(w)
                return append_jsonl(args.out_pz, [ex] if ex else [])

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_pz_worker, w): w for w in pool[:needed]}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="pz"):
                    total_written += fut.result()
        else:
            print("  PZ fetch failed — supplementing with synth examples")
            def _pz_synth_worker(_: int) -> int:
                ex = generate_synth_chain()
                if ex:
                    ex["type"] = "pz_synth"
                return append_jsonl(args.out_pz, [ex] if ex else [])

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_pz_synth_worker, i): i for i in range(needed)}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="pz-synth"):
                    total_written += fut.result()

    if args.type in ("ctf", "all"):
        existing = count_written(args.out_ctf)
        needed   = TARGET_CTF - existing
        print(f"\n[C] CTF writeups: {existing}/{TARGET_CTF}")
        writeups = fetch_ctf_writeups()
        print(f"  Fetched {len(writeups)} novel CTF writeups")

        if writeups:
            pool = writeups * (needed // max(len(writeups), 1) + 1)

            def _ctf_worker(w: dict) -> int:
                ex = generate_ctf_example(w)
                return append_jsonl(args.out_ctf, [ex] if ex else [])

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_ctf_worker, w): w for w in pool[:needed]}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="ctf"):
                    total_written += fut.result()
        else:
            print("  CTF fetch failed — supplementing with synth examples")
            def _ctf_synth_worker(_: int) -> int:
                ex = generate_synth_chain()
                if ex:
                    ex["type"] = "ctf_synth"
                return append_jsonl(args.out_ctf, [ex] if ex else [])

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_ctf_synth_worker, i): i for i in range(needed)}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="ctf-synth"):
                    total_written += fut.result()

    print(f"\nWrote {total_written} new examples total")
    print(f"  Synth : {args.out_synth}")
    print(f"  PZ    : {args.out_pz}")
    print(f"  CTF   : {args.out_ctf}")


if __name__ == "__main__":
    main()
