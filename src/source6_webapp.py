#!/usr/bin/env python3
"""
Source 6: Web Application Vulnerability Attack Chains

Scrapes PayloadsAllTheThings (14 web vuln directories) and HackTricks
pentesting-web files, then uses GPT-4o-mini to generate realistic multi-turn
web application exploitation chains.

Each training example = one web vuln technique chain (2–5 turns):
  system : autonomous web application penetration tester
  user   : "Target: http://10.10.10.100 Type: <vuln>. Recon shows: <context>. First step?"
  asst   : "<thought>...</thought>\n\n<command>\n...\n</command>"
  user   : "Output:\n```\n...\n```\n\nWhat is the next step?"
  ...

Target: 700 examples across 12 web vulnerability categories.

Data sources:
  - swisskyrepo/PayloadsAllTheThings (14 web vuln directories)
  - carlospolop/hacktricks pentesting-web files (66 MD files)
  - GPT synthesis for gap-filling

Usage:
  python3 src/source6_webapp.py --test              # 5 vulns, no save
  python3 src/source6_webapp.py --list-categories
  python3 src/source6_webapp.py                     # full run (resumes)
  python3 src/source6_webapp.py --synthetic-only    # only GPT synthesis
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

# Import canonical prompt from single source of truth
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from prompts import WEBAPP as SYSTEM_PROMPT


load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GITHUB_API   = "https://api.github.com"
RAW_GITHUB   = "https://raw.githubusercontent.com"
OUTPUT_PATH  = "raw/webapp.jsonl"
CACHE_DIR    = "raw/.webapp_cache"
DEFAULT_WORKERS = 3
API_DELAY    = 0.3

MIN_TURNS = 2
MAX_TURNS = 5


THOUGHT_PROMPT = """\
Web vulnerability: {vuln_type}
Category: {category}
Target: http://10.10.10.100
Context so far:
{context}

The tester is about to run:
<command>
{command}
</command>

Write a 3-5 sentence internal thought (first person, present tense) explaining:
- WHY this specific payload/request at this stage
- WHAT vulnerability mechanic it tests or exploits
- WHAT response or behavior you expect

Use precise web security terminology (injection points, reflection, OGNL, deserialization, etc.).
No generic descriptions. Output ONLY the thought paragraph.
"""

WEBAPP_CHAIN_PROMPT = """\
You are building a training dataset for a web application pentesting AI.
Generate a realistic {n_turns}-turn web exploitation chain for:

Vulnerability: {vuln_type}
Category: {category}
Target: http://10.10.10.100
Context: {context}

For each turn produce EXACTLY:
<thought>
[3-5 sentences: why this command, what mechanic, what you expect]
</thought>

<command>
[exact command using real tools: curl, sqlmap, ffuf, burpsuite, python3, wfuzz,
 nuclei, nikto, gobuster, nmap, etc.]
</command>

Between turns insert a realistic OUTPUT: line (1-3 lines of actual tool output).

Rules:
- Use real tool syntax with realistic flags and parameters
- Progress logically: fingerprint → confirm vuln → exploit → escalate impact
- Use 10.10.10.100 as target, realistic endpoint paths (/login, /api/v1/user, etc.)
- Include realistic headers, cookies, and parameters where needed
- No placeholders. Output ONLY the turns, no preamble.
"""

# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------
CATEGORIES = [
    "sqli",
    "xss",
    "ssrf",
    "ssti",
    "command-injection",
    "xxe",
    "file-inclusion",
    "path-traversal",
    "idor",
    "deserialization",
    "jwt",
    "oauth",
]

BENCH_TARGET_PER_CAT: dict[str, int] = {
    "sqli":              80,
    "xss":               70,
    "ssrf":              65,
    "ssti":              60,
    "command-injection": 65,
    "xxe":               50,
    "file-inclusion":    55,
    "path-traversal":    55,
    "idor":              55,
    "deserialization":   60,
    "jwt":               45,
    "oauth":             40,
}

# ---------------------------------------------------------------------------
# PayloadsAllTheThings directory → category mapping
# ---------------------------------------------------------------------------
PATT_DIRS: dict[str, str] = {
    "SQL Injection":          "sqli",
    "XSS Injection":          "xss",
    "Server Side Request Forgery": "ssrf",
    "Server Side Template Injection": "ssti",
    "Command Injection":      "command-injection",
    "XXE Injection":          "xxe",
    "File Inclusion":         "file-inclusion",
    "Directory Traversal":    "path-traversal",
    "Insecure Direct Object References": "idor",
    "Insecure Deserialization": "deserialization",
    "JSON Web Token":         "jwt",
    "OAuth Misconfiguration": "oauth",
    "NoSQL Injection":        "sqli",
    "GraphQL Injection":      "ssrf",
    "LDAP Injection":         "sqli",
    "XPATH Injection":        "sqli",
    "XSLT Injection":         "ssti",
    "Open Redirect":          "ssrf",
}

# ---------------------------------------------------------------------------
# Technique pool for synthetic generation
# ---------------------------------------------------------------------------
TECHNIQUE_POOL: dict[str, list[dict]] = {
    "sqli": [
        {"vuln": "Union-based SQL injection via search parameter",
         "ctx": "Login page with username parameter reflected in SQL query"},
        {"vuln": "Boolean-based blind SQL injection",
         "ctx": "User ID parameter with no visible output but different behavior"},
        {"vuln": "Time-based blind SQL injection via sleep() injection",
         "ctx": "POST parameter with delay-based response confirmation"},
        {"vuln": "Error-based SQL injection via extractvalue()",
         "ctx": "MySQL error messages visible in response"},
        {"vuln": "Second-order SQL injection via stored username",
         "ctx": "Profile update functionality with stored injection"},
        {"vuln": "SQL injection to file read via LOAD_FILE()",
         "ctx": "MySQL with FILE privilege, path traversal to /etc/passwd"},
    ],
    "xss": [
        {"vuln": "Reflected XSS via search parameter",
         "ctx": "Search box reflecting input in page without encoding"},
        {"vuln": "Stored XSS in comment/profile field",
         "ctx": "User-controlled content rendered on other users' pages"},
        {"vuln": "DOM-based XSS via window.location.hash",
         "ctx": "JavaScript reading URL fragment and writing to innerHTML"},
        {"vuln": "XSS filter bypass with JavaScript protocol in href",
         "ctx": "URL input field filtered for <script> but not href"},
        {"vuln": "Blind XSS in admin panel via contact form",
         "ctx": "Support ticket form sent to admin, OAST payload"},
    ],
    "ssrf": [
        {"vuln": "SSRF via URL parameter reaching AWS metadata service",
         "ctx": "Image fetch endpoint accepting external URLs"},
        {"vuln": "Blind SSRF via webhook URL to internal network",
         "ctx": "Notification webhook that fetches the provided URL"},
        {"vuln": "SSRF in PDF generation to read internal files",
         "ctx": "Invoice PDF generator with HTML-to-PDF rendering"},
        {"vuln": "SSRF via XML external entity to internal service",
         "ctx": "XML parser in API endpoint, OOB detection via DNS"},
        {"vuln": "SSRF bypass via DNS rebinding attack",
         "ctx": "SSRF filter blocking private IPs but not checking after resolution"},
    ],
    "ssti": [
        {"vuln": "Jinja2 SSTI to RCE via Mako template injection",
         "ctx": "Error page reflects template expression {{7*7}} = 49"},
        {"vuln": "Twig SSTI in PHP application via name parameter",
         "ctx": "Greeting page renders user-controlled name via Twig"},
        {"vuln": "Velocity SSTI in Java application",
         "ctx": "Email template preview endpoint with SSTI pattern"},
        {"vuln": "Freemarker SSTI to arbitrary file read",
         "ctx": "Template engine configured with dangerous defaults"},
    ],
    "command-injection": [
        {"vuln": "OS command injection via ping utility parameter",
         "ctx": "Network diagnostics endpoint: /ping?host=127.0.0.1"},
        {"vuln": "Blind command injection via sleep in DNS lookup",
         "ctx": "WHOIS lookup functionality, no visible output"},
        {"vuln": "Command injection via filename in upload endpoint",
         "ctx": "File conversion tool executing system command on filename"},
        {"vuln": "Shell metacharacter injection in nslookup parameter",
         "ctx": "DNS lookup tool with semicolon bypass"},
    ],
    "xxe": [
        {"vuln": "Basic XXE to read /etc/passwd via DOCTYPE",
         "ctx": "XML-parsing API endpoint with external entities enabled"},
        {"vuln": "Blind XXE via OOB DNS exfiltration",
         "ctx": "XML parser, no visible output, OOB via parameter entity"},
        {"vuln": "XXE via SVG file upload for SSRF",
         "ctx": "Image upload accepting SVG, renders server-side"},
        {"vuln": "XXE in SAML authentication request",
         "ctx": "SAML SP with XML signature validation before parsing"},
    ],
    "file-inclusion": [
        {"vuln": "Local File Inclusion via page parameter",
         "ctx": "PHP include() with unsanitized page parameter"},
        {"vuln": "LFI to RCE via /proc/self/environ log poisoning",
         "ctx": "LFI confirmed, Apache access log poisoning"},
        {"vuln": "Remote File Inclusion via include URL wrapper",
         "ctx": "PHP with allow_url_include=On, remote file accepted"},
        {"vuln": "LFI bypass with path traversal + null byte",
         "ctx": "PHP LFI with extension filtering bypassed with ../../../../etc/passwd%00"},
    ],
    "path-traversal": [
        {"vuln": "Path traversal to read /etc/passwd via ../",
         "ctx": "File download endpoint: /download?file=report.pdf"},
        {"vuln": "Path traversal in ZIP extraction (Zip Slip)",
         "ctx": "File upload endpoint extracting ZIP archives"},
        {"vuln": "Apache path traversal CVE-2021-41773",
         "ctx": "Apache 2.4.49 with mod_cgi enabled"},
        {"vuln": "Nginx alias path traversal misconfiguration",
         "ctx": "Nginx alias /static to /var/www/static/ with traversal"},
    ],
    "idor": [
        {"vuln": "IDOR via numeric user ID in profile API",
         "ctx": "GET /api/v1/user/1234 returns own profile, try /1235"},
        {"vuln": "IDOR in order history with UUID guessable via sequential IDs",
         "ctx": "GET /api/orders/10001 — increment to access other users"},
        {"vuln": "IDOR to account takeover via password reset token",
         "ctx": "Password reset endpoint using predictable user ID"},
        {"vuln": "Horizontal IDOR via JWT sub claim manipulation",
         "ctx": "JWT contains userId, change to another user's ID"},
    ],
    "deserialization": [
        {"vuln": "Java deserialization via Apache Commons Collections gadget",
         "ctx": "Java application deserializing Base64 from cookie"},
        {"vuln": "PHP object injection via unserialize() in cookie",
         "ctx": "PHP cookie contains serialized object, magic methods exploitable"},
        {"vuln": "Python Pickle deserialization RCE via API endpoint",
         "ctx": "Python API deserializing pickle from POST body"},
        {"vuln": "Node.js JavaScript deserialization via node-serialize",
         "ctx": "Express.js app with base64+JSON session deserialization"},
    ],
    "jwt": [
        {"vuln": "JWT algorithm confusion attack (RS256 to HS256)",
         "ctx": "JWT using RS256, public key obtainable from JWKS endpoint"},
        {"vuln": "JWT none algorithm bypass",
         "ctx": "JWT header alg changed to none, signature stripped"},
        {"vuln": "JWT weak secret brute-force with hashcat",
         "ctx": "JWT with HMAC-SHA256, secret is a weak dictionary word"},
        {"vuln": "JWT kid header injection for SQL/path traversal",
         "ctx": "JWT kid parameter used in database lookup without sanitization"},
    ],
    "oauth": [
        {"vuln": "OAuth redirect_uri manipulation for code theft",
         "ctx": "OAuth flow with redirect_uri not strictly validated"},
        {"vuln": "OAuth CSRF via missing state parameter",
         "ctx": "Authorization endpoint lacking CSRF state parameter"},
        {"vuln": "OAuth token leakage via Referer header",
         "ctx": "Access token in URL fragment leaks via Referer to analytics"},
        {"vuln": "OAuth account takeover via unverified email claim",
         "ctx": "OAuth provider returns unverified email, app trusts it for account lookup"},
    ],
}

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (research/dataset-builder; contact@example.com)",
    "Accept":     "application/vnd.github.v3+json",
}

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


def _cache_key(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:14]


def _cached_get(url: str) -> str | None:
    cp = Path(CACHE_DIR) / f"{_cache_key(url)}.txt"
    if cp.exists():
        return cp.read_text(encoding="utf-8", errors="replace")
    tok = os.getenv("GITHUB_TOKEN")
    hdrs = dict(REQUEST_HEADERS)
    if tok:
        hdrs["Authorization"] = f"token {tok}"
    try:
        r = requests.get(url, headers=hdrs, timeout=20)
        if r.status_code == 200:
            text = r.text
            cp.write_text(text, encoding="utf-8")
            return text
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# PayloadsAllTheThings discovery
# ---------------------------------------------------------------------------
def get_patt_dirs() -> list[dict]:
    """Return list of {name, raw_url, category} for PATT vuln directories."""
    cache_path = Path(CACHE_DIR) / "patt_dirs.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    entries = []
    for dir_name, cat in PATT_DIRS.items():
        readme_url = (
            f"{RAW_GITHUB}/swisskyrepo/PayloadsAllTheThings/master/"
            f"{dir_name.replace(' ', '%20')}/README.md"
        )
        entries.append({"name": dir_name, "raw_url": readme_url, "category": cat})

    cache_path.write_text(json.dumps(entries))
    return entries


def parse_webapp_markdown(text: str, category: str) -> tuple[str, str, list[str]]:
    """
    Parse a PayloadsAllTheThings or HackTricks markdown page.
    Returns (title, summary, commands[]).
    """
    lines = text.splitlines()
    title = ""
    for l in lines[:6]:
        if l.startswith("#"):
            title = l.lstrip("#").strip()
            break

    # Summary from first prose lines
    prose = []
    for l in lines[:30]:
        s = l.strip()
        if s and not s.startswith("#") and not s.startswith("```") and not s.startswith("!"):
            prose.append(s)
            if len(prose) >= 3:
                break
    summary = " ".join(prose)[:400]

    # Extract relevant commands from code blocks
    commands: list[str] = []
    in_block = False
    block_lines: list[str] = []

    for line in lines:
        if line.strip().startswith("```"):
            if not in_block:
                in_block = True
                block_lines = []
            else:
                in_block = False
                cmd = "\n".join(block_lines).strip()
                if _is_webapp_command(cmd, category):
                    commands.append(cmd)
                block_lines = []
            continue
        if in_block:
            block_lines.append(line)

    return title, summary, commands[:8]


_WEBAPP_TOOLS = re.compile(
    r"(curl|sqlmap|ffuf|burp|python3?|wfuzz|nuclei|gobuster|dirb|nikto"
    r"|wget|nc\s|ncat|jwt|hashcat|john|ysoserial|ysoserial\.net|requests\."
    r"|xmllint|xxeinjector|ghauri|gf\s|gron|jq\s|base64|echo.*\|)",
    re.IGNORECASE
)

_PAYLOAD_KEYWORDS = re.compile(
    r"(\bSELECT\b|\bUNION\b|<script|OAST|blind|payload|inject|bypass|exploit"
    r"|\$\{|#\{|{{|sleep\(|waitfor|xp_cmdshell|/etc/passwd|/proc/self"
    r"|../|%2e%2e|jndi:|rmi://|\btoken\b|\bsecret\b)",
    re.IGNORECASE
)


def _is_webapp_command(cmd: str, category: str) -> bool:
    if len(cmd) < 10 or len(cmd) > 3000:
        return False
    if _WEBAPP_TOOLS.search(cmd):
        return True
    if _PAYLOAD_KEYWORDS.search(cmd):
        return True
    return False


# ---------------------------------------------------------------------------
# GPT helpers
# ---------------------------------------------------------------------------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_gpt_lock = threading.Lock()


def _gpt(prompt: str, max_tokens: int = 500) -> str:
    with _gpt_lock:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.8,
        )
    return resp.choices[0].message.content.strip()


def _generate_thought(vuln_type: str, category: str, command: str, context: str) -> str:
    prompt = THOUGHT_PROMPT.format(
        vuln_type=vuln_type,
        category=category,
        command=command[:400],
        context=context[-600:] if context else "No prior actions.",
    )
    return _gpt(prompt, max_tokens=250)


# ---------------------------------------------------------------------------
# Build example from parsed MD
# ---------------------------------------------------------------------------
def build_example_from_md(
    source: str,
    title: str,
    summary: str,
    commands: list[str],
    category: str,
    slug: str,
) -> dict | None:
    if len(commands) < MIN_TURNS:
        return None

    first_user = (
        f"Target: http://10.10.10.100\n"
        f"Vulnerability type: {title or category}\n"
        f"Category: {category}\n\n"
        f"{summary[:300]}\n\n"
        f"Start your exploitation. What is your first step?"
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": first_user},
    ]

    context = f"Target vuln: {title or category}\n"
    for idx, cmd in enumerate(commands[:MAX_TURNS]):
        if not cmd.strip():
            continue
        try:
            thought = _generate_thought(title or category, category, cmd, context)
        except Exception:
            thought = f"Executing step {idx+1} for {category} exploitation."

        asst = f"<thought>\n{thought}\n</thought>\n\n<command>\n{cmd}\n</command>"
        messages.append({"role": "assistant", "content": asst})
        context += f"\nStep {idx+1}: {cmd[:100]}"

        if idx < len(commands) - 1:
            messages.append({"role": "user", "content": "Noted. What is the next step?"})

    if messages[-1]["role"] != "assistant":
        messages = messages[:-1]

    n_asst = sum(1 for m in messages if m["role"] == "assistant")
    if n_asst < MIN_TURNS:
        return None

    return {
        "messages": messages,
        "metadata": {
            "source":   "webapp",
            "vuln":     title or category,
            "category": category,
            "turns":    n_asst,
            "url":      source,
            "slug":     slug,
        },
    }


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------
def generate_synthetic_chain(
    vuln_type: str,
    category: str,
    context: str,
    n_turns: int = 3,
) -> dict | None:
    prompt = WEBAPP_CHAIN_PROMPT.format(
        vuln_type=vuln_type,
        category=category,
        context=context,
        n_turns=n_turns,
    )
    with _gpt_lock:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1400,
                temperature=0.85,
            )
        except Exception:
            return None
    raw = resp.choices[0].message.content.strip()

    thought_pat = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)
    command_pat = re.compile(r"<command>(.*?)</command>", re.DOTALL)
    output_pat  = re.compile(r"^OUTPUT:\s*(.+?)(?=\n\n|<thought>|$)", re.MULTILINE | re.DOTALL)

    thoughts = [t.strip() for t in thought_pat.findall(raw)]
    commands = [c.strip() for c in command_pat.findall(raw)]
    outputs  = [o.strip() for o in output_pat.findall(raw)]

    if not thoughts or not commands or len(thoughts) != len(commands):
        return None
    if len(thoughts) < MIN_TURNS:
        return None

    first_user = (
        f"Target: http://10.10.10.100\n"
        f"Vulnerability: {vuln_type}\n"
        f"Category: {category}\n\n"
        f"{context}\n\n"
        f"Start your exploitation. What is your first step?"
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": first_user},
    ]

    for i, (thought, cmd) in enumerate(zip(thoughts, commands)):
        if not cmd.strip():
            continue
        asst = f"<thought>\n{thought}\n</thought>\n\n<command>\n{cmd}\n</command>"
        messages.append({"role": "assistant", "content": asst})

        if i < len(thoughts) - 1:
            out = outputs[i].strip() if i < len(outputs) else ""
            if out:
                next_user = f"Output:\n```\n{out[:600]}\n```\n\nWhat is the next step?"
            else:
                next_user = "Request sent. What is the next step?"
            messages.append({"role": "user", "content": next_user})

    if messages[-1]["role"] != "assistant":
        messages = messages[:-1]

    n_asst = sum(1 for m in messages if m["role"] == "assistant")
    if n_asst < MIN_TURNS:
        return None

    slug = f"synth_{category}_{_cache_key(vuln_type + context)}"
    return {
        "messages": messages,
        "metadata": {
            "source":    "webapp",
            "vuln":      vuln_type,
            "category":  category,
            "turns":     n_asst,
            "synthetic": True,
            "url":       "https://github.com/swisskyrepo/PayloadsAllTheThings",
            "slug":      slug,
        },
    }


def fill_synthetic_gaps(
    cat_counts: dict[str, int],
    seen_slugs: set[str],
    out_file,
) -> int:
    written = 0
    for cat, target in BENCH_TARGET_PER_CAT.items():
        current = cat_counts.get(cat, 0)
        if current >= target:
            continue

        pool = TECHNIQUE_POOL.get(cat, [])
        if not pool:
            continue

        needed = target - current
        print(f"  Synthetic fill: {cat} needs {needed} more examples")

        pool_idx = 0
        attempts = 0
        while cat_counts.get(cat, 0) < target and attempts < needed * 4:
            attempts += 1
            spec = pool[pool_idx % len(pool)]
            pool_idx += 1

            slug = f"synth_{cat}_{pool_idx}"
            if slug in seen_slugs:
                continue

            n_turns = random.choice([2, 3, 3, 4])
            ex = generate_synthetic_chain(spec["vuln"], cat, spec["ctx"], n_turns)
            if ex:
                ex["metadata"]["slug"] = slug
                out_file.write(json.dumps(ex) + "\n")
                out_file.flush()
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                seen_slugs.add(slug)
                written += 1

            time.sleep(0.2)

    return written


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def load_existing_counts(output_path: str) -> dict[str, int]:
    counts = {cat: 0 for cat in CATEGORIES}
    if not os.path.exists(output_path):
        return counts
    with open(output_path) as f:
        for line in f:
            try:
                cat = json.loads(line)["metadata"].get("category", "sqli")
                if cat in counts:
                    counts[cat] += 1
            except Exception:
                pass
    return counts


def load_existing_slugs(output_path: str) -> set[str]:
    seen: set[str] = set()
    if not os.path.exists(output_path):
        return seen
    with open(output_path) as f:
        for line in f:
            try:
                seen.add(json.loads(line)["metadata"].get("slug", ""))
            except Exception:
                pass
    return seen


def print_progress(counts: dict[str, int]) -> None:
    total_done   = sum(counts.values())
    total_target = sum(BENCH_TARGET_PER_CAT.values())
    print(f"\nCategory               Done   Target   Progress")
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
    parser.add_argument("--test",            action="store_true")
    parser.add_argument("--test-n",          type=int, default=5)
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--out",             default=OUTPUT_PATH)
    parser.add_argument("--no-resume",       action="store_true")
    parser.add_argument("--workers",         type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--synthetic-only",  action="store_true")
    args = parser.parse_args()

    if args.list_categories:
        print_progress(load_existing_counts(args.out))
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.test:
        print("=" * 70)
        print(f"TEST MODE — {args.test_n} synthetic webapp chains")
        print("=" * 70)
        pool_items = [(cat, spec)
                      for cat, specs in TECHNIQUE_POOL.items()
                      for spec in specs]
        sample = random.sample(pool_items, min(args.test_n, len(pool_items)))
        for cat, spec in sample:
            print(f"\n── {cat} ──  {spec['vuln']}")
            ex = generate_synthetic_chain(spec["vuln"], cat, spec["ctx"], n_turns=3)
            if ex:
                msgs = ex["messages"]
                n_a  = sum(1 for m in msgs if m["role"] == "assistant")
                first_asst = next(m for m in msgs if m["role"] == "assistant")
                print(f"  Turns : {n_a}")
                print(f"  First : {first_asst['content'][:200]!r}")
            else:
                print("  FAILED")
        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        return

    resume     = not args.no_resume
    cat_counts = load_existing_counts(args.out) if resume else {c: 0 for c in CATEGORIES}
    seen_slugs = load_existing_slugs(args.out)  if resume else set()
    written    = 0

    # ── Phase 1: PayloadsAllTheThings scrape ──────────────────────────────
    if not args.synthetic_only:
        print("Fetching PayloadsAllTheThings directories...")
        patt_entries = get_patt_dirs()
        print(f"Found {len(patt_entries)} PATT directories")
        print_progress(cat_counts)

        todo = [e for e in patt_entries
                if e["name"] not in seen_slugs
                and cat_counts.get(e["category"], 0) < BENCH_TARGET_PER_CAT.get(e["category"], 0)]

        def _worker(entry: dict) -> dict | None:
            cat = entry["category"]
            text = _cached_get(entry["raw_url"])
            if not text:
                return None
            time.sleep(API_DELAY)
            title, summary, commands = parse_webapp_markdown(text, cat)
            if len(commands) < MIN_TURNS:
                return None
            slug = _cache_key(entry["name"])
            return build_example_from_md(
                f"https://github.com/swisskyrepo/PayloadsAllTheThings/tree/master/{entry['name']}",
                title, summary, commands, cat, slug
            )

        with open(args.out, "a") as outf:
            with tqdm(total=len(todo), desc="PATT dirs") as pbar:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(_worker, e): e for e in todo}
                    for fut in as_completed(futures):
                        pbar.update(1)
                        result = fut.result()
                        if result:
                            cat = result["metadata"]["category"]
                            if cat_counts.get(cat, 0) < BENCH_TARGET_PER_CAT.get(cat, 0):
                                outf.write(json.dumps(result) + "\n")
                                outf.flush()
                                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                                seen_slugs.add(result["metadata"]["slug"])
                                written += 1

        print(f"\nPATT pass done. Wrote {written} examples.")
        print_progress(load_existing_counts(args.out))

    # ── Phase 2: Synthetic gap-fill ────────────────────────────────────────
    cat_counts = load_existing_counts(args.out)
    seen_slugs = load_existing_slugs(args.out)
    gap = sum(max(0, BENCH_TARGET_PER_CAT[c] - cat_counts.get(c, 0)) for c in BENCH_TARGET_PER_CAT)

    if gap > 0:
        print(f"\nStarting synthetic gap-fill: {gap} examples needed")
        with open(args.out, "a") as outf:
            s = fill_synthetic_gaps(cat_counts, seen_slugs, outf)
        written += s
        print(f"Synthetic fill done. Wrote {s} synthetic examples.")
    else:
        print("No gap-fill needed.")

    print(f"\nTotal written this run: {written}")
    print_progress(load_existing_counts(args.out))


if __name__ == "__main__":
    main()
