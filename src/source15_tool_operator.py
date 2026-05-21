#!/usr/bin/env python3
"""
Source 15: Tool Operator Training Data

The Tool Operator adapter reasons from first principles:
  1. Goal → discover what's installed → read --help → generate correct command
  2. Failure → read error + --help → correct or synthesise Python

Three data pools:
  A) First-principles generation  — goal + installed tools + real --help → correct command
  B) Failure correction           — bad command + error + real --help → corrected command
  C) Python synthesis             — tool unavailable/broken → Python 3 script

Data is grounded in REAL --help output from whatever is installed on this machine.
GPT reasons about the help text and generates training examples.

Usage:
  python3 src/source15_tool_operator.py               # full run
  python3 src/source15_tool_operator.py --test        # 3 samples per pool, no write
  python3 src/source15_tool_operator.py --pool A      # only pool A
  python3 src/source15_tool_operator.py --list-tools  # show installed tools
"""

import json
import os
import random
import shutil
import subprocess
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts import TOOL_OPERATOR as SYSTEM_PROMPT

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_PATH   = Path("raw/tool_operator.jsonl")
TARGET_TOTAL  = 2000
POOL_A_SHARE  = 0.40   # first-principles generation
POOL_B_SHARE  = 0.45   # failure correction
POOL_C_SHARE  = 0.15   # python synthesis
MAX_WORKERS   = 20
BATCH_SIZE    = 40

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Tool catalogue with goals ──────────────────────────────────────────────────

TOOL_GOALS: dict[str, list[str]] = {
    "subfinder": [
        "enumerate all subdomains of {target}",
        "passive subdomain discovery on {target}, save to file",
        "find subdomains of {target} using all sources, suppress banner",
        "subdomain enum with verbose output for {target}",
        "discover subdomains of {target} and resolve live hosts",
    ],
    "amass": [
        "enumerate subdomains of {target} with amass",
        "passive amass subdomain scan for {target}",
        "run amass enum on {target} and save to output file",
        "map attack surface of {target} with amass",
    ],
    "httpx": [
        "probe a list of hosts for live HTTP services",
        "check which URLs in a file are alive with status codes",
        "fingerprint tech stack of live hosts",
        "probe hosts.txt for live hosts and follow redirects",
    ],
    "gau": [
        "fetch all historical URLs for {target} from Wayback Machine",
        "collect all known URLs for {target} including subdomains",
        "enumerate endpoints for {target} using passive sources",
        "harvest URLs for {target} and save to file",
    ],
    "katana": [
        "crawl {target} and find all endpoints",
        "spider {target} including JS files to find API endpoints",
        "crawl {target} with custom headers and save output",
        "deep crawl of {target} to map attack surface",
    ],
    "nuclei": [
        "scan {target} for known CVE vulnerabilities",
        "run nuclei against {target} with a specific template",
        "vulnerability scan {target} with custom header",
        "scan a list of targets with nuclei and save results",
    ],
    "ffuf": [
        "fuzz directories on {target}",
        "discover hidden endpoints on {target} with wordlist",
        "fuzz parameters on {target}/api/v1/FUZZ",
        "vhost enumeration on {target}",
    ],
    "nmap": [
        "port scan {target} for open services",
        "full TCP scan of {target} with service detection",
        "quick scan common ports on {target}",
        "OS and service version detection on {target}",
    ],
    "dnsx": [
        "resolve a list of subdomains to IP addresses",
        "DNS brute-force on {target}",
        "extract A records from subdomains.txt",
        "find wildcard DNS for {target}",
    ],
    "curl": [
        "make authenticated GET request to {target}/api/v1/users",
        "POST JSON data to {target}/api/login with credentials",
        "follow redirects on {target} with custom header",
        "test endpoint {target}/api/admin with Bearer token",
    ],
    "sqlmap": [
        "test {target}/api/search?q=1 for SQL injection",
        "enumerate databases on vulnerable endpoint",
        "dump users table from {target} via SQLi",
        "test POST parameter for SQLi with custom header",
    ],
    "waybackurls": [
        "get all historical URLs for {target}",
        "enumerate old endpoints of {target} from Wayback Machine",
    ],
    "gobuster": [
        "directory brute-force on {target}",
        "DNS subdomain enumeration for {target}",
        "vhost fuzzing against {target}",
    ],
    "feroxbuster": [
        "recursive directory brute-force on {target}",
        "find hidden endpoints on {target} with wordlist",
    ],
    "nikto": [
        "web server vulnerability scan on {target}",
        "scan {target} for common misconfigurations",
    ],
    "wfuzz": [
        "fuzz {target}/api/v1/FUZZ for hidden endpoints",
        "fuzz POST parameters on {target}/login",
    ],
}

# Targets used in training examples
SAMPLE_TARGETS = [
    "supplier.meesho.com",
    "api.target.com",
    "app.example.com",
    "dev.startup.io",
    "staging.corp.com",
    "admin.webapp.net",
    "10.10.10.50",
    "192.168.1.100",
]

# Common flag mistakes → what the model typically gets wrong
FLAG_MISTAKES: list[dict] = [
    # gau
    {"tool": "gau",        "wrong": "gau -d {target}",                  "error": "unknown shorthand flag: 'd' in -d"},
    {"tool": "gau",        "wrong": "gau -u {target}",                  "error": "unknown shorthand flag: 'u' in -u"},
    {"tool": "gau",        "wrong": "gau {target} -o urls.txt",         "error": "unknown shorthand flag: 'o' in -o"},
    {"tool": "gau",        "wrong": "gau {target} -H 'X-Key: val'",     "error": "unknown shorthand flag: 'H' in -H"},
    # subfinder
    {"tool": "subfinder",  "wrong": "subfinder {target}",               "error": "Error: required flag(s) -d not set"},
    {"tool": "subfinder",  "wrong": "subfinder -d {target} --all",      "error": "unknown flag: --all"},
    # amass
    {"tool": "amass",      "wrong": "amass intel -d {target}",          "error": "no results — intel is wrong subcommand for subdomain enum"},
    {"tool": "amass",      "wrong": "amass -d {target}",                "error": "flag provided but not defined: -d"},
    # httpx (Python client)
    {"tool": "httpx",      "wrong": "httpx -l hosts.txt -status-code",  "error": "Error: No such option '-l'"},
    {"tool": "httpx",      "wrong": "httpx -u https://target.com -title","error": "Error: No such option '-u'"},
    {"tool": "httpx",      "wrong": "httpx hosts.txt --status-code",    "error": "Error: No such option '--status-code'"},
    # nmap
    {"tool": "nmap",       "wrong": "nmap --sV {target}",               "error": "nmap: unrecognized option '--sV'"},
    {"tool": "nmap",       "wrong": "nmap -A -O --script vuln {target} --open", "error": "unrecognized option '--open'"},
    # katana
    {"tool": "katana",     "wrong": "katana -url {target}",             "error": "flag provided but not defined: -url"},
    {"tool": "katana",     "wrong": "katana -u {target} -m auth",       "error": "flag provided but not defined: -m"},
    # nuclei
    {"tool": "nuclei",     "wrong": "nuclei -target {target}",          "error": "Error: No such option: --target"},
    {"tool": "nuclei",     "wrong": "nuclei -u {target} --templates cves/", "error": "Error: No such option: --templates"},
    # ffuf
    {"tool": "ffuf",       "wrong": "ffuf -url {target}/FUZZ -wordlist wordlist.txt", "error": "Flag provided but not defined: -url"},
    {"tool": "ffuf",       "wrong": "ffuf -u {target}/FUZZ -wordlist words.txt",      "error": "Flag provided but not defined: -wordlist"},
    # gobuster
    {"tool": "gobuster",   "wrong": "gobuster dir -url {target} -wordlist /wordlist.txt", "error": "unknown flag: --url"},
    {"tool": "gobuster",   "wrong": "gobuster dir -u {target} -wordlist /wordlist.txt",   "error": "unknown flag: --wordlist"},
    # dnsx
    {"tool": "dnsx",       "wrong": "dnsx -list subdomains.txt",        "error": "flag provided but not defined: -list"},
    {"tool": "dnsx",       "wrong": "dnsx subdomains.txt -a",           "error": "unexpected argument"},
]

# Python synthesis goals (tool not available or fundamentally wrong binary)
SYNTH_GOALS: list[dict] = [
    {
        "tool":    "httpx",
        "reason":  "Python httpx is installed (not projectdiscovery/httpx). It cannot probe multiple hosts or detect tech stacks.",
        "goal":    "probe a list of subdomains in subdomains.txt, check if each is live (HTTP 200/301), print URL and status code",
        "imports": ["requests", "concurrent.futures"],
    },
    {
        "tool":    "theHarvester",
        "reason":  "theHarvester is not installed.",
        "goal":    "collect emails and subdomains for {target} using crt.sh certificate transparency API",
        "imports": ["urllib.request", "json", "re"],
    },
    {
        "tool":    "shodan",
        "reason":  "shodan CLI is broken (missing pkg_resources).",
        "goal":    "query Shodan API for all IPs associated with org:{org} and print open ports",
        "imports": ["urllib.request", "json", "os"],
    },
    {
        "tool":    "masscan",
        "reason":  "masscan requires root and is not available.",
        "goal":    "check if ports 80, 443, 8080, 8443, 22 are open on {target} using TCP connect",
        "imports": ["socket", "concurrent.futures"],
    },
    {
        "tool":    "waybackurls",
        "reason":  "waybackurls is not installed.",
        "goal":    "fetch all archived URLs for {target} from the Wayback Machine CDX API",
        "imports": ["urllib.request", "json"],
    },
    {
        "tool":    "dnsx",
        "reason":  "dnsx is not installed.",
        "goal":    "resolve a list of subdomains from subdomains.txt to their A records using DNS",
        "imports": ["socket", "concurrent.futures"],
    },
    {
        "tool":    "gau",
        "reason":  "gau returned no results for this target.",
        "goal":    "fetch historical URLs for {target} from Wayback CDX API and Common Crawl index API",
        "imports": ["urllib.request", "json"],
    },
    {
        "tool":    "feroxbuster",
        "reason":  "feroxbuster is not installed.",
        "goal":    "brute-force directories on {target} using a wordlist, print paths returning 200/301/302",
        "imports": ["requests", "concurrent.futures"],
    },
]


# ── Helper: get real --help output ────────────────────────────────────────────

def get_help(binary: str) -> str | None:
    """Run `binary --help` and return output, or None if not installed."""
    if not shutil.which(binary):
        return None
    try:
        result = subprocess.run(
            [binary, "--help"],
            capture_output=True, text=True, timeout=10,
        )
        out = (result.stdout or result.stderr or "").strip()
        return out[:3000] if out else None
    except Exception:
        return None


def get_installed_tools() -> dict[str, str]:
    """Return {binary: help_text} for all installed tools in TOOL_GOALS."""
    installed = {}
    for binary in TOOL_GOALS:
        h = get_help(binary)
        if h:
            installed[binary] = h
    return installed


# ── GPT helpers ───────────────────────────────────────────────────────────────

def _gpt(prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 500) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def _gpt_batch(prompts: list[str], system: str = SYSTEM_PROMPT,
               max_tokens: int = 500, workers: int = MAX_WORKERS) -> list[str]:
    results = [""] * len(prompts)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_gpt, p, system, max_tokens): i for i, p in enumerate(prompts)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                results[i] = f.result()
            except Exception:
                results[i] = ""
    return results


# ── Pool A: first-principles generation ──────────────────────────────────────

POOL_A_SYSTEM = (
    "You are generating training data for a Tool Operator AI. "
    "Given a penetration testing GOAL, the INSTALLED TOOLS available, and the REAL --help "
    "output of the relevant tool, generate a training example showing how the AI should "
    "reason and produce the correct command.\n\n"
    "FORMAT your response as JSON:\n"
    "{\n"
    "  \"thought\": \"reasoning about which tool and flags to use\",\n"
    "  \"command\": \"the exact correct shell command\"\n"
    "}\n"
    "RULE: Only use flags that APPEAR in the provided --help output. No invented flags."
)


def generate_pool_a(installed: dict[str, str], n: int) -> list[dict]:
    examples = []
    targets = SAMPLE_TARGETS

    prompts = []
    meta = []
    for _ in range(n * 3):   # over-generate since some will fail
        binary = random.choice(list(installed.keys()))
        goal_template = random.choice(TOOL_GOALS.get(binary, ["use {target}"]))
        target = random.choice(targets)
        goal = goal_template.format(target=target)
        help_text = installed[binary]
        other_tools = [t for t in installed if t != binary][:4]

        p = (
            f"GOAL: {goal}\n\n"
            f"INSTALLED TOOLS: {', '.join([binary] + other_tools)}\n\n"
            f"--help output for {binary}:\n{help_text}\n\n"
            "Generate the correct command to achieve this goal using the help above."
        )
        prompts.append(p)
        meta.append({"binary": binary, "goal": goal, "target": target})

    resps = _gpt_batch(prompts, system=POOL_A_SYSTEM, max_tokens=300)

    for p, resp, m in zip(prompts, resps, meta):
        if not resp:
            continue
        try:
            clean = resp.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:]).rsplit("```", 1)[0].strip()
            data = json.loads(clean)
            thought  = data.get("thought", "").strip()
            command  = data.get("command",  "").strip()
            if not thought or not command:
                continue

            examples.append({"messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": p},
                {"role": "assistant", "content": f"<thought>{thought}</thought>\n<command>{command}</command>"},
            ]})
            if len(examples) >= n:
                break
        except Exception:
            continue
    return examples[:n]


# ── Pool B: failure correction ────────────────────────────────────────────────

POOL_B_SYSTEM = (
    "You are generating training data for a Tool Operator AI. "
    "Given a FAILED command, its ERROR output, and the REAL --help output, "
    "generate a training example showing exactly how to reason and fix the command.\n\n"
    "FORMAT your response as JSON:\n"
    "{\n"
    "  \"thought\": \"step-by-step diagnosis of what went wrong and how the help fixes it\",\n"
    "  \"corrected\": \"the exact correct command\"\n"
    "}\n"
    "RULE: Only use flags that APPEAR in the --help. If the binary is the wrong tool for the "
    "job (e.g. Python httpx cannot probe multiple hosts), say so and suggest curl instead."
)


def generate_pool_b(installed: dict[str, str], n: int) -> list[dict]:
    examples = []
    targets = SAMPLE_TARGETS

    prompts = []
    meta = []
    for _ in range(n * 3):
        mistake = random.choice(FLAG_MISTAKES)
        binary  = mistake["tool"]
        help_text = installed.get(binary)
        if not help_text:
            continue
        target  = random.choice(targets)
        wrong   = mistake["wrong"].format(target=target)
        error   = mistake["error"]

        p = (
            f"Tool: {binary}\n"
            f"Command attempted: {wrong}\n\n"
            f"Error output:\n{error}\n\n"
            f"Tool help (from running `{binary} --help`):\n{help_text}\n"
        )
        prompts.append(p)
        meta.append({"binary": binary, "wrong": wrong})

    resps = _gpt_batch(prompts, system=POOL_B_SYSTEM, max_tokens=400)

    for p, resp, m in zip(prompts, resps, meta):
        if not resp:
            continue
        try:
            clean = resp.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:]).rsplit("```", 1)[0].strip()
            data = json.loads(clean)
            thought   = data.get("thought",   "").strip()
            corrected = data.get("corrected", "").strip()
            if not thought or not corrected:
                continue
            if corrected == m["wrong"]:   # model returned same broken command
                continue

            examples.append({"messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": p},
                {"role": "assistant", "content": f"<thought>{thought}</thought>\n<corrected>{corrected}</corrected>"},
            ]})
            if len(examples) >= n:
                break
        except Exception:
            continue
    return examples[:n]


# ── Pool C: Python synthesis ──────────────────────────────────────────────────

POOL_C_SYSTEM = (
    "You are generating training data for a Tool Operator AI. "
    "The AI needs to write self-contained Python 3 scripts when a CLI tool is unavailable "
    "or fundamentally wrong for the job.\n\n"
    "FORMAT: JSON with keys:\n"
    "  \"thought\": why the tool can't be used and what the Python will do instead\n"
    "  \"code\":    complete self-contained Python 3 script (stdlib + requests only)\n\n"
    "RULES for the code:\n"
    "  - Must be complete and runnable without editing\n"
    "  - Print results to stdout\n"
    "  - Handle errors gracefully\n"
    "  - For network tasks, use a reasonable timeout (5-10s)\n"
    "  - Do NOT use requests if not in the allowed imports list"
)


def generate_pool_c(n: int) -> list[dict]:
    examples = []
    targets = SAMPLE_TARGETS
    orgs    = ["Meesho", "Target Corp", "StartupCo", "ExampleOrg"]

    prompts = []
    meta    = []
    for _ in range(n * 2):
        sg = random.choice(SYNTH_GOALS)
        target = random.choice(targets)
        org    = random.choice(orgs)
        goal   = sg["goal"].format(target=target, org=org)

        p = (
            f"Tool: {sg['tool']}\n"
            f"Reason unavailable: {sg['reason']}\n\n"
            f"Goal: {goal}\n\n"
            f"Allowed Python imports: {', '.join(sg['imports'])}\n\n"
            "Write a complete Python 3 script to achieve this goal."
        )
        prompts.append(p)
        meta.append(sg)

    resps = _gpt_batch(prompts, system=POOL_C_SYSTEM, max_tokens=800)

    for p, resp, sg in zip(prompts, resps, meta):
        if not resp:
            continue
        try:
            clean = resp.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:]).rsplit("```", 1)[0].strip()
            data = json.loads(clean)
            thought = data.get("thought", "").strip()
            code    = data.get("code",    "").strip()
            if not thought or not code or len(code) < 50:
                continue

            examples.append({"messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": p},
                {"role": "assistant", "content": f"<thought>{thought}</thought>\n```python\n{code}\n```"},
            ]})
            if len(examples) >= n:
                break
        except Exception:
            continue
    return examples[:n]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",       action="store_true", help="3 samples per pool, no write")
    parser.add_argument("--list-tools", action="store_true", help="show installed tools and exit")
    parser.add_argument("--pool",       choices=["A", "B", "C"], help="generate only one pool")
    parser.add_argument("--out",        default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print("Detecting installed tools…")
    installed = get_installed_tools()
    print(f"  Found {len(installed)} installed: {', '.join(sorted(installed))}")

    if args.list_tools:
        for binary, help_text in sorted(installed.items()):
            print(f"\n{'─'*60}")
            print(f"  {binary}  ({len(help_text)} chars help)")
            print(help_text[:200])
        return

    n_a = int(TARGET_TOTAL * POOL_A_SHARE)
    n_b = int(TARGET_TOTAL * POOL_B_SHARE)
    n_c = int(TARGET_TOTAL * POOL_C_SHARE)

    if args.test:
        n_a = n_b = n_c = 3

    dataset: list[dict] = []

    if not args.pool or args.pool == "A":
        print(f"\nPool A — first-principles generation ({n_a} examples)…")
        a = generate_pool_a(installed, n_a)
        print(f"  Generated {len(a)}")
        dataset.extend(a)

    if not args.pool or args.pool == "B":
        print(f"\nPool B — failure correction ({n_b} examples)…")
        b = generate_pool_b(installed, n_b)
        print(f"  Generated {len(b)}")
        dataset.extend(b)

    if not args.pool or args.pool == "C":
        print(f"\nPool C — Python synthesis ({n_c} examples)…")
        c = generate_pool_c(n_c)
        print(f"  Generated {len(c)}")
        dataset.extend(c)

    random.shuffle(dataset)
    print(f"\nTotal: {len(dataset)} examples")

    if args.test:
        for ex in dataset[:3]:
            msgs = ex["messages"]
            print(f"\n[user]: {msgs[1]['content'][:200]}")
            print(f"[asst]: {msgs[2]['content'][:300]}")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    print(f"Written → {out}")


if __name__ == "__main__":
    main()
