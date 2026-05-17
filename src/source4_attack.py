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
    "execution":              95,   # increased for synthetic fill
    "discovery":              95,
    "credential-access":      85,
    "persistence":            85,
    "privilege-escalation":   85,
    "defense-evasion":       105,
    "command-and-control":    65,
    "collection":             55,
    "impact":                 55,
    "lateral-movement":       55,   # sparse in Atomic Red Team → synthetic fill
    "exfiltration":           45,
    "initial-access":         40,   # sparse → synthetic fill
    "reconnaissance":         30,   # very sparse → synthetic fill
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
# Synthetic gap-fill for sparse categories
# ---------------------------------------------------------------------------

# Concrete technique seeds for categories that are sparse in Atomic Red Team
_SYNTH_TECHNIQUE_SEEDS: dict[str, list[tuple[str, str, str, str]]] = {
    "initial-access": [
        ("T1566.001", "Spearphishing Attachment",          "windows", "Send macro-laden Office doc via email; victim opens attachment triggering mshta.exe"),
        ("T1190",     "Exploit Public-Facing Application", "linux",   "Exploit Apache Log4j RCE (CVE-2021-44228) via crafted JNDI LDAP payload in User-Agent header"),
        ("T1133",     "External Remote Services",          "windows", "Gain initial access via exposed RDP with weak credentials; use Crowbar bruteforce"),
        ("T1078",     "Valid Accounts",                    "linux",   "Authenticate via SSH with leaked credentials found in GitHub repository"),
        ("T1195.002", "Compromise Software Supply Chain",  "linux",   "Inject malicious npm package version that executes reverse shell on install"),
        ("T1566.002", "Spearphishing Link",                "windows", "Send link to malicious HTA file hosted on attacker infra; HTA executes PowerShell stager"),
        ("T1091",     "Replication Through Removable Media", "windows", "USB drop with AutoRun.inf and malicious LNK file that spawns reverse shell"),
        ("T1190",     "Exploit Public-Facing Application", "linux",   "Exploit Apache Struts CVE-2017-5638 RCE via crafted Content-Type header"),
        ("T1190",     "Exploit Public-Facing Application", "linux",   "Exploit Atlassian Confluence CVE-2022-26134 OGNL injection for unauthenticated RCE"),
        ("T1078.003", "Valid Accounts: Local Accounts",    "linux",   "Spray discovered /etc/passwd users against SSH using hydra with common passwords"),
        ("T1566.001", "Spearphishing Attachment",          "windows", "Send ISO file with LNK inside to bypass Mark-of-the-Web; LNK calls rundll32 loader"),
        ("T1190",     "Exploit Public-Facing Application", "windows", "Exploit Exchange ProxyLogon CVE-2021-26855 + CVE-2021-27065 to write webshell"),
        ("T1078.001", "Valid Accounts: Default Accounts",  "linux",   "Access Tomcat manager with default admin:admin credentials and deploy WAR webshell"),
        ("T1133",     "External Remote Services",          "linux",   "Exploit exposed Citrix ADC (CVE-2019-19781) path traversal to gain initial foothold"),
        ("T1190",     "Exploit Public-Facing Application", "linux",   "Exploit GitLab CE RCE CVE-2021-22205 via unsafe ExifTool image processing"),
        ("T1566.003", "Spearphishing via Service",         "windows", "Deliver payload via Teams message with malicious link exploiting CVE-2023-36745"),
    ],
    "reconnaissance": [
        ("T1595.001", "Active Scanning: Scanning IP Blocks",           "linux", "Masscan the target /24 then nmap service scan on discovered open ports"),
        ("T1592.002", "Gather Victim Host Information: Software",      "linux", "Fingerprint target web stack via HTTP headers and Wappalyzer-style probing"),
        ("T1589.001", "Gather Victim Identity Information: Credentials","linux", "Search GitHub/Pastebin for leaked org credentials using custom dork queries"),
        ("T1596.005", "Search Open Technical Databases: Scan Databases","linux", "Query Shodan and Censys for exposed services and CVE fingerprints on target IP range"),
        ("T1593.001", "Search Open Websites/Domains: Social Media",    "linux", "Harvest employee names/roles via LinkedIn scraping using linkedin2username"),
        ("T1598.003", "Phishing for Information: Spearphishing Link",  "windows","Craft credential harvesting page mimicking VPN portal; gather credentials"),
        ("T1590.001", "Gather Victim Network Information: Domain Properties","linux","Passive DNS enumeration using dnsx, subfinder, and amass for subdomain discovery"),
        ("T1595.002", "Active Scanning: Vulnerability Scanning",       "linux", "Run nuclei against discovered web endpoints to fingerprint CVEs and misconfigs"),
        ("T1592.001", "Gather Victim Host Information: Hardware",       "linux", "SNMP walk against target network devices to enumerate hardware and software inventory"),
        ("T1591.002", "Gather Victim Org Information: Business Relationships","linux","Map third-party vendors and supply chain partners via WHOIS/ASN lookups and job postings"),
        ("T1596.001", "Search Open Technical Databases: DNS/Passive DNS","linux","Use SecurityTrails and DNSDB to enumerate historical DNS records and subdomains"),
        ("T1594",     "Search Victim-Owned Websites",                  "linux", "Spider target website with gospider collecting URLs, forms, and JavaScript endpoints"),
        ("T1589.002", "Gather Victim Identity Information: Email Addresses","linux","Run theHarvester against target domain to collect email addresses from public sources"),
        ("T1590.002", "Gather Victim Network Information: DNS",         "linux", "Zone transfer attempt with dig AXFR, fallback to dnsx brute-force subdomain enumeration"),
        ("T1593.002", "Search Open Websites/Domains: Search Engines",  "linux", "Google dork for exposed admin panels, config files, and credentials for target domain"),
        ("T1597.001", "Search Closed Sources: Threat Intel Vendors",   "linux", "Query VirusTotal and AlienVault OTX for known malware hashes and C2s linked to target"),
    ],
    "lateral-movement": [
        ("T1021.001", "Remote Services: Remote Desktop Protocol",       "windows", "Use harvested credentials to RDP into lateral target; enable clipboard for data transfer"),
        ("T1021.002", "Remote Services: SMB/Windows Admin Shares",      "windows", "Use impacket psexec with NTLM hash to spawn shell on lateral target via C$ share"),
        ("T1550.002", "Use Alternate Authentication Material: Pass the Hash","windows","Use mimikatz NTLM hash with wmiexec for lateral movement without plaintext password"),
        ("T1021.006", "Remote Services: Windows Remote Management",     "windows", "Invoke-Command over WinRM with harvested credentials for lateral code execution"),
        ("T1534",     "Internal Spearphishing",                         "windows", "Send phishing email from compromised internal mailbox to high-value target"),
        ("T1570",     "Lateral Tool Transfer",                          "linux",   "Use SCP from foothold to upload tools to lateral pivot over SSH tunnel"),
        ("T1563.002", "Remote Service Session Hijacking: RDP Hijacking","windows", "Hijack disconnected RDP session using tscon without password via SYSTEM token"),
        ("T1550.003", "Use Alternate Authentication Material: Pass the Ticket","windows","Import harvested Kerberos TGT with mimikatz ptt for lateral movement to DC"),
        ("T1021.004", "Remote Services: SSH",                           "linux",   "SSH from compromised host using private key found in /home/user/.ssh to lateral target"),
        ("T1021.003", "Remote Services: Distributed Component Object Model","windows","Use DCOM MMC20.Application method to execute code on remote host via impacket dcomexec"),
        ("T1080",     "Taint Shared Content",                           "windows", "Drop malicious macro-enabled template into network share accessed by target users"),
        ("T1021.005", "Remote Services: VNC",                           "linux",   "Connect to internal VNC server with extracted password from config file"),
        ("T1550.001", "Use Alternate Authentication Material: Application Access Token","linux","Reuse harvested JWT to authenticate to internal API service as admin user"),
        ("T1210",     "Exploitation of Remote Services",                "windows", "Use EternalBlue (MS17-010) against unpatched lateral host within internal network"),
        ("T1563.001", "Remote Service Session Hijacking: SSH Hijacking","linux",   "Hijack SSH agent socket via ControlMaster to connect to target without credentials"),
        ("T1072",     "Software Deployment Tools",                      "windows", "Abuse SCCM distribution point to deploy malicious package to target endpoint"),
    ],
    "exfiltration": [
        ("T1041",     "Exfiltration Over C2 Channel",                   "linux",   "Chunk and base64-encode sensitive files then exfil via existing C2 beacon traffic"),
        ("T1048.003", "Exfiltration Over Alternative Protocol: DNS",    "linux",   "Exfil data via DNS TXT record queries using dnscat2 tunnel to bypass firewall"),
        ("T1052.001", "Exfiltration Over Physical Medium: USB",         "windows", "Copy staged loot to mounted USB device via robocopy with volume shadow bypass"),
        ("T1567.002", "Exfiltration Over Web Service: Cloud Storage",   "linux",   "Upload encrypted loot to attacker-controlled S3 bucket using aws cli with stolen keys"),
        ("T1020",     "Automated Exfiltration",                         "linux",   "Scheduled cron job archives /home/**/.ssh and POSTs archive to attacker webhook nightly"),
        ("T1030",     "Data Transfer Size Limits",                      "linux",   "Split archive into 50MB chunks and exfil sequentially to avoid DLP size thresholds"),
        ("T1048.001", "Exfiltration Over Alternative Protocol: ICMP",   "linux",   "Encode loot in ICMP ping payload using icmpsh to bypass outbound TCP/UDP restrictions"),
        ("T1567.001", "Exfiltration Over Web Service: Exfil to Code Repo","linux", "Push sensitive files to attacker-controlled private GitHub repo using git push"),
        ("T1011.001", "Exfiltration Over Other Network Medium: Bluetooth","windows","Pair attacker phone via Bluetooth and transfer staged archive over OBEX FTP"),
        ("T1048.002", "Exfiltration Over Alternative Protocol: HTTPS",  "linux",   "POST base64-encoded data to attacker HTTPS server mimicking legitimate analytics beacon"),
        ("T1029",     "Scheduled Transfer",                             "linux",   "Set nightly cron to archive new files in /var/log and POST to webhook at 03:00"),
        ("T1537",     "Transfer Data to Cloud Account",                 "linux",   "Use rclone with attacker-controlled cloud provider credentials to sync loot directory"),
    ],
}

SYNTH_ATTACK_PROMPT = """\
You are simulating a MITRE ATT&CK red team exercise.

Technique: {technique_id} — {technique_name}
Tactic: {tactic}
Platform: {platform}
Scenario context: {hint}

Generate a realistic {n_turns}-turn attacker dialogue executing this technique.

For EACH turn output exactly:
THOUGHT: <2-3 sentence first-person reasoning about what you're doing and why>
COMMAND: <exact shell/CLI command — no placeholders, use realistic IPs/domains>
OUTPUT: <realistic terminal output, 3-8 lines>
---

Start directly with TURN 1.
"""


def _load_cat_counts_attack(output_path: str, targets: dict) -> dict[str, int]:
    counts: dict[str, int] = {c: 0 for c in targets}
    if Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ex = json.loads(line)
                    cat = ex["metadata"].get("tactic", "")
                    if cat in counts:
                        counts[cat] += 1
                except Exception:
                    pass
    return counts


def _parse_synth_attack_turns(raw: str) -> list[dict]:
    turns = []
    blocks = re.split(r"---+", raw)
    for block in blocks:
        thought = re.search(r"THOUGHT:\s*(.+?)(?=COMMAND:|$)", block, re.S)
        command = re.search(r"COMMAND:\s*(.+?)(?=OUTPUT:|$)", block, re.S)
        output  = re.search(r"OUTPUT:\s*(.+?)$", block, re.S)
        t = thought.group(1).strip() if thought else ""
        c = command.group(1).strip() if command else ""
        o = output.group(1).strip()  if output  else ""
        if t and c:
            turns.append({"thought": t, "command": c, "output": o})
    return turns


def _gpt_attack(prompt: str, max_tokens: int = 1000) -> str:
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.85,
    )
    return resp.choices[0].message.content or ""


def generate_synthetic_attack_example(
    technique_id: str, technique_name: str, tactic: str,
    platform: str, hint: str,
) -> dict | None:
    n_turns = random.randint(2, 4)
    try:
        raw = _gpt_attack(SYNTH_ATTACK_PROMPT.format(
            technique_id=technique_id, technique_name=technique_name,
            tactic=tactic, platform=platform, hint=hint, n_turns=n_turns,
        ), max_tokens=1100)
    except Exception:
        return None

    turns = _parse_synth_attack_turns(raw)
    if len(turns) < 2:
        return None

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "user",
        "content": (
            f"ATT&CK Technique: {technique_id} — {technique_name}\n"
            f"Tactic: {tactic}\nPlatform: {platform}\n\n{hint}\n\n"
            "What is your first action?"
        ),
    })

    for i, turn in enumerate(turns[:MAX_TURNS]):
        asst = (
            f"<thought>\n{turn['thought']}\n</thought>\n\n"
            f"<command>\n{turn['command']}\n</command>"
        )
        messages.append({"role": "assistant", "content": asst})
        if i < len(turns) - 1:
            nxt_out = turns[i + 1]["output"] or "(no output)"
            messages.append({
                "role": "user",
                "content": f"Output:\n```\n{nxt_out}\n```\n\nWhat is the next step?",
            })

    return {
        "messages": messages,
        "metadata": {
            "source":       "attack_synthetic",
            "technique_id": technique_id,
            "test_name":    f"synthetic_{technique_id}",
            "tactic":       tactic,
            "platform":     platform,
            "synthetic":    True,
        },
    }


def fill_synthetic_attack_gaps(
    output_path: str,
    targets: dict[str, int],
    workers: int = DEFAULT_WORKERS,
) -> None:
    """Top up sparse ATT&CK categories using seeded synthetic technique chains."""
    cat_counts = _load_cat_counts_attack(output_path, targets)
    gap_cats   = {c for c in targets if cat_counts.get(c, 0) < targets.get(c, 0)}
    if not gap_cats:
        print("\nAll ATT&CK categories at target — no synthetic fill needed.")
        return

    total_gap = sum(max(0, targets[c] - cat_counts.get(c, 0)) for c in gap_cats)
    print(f"\n[ATT&CK synthetic fill] {len(gap_cats)} categories below target, generating {total_gap} examples")
    print(f"  Under-target: {sorted(gap_cats)}")

    # Build work queue from seeds; cycle until categories are filled
    seeds: list[tuple[str, str, str, str, str]] = []
    for cat in gap_cats:
        for tid, tname, platform, hint in _SYNTH_TECHNIQUE_SEEDS.get(cat, []):
            seeds.append((tid, tname, cat, platform, hint))
    # For cats without specific seeds, generate generic ones
    for cat in gap_cats:
        if cat not in _SYNTH_TECHNIQUE_SEEDS:
            seeds.extend([
                (f"SYNTH-{cat}", cat.replace("-", " ").title(), cat, "linux", f"Generic {cat} technique scenario"),
                (f"SYNTH-{cat}", cat.replace("-", " ").title(), cat, "windows", f"Windows-specific {cat} technique scenario"),
            ])

    if not seeds:
        print("  No seeds available for synthetic generation.")
        return

    counts_lock = threading.Lock()
    pbar = tqdm(total=total_gap, desc="ATT&CK synthetic", unit="ex")

    def _gen(seed: tuple) -> int:
        tid, tname, cat, platform, hint = seed
        with counts_lock:
            if cat_counts.get(cat, 0) >= targets.get(cat, 0):
                return 0
        ex = generate_synthetic_attack_example(tid, tname, cat, platform, hint)
        if ex is None:
            return 0
        with counts_lock:
            if cat_counts.get(cat, 0) >= targets.get(cat, 0):
                return 0
            with open(output_path, "a") as f:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        return 1

    passes = 0
    while True:
        still_short = {c for c in targets if cat_counts.get(c, 0) < targets.get(c, 0)}
        if not still_short:
            break
        if passes > 20:
            print(f"  [warn] Pass limit reached. Remaining: {still_short}")
            break
        passes += 1
        random.shuffle(seeds)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = [executor.submit(_gen, seed) for seed in seeds]
            for fut in as_completed(futs):
                n = fut.result()
                if n:
                    pbar.update(n)

    pbar.close()
    print("\nFinal ATT&CK distribution after synthetic fill:")
    show_progress(cat_counts, targets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",            action="store_true",
                        help="Process 5 techniques, no file output")
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--workers",         type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--synthetic-only",  action="store_true",
                        help="Skip Atomic Red Team fetch; only run synthetic gap-fill")
    args = parser.parse_args()

    Path("raw").mkdir(exist_ok=True)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    if args.synthetic_only:
        print("[--synthetic-only] Skipping Atomic Red Team fetch, running synthetic gap-fill only")
        targets = dict(BENCH_TARGET_PER_CAT)
        fill_synthetic_attack_gaps(
            output_path = OUTPUT_PATH,
            targets     = targets,
            workers     = args.workers,
        )
        return

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
        return

    # Top up sparse categories with synthetic examples
    fill_synthetic_attack_gaps(
        output_path = OUTPUT_PATH,
        targets     = targets,
        workers     = args.workers,
    )


if __name__ == "__main__":
    main()
