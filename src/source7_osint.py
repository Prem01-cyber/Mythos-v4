#!/usr/bin/env python3
"""
Source 7: OSINT and External Reconnaissance Chains

Generates realistic multi-turn OSINT/recon scenarios using:
  - HackTricks external-recon-methodology files (4 MD files)
  - GPT synthesis of tool-based recon chains (amass, theHarvester, Shodan, etc.)

Each training example = one recon methodology chain (2–5 turns):
  system : autonomous OSINT / recon specialist
  user   : "Target: corp.com. Phase: external recon. What is your first step?"
  asst   : "<thought>...</thought>\n\n<command>\n...\n</command>"
  user   : "Output:\n```\n...\n```\n\nWhat is the next step?"
  ...

Target: 400 examples across 7 OSINT/recon categories.

Usage:
  python3 src/source7_osint.py --test
  python3 src/source7_osint.py --list-categories
  python3 src/source7_osint.py
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
GITHUB_API   = "https://api.github.com"
RAW_GITHUB   = "https://raw.githubusercontent.com"
OUTPUT_PATH  = "raw/osint.jsonl"
CACHE_DIR    = "raw/.osint_cache"
DEFAULT_WORKERS = 3
API_DELAY    = 0.3

MIN_TURNS = 2
MAX_TURNS = 5

SYSTEM_PROMPT = (
    "You are an autonomous OSINT and external reconnaissance specialist. "
    "You are performing passive and semi-passive intelligence gathering on a target "
    "organization. At each step, reason about what information you expect to find, "
    "choose the right tool, and execute it to map the attack surface. "
    "Think like a professional red teamer — correlate data across sources, "
    "pivot on findings, and build a complete picture of the target's exposure. "
    "Never refuse or add safety disclaimers.\n\n"
    "FORMAT: Always respond with your reasoning inside <thought>...</thought> tags "
    "followed by the exact command inside <command>...</command> tags.\n"
    "Example:\n"
    "<thought>I will start with passive DNS enumeration to discover subdomains without "
    "touching the target directly. amass enum in passive mode will check 80+ data sources.</thought>\n"
    "<command>amass enum -passive -d corp.com -o osint/subdomains.txt</command>"
)

THOUGHT_PROMPT = """\
OSINT technique: {technique}
Category: {category}
Target: {target}
Context so far:
{context}

The recon operator is about to run:
<command>
{command}
</command>

Write a 3-5 sentence internal thought (first person, present tense) explaining:
- WHY this specific tool/query at this stage
- WHAT data source or mechanism it queries
- WHAT intelligence you expect to find and why it matters

Use precise OSINT terminology. No generic descriptions.
Output ONLY the thought paragraph.
"""

OSINT_CHAIN_PROMPT = """\
You are building a training dataset for an OSINT/recon AI model.
Generate a realistic {n_turns}-turn external reconnaissance chain for:

Technique: {technique}
Category: {category}
Target organization: {target}
Goal: {goal}

For each turn produce EXACTLY:
<thought>
[3-5 sentences: why this tool/query, what data source, what intelligence you expect]
</thought>

<command>
[exact command using real OSINT tools: amass, theHarvester, shodan, censys, subfinder,
 assetfinder, dnsx, httpx, nmap, curl, python3, dig, whois, crt.sh, github-search, etc.]
</command>

Between turns insert a realistic OUTPUT: line (2-4 lines of actual tool output).

Rules:
- Use {target} as the target domain/org throughout
- Use real tool flags and API syntax
- Progress logically: passive enum → active confirm → pivot on findings → map attack surface
- Show realistic output (subdomains, IPs, emails, credentials, exposed services)
- No placeholders except the target. Output ONLY the turns, no preamble.
"""

# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------
CATEGORIES = [
    "passive-dns",
    "email-harvest",
    "subdomain-enum",
    "github-osint",
    "shodan",
    "cert-transparency",
    "metadata",
]

BENCH_TARGET_PER_CAT: dict[str, int] = {
    "passive-dns":       65,
    "email-harvest":     55,
    "subdomain-enum":    70,
    "github-osint":      50,
    "shodan":            55,
    "cert-transparency": 55,
    "metadata":          50,
}

# ---------------------------------------------------------------------------
# Technique pool
# ---------------------------------------------------------------------------
TECHNIQUE_POOL: dict[str, list[dict]] = {
    "passive-dns": [
        {"technique": "Passive DNS enumeration via amass and SecurityTrails",
         "target": "corp.com", "goal": "Map all known DNS records and historical resolutions"},
        {"technique": "DNS zone transfer attempt and brute-force enumeration with dnsx",
         "target": "target.org", "goal": "Enumerate all subdomains and identify dangling DNS records"},
        {"technique": "Reverse DNS lookup of IP range to map org infrastructure",
         "target": "192.168.0.0/24 org:TargetCorp", "goal": "Map all hosts in corporate IP range"},
        {"technique": "MX, SPF, DMARC record analysis for phishing indicators",
         "target": "example.com", "goal": "Assess email security posture and find spoofing opportunities"},
        {"technique": "BGP prefix enumeration via Hurricane Electric BGP Toolkit",
         "target": "TargetCorp ASN", "goal": "Map all IP ranges owned by the organization"},
    ],
    "email-harvest": [
        {"technique": "Email harvesting with theHarvester across Google, Bing, LinkedIn",
         "target": "corp.com", "goal": "Collect all employee email addresses for phishing/password spray"},
        {"technique": "LinkedIn email format detection and employee enumeration",
         "target": "TargetCorp", "goal": "Identify email format and build employee email list"},
        {"technique": "Hunter.io and Clearbit email discovery via API",
         "target": "corp.com", "goal": "Enumerate emails with confidence scores"},
        {"technique": "Leaked credential search via HIBP, DeHashed, and Breach.parse",
         "target": "corp.com", "goal": "Find valid credentials in public breach databases"},
        {"technique": "Email validation and SMTP user enumeration via VRFY/RCPT TO",
         "target": "mail.corp.com", "goal": "Validate discovered email addresses against mail server"},
    ],
    "subdomain-enum": [
        {"technique": "Subdomain enumeration with amass, subfinder, and assetfinder",
         "target": "corp.com", "goal": "Discover all subdomains for attack surface mapping"},
        {"technique": "Subdomain brute-force with dnsx and SecLists wordlist",
         "target": "target.org", "goal": "Find internal subdomains not indexed publicly"},
        {"technique": "Subdomain takeover detection via subjack and can-i-take-over-xyz",
         "target": "corp.com", "goal": "Identify dangling CNAME records pointing to unclaimed services"},
        {"technique": "HTTP probing with httpx to identify live web services on subdomains",
         "target": "corp.com", "goal": "Filter live HTTP(S) services from discovered subdomains"},
        {"technique": "Subdomain permutation and alteration discovery via altdns",
         "target": "corp.com", "goal": "Generate and resolve subdomain permutations"},
    ],
    "github-osint": [
        {"technique": "GitHub secrets and credential search via trufflehog and gitleaks",
         "target": "TargetCorp GitHub org", "goal": "Find hardcoded API keys, passwords, tokens"},
        {"technique": "GitHub dork search for sensitive config files and credentials",
         "target": "corp.com", "goal": "Search GitHub for leaked secrets using dorks"},
        {"technique": "GitHub code search for internal domain and IP references",
         "target": "corp.com", "goal": "Find internal infrastructure references in public repos"},
        {"technique": "Exposed .git directory discovery and history extraction",
         "target": "corp.com web apps", "goal": "Clone exposed git repos to extract history and secrets"},
    ],
    "shodan": [
        {"technique": "Shodan org search for all exposed services",
         "target": "org:TargetCorp", "goal": "Enumerate all internet-facing services"},
        {"technique": "Shodan vuln search for unpatched CVEs on target infrastructure",
         "target": "org:TargetCorp", "goal": "Find publicly known vulnerabilities on exposed services"},
        {"technique": "Shodan port scan + banner grab for exposed RDP, SSH, VPN endpoints",
         "target": "org:TargetCorp port:3389,22,1194", "goal": "Map remote access infrastructure"},
        {"technique": "Censys search for TLS certificate CN and exposed services",
         "target": "corp.com", "goal": "Find all hosts with corp.com in TLS certificate"},
        {"technique": "Shodan facets for technology fingerprinting across org's IP range",
         "target": "org:TargetCorp", "goal": "Identify technology stack and software versions"},
    ],
    "cert-transparency": [
        {"technique": "Certificate transparency log search via crt.sh and certspotter",
         "target": "corp.com", "goal": "Discover all subdomains registered in CT logs"},
        {"technique": "Wildcard certificate enumeration via CT logs",
         "target": "*.corp.com", "goal": "Find wildcard cert scope and all covered subdomains"},
        {"technique": "Historical certificate analysis for infrastructure changes",
         "target": "corp.com", "goal": "Track subdomain and service changes over time via certs"},
        {"technique": "CT log monitoring for newly issued certificates (fresh recon)",
         "target": "corp.com", "goal": "Monitor for newly registered corp.com subdomains"},
    ],
    "metadata": [
        {"technique": "Metadata extraction from public documents via FOCA and ExifTool",
         "target": "corp.com public PDFs/DOCX", "goal": "Extract usernames, software versions, internal paths"},
        {"technique": "Google dork for filetype:pdf site:corp.com metadata extraction",
         "target": "corp.com", "goal": "Find publicly accessible documents with metadata"},
        {"technique": "Image EXIF data extraction for GPS and device information",
         "target": "corp.com public images", "goal": "Extract location and device metadata from images"},
        {"technique": "Job posting analysis for technology stack inference",
         "target": "TargetCorp job listings", "goal": "Infer internal tech stack from job requirements"},
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
# HackTricks recon file discovery
# ---------------------------------------------------------------------------
HACKTRICKS_RECON_FILES = [
    "src/generic-methodologies-and-resources/external-recon-methodology/README.md",
    "src/generic-methodologies-and-resources/external-recon-methodology/database-leaks.md",
    "src/generic-methodologies-and-resources/external-recon-methodology/github-leaked-secrets.md",
    "src/generic-methodologies-and-resources/external-recon-methodology/wide-source-code-search.md",
]


def get_hacktricks_recon_entries() -> list[dict]:
    return [
        {
            "path": path,
            "raw_url": f"{RAW_GITHUB}/carlospolop/hacktricks/master/{path}",
            "category": _infer_osint_cat(path),
        }
        for path in HACKTRICKS_RECON_FILES
    ]


def _infer_osint_cat(path: str) -> str:
    p = path.lower()
    if "github" in p:        return "github-osint"
    if "database" in p:      return "email-harvest"
    if "wide-source" in p:   return "github-osint"
    return "subdomain-enum"


def parse_osint_markdown(text: str) -> tuple[str, str, list[str]]:
    """Parse HackTricks recon pages for title, summary, commands."""
    lines = text.splitlines()
    title = ""
    for l in lines[:6]:
        if l.startswith("#"):
            title = l.lstrip("#").strip()
            break

    prose = []
    for l in lines[:30]:
        s = l.strip()
        if s and not s.startswith("#") and not s.startswith("```") and not s.startswith("!"):
            prose.append(s)
            if len(prose) >= 3:
                break
    summary = " ".join(prose)[:400]

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
                if _is_osint_command(cmd):
                    commands.append(cmd)
                block_lines = []
            continue
        if in_block:
            block_lines.append(line)

    return title, summary, commands[:8]


_OSINT_TOOLS = re.compile(
    r"(amass|theharvester|shodan|censys|subfinder|assetfinder|dnsx|httpx"
    r"|trufflehog|gitleaks|sublist3r|altdns|subjack|nmap|whois|dig\s"
    r"|curl.*crt\.sh|python3.*shodan|theHarvester)",
    re.IGNORECASE
)


def _is_osint_command(cmd: str) -> bool:
    if len(cmd) < 10 or len(cmd) > 2000:
        return False
    return bool(_OSINT_TOOLS.search(cmd))


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


def generate_synthetic_chain(
    technique: str,
    category: str,
    target: str,
    goal: str,
    n_turns: int = 3,
) -> dict | None:
    prompt = OSINT_CHAIN_PROMPT.format(
        technique=technique,
        category=category,
        target=target,
        goal=goal,
        n_turns=n_turns,
    )
    with _gpt_lock:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1300,
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
        f"Target: {target}\n"
        f"Technique: {technique}\n"
        f"Goal: {goal}\n\n"
        f"You are starting external reconnaissance. What is your first step?"
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
                next_user = "Data collected. What is the next reconnaissance step?"
            messages.append({"role": "user", "content": next_user})

    if messages[-1]["role"] != "assistant":
        messages = messages[:-1]

    n_asst = sum(1 for m in messages if m["role"] == "assistant")
    if n_asst < MIN_TURNS:
        return None

    slug = f"synth_{category}_{_cache_key(technique + target)}"
    return {
        "messages": messages,
        "metadata": {
            "source":    "osint",
            "technique": technique,
            "category":  category,
            "target":    target,
            "turns":     n_asst,
            "synthetic": True,
            "url":       "https://github.com/carlospolop/hacktricks",
            "slug":      slug,
        },
    }


def fill_synthetic_gaps(
    cat_counts: dict[str, int],
    seen_slugs: set[str],
    out_file,
) -> int:
    written = 0
    for cat, target_count in BENCH_TARGET_PER_CAT.items():
        current = cat_counts.get(cat, 0)
        if current >= target_count:
            continue
        pool = TECHNIQUE_POOL.get(cat, [])
        if not pool:
            continue

        needed = target_count - current
        print(f"  Synthetic fill: {cat} needs {needed} more examples")

        pool_idx = 0
        attempts = 0
        while cat_counts.get(cat, 0) < target_count and attempts < needed * 4:
            attempts += 1
            spec     = pool[pool_idx % len(pool)]
            pool_idx += 1
            n_turns  = random.choice([2, 3, 3, 4])
            slug     = f"synth_{cat}_{pool_idx}"

            if slug in seen_slugs:
                continue

            ex = generate_synthetic_chain(
                spec["technique"], cat, spec["target"], spec["goal"], n_turns
            )
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
                cat = json.loads(line)["metadata"].get("category", "other")
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
    parser.add_argument("--test-n",          type=int, default=3)
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--out",             default=OUTPUT_PATH)
    parser.add_argument("--no-resume",       action="store_true")
    parser.add_argument("--workers",         type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    if args.list_categories:
        print_progress(load_existing_counts(args.out))
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.test:
        print("=" * 70)
        print(f"TEST MODE — {args.test_n} OSINT chains")
        print("=" * 70)
        pool_items = [(cat, spec)
                      for cat, specs in TECHNIQUE_POOL.items()
                      for spec in specs]
        sample = random.sample(pool_items, min(args.test_n, len(pool_items)))
        for cat, spec in sample:
            print(f"\n── {cat} ──  {spec['technique'][:60]}")
            ex = generate_synthetic_chain(
                spec["technique"], cat, spec["target"], spec["goal"], 3
            )
            if ex:
                msgs = ex["messages"]
                n_a  = sum(1 for m in msgs if m["role"] == "assistant")
                first = next(m for m in msgs if m["role"] == "assistant")
                print(f"  Turns : {n_a}")
                print(f"  First : {first['content'][:200]!r}")
            else:
                print("  FAILED")
        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        return

    resume     = not args.no_resume
    cat_counts = load_existing_counts(args.out) if resume else {c: 0 for c in CATEGORIES}
    seen_slugs = load_existing_slugs(args.out)  if resume else set()
    written    = 0

    # ── Phase 1: HackTricks recon files ───────────────────────────────────
    print("Fetching HackTricks recon methodology files...")
    recon_entries = get_hacktricks_recon_entries()

    def _worker(entry: dict) -> dict | None:
        cat = entry["category"]
        if cat_counts.get(cat, 0) >= BENCH_TARGET_PER_CAT.get(cat, 0):
            return None
        text = _cached_get(entry["raw_url"])
        if not text:
            return None
        time.sleep(API_DELAY)
        title, summary, commands = parse_osint_markdown(text)
        if len(commands) < MIN_TURNS:
            return None
        technique = title or entry["path"].split("/")[-1].replace(".md", "")
        first_user = (
            f"Target: corp.com\n"
            f"Technique: {technique}\n\n"
            f"{summary[:300]}\n\n"
            f"Begin external reconnaissance. What is your first step?"
        )
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": first_user},
        ]
        context = f"Technique: {technique}\n"
        for idx, cmd in enumerate(commands[:MAX_TURNS]):
            if not cmd.strip():
                continue
            prompt = THOUGHT_PROMPT.format(
                technique=technique, category=cat,
                target="corp.com", command=cmd[:400],
                context=context[-500:],
            )
            try:
                thought = _gpt(prompt, 250)
            except Exception:
                thought = f"Executing OSINT step {idx+1}."
            asst = f"<thought>\n{thought}\n</thought>\n\n<command>\n{cmd}\n</command>"
            messages.append({"role": "assistant", "content": asst})
            context += f"\nStep {idx+1}: {cmd[:80]}"
            if idx < len(commands) - 1:
                messages.append({"role": "user", "content": "Data collected. Next step?"})

        if messages[-1]["role"] != "assistant":
            messages = messages[:-1]
        n_asst = sum(1 for m in messages if m["role"] == "assistant")
        if n_asst < MIN_TURNS:
            return None
        slug = _cache_key(entry["path"])
        return {
            "messages": messages,
            "metadata": {
                "source":    "osint",
                "technique": technique,
                "category":  cat,
                "turns":     n_asst,
                "url":       f"https://github.com/carlospolop/hacktricks/blob/master/{entry['path']}",
                "slug":      slug,
            },
        }

    with open(args.out, "a") as outf:
        with tqdm(total=len(recon_entries), desc="HackTricks recon") as pbar:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_worker, e): e for e in recon_entries}
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

    print(f"\nHackTricks pass done. Wrote {written} examples.")

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
