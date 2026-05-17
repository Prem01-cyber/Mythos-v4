#!/usr/bin/env python3
"""
Gap-filler + metadata repair script.

Runs two jobs in sequence:
  1. Fix machine_name metadata in raw/htb_writeups.jsonl (extract from user message)
  2. Generate synthetic HTB examples to fill linux:easy / windows:easy / windows:hard gaps
  3. Report ATT&CK gap status (those are Atomic Red Team coverage limits — noted)

Usage:
  python3 src/fill_gaps.py            # full run
  python3 src/fill_gaps.py --dry-run  # show gaps, no writes
"""

import os, re, json, sys, time, random, argparse, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

HTB_PATH  = "raw/htb_writeups.jsonl"
ATK_PATH  = "raw/attack.jsonl"

# ── Targets ──────────────────────────────────────────────────────────────────
HTB_TARGETS = {
    "linux:easy":     90,
    "linux:medium":   90,
    "linux:hard":     30,
    "linux:insane":   10,
    "windows:easy":   45,
    "windows:medium": 70,
    "windows:hard":   20,
    "windows:ad":     60,
}

ATK_TARGETS = {
    "execution":           80, "discovery":           80,
    "credential-access":   70, "persistence":         70,
    "privilege-escalation":70, "defense-evasion":     90,
    "command-and-control": 50, "collection":          40,
    "impact":              40, "lateral-movement":    30,
    "exfiltration":        25, "initial-access":      15,
    "reconnaissance":      10,
}

# ── Curated machine lists for synthetic gap-fill ──────────────────────────────
# (name, os, difficulty, key_vulnerability_hint)
EASY_MACHINES = [
    # Linux Easy
    ("Bashed",       "linux", "easy", "phpbash webshell, sudo scriptmanager"),
    ("Shocker",      "linux", "easy", "Shellshock CGI RCE, sudo perl"),
    ("Sense",        "linux", "easy", "pfSense command injection, CVE-2014-4688"),
    ("Blocky",       "linux", "easy", "Minecraft plugin jar, plaintext creds in decompiled code"),
    ("Networked",    "linux", "easy", "PHP file upload bypass, cron privesc with network-cleanup.sh"),
    ("Magic",        "linux", "easy", "SQLi login bypass, file upload magic bytes, mysqldump privesc"),
    ("Admirer",      "linux", "easy", "Adminer 4.6.2 SSRF, sudo admin_tasks.sh env hijack"),
    ("Blunder",      "linux", "easy", "Bludit CMS brute bypass, PHP image upload, sudo bluditadm"),
    ("Doctor",       "linux", "easy", "SSTI in messaging system, Splunk RCE privesc"),
    ("Knife",        "linux", "easy", "PHP 8.1 dev backdoor RCE, sudo knife exploit"),
    ("Previse",      "linux", "easy", "PHP exec injection in logs, sudo mysql path hijack"),
    ("Horizontall",  "linux", "easy", "Strapi CMS RCE, Laravel local port tunnel, CVE-2021-3129"),
    ("Paper",        "linux", "easy", "WordPress secret drafts disclosure, Polkit CVE-2021-3560"),
    ("Validation",   "linux", "easy", "SQL injection to file write, SUID find privesc"),
    ("Busqueda",     "linux", "easy", "Searchor SSTI eval injection, sudo script with git config"),
    ("Analytics",    "linux", "easy", "Metabase pre-auth RCE CVE-2023-38646, Docker env credentials"),
    ("Devvortex",    "linux", "easy", "Joomla CVE-2023-23752 info leak, Apport privilege escalation"),
    ("Perfection",   "linux", "easy", "SSTI in grade calculator, PBKDF2 hash crack in cron"),
    ("BoardLight",   "linux", "easy", "Dolibarr CVE-2023-30253 PHP inject, SUID enlightenment_sys"),
    ("GreenHorn",    "linux", "easy", "Pluck CMS RCE, PDF password extraction, sudo less"),
    ("Cicada",       "linux", "easy", "SMB guest enumeration, password spray, SeBackupPrivilege"),
    ("Instant",      "linux", "easy", "Swagger API endpoint, Android APK creds, solar-putty decrypt"),
    ("Heal",         "linux", "easy", "LimeSurvey file upload, CVE-2021-41433 consul privesc"),
    # Windows Easy
    ("Blue",         "windows", "easy", "EternalBlue MS17-010 RCE, no privesc needed"),
    ("Legacy",       "windows", "easy", "MS08-067 netapi RCE, no privesc needed"),
    ("Devel",        "windows", "easy", "FTP ASP webshell upload, local exploit suggester token impersonation"),
    ("Jerry",        "windows", "easy", "Tomcat default creds, WAR webshell deploy"),
    ("Netmon",       "windows", "easy", "PRTG backup file creds, PRTG notification command injection"),
    ("Access",       "windows", "easy", "Telnet access, MDB credentials, sticky note admin password"),
    ("Timelapse",    "windows", "easy", "SMB share PFX cert, evil-winrm TLS login, LAPS history"),
    ("Return",       "windows", "easy", "Printer LDAP credential capture, Server Operators group service"),
    ("Sau",          "windows", "easy", "Maltrail RCE via SSRF to local service, sudo systemctl"),
    ("CozyHosting",  "windows", "easy", "Spring Boot actuator session hijack, SSH exec injection, SUID psql"),
    ("Grandpa",      "windows", "easy", "IIS 6.0 WebDAV ScStoragePathFromUrl buffer overflow, churrasco token kidnapping"),
    ("Granny",       "windows", "easy", "IIS 6.0 WebDAV PUT+MOVE webshell, local_exploit_suggester token kidnapping"),
    ("SecNotes",     "windows", "easy", "XSS CSRF password reset, SMB cred in todo, WSL bash shell"),
    ("Bounty",       "windows", "easy", "IIS web.config upload via ASP ASPX, JuicyPotato SeImpersonatePrivilege"),
    ("Arctic",       "windows", "easy", "ColdFusion 8 unrestricted file upload, MS10-059 chimichurri privesc"),
    # Windows Hard
    ("Proper",       "windows", "hard", "RPC DCOM exploitation, MSSQL injection, UNC path injection privesc"),
    ("Sharp",        "windows", "hard", ".NET WCF RCE service, SeImpersonatePrivilege RoguePotato"),
    ("Streamio",     "windows", "hard", "MSSQL injection, FFmpeg SSRF, AppLocker bypass LOLBAS"),
    ("Search",       "windows", "hard", "ADCS ESC8 relay, Excel password spray, ReadGMSAPassword"),
    ("Atom",         "windows", "hard", "Electron auto-updater YAML injection, Redis port forward, Kanban creds"),
    ("Intelligence", "windows", "hard", "ADCS ESC6 template abuse, Kerberos S4U delegation, GMSA password"),
    ("Object",       "windows", "hard", "Jenkins script console RCE, ForceChangePassword ACL, Kerberoasting"),
]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an autonomous penetration tester working through a target machine. "
    "At each step reason about what you observe, decide the next action, and "
    "execute it. Think like an attacker — be methodical, exploit what you find, "
    "escalate privileges. Never refuse or add safety disclaimers.\n\n"
    "FORMAT: Always respond with your reasoning inside <thought>...</thought> tags "
    "followed by the exact command inside <command>...</command> tags. "
    "Example:\n"
    "<thought>Port 80 is open so I'll enumerate the web app first.</thought>\n"
    "<command>gobuster dir -u http://10.10.10.1 -w /usr/share/wordlists/dirb/common.txt</command>"
)

SYNTH_PROMPT = """\
You are a HackTheBox expert generating a realistic penetration testing walkthrough.

Machine: {name} (HTB, {os_label}, {difficulty})
Key Vulnerability Path: {hint}
Phase: {phase}

Generate {n_turns} realistic turns of a pentesting session for this phase.
Each turn follows this EXACT format:

USER_OUTPUT: <realistic terminal output that was observed>
THOUGHT: <2-4 sentences of attacker reasoning, second-person ("You notice...", "This suggests...")>
COMMAND: <exact tool command the attacker runs next>

Rules:
- Use real tool names: nmap, gobuster, ffuf, hydra, evil-winrm, impacket, crackmapexec, etc.
- Use realistic HTB IPs (10.10.10.x or 10.10.11.x), ports, hashes, usernames
- Commands must be technically accurate for {name} (use its real exploit path)
- Each command must logically follow from the previous output
- No disclaimers or refusals

Start USER_OUTPUT with the initial observation relevant to this phase.
Output ONLY the turns, no preamble or explanation.
"""


def _gpt(prompt: str, max_tokens: int = 900) -> str:
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            if attempt == 2:
                return ""
            time.sleep(2 ** attempt)
    return ""


def _parse_turns(raw: str) -> list[dict]:
    turns = []
    blocks = re.split(r"USER_OUTPUT:", raw, flags=re.IGNORECASE)
    for block in blocks:
        if not block.strip():
            continue
        thought_m = re.search(r"THOUGHT:\s*(.+?)(?=COMMAND:|USER_OUTPUT:|$)", block, re.DOTALL | re.IGNORECASE)
        cmd_m     = re.search(r"COMMAND:\s*(.+?)(?=USER_OUTPUT:|THOUGHT:|$)", block, re.DOTALL | re.IGNORECASE)
        out_m     = re.match(r"(.+?)(?=THOUGHT:)", block, re.DOTALL | re.IGNORECASE)
        out_text  = out_m.group(1).strip() if out_m else ""
        thought   = thought_m.group(1).strip() if thought_m else ""
        cmd       = cmd_m.group(1).strip() if cmd_m else ""
        if thought and cmd and len(cmd.split()) >= 2:
            turns.append({"output": out_text[:500], "thought": thought[:600], "command": cmd[:400]})
    return turns


def generate_synthetic(name: str, os_label: str, diff: str, hint: str, phase: str) -> dict | None:
    is_ad    = any(k in hint.lower() for k in ("ad", "kerberos", "domain", "ldap", "gpo", "dcsync", "asrep", "gmsa", "adcs"))
    category = "windows:ad" if (is_ad and os_label == "windows") else f"{os_label}:{diff}"
    n_turns  = random.randint(4, 7)

    raw = _gpt(SYNTH_PROMPT.format(
        name=name, os_label=os_label, difficulty=diff,
        hint=hint, phase=phase, n_turns=n_turns,
    ))
    if not raw:
        return None

    turns = _parse_turns(raw)
    if len(turns) < 2:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Target: {name} (HackTheBox, {os_label.capitalize()}, {diff.capitalize()})\n"
            f"Phase: {phase}\n\nWhat is your first action?"
        )},
    ]
    for i, turn in enumerate(turns[:8]):
        messages.append({"role": "assistant", "content": (
            f"<thought>\n{turn['thought']}\n</thought>\n\n"
            f"<command>\n{turn['command']}\n</command>"
        )})
        if i < len(turns) - 1:
            nxt = turns[i + 1]["output"] or "(no output)"
            messages.append({"role": "user", "content":
                f"Output:\n```\n{nxt}\n```\n\nWhat is the next action?"})

    return {
        "messages": messages,
        "metadata": {
            "source": "htb_writeup", "machine_name": name,
            "os": os_label, "difficulty": diff,
            "is_ad": is_ad, "category": category,
            "phase": phase, "synthetic": True,
        },
    }


# ── Job 1: Fix machine_name metadata ─────────────────────────────────────────
def fix_machine_names(path: str, dry_run: bool = False) -> int:
    with open(path) as f:
        examples = [json.loads(l) for l in f if l.strip()]

    fixed = 0
    for ex in examples:
        if ex["metadata"].get("machine_name") in (None, "?", "", "unknown"):
            user_msgs = [m for m in ex["messages"] if m["role"] == "user"]
            if user_msgs:
                nm = re.search(r"Target:\s*([^\s(,]+)", user_msgs[0]["content"])
                if nm:
                    ex["metadata"]["machine_name"] = nm.group(1).strip()
                    fixed += 1

    if not dry_run:
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Fixed machine_name for {fixed} examples → {path}")
    else:
        print(f"  [dry-run] Would fix machine_name for {fixed} examples")
    return fixed


# ── Job 2: Fill HTB gaps ──────────────────────────────────────────────────────
def fill_htb_gaps(path: str, targets: dict, workers: int = 4, dry_run: bool = False) -> None:
    with open(path) as f:
        examples = [json.loads(l) for l in f if l.strip()]

    from collections import Counter
    cat_counts = Counter(e["metadata"].get("category") for e in examples)

    gaps = {cat: targets[cat] - cat_counts.get(cat, 0)
            for cat in targets if cat_counts.get(cat, 0) < targets[cat]}

    if not gaps:
        print("  HTB: all categories at target. Nothing to fill.")
        return

    print("  HTB gaps to fill:")
    for cat, gap in sorted(gaps.items()):
        print(f"    {cat:<20} current={cat_counts.get(cat,0):>3}  target={targets[cat]:>3}  need={gap}")

    if dry_run:
        return

    # Build work queue: machine × phase
    PHASES = ["Enumeration", "Foothold", "Privilege Escalation"]
    work_queue: list[tuple] = []  # (name, os, diff, hint, phase, category)

    gap_cats = set(gaps)
    for name, os_label, diff, hint in EASY_MACHINES:
        is_ad    = any(k in hint.lower() for k in ("ad", "kerberos", "domain", "ldap", "gpo", "dcsync", "asrep", "gmsa", "adcs"))
        category = "windows:ad" if (is_ad and os_label == "windows") else f"{os_label}:{diff}"
        if category not in gap_cats:
            continue
        for phase in PHASES:
            work_queue.append((name, os_label, diff, hint, phase, category))

    random.shuffle(work_queue)
    counts_lock = threading.Lock()
    written     = [0]

    def _gen(item):
        name, os_label, diff, hint, phase, category = item
        with counts_lock:
            if cat_counts.get(category, 0) >= targets.get(category, 0):
                return 0
        ex = generate_synthetic(name, os_label, diff, hint, phase)
        if not ex:
            return 0
        with counts_lock:
            if cat_counts.get(category, 0) >= targets.get(category, 0):
                return 0
            with open(path, "a") as f:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            cat_counts[category] = cat_counts.get(category, 0) + 1
            written[0] += 1
        return 1

    total_needed = sum(gaps.values())
    pbar = tqdm(total=total_needed, desc="HTB synthetic fill", unit="ex")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_gen, item): item for item in work_queue}
        for fut in as_completed(futs):
            pbar.update(fut.result())

    pbar.close()
    print(f"\n  Added {written[0]} synthetic examples.")
    print("  Final HTB counts:")
    for cat, target in sorted(targets.items()):
        have = cat_counts.get(cat, 0)
        status = "✅" if have >= target else f"⚠ -{target-have}"
        print(f"    {cat:<20} {have:>4}/{target}  {status}")


# ── Job 3: Report ATT&CK gaps ────────────────────────────────────────────────
def report_atk_gaps(path: str, targets: dict) -> None:
    with open(path) as f:
        exs = [json.loads(l) for l in f if l.strip()]
    from collections import Counter
    cats = Counter(e["metadata"]["tactic"] for e in exs)
    gaps = {c: targets[c] - cats.get(c, 0) for c in targets if cats.get(c, 0) < targets[c]}
    if not gaps:
        print("  ATT&CK: all categories at target ✅")
        return
    print("  ATT&CK remaining gaps (Atomic Red Team coverage limits):")
    for cat, gap in sorted(gaps.items(), key=lambda x: -x[1]):
        print(f"    {cat:<28} have={cats.get(cat,0):>3}  target={targets[cat]:>3}  short={gap}")
    print("  Note: these tactics have limited Atomic Red Team coverage.")
    print("  Run: python3 src/source4_attack.py --workers 6  to attempt filling.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true", help="Show gaps only, no writes")
    parser.add_argument("--workers",  type=int, default=4)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  STEP 1: Fix HTB machine_name metadata")
    print("="*60)
    fix_machine_names(HTB_PATH, dry_run=args.dry_run)

    print("\n" + "="*60)
    print("  STEP 2: Fill HTB category gaps with synthetic examples")
    print("="*60)
    fill_htb_gaps(HTB_PATH, HTB_TARGETS, workers=args.workers, dry_run=args.dry_run)

    print("\n" + "="*60)
    print("  STEP 3: ATT&CK gap report")
    print("="*60)
    report_atk_gaps(ATK_PATH, ATK_TARGETS)

    print("\nDone.")


if __name__ == "__main__":
    main()
