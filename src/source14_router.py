#!/usr/bin/env python3
"""
Source 14: Router Classifier Training Data

Generates labelled (prompt → adapter) examples for training a learned routing
classifier to replace / back the regex-based AdapterRouter.

Three pools of examples:
  A) Extracted from processed/*.jsonl user turns — labelled by the ADAPTER each
     file was trained on. These are "clean" single-domain examples.
  B) Synthetic ambiguous cases — prompts that would confuse regex rules (e.g.
     SSRF on cloud metadata, AD pivoting from a web shell, etc.). Labelled
     manually with the correct adapter.
  C) Cross-domain prompts — real security scenarios that span two adapters;
     labelled with the adapter that should OWN the current step.

Output: raw/router_labels.jsonl  — one JSON per line: {prompt, adapter}

Usage:
  python3 src/source14_router.py          # generate to raw/router_labels.jsonl
  python3 src/source14_router.py --test   # print 5 samples per pool, no write
  python3 src/source14_router.py --stats  # count per adapter, no write
"""

import json
import random
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("processed")
OUTPUT_PATH   = Path("raw/router_labels.jsonl")

# How many user-turn prompts to sample per processed file (pool A)
POOL_A_PER_FILE = 150

# Adapter label for each processed file
FILE_TO_ADAPTER: dict[str, str] = {
    "htb_writeups.jsonl":   "htb",
    "vulhub.jsonl":         "vulhub",
    "attack.jsonl":         "attack",
    "ad.jsonl":             "ad",
    "webapp.jsonl":         "webapp",
    "osint.jsonl":          "osint",
    "cloud.jsonl":          "cloud",
    "exploitdb.jsonl":      "exploitdb",
    "researcher.jsonl":     "researcher",
    "researcher_synth.jsonl":  "researcher",
    "researcher_pz.jsonl":     "researcher",
    "researcher_ctf.jsonl":    "researcher",
    "planner.jsonl":        "planner",
    "planner_decomp.jsonl": "planner",
    "planner_replan.jsonl": "planner",
    "executor.jsonl":       "executor",
    "executor_correction.jsonl": "executor",
    "executor_filtering.jsonl":  "executor",
    "analyst.jsonl":        "analyst",
    "analyst_h1.jsonl":     "analyst",
    "analyst_synth.jsonl":  "analyst",
}

# ── Pool B: synthetic ambiguous / hard cases ─────────────────────────────────
# These are cases where the regex router either ties or picks wrong.
# Format: (prompt_text, correct_adapter)
AMBIGUOUS_CASES: list[tuple[str, str]] = [
    # SSRF to cloud metadata — looks like webapp (SSRF) but the TARGET is cloud infra
    ("The target at 169.254.169.254 is returning AWS IAM credentials via SSRF. "
     "What can we do with these credentials?", "cloud"),
    ("SSRF confirmed on api.target.com — can read http://169.254.169.254/latest/meta-data/iam/. "
     "Retrieve the instance role credentials and enumerate S3 buckets.", "cloud"),
    ("Server-side request forgery on the image upload endpoint allows fetching "
     "http://metadata.google.internal/computeMetadata/v1/. Exploit this for GCP privilege escalation.", "cloud"),
    # AD from a web shell — the entry was webapp but the GOAL is AD
    ("We have RCE via a PHP webshell on the IIS server. The box is domain-joined. "
     "Run SharpHound to enumerate the Active Directory environment.", "ad"),
    ("Got a meterpreter shell on a Windows host. The host has impacket installed. "
     "Use secretsdump to dump NTLM hashes from the domain controller.", "ad"),
    # CVE in a cloud service — CVE pattern → vulhub, but context is cloud
    ("CVE-2019-0708 (BlueKeep) is patched. The EC2 instance has IMDSv1 enabled. "
     "Use the metadata service to get the IAM role and escalate to admin.", "cloud"),
    ("CVE-2021-44228 (Log4Shell) was confirmed on the application. "
     "The server runs on AWS Lambda. Chain the RCE to exfiltrate the Lambda execution role.", "cloud"),
    # Timing oracle — researcher, not webapp
    ("The /api/login endpoint returns 200ms for valid usernames and 50ms for invalid ones. "
     "Standard tools found nothing. What does this timing difference suggest?", "researcher"),
    ("I notice that GET /api/export?format=pdf takes 3x longer than format=csv. "
     "The error on format=x leaks: template: pdf.tmpl:14. sqlmap and nikto found nothing.", "researcher"),
    # Prototype pollution — researcher, not webapp
    ("POST /api/settings with {\"__proto__\":{\"isAdmin\":true}} returns 200 without error. "
     "Standard scanners missed this. How do we confirm and exploit prototype pollution?", "researcher"),
    # HTB post-shell with AD — htb owns the chain, not ad
    ("We have a low-priv shell on Cascade (Windows HTB box). "
     "linpeas found AD CS misconfiguration. What's the privesc path?", "htb"),
    ("Foothold on Forest HTB box via AS-REP roasting. We have a shell as svc-alfresco. "
     "Enumerate WriteDACL rights for privilege escalation.", "htb"),
    # OSINT that pivots to cloud
    ("Subdomain enumeration found dev.target-corp.com pointing to an S3 bucket. "
     "The bucket returns a 403. Check for subdomain takeover.", "osint"),
    ("Google dorking found an exposed .git directory. Inside is an AWS access key. "
     "What's the recon methodology to assess blast radius?", "osint"),
    # Webapp IDOR that needs analyst interpretation
    ("GET /api/users/1337 returns another user's full profile including email and phone. "
     "sqlmap shows no SQLi. Classify this finding and assess impact.", "analyst"),
    ("The response to POST /api/order/submit contains other users' order IDs in the "
     "JSON payload. This looks like an IDOR. Write the H1 report summary.", "analyst"),
    # Executor role — fixing a broken command
    ("subfinder -x -d target.com returned: Error: unknown flag '-x'. Fix this command.", "executor"),
    ("nmap --sV -p- target.com returned: nmap: unrecognized option '--sV'. "
     "The correct flag is -sV. What is the corrected command?", "executor"),
    # Planner role — decomposing goals
    ("We are in the recon phase of a bug bounty against api.target.com. "
     "Create a step-by-step attack plan for testing the OAuth2 implementation.", "planner"),
    ("We just found that the /api/v1/admin endpoint returns 401 instead of 403. "
     "Our original plan assumed no admin panel. Revise the plan.", "planner"),
    # Researcher cross-domain (CTF-style)
    ("CTF challenge: JWT with kid header. jwt_tool --crack found nothing. "
     "Requesting /api/keys?kid=../../../../etc/passwd returns 'invalid PEM'. "
     "What does this suggest?", "researcher"),
    # Cloud misconfig that looks like osint
    ("S3 bucket target-prod is publicly listable via aws s3 ls. "
     "It contains database backups. What's the exploitation path?", "cloud"),
    ("The Azure storage account has anonymous blob access enabled. "
     "amass found the endpoint via certificate transparency. Exploit the misconfiguration.", "cloud"),
    # Generic AD that looks like attack
    ("T1558.003 Kerberoasting: extract service tickets for SPN accounts and crack offline. "
     "Use Rubeus to perform the attack.", "ad"),
    ("Lateral movement via pass-the-hash to the domain controller using crackmapexec.", "ad"),
    # WebApp with JWT that's not AD
    ("The JWT uses alg:none. Modify the token to set isAdmin=true and access /api/admin.", "webapp"),
    ("CORS misconfiguration on api.target.com allows credentials from null origin. "
     "Craft a PoC to steal the victim's session token.", "webapp"),
    # Attack (ATT&CK) vs HTB
    ("T1003 OS Credential Dumping via lsass.exe process memory. "
     "Use mimikatz sekurlsa::logonpasswords.", "attack"),
    ("Persistence via T1053.005 Scheduled Task. Create a task that calls back to C2 "
     "every 5 minutes using schtasks.", "attack"),
    # ExploitDB — explicit exploit creation
    ("Write a Python exploit for CVE-2023-46604 (Apache ActiveMQ RCE). "
     "The target is running ActiveMQ 5.15.3.", "exploitdb"),
    ("Generate a shellcode payload for a buffer overflow in a 32-bit Linux binary. "
     "Use msfvenom with x86/shikata_ga_nai encoder.", "exploitdb"),
    # More researcher edge cases
    ("Race condition: 15 simultaneous POST /api/transfer requests all passed the balance "
     "check before any deduction committed. The flag endpoint checks total_transferred >= 1000.", "researcher"),
    ("HTTP request smuggling via CL.TE desync. HAProxy uses Content-Length, Nginx uses "
     "Transfer-Encoding. Standard smuggler.py didn't detect it.", "researcher"),
    # Vulhub CVE exploitation
    ("CVE-2017-5638 Apache Struts OGNL injection. The target runs Struts 2.3.5. "
     "Generate the exploit payload.", "vulhub"),
    ("CVE-2021-3129 Laravel debug mode RCE. The /ignition/execute-solution endpoint "
     "is accessible. Exploit it.", "vulhub"),
]

# ── Pool C: cross-domain prompts with clear ownership ────────────────────────
CROSS_DOMAIN: list[tuple[str, str]] = [
    # Web entry → cloud pivot: the STEP is cloud
    ("We exploited SSRF on api.startup.io to reach the AWS metadata service. "
     "Now enumerate IAM roles, S3 buckets, and check for privilege escalation paths "
     "using the retrieved credentials.", "cloud"),
    # Network scan → service detection: htb owns this step
    ("nmap -sV -p- returned ports 22,80,443,8080 open on 10.10.10.5. "
     "Port 8080 runs Apache Tomcat 9.0.30. What vulnerability should we investigate?", "htb"),
    # CVE found during OSINT: vulhub owns exploitation
    ("Shodan shows the target runs Elasticsearch 7.5.1 with no authentication. "
     "CVE-2021-22145 applies. How do we exploit this?", "vulhub"),
    # Shell obtained, moving to AD enumeration: ad owns this step
    ("Got a reverse shell as www-data on 192.168.1.50. The /etc/hosts shows "
     "CORP-DC01.corp.local. Run BloodHound to map AD attack paths.", "ad"),
    # Finding anomaly during recon: researcher owns interpretation
    ("During ffuf enumeration, /api/v2/users returned 2341 bytes for IDs 1-50 "
     "but 89 bytes for ID 51. All standard vuln scans returned nothing. "
     "What does this behavioral inconsistency suggest?", "researcher"),
    # Reporting phase: analyst owns it
    ("We confirmed an IDOR at /api/v1/invoice/{id} that exposes other users' "
     "invoice data. No authentication required. Write the vulnerability report "
     "with CVSS score and remediation steps.", "analyst"),
    # Planning new phase after unexpected result
    ("We expected the admin panel at /admin but got 404. The Burp sitemap shows "
     "/management/dashboard returns 200 with a login form. Revise our test plan.", "planner"),
    # Command failed, needs correction
    ("httpx -l subdomains.txt --status-code --title failed with: Error: unknown flag "
     "'--status-code'. The correct flag is -sc. What is the corrected command?", "executor"),
    # Cloud IAM enumeration
    ("The Lambda function's execution role has iam:PassRole and ec2:RunInstances. "
     "Explain the privilege escalation path and provide the exploit commands.", "cloud"),
    # OSINT → subdomain takeover
    ("subfinder found dev.target.com pointing to a Heroku app that no longer exists. "
     "Verify the subdomain takeover and claim the DNS record.", "osint"),
    # Webapp XSS chaining
    ("Reflected XSS confirmed in the search parameter. The app uses HttpOnly cookies "
     "but the admin panel at /admin has no CSRF protection. Chain XSS to CSRF "
     "to create a new admin account.", "webapp"),
    # ATT&CK technique in a post-exploitation context
    ("Post-exploitation on a Windows host: disable Windows Defender using "
     "Set-MpPreference -DisableRealtimeMonitoring $true, then use Cobalt Strike "
     "to establish persistence via T1543.003.", "attack"),
]


def _extract_user_turns(path: Path, adapter: str, n: int) -> list[dict]:
    """Sample user-turn prompts from a processed JSONL file."""
    samples: list[dict] = []
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    for ex in rows:
        msgs = ex.get("messages", [])
        for m in msgs:
            if m["role"] == "user":
                content = m["content"].strip()
                # Skip very short prompts (likely just a target name) or very long ones
                if 30 <= len(content) <= 600:
                    samples.append({"prompt": content, "adapter": adapter})

    random.shuffle(samples)
    return samples[:n]


def build_dataset(seed: int = 42) -> list[dict]:
    random.seed(seed)
    dataset: list[dict] = []

    # Pool A — extract from processed files
    for filename, adapter in FILE_TO_ADAPTER.items():
        path = PROCESSED_DIR / filename
        if not path.exists():
            continue
        examples = _extract_user_turns(path, adapter, POOL_A_PER_FILE)
        dataset.extend(examples)

    # Pool B — hand-crafted ambiguous cases (repeat 3x with minor variation)
    for prompt, adapter in AMBIGUOUS_CASES:
        dataset.append({"prompt": prompt, "adapter": adapter})

    # Pool C — cross-domain prompts
    for prompt, adapter in CROSS_DOMAIN:
        dataset.append({"prompt": prompt, "adapter": adapter})

    random.shuffle(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  action="store_true", help="Print 5 samples per pool, no write")
    parser.add_argument("--stats", action="store_true", help="Print per-adapter counts, no write")
    parser.add_argument("--out",   default=str(OUTPUT_PATH), help="Output path")
    args = parser.parse_args()

    dataset = build_dataset()

    if args.stats or args.test:
        from collections import Counter
        counts = Counter(d["adapter"] for d in dataset)
        print(f"Total examples: {len(dataset)}")
        print("\nPer-adapter distribution:")
        for adapter, n in sorted(counts.items()):
            bar = "█" * (n // 20)
            print(f"  {adapter:<12} {n:>5}  {bar}")

    if args.test:
        print("\n--- Sample prompts ---")
        shown: set[str] = set()
        for ex in dataset:
            if ex["adapter"] not in shown:
                shown.add(ex["adapter"])
                print(f"\n[{ex['adapter']}] {ex['prompt'][:200]!r}")
            if len(shown) >= 12:
                break
        return

    if args.stats:
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    print(f"Written {len(dataset)} examples → {out}")


if __name__ == "__main__":
    main()
