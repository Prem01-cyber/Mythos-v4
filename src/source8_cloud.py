#!/usr/bin/env python3
"""
Source 8: Cloud Security Attack Chains (AWS / Azure / GCP)

Scrapes CloudGoat scenario READMEs from RhinoSecurityLabs/cloudgoat and
generates synthetic cloud attack chains covering AWS, Azure, and GCP.

Each training example = one cloud attack scenario (2–5 turns):
  system : autonomous cloud security red team operator
  user   : "Target: AWS account 123456789. Phase: <phase>. First step?"
  asst   : "<thought>...</thought>\n\n<command>\n...\n</command>"
  user   : "Output:\n```\n...\n```\n\nWhat next?"
  ...

Target: 400 examples across 6 cloud attack categories.

Data sources:
  - RhinoSecurityLabs/cloudgoat scenario READMEs (scenario dirs)
  - GPT synthesis for gap-filling AWS/Azure/GCP attack chains

Usage:
  python3 src/source8_cloud.py --test
  python3 src/source8_cloud.py --list-categories
  python3 src/source8_cloud.py
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
from prompts import CLOUD as SYSTEM_PROMPT


load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GITHUB_API   = "https://api.github.com"
RAW_GITHUB   = "https://raw.githubusercontent.com"
OUTPUT_PATH  = "raw/cloud.jsonl"
CACHE_DIR    = "raw/.cloud_cache"
DEFAULT_WORKERS = 3
API_DELAY    = 0.3

MIN_TURNS = 2
MAX_TURNS = 5


THOUGHT_PROMPT = """\
Cloud attack technique: {technique}
Category: {category}
Target environment: {target}
Context so far:
{context}

The attacker is about to run:
<command>
{command}
</command>

Write a 3-5 sentence internal thought (first person, present tense) explaining:
- WHY this specific AWS/Azure/GCP command at this stage
- WHAT cloud service/permission/misconfiguration it targets
- WHAT access or data you expect to find

Use precise cloud security terminology (IAM, STS, S3 ACL, resource policies, etc.).
No generic descriptions. Output ONLY the thought paragraph.
"""

CLOUD_CHAIN_PROMPT = """\
You are building a training dataset for a cloud security red team AI.
Generate a realistic {n_turns}-turn cloud attack chain for:

Technique: {technique}
Category: {category}
Target: {target}
Scenario: {scenario}

For each turn produce EXACTLY:
<thought>
[3-5 sentences: why this command, what cloud mechanic, what access/data you expect]
</thought>

<command>
[exact command using real tools: aws cli, az cli, gcloud, boto3, pacu, prowler,
 scoutsuite, trufflehog, curl with metadata endpoints, etc.]
</command>

Between turns insert a realistic OUTPUT: line (2-4 lines of actual CLI output).

Rules:
- Use realistic AWS account IDs (123456789012), resource ARNs, and region us-east-1
- Use realistic Azure tenant/subscription IDs for Azure scenarios
- Progress logically: initial access → enumerate → escalate → impact
- Show realistic CLI output in OUTPUT: lines (actual ARNs, policy names, etc.)
- No placeholders except standard IDs. Output ONLY the turns, no preamble.
"""

# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------
CATEGORIES = [
    "aws-iam",
    "aws-s3",
    "azure-ad",
    "cloud-metadata",
    "cloud-iam",
    "serverless",
]

BENCH_TARGET_PER_CAT: dict[str, int] = {
    "aws-iam":        80,
    "aws-s3":         70,
    "azure-ad":       65,
    "cloud-metadata": 65,
    "cloud-iam":      65,
    "serverless":     55,
}

# ---------------------------------------------------------------------------
# Technique pool for synthesis
# ---------------------------------------------------------------------------
TECHNIQUE_POOL: dict[str, list[dict]] = {
    "aws-iam": [
        {"technique": "AWS IAM privilege escalation via iam:CreatePolicyVersion",
         "target": "AWS account 123456789012",
         "scenario": "Leaked access key with limited IAM permissions, escalate to AdministratorAccess"},
        {"technique": "AWS IAM privilege escalation via iam:PassRole + ec2:RunInstances",
         "target": "AWS EC2 + IAM",
         "scenario": "User can pass admin role to new EC2 instance for privilege escalation"},
        {"technique": "AWS IAM role chaining via sts:AssumeRole for lateral movement",
         "target": "AWS multi-account environment",
         "scenario": "Initial role can assume cross-account roles for lateral movement"},
        {"technique": "AWS IAM confused deputy via resource-based policies",
         "target": "AWS Lambda + S3",
         "scenario": "Lambda execution role with overly permissive resource policy"},
        {"technique": "AWS IAM credential exfiltration via instance metadata v1",
         "target": "EC2 instance running vulnerable web app",
         "scenario": "SSRF vulnerability allows reading IMDSv1 credentials at 169.254.169.254"},
        {"technique": "AWS SCP bypass via organization service control policy gaps",
         "target": "AWS Organizations multi-account",
         "scenario": "SCP does not restrict all dangerous IAM actions in member accounts"},
    ],
    "aws-s3": [
        {"technique": "S3 bucket public access misconfiguration data exfiltration",
         "target": "AWS S3 public bucket",
         "scenario": "S3 bucket ACL set to public-read with sensitive data inside"},
        {"technique": "S3 server-side request forgery via presigned URL abuse",
         "target": "S3 presigned URL endpoint",
         "scenario": "Attacker-controlled presigned URL redirected to internal S3 metadata"},
        {"technique": "S3 bucket policy misconfiguration allowing cross-account access",
         "target": "S3 bucket with overly permissive bucket policy",
         "scenario": "Bucket policy allows s3:GetObject for principal: * without conditions"},
        {"technique": "S3 versioning abuse to access deleted sensitive objects",
         "target": "S3 bucket with versioning enabled",
         "scenario": "Previous versions of deleted config files contain credentials"},
        {"technique": "S3 object ACL manipulation for data exfiltration",
         "target": "S3 bucket with write access",
         "scenario": "User has s3:PutObjectAcl, can make any object public"},
    ],
    "azure-ad": [
        {"technique": "Azure AD Service Principal credential theft and abuse",
         "target": "Azure tenant abc123-def456",
         "scenario": "Service principal secret leaked in GitHub, enumerate permissions"},
        {"technique": "Azure AD privilege escalation via app registration permissions",
         "target": "Azure AD application with API permissions",
         "scenario": "App registration has Application permissions for MS Graph, escalate"},
        {"technique": "Azure RBAC privilege escalation via Owner role on subscription",
         "target": "Azure subscription compromised user",
         "scenario": "Low-priv user has Contributor on resource group, Owner on subscription"},
        {"technique": "Azure managed identity abuse for lateral movement",
         "target": "Azure VM with system-assigned managed identity",
         "scenario": "Compromised VM has managed identity with subscription-level access"},
        {"technique": "Azure storage account SAS token abuse",
         "target": "Azure storage account",
         "scenario": "Leaked SAS token with read/write permissions to storage containers"},
    ],
    "cloud-metadata": [
        {"technique": "AWS EC2 instance metadata service v1 credential theft via SSRF",
         "target": "http://169.254.169.254/latest/meta-data/",
         "scenario": "SSRF in web app allows reading AWS metadata for IAM role credentials"},
        {"technique": "GCP metadata service credential theft via SSRF",
         "target": "http://metadata.google.internal/computeMetadata/v1/",
         "scenario": "SSRF reaches GCP metadata endpoint, steals service account token"},
        {"technique": "Azure IMDS credential theft via container escape",
         "target": "http://169.254.169.254/metadata/instance",
         "scenario": "Container breakout reaches Azure IMDS for managed identity token"},
        {"technique": "Kubernetes service account token theft from pod environment",
         "target": "/var/run/secrets/kubernetes.io/serviceaccount/token",
         "scenario": "RCE in k8s pod, extract and abuse service account JWT token"},
    ],
    "cloud-iam": [
        {"technique": "GCP service account key generation for persistent access",
         "target": "GCP project my-project-123",
         "scenario": "Compromised user with iam.serviceAccountKeys.create permission"},
        {"technique": "GCP privilege escalation via iam.serviceAccounts.actAs",
         "target": "GCP project with IAM binding",
         "scenario": "User can impersonate high-privilege service account via actAs"},
        {"technique": "AWS cross-account role assumption via misconfigured trust policy",
         "target": "AWS cross-account trust relationship",
         "scenario": "Trust policy allows assume role from any account in organization"},
        {"technique": "Azure DevOps service connection secret extraction",
         "target": "Azure DevOps pipeline with cloud service connections",
         "scenario": "Access to Azure DevOps allows extracting service connection secrets"},
    ],
    "serverless": [
        {"technique": "AWS Lambda environment variable exfiltration via code injection",
         "target": "AWS Lambda function",
         "scenario": "Lambda with unrestricted execution, extract env vars containing secrets"},
        {"technique": "AWS Lambda function URL SSRF to access internal services",
         "target": "Lambda function URL endpoint",
         "scenario": "Lambda accessible via function URL, SSRF to internal VPC services"},
        {"technique": "GCP Cloud Function environment variable theft",
         "target": "GCP Cloud Function",
         "scenario": "Cloud Function has RCE, extract GOOGLE_APPLICATION_CREDENTIALS"},
        {"technique": "Azure Function App managed identity token abuse",
         "target": "Azure Function App with managed identity",
         "scenario": "Function app has managed identity with Key Vault read access"},
        {"technique": "AWS API Gateway + Lambda injection for privilege escalation",
         "target": "Serverless API backend",
         "scenario": "Injection in API Gateway parameter flows to Lambda IAM operations"},
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
# CloudGoat scenario discovery
# ---------------------------------------------------------------------------
def get_cloudgoat_scenarios() -> list[dict]:
    """Fetch list of CloudGoat scenario directories with README paths."""
    cache_path = Path(CACHE_DIR) / "cloudgoat_scenarios.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    tok = os.getenv("GITHUB_TOKEN")
    hdrs = dict(REQUEST_HEADERS)
    if tok:
        hdrs["Authorization"] = f"token {tok}"

    url = f"{GITHUB_API}/repos/RhinoSecurityLabs/cloudgoat/contents"
    try:
        r = requests.get(url, headers=hdrs, timeout=20)
        if r.status_code != 200:
            return []
        items = r.json()
    except Exception:
        return []

    scenarios = []
    for item in items:
        if item.get("type") != "dir":
            continue
        name = item["name"]
        if name.startswith(".") or name in ("core", "docs", "test"):
            continue
        readme_url = (
            f"{RAW_GITHUB}/RhinoSecurityLabs/cloudgoat/master/{name}/README.md"
        )
        category = _infer_cloud_category(name)
        scenarios.append({
            "name": name,
            "readme_url": readme_url,
            "category": category,
        })

    cache_path.write_text(json.dumps(scenarios))
    return scenarios


def _infer_cloud_category(name: str) -> str:
    n = name.lower()
    if "iam" in n:              return "aws-iam"
    if "s3" in n:               return "aws-s3"
    if "lambda" in n or "serverless" in n: return "serverless"
    if "ec2" in n or "ssrf" in n: return "cloud-metadata"
    if "azure" in n:            return "azure-ad"
    return "aws-iam"


def parse_cloudgoat_readme(text: str) -> tuple[str, str, list[str]]:
    """Parse CloudGoat README for scenario title, summary, and attack steps."""
    lines = text.splitlines()
    title = ""
    for l in lines[:6]:
        if l.startswith("#"):
            title = l.lstrip("#").strip()
            break

    prose = []
    for l in lines[:40]:
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
                if _is_cloud_command(cmd):
                    commands.append(cmd)
                block_lines = []
            continue
        if in_block:
            block_lines.append(line)

    return title, summary, commands[:8]


_CLOUD_TOOLS = re.compile(
    r"(aws\s|az\s|gcloud\s|boto3|pacu|prowler|scoutsuite|trufflehog"
    r"|curl.*169\.254|curl.*metadata\.google|kubectl|terraform|cloudmapper"
    r"|enumerate-iam|WeirdAAL)",
    re.IGNORECASE
)


def _is_cloud_command(cmd: str) -> bool:
    if len(cmd) < 10 or len(cmd) > 2000:
        return False
    return bool(_CLOUD_TOOLS.search(cmd))


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
    scenario: str,
    n_turns: int = 3,
) -> dict | None:
    prompt = CLOUD_CHAIN_PROMPT.format(
        technique=technique,
        category=category,
        target=target,
        scenario=scenario,
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
        f"Target: {target}\n"
        f"Technique: {technique}\n"
        f"Scenario: {scenario}\n\n"
        f"Begin the cloud attack chain. What is your first step?"
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
                next_user = "Command completed. What is the next step?"
            messages.append({"role": "user", "content": next_user})

    if messages[-1]["role"] != "assistant":
        messages = messages[:-1]

    n_asst = sum(1 for m in messages if m["role"] == "assistant")
    if n_asst < MIN_TURNS:
        return None

    slug = f"synth_{category}_{_cache_key(technique + scenario)}"
    return {
        "messages": messages,
        "metadata": {
            "source":    "cloud",
            "technique": technique,
            "category":  category,
            "turns":     n_asst,
            "synthetic": True,
            "url":       "https://github.com/RhinoSecurityLabs/cloudgoat",
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
                spec["technique"], cat, spec["target"], spec["scenario"], n_turns
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
# Build from CloudGoat README
# ---------------------------------------------------------------------------
def build_example_from_readme(
    scenario: dict,
    title: str,
    summary: str,
    commands: list[str],
    category: str,
) -> dict | None:
    if len(commands) < MIN_TURNS:
        return None

    technique = title or scenario["name"]
    first_user = (
        f"Target: AWS account 123456789012 (us-east-1)\n"
        f"Scenario: {technique}\n\n"
        f"{summary[:300]}\n\n"
        f"Begin the cloud attack. What is your first step?"
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": first_user},
    ]

    context = f"Scenario: {technique}\n"
    for idx, cmd in enumerate(commands[:MAX_TURNS]):
        if not cmd.strip():
            continue
        prompt = THOUGHT_PROMPT.format(
            technique=technique, category=category,
            target="AWS account 123456789012",
            command=cmd[:400], context=context[-500:],
        )
        try:
            thought = _gpt(prompt, 250)
        except Exception:
            thought = f"Executing cloud attack step {idx+1}."
        asst = f"<thought>\n{thought}\n</thought>\n\n<command>\n{cmd}\n</command>"
        messages.append({"role": "assistant", "content": asst})
        context += f"\nStep {idx+1}: {cmd[:80]}"
        if idx < len(commands) - 1:
            messages.append({"role": "user", "content": "Output received. What is the next step?"})

    if messages[-1]["role"] != "assistant":
        messages = messages[:-1]

    n_asst = sum(1 for m in messages if m["role"] == "assistant")
    if n_asst < MIN_TURNS:
        return None

    slug = _cache_key(scenario["name"])
    return {
        "messages": messages,
        "metadata": {
            "source":    "cloud",
            "technique": technique,
            "category":  category,
            "turns":     n_asst,
            "url":       f"https://github.com/RhinoSecurityLabs/cloudgoat/tree/master/{scenario['name']}",
            "slug":      slug,
        },
    }


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
                cat = json.loads(line)["metadata"].get("category", "aws-iam")
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
    parser.add_argument("--synthetic-only",  action="store_true")
    args = parser.parse_args()

    if args.list_categories:
        print_progress(load_existing_counts(args.out))
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.test:
        print("=" * 70)
        print(f"TEST MODE — {args.test_n} cloud attack chains")
        print("=" * 70)
        pool_items = [(cat, spec)
                      for cat, specs in TECHNIQUE_POOL.items()
                      for spec in specs]
        sample = random.sample(pool_items, min(args.test_n, len(pool_items)))
        for cat, spec in sample:
            print(f"\n── {cat} ──  {spec['technique'][:65]}")
            ex = generate_synthetic_chain(
                spec["technique"], cat, spec["target"], spec["scenario"], 3
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

    # ── Phase 1: CloudGoat README scrape ──────────────────────────────────
    if not args.synthetic_only:
        print("Fetching CloudGoat scenarios...")
        scenarios = get_cloudgoat_scenarios()
        print(f"Found {len(scenarios)} CloudGoat scenarios")
        print_progress(cat_counts)

        todo = [s for s in scenarios if _cache_key(s["name"]) not in seen_slugs]

        def _worker(scenario: dict) -> dict | None:
            cat = scenario["category"]
            if cat_counts.get(cat, 0) >= BENCH_TARGET_PER_CAT.get(cat, 0):
                return None
            text = _cached_get(scenario["readme_url"])
            if not text:
                return None
            time.sleep(API_DELAY)
            title, summary, commands = parse_cloudgoat_readme(text)
            if len(commands) < MIN_TURNS:
                return None
            return build_example_from_readme(scenario, title, summary, commands, cat)

        with open(args.out, "a") as outf:
            with tqdm(total=len(todo), desc="CloudGoat") as pbar:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(_worker, s): s for s in todo}
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

        print(f"\nCloudGoat pass done. Wrote {written} examples.")
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
