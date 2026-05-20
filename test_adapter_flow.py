#!/usr/bin/env python3
"""Adapter flow test — runs all adapters individually and chains outputs.

Tests two things:
  1. SOLO: each adapter with a relevant standalone prompt (verify it loads
     and produces coherent output for its domain)
  2. CHAINED: osint → webapp → analyst → researcher (simulate a real
     bug bounty session where each adapter's output feeds the next)

Usage:
    # All adapters, full chain (GPU)
    python test_adapter_flow.py --target supplier.meesho.com

    # Just specific adapters
    python test_adapter_flow.py --adapters osint webapp analyst

    # Solo tests only (no chain)
    python test_adapter_flow.py --solo-only

    # Chain only
    python test_adapter_flow.py --chain-only

    # Limit output tokens (faster)
    python test_adapter_flow.py --max-tokens 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make mythos_engine importable
sys.path.insert(0, str(Path(__file__).parent / "mythos_engine"))

from pentestgpt.core.config import DEFAULT_ADAPTERS, load_state
from pentestgpt.core.multi_adapter_model import AdapterSpec, MultiAdapterModel, QWEN3_ADAPTERS
from pentestgpt.prompts.mythos_prompts import SYSTEM_PROMPTS, BUG_BOUNTY_PROMPTS, FORMAT_INSTRUCTION

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"

ADAPTER_COLOUR: dict[str, str] = {
    "osint":      "\033[34m",   # blue
    "webapp":     "\033[35m",   # magenta
    "analyst":    "\033[36m",   # cyan
    "researcher": "\033[33m",   # yellow
    "htb":        "\033[32m",   # green
    "attack":     "\033[31m",   # red
    "vulhub":     "\033[91m",   # bright red
    "exploitdb":  "\033[95m",   # bright magenta
    "ad":         "\033[96m",   # bright cyan
    "cloud":      "\033[94m",   # bright blue
    "planner":    "\033[92m",   # bright green
    "executor":   "\033[93m",   # bright yellow
}

# ── Scope context (mirrors meesho/scope.json) ─────────────────────────────────
MEESHO_SCOPE = """\
[BUG BOUNTY SCOPE — Meesho Bug Bounty (HackerOne)]

⚠️  MANDATORY: Add this header to ALL HTTP requests or WAF will block you:
  -H "X-Hackerone: aquamarine_skeleton"
  Example: curl -s -H "X-Hackerone: aquamarine_skeleton" https://supplier.meesho.com/

In scope:
  + meesho.com (web)
  + supplier.meesho.com (web)

Out of scope (do NOT test):
  - rate-limiting
  - clickjacking
  - self-xss

Test accounts (use these — do NOT create new accounts):
  [supplier] user1=suppliertest-1@meeshoai.com  user2=suppliertest-2@meeshoai.com  password=Hackerone@123$
  [consumer] mobile1=6666666661  mobile2=6666666662  otp=999999
"""

# ── Solo test prompts — one per adapter, realistic for its domain ─────────────
SOLO_PROMPTS: dict[str, str] = {
    "osint": (
        "Enumerate the attack surface of supplier.meesho.com. "
        "Use subfinder for subdomain discovery, then probe live hosts with httpx. "
        "Include the mandatory X-Hackerone header in all HTTP requests."
    ),
    "webapp": (
        "The endpoint https://supplier.meesho.com/api/supplier/order/list "
        "returns order details. Test it for IDOR by changing the supplier ID "
        "in the Authorization header. Use test account: suppliertest-1@meeshoai.com / Hackerone@123$. "
        "Always include -H 'X-Hackerone: aquamarine_skeleton' in curl commands."
    ),
    "analyst": (
        "Tool output:\n"
        "[subfinder]\nsupplier.meesho.com\napi.supplier.meesho.com\nstatic.supplier.meesho.com\n\n"
        "[httpx]\nhttps://supplier.meesho.com [200] [nginx/1.18] [React]\n"
        "https://api.supplier.meesho.com [401] [nginx/1.18]\n"
        "https://static.supplier.meesho.com [200] [nginx/1.18] [CloudFront]\n\n"
        "Extract all intelligence: subdomains, live endpoints, tech stack, findings, anomalies."
    ),
    "researcher": (
        "HYPOTHESIS: The /api/supplier/order/export endpoint accepts a {\"format\": \"pdf\"} parameter. "
        "Response time for format=pdf is 3x longer than format=csv. "
        "Error on format=x leaks: template: pdf.tmpl:14. "
        "Design a minimal probe to test for Go SSTI via the format parameter."
    ),
    "htb": (
        "I have SSH access to a Linux machine at 10.10.10.50. "
        "Running `sudo -l` shows I can run /usr/bin/python3 as root. "
        "Escalate privileges to root."
    ),
    "vulhub": (
        "CVE-2021-44228 (Log4Shell) — the target is running Apache Solr 8.11.0. "
        "Craft a payload to test for JNDI injection via the User-Agent header."
    ),
    "exploitdb": (
        "Write a Python exploit for CVE-2019-0708 (BlueKeep) RDP vulnerability. "
        "Target: Windows Server 2008 R2 at 10.10.10.50:3389."
    ),
    "attack": (
        "Implement MITRE ATT&CK T1003.001 (LSASS Memory Dump) on a compromised "
        "Windows host where you have SYSTEM privileges. "
        "Use the Mimikatz sekurlsa::logonpasswords module."
    ),
    "ad": (
        "I have a foothold as user jdoe@corp.local. BloodHound shows "
        "jdoe has GenericWrite on the ServiceAccount object. "
        "Perform a targeted Kerberoasting attack to escalate privileges."
    ),
    "cloud": (
        "We found an SSRF in the export feature at https://app.target.com/export?url=. "
        "The server appears to be on AWS. "
        "Exploit the SSRF to retrieve AWS IAM credentials from the metadata service "
        "and enumerate accessible S3 buckets."
    ),
    "planner": (
        "Current state: supplier.meesho.com identified, 3 subdomains found, "
        "API at api.supplier.meesho.com returns 401 without auth. "
        "We have test credentials. Phase: web_vuln_scan. "
        "What is the highest-value next action?"
    ),
    "executor": (
        "Generate a correct subfinder command to enumerate subdomains of supplier.meesho.com, "
        "outputting results to subdomains.txt, using all sources, verbose mode."
    ),
}

# ── Chain scenario ─────────────────────────────────────────────────────────────
# Simulates: osint recon → webapp vuln testing → analyst interprets → researcher probes
CHAIN_TARGET = "supplier.meesho.com"
CHAIN_STEPS: list[tuple[str, str]] = [
    ("osint", (
        f"Begin reconnaissance on {CHAIN_TARGET}. "
        "Run subfinder to find subdomains, then probe live hosts. "
        "Include -H 'X-Hackerone: aquamarine_skeleton' in all HTTP requests."
    )),
    ("webapp", (
        "PREVIOUS_OUTPUT_HERE\n\n"
        "Given the reconnaissance above, test supplier.meesho.com for IDOR vulnerabilities. "
        "Check the order API endpoints. Use credentials: suppliertest-1@meeshoai.com / Hackerone@123$. "
        "Always include -H 'X-Hackerone: aquamarine_skeleton'."
    )),
    ("analyst", (
        "PREVIOUS_OUTPUT_HERE\n\n"
        "Extract structured findings from the above recon and testing session. "
        "Return: subdomains found, live endpoints, potential vulnerabilities, and anomalies."
    )),
    ("researcher", (
        "PREVIOUS_OUTPUT_HERE\n\n"
        "Standard vulnerability scanning found no confirmed findings. "
        "Form a hypothesis from any behavioral anomalies observed and design a minimal probe."
    )),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def separator(label: str = "", colour: str = CYAN) -> None:
    width = 88
    if label:
        pad = (width - len(label) - 4) // 2
        print(f"\n{colour}{'─' * pad}  {BOLD}{label}{RESET}{colour}  {'─' * pad}{RESET}")
    else:
        print(f"{DIM}{'─' * width}{RESET}")


def build_messages(adapter: str, prompt: str, history: list[dict], mode: str = "bug_bounty") -> list[dict]:
    """Build a messages list for inference."""
    prompt_map = BUG_BOUNTY_PROMPTS if mode == "bug_bounty" else SYSTEM_PROMPTS
    fallback   = BUG_BOUNTY_PROMPTS.get("htb", SYSTEM_PROMPTS["htb"])
    base       = prompt_map.get(adapter) or SYSTEM_PROMPTS.get(adapter) or fallback
    system     = f"{MEESHO_SCOPE}\n\n{base}{FORMAT_INSTRUCTION}"
    msgs: list[dict] = [{"role": "system", "content": system}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": prompt})
    return msgs


def run_adapter(
    model:      MultiAdapterModel,
    adapter:    str,
    prompt:     str,
    history:    list[dict],
    max_tokens: int,
    step_label: str = "",
) -> tuple[str, float, int]:
    """Run one adapter inference. Returns (response, elapsed_s, output_tokens)."""
    msgs = build_messages(adapter, prompt, history)
    t0   = time.time()
    resp = model.generate(adapter, msgs, max_new_tokens=max_tokens)
    elapsed = time.time() - t0
    out_tokens = model.count_tokens(resp)
    return resp, elapsed, out_tokens


def print_result(
    adapter:    str,
    prompt:     str,
    response:   str,
    elapsed:    float,
    out_tokens: int,
    max_display: int = 600,
) -> None:
    colour = ADAPTER_COLOUR.get(adapter, CYAN)
    separator(f" {adapter.upper()} ", colour)
    print(f"{DIM}PROMPT: {prompt[:120]}{'...' if len(prompt) > 120 else ''}{RESET}")
    separator()
    display = response[:max_display]
    if len(response) > max_display:
        display += f"\n{DIM}... [{len(response) - max_display} chars truncated]{RESET}"
    print(display)
    separator()
    status_colour = GREEN if response.strip() and "<command>" in response else YELLOW
    print(
        f"{status_colour}⏱ {elapsed:.1f}s{RESET}  "
        f"{DIM}| {out_tokens} output tokens{RESET}  "
        f"| {'✓ has <command>' if '<command>' in response else '✗ no <command>'}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def load_model(project_root: str, max_tokens: int) -> MultiAdapterModel:
    """Load the Qwen3-14B base with all available adapters."""
    print(f"\n{BOLD}{CYAN}Loading MultiAdapterModel...{RESET}")
    state_adapters = load_state().get("adapters", {})
    specs: dict[str, AdapterSpec] = {}

    for name, ac in DEFAULT_ADAPTERS.items():
        path = ac.path
        # Prefer absolute path from state.json
        state_entry = state_adapters.get(name, {})
        if state_entry.get("exists") and state_entry.get("local_path"):
            path = state_entry["local_path"]

        adapter_path = Path(path) if Path(path).is_absolute() else Path(project_root) / path
        if adapter_path.exists():
            specs[name] = AdapterSpec(path=str(adapter_path), base=ac.base, is_qwen3=ac.is_qwen3, max_seq=ac.max_seq)
            print(f"  {GREEN}✓{RESET} {name:<12} {DIM}{adapter_path}{RESET}")
        else:
            print(f"  {YELLOW}✗{RESET} {name:<12} {DIM}NOT FOUND: {adapter_path}{RESET}")

    if not specs:
        print(f"{RED}No adapters found. Check paths and run bootstrap.py.{RESET}")
        sys.exit(1)

    model = MultiAdapterModel(
        adapter_specs=specs,
        project_root=project_root,
        mock=False,
        max_new_tokens=max_tokens,
        temperature=0.3,
    )
    loaded = model.loaded_adapters()
    print(f"\n{GREEN}✓ Pool A loaded: {loaded}{RESET}")
    print(f"{DIM}Model context limit: {model.max_context_length()} tokens{RESET}\n")
    return model


def run_solo_tests(
    model:      MultiAdapterModel,
    adapters:   list[str],
    max_tokens: int,
    results:    dict,
) -> None:
    separator("SOLO ADAPTER TESTS", CYAN)
    print(f"Testing {len(adapters)} adapters individually\n")

    for adapter in adapters:
        if adapter not in model.loaded_adapters():
            print(f"{YELLOW}  ⚠ {adapter} not loaded — skipping{RESET}")
            results[adapter] = {"status": "not_loaded"}
            continue

        prompt = SOLO_PROMPTS.get(adapter, f"Perform a security assessment of supplier.meesho.com using your {adapter} expertise.")
        print(f"\n{BOLD}Testing {ADAPTER_COLOUR.get(adapter, CYAN)}{adapter}{RESET}...", end=" ", flush=True)

        try:
            response, elapsed, out_tokens = run_adapter(
                model, adapter, prompt, [], max_tokens,
            )
            has_cmd = "<command>" in response
            has_thought = "<thought>" in response
            print(f"{GREEN}done ({elapsed:.1f}s, {out_tokens}t){RESET}")

            results[adapter] = {
                "status":      "ok",
                "elapsed_s":   round(elapsed, 2),
                "out_tokens":  out_tokens,
                "has_command": has_cmd,
                "has_thought": has_thought,
                "response":    response[:2000],
            }
            print_result(adapter, prompt, response, elapsed, out_tokens)

        except Exception as e:
            print(f"{RED}FAILED: {e}{RESET}")
            results[adapter] = {"status": "error", "error": str(e)}


def run_chain_test(
    model:      MultiAdapterModel,
    max_tokens: int,
    results:    dict,
) -> None:
    separator("CHAINED ADAPTER FLOW", CYAN)
    print(
        f"Chain: {' → '.join(a for a, _ in CHAIN_STEPS)}\n"
        f"Each adapter's output becomes context for the next.\n"
    )

    history:   list[dict] = []
    prev_output = ""
    chain_results = []

    for i, (adapter, prompt_template) in enumerate(CHAIN_STEPS, 1):
        # Inject previous adapter's output into the prompt
        prompt = prompt_template.replace("PREVIOUS_OUTPUT_HERE", prev_output) if prev_output else prompt_template

        loaded = model.loaded_adapters()
        if adapter not in loaded:
            print(f"{YELLOW}  Step {i}: {adapter} not loaded — using first available fallback{RESET}")
            adapter = loaded[0] if loaded else adapter

        separator(f"STEP {i}/{len(CHAIN_STEPS)} — {adapter.upper()}", ADAPTER_COLOUR.get(adapter, CYAN))
        print(f"{DIM}Prompt: {prompt[:150]}...{RESET}\n")

        t0 = time.time()
        try:
            response, elapsed, out_tokens = run_adapter(
                model, adapter, prompt, history, max_tokens,
            )
            print(f"{BOLD}Response:{RESET}")
            display = response[:800]
            if len(response) > 800:
                display += f"\n{DIM}... [{len(response) - 800} more chars]{RESET}"
            print(display)
            separator()
            print(
                f"{GREEN}✓{RESET} {elapsed:.1f}s  |  {out_tokens} tokens  "
                f"| {'✓ <command>' if '<command>' in response else '✗ no command'}  "
                f"| {'✓ <thought>' if '<thought>' in response else '✗ no thought'}"
            )

            history.append({"role": "user",      "content": prompt})
            history.append({"role": "assistant", "content": response})
            prev_output = response

            chain_results.append({
                "step":       i,
                "adapter":    adapter,
                "elapsed_s":  round(elapsed, 2),
                "out_tokens": out_tokens,
                "has_command": "<command>" in response,
                "has_thought": "<thought>" in response,
            })

        except Exception as e:
            print(f"{RED}Step {i} FAILED: {e}{RESET}")
            chain_results.append({"step": i, "adapter": adapter, "status": "error", "error": str(e)})

    results["chain"] = chain_results


def print_summary(results: dict) -> None:
    separator("SUMMARY", BOLD)

    # Solo results
    solo_adapters = [k for k in results if k != "chain"]
    if solo_adapters:
        print(f"\n{'Adapter':<14} {'Status':<10} {'Time':<8} {'Tokens':<8} {'<cmd>?':<8} {'<thought>?'}")
        print(f"{'─'*14} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
        for adapter in solo_adapters:
            r = results[adapter]
            if r.get("status") == "not_loaded":
                print(f"{YELLOW}{adapter:<14}{RESET} {'NOT LOADED':<10}")
            elif r.get("status") == "error":
                print(f"{RED}{adapter:<14}{RESET} {'ERROR':<10} {r.get('error', '')[:40]}")
            else:
                cmd_ok  = GREEN + "✓" + RESET if r.get("has_command") else RED + "✗" + RESET
                thk_ok  = GREEN + "✓" + RESET if r.get("has_thought") else YELLOW + "~" + RESET
                t_str   = f"{r['elapsed_s']:.1f}s"
                tok_str = str(r.get("out_tokens", "?"))
                print(f"{ADAPTER_COLOUR.get(adapter,CYAN)}{adapter:<14}{RESET} {'OK':<10} {t_str:<8} {tok_str:<8} {cmd_ok}       {thk_ok}")

    # Chain results
    if "chain" in results:
        print(f"\n{BOLD}Chain:{RESET}")
        for step in results["chain"]:
            if step.get("status") == "error":
                print(f"  {RED}Step {step['step']} ({step['adapter']}): ERROR — {step.get('error','')[:60]}{RESET}")
            else:
                cmd_ok = GREEN + "✓" + RESET if step.get("has_command") else RED + "✗" + RESET
                print(
                    f"  Step {step['step']} {ADAPTER_COLOUR.get(step['adapter'],CYAN)}{step['adapter']:<12}{RESET} "
                    f"{step['elapsed_s']:.1f}s  {step['out_tokens']}t  cmd={cmd_ok}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mythos adapter flow test")
    parser.add_argument("--target",     default="supplier.meesho.com",
                        help="Target for the test scenario")
    parser.add_argument("--adapters",   nargs="+", default=None,
                        help="Specific adapters to solo-test (default: all loaded)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max new tokens per generation (default: 512, use 256 for speed)")
    parser.add_argument("--solo-only",  action="store_true",
                        help="Run only solo adapter tests (no chain)")
    parser.add_argument("--chain-only", action="store_true",
                        help="Run only the chained flow test")
    parser.add_argument("--save",       default="",
                        help="Save results JSON to this file")
    parser.add_argument("--project-root", default="",
                        help="Project root (default: directory of this script)")
    args = parser.parse_args()

    project_root = args.project_root or str(Path(__file__).parent)
    CHAIN_STEPS[0] = (CHAIN_STEPS[0][0], CHAIN_STEPS[0][1].replace("supplier.meesho.com", args.target))

    print(f"""
{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MYTHOS ENGINE — ADAPTER FLOW TEST
  target: {args.target}
  max-tokens: {args.max_tokens}
  mode: {'solo only' if args.solo_only else 'chain only' if args.chain_only else 'solo + chain'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
""")

    model   = load_model(project_root, args.max_tokens)
    loaded  = model.loaded_adapters()
    results: dict = {}

    adapters_to_test = args.adapters or sorted(loaded)

    if not args.chain_only:
        run_solo_tests(model, adapters_to_test, args.max_tokens, results)

    if not args.solo_only:
        run_chain_test(model, args.max_tokens, results)

    print_summary(results)

    if args.save:
        out_path = Path(args.save)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\n{GREEN}Results saved to: {out_path}{RESET}")


if __name__ == "__main__":
    main()
