#!/usr/bin/env python3
"""Mythos Engine Bootstrap — portable environment initialiser.

Run once on any machine (local workstation, rented GPU, cloud VM) before
starting the engine.  Detects the project root from its own location,
discovers or downloads all required resources, and writes a resolved
state file to ~/.mythosengine/state.json that every other script reads.

Usage:
    python3 bootstrap.py              # full init + report
    python3 bootstrap.py --check      # verify state, exit 1 if anything required is missing
    python3 bootstrap.py --pull       # re-pull all missing adapters from HF Hub
    python3 bootstrap.py --no-hf      # skip HuggingFace download (offline / air-gapped)
    python3 bootstrap.py --json       # print state.json to stdout (for scripting)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Resolve project root from this file — never CWD ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
STATE_FILE   = Path.home() / ".mythosengine" / "state.json"

# ── Adapter registry ─────────────────────────────────────────────────────────
# (name, relative-path-candidate-1, relative-path-candidate-2, HF repo ID)
ADAPTER_SPECS = [
    ("htb",       "adapters/mythos-v4-htb/best-adapter",       "outputs_new/mythos-v4-htb/best-adapter",       ""),
    ("vulhub",    "adapters/mythos-v4-vulhub/best-adapter",    "outputs_new/mythos-v4-vulhub/best-adapter",    ""),
    ("attack",    "adapters/mythos-v4-attack/best-adapter",    "outputs_new/mythos-v4-attack/best-adapter",    ""),
    ("exploitdb", "adapters/mythos-v4-exploitdb/best-adapter", "outputs_new/mythos-v4-exploitdb/best-adapter", ""),
    ("ad",        "adapters/mythos-v4-ad/best-adapter",        "outputs_new/mythos-v4-ad/best-adapter",        ""),
    ("webapp",    "adapters/mythos-v4-webapp/best-adapter",    "outputs_new/mythos-v4-webapp/best-adapter",    ""),
    ("osint",     "adapters/mythos-v4-osint/best-adapter",     "outputs_new/mythos-v4-osint/best-adapter",     ""),
    ("cloud",     "adapters/mythos-v4-cloud/best-adapter",     "outputs_new/mythos-v4-cloud/best-adapter",     ""),
    # New adapters (trained separately — orchestration / reasoning layer)
    ("executor",  "adapters/mythos-v4-executor/best-adapter",  "outputs_new/mythos-v4-executor/best-adapter",  ""),
    ("analyst",   "adapters/mythos-v4-analyst/best-adapter",   "outputs_new/mythos-v4-analyst/best-adapter",   ""),
    ("planner",   "adapters/mythos-v4-planner/best-adapter",   "outputs_new/mythos-v4-planner/best-adapter",   ""),
    ("researcher","adapters/mythos-v4-researcher/best-adapter","outputs_new/mythos-v4-researcher/best-adapter",""),
]

# ── Tools expected by the engine ─────────────────────────────────────────────
TOOLS = [
    # port scanning
    "nmap", "masscan", "rustscan",
    # subdomain enum
    "subfinder", "amass", "assetfinder", "dnsx", "sublist3r",
    # HTTP probing
    "httpx", "katana", "waybackurls", "gau", "whatweb", "wafw00f", "nikto",
    # fuzzing
    "ffuf", "gobuster", "feroxbuster", "dirsearch",
    # web vuln
    "sqlmap", "nuclei", "dalfox", "wpscan",
    # OSINT
    "theHarvester", "shodan", "recon-ng",
    # AD / Windows
    "kerbrute", "evil-winrm",
    # exploitation
    "msfconsole",
    # credentials
    "hydra", "john", "hashcat",
    # tunnelling
    "chisel",
    # cloud
    "aws", "az",
    # core utils
    "curl", "wget", "jq", "git", "python3",
]

# ── Wordlist candidate paths (checked in order, first found wins) ─────────────
WORDLIST_CANDIDATES: dict[str, list[str]] = {
    "seclists": [
        "/usr/share/seclists",
        "/usr/share/SecLists",
        str(Path.home() / "wordlists" / "SecLists"),
        str(Path.home() / "SecLists"),
    ],
    "rockyou": [
        "/usr/share/wordlists/rockyou.txt",
        "/usr/share/seclists/Passwords/Leaked-Databases/rockyou.txt",
        "/usr/share/SecLists/Passwords/Leaked-Databases/rockyou.txt",
        str(Path.home() / "wordlists" / "rockyou.txt"),
    ],
    "common_dirs": [
        "/usr/share/seclists/Discovery/Web-Content/common.txt",
        "/usr/share/SecLists/Discovery/Web-Content/common.txt",
        "/usr/share/dirb/common.txt",
        str(Path.home() / "wordlists" / "common.txt"),
    ],
    "api_paths": [
        "/usr/share/seclists/Discovery/Web-Content/api/objects.txt",
        "/usr/share/SecLists/Discovery/Web-Content/api/objects.txt",
    ],
}

# ── Runtime directories to ensure exist ──────────────────────────────────────
RUNTIME_DIRS = [
    Path.home() / ".mythosengine",
    Path.home() / ".mythosengine" / "workspace",
    Path.home() / ".mythosengine" / "reports",
    Path.home() / ".mythosengine" / "tools",
    Path.home() / ".mythosengine" / "rag_cache",
    Path.home() / ".mythosengine" / "feedback",
    Path.home() / ".mythosengine" / "engagements",
    Path.home() / ".pentestgpt" / "sessions",
]


# ── Rich (optional — graceful fallback if not installed) ─────────────────────

def _try_rich() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


class _FallbackConsole:
    """Minimal fallback console when Rich is not available."""
    def print(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        # Strip rich markup like [bold], [green], etc.
        import re
        text = re.sub(r"\[/?[^\]]*\]", "", text)
        print(text)

    def rule(self, title="", **kwargs):
        print(f"\n{'─' * 20} {title} {'─' * 20}\n")


def _get_console():
    if _try_rich():
        from rich.console import Console
        return Console()
    return _FallbackConsole()


def _status_symbol(ok: bool, required: bool = True) -> str:
    if ok:
        return "[bold green]✓[/]"
    return "[bold red]✗[/]" if required else "[bold yellow]–[/]"


# ── State helpers ─────────────────────────────────────────────────────────────

def load_existing_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["bootstrapped_at"] = datetime.now(timezone.utc).isoformat()
    state["project_root"]    = str(PROJECT_ROOT)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Step 1: Runtime directories ───────────────────────────────────────────────

def ensure_runtime_dirs(console) -> dict[str, str]:
    console.rule("Runtime Directories")
    created = {}
    for d in RUNTIME_DIRS:
        existed = d.exists()
        d.mkdir(parents=True, exist_ok=True)
        label = str(d).replace(str(Path.home()), "~")
        sym   = "[dim]·[/]" if existed else "[bold green]+[/]"
        console.print(f"  {sym} {label}")
        created[label] = str(d)
    return created


# ── Step 2: Adapter scan / pull ───────────────────────────────────────────────

def _find_adapter_local(candidates: list[str]) -> str | None:
    """Return the first candidate path that contains adapter_config.json or config.json."""
    for rel in candidates:
        p = PROJECT_ROOT / rel
        if p.exists() and (
            (p / "adapter_config.json").exists()
            or (p / "config.json").exists()
            or any(p.glob("*.safetensors"))
            or any(p.glob("adapter_model.bin"))
        ):
            return str(p)
    return None


def _pull_from_hf(name: str, hf_repo: str, dest: Path, console) -> bool:
    if not hf_repo:
        console.print(f"    [yellow]No HF repo configured for {name} — skipping pull[/]")
        return False
    try:
        from huggingface_hub import snapshot_download
        console.print(f"    [cyan]Pulling {name} from {hf_repo}…[/]")
        snapshot_download(repo_id=hf_repo, local_dir=str(dest))
        return True
    except ImportError:
        console.print("    [yellow]huggingface_hub not installed — run: pip install huggingface_hub[/]")
        return False
    except Exception as exc:
        console.print(f"    [red]HF pull failed for {name}: {exc}[/]")
        return False


def scan_adapters(console, pull: bool = False, no_hf: bool = False) -> dict[str, dict]:
    console.rule("Adapters")
    result = {}
    for name, path1, path2, hf_repo in ADAPTER_SPECS:
        local = _find_adapter_local([path1, path2])
        if local:
            console.print(f"  {_status_symbol(True)} {name:<12} {local.replace(str(PROJECT_ROOT) + '/', '')}")
            result[name] = {"local_path": local, "hf_repo": hf_repo, "exists": True, "verified_at": datetime.now(timezone.utc).isoformat()}
        else:
            dest = PROJECT_ROOT / path1
            if pull and not no_hf:
                console.print(f"  [yellow]↓[/] {name:<12} not found — attempting HF pull…")
                ok = _pull_from_hf(name, hf_repo, dest, console)
                if ok:
                    local = str(dest)
                    result[name] = {"local_path": local, "hf_repo": hf_repo, "exists": True, "verified_at": datetime.now(timezone.utc).isoformat()}
                    continue
            console.print(f"  {_status_symbol(False, required=False)} {name:<12} [dim]not found  (expected: {path1})[/]")
            result[name] = {"local_path": str(PROJECT_ROOT / path1), "hf_repo": hf_repo, "exists": False, "verified_at": None}
    return result


# ── Step 3: Tool scan ─────────────────────────────────────────────────────────

def scan_tools(console) -> dict[str, dict]:
    console.rule("Security Tools")
    result = {}
    col = 0
    lines = []
    for tool in TOOLS:
        path = shutil.which(tool)
        ok   = path is not None
        sym  = "[green]✓[/]" if ok else "[dim]–[/]"
        entry = f"  {sym} {tool:<20}"
        lines.append((entry, ok))
        result[tool] = {"path": path, "installed": ok}

    # Print in 2 columns
    for i in range(0, len(lines), 2):
        left  = lines[i][0]
        right = lines[i + 1][0] if i + 1 < len(lines) else ""
        console.print(left + right)

    missing = [t for t, d in result.items() if not d["installed"]]
    if missing:
        console.print(f"\n  [yellow]Missing ({len(missing)}):[/] {', '.join(missing)}")
        console.print("  Run [bold]sudo bash install_tools.sh[/] to install all tools.")
    return result


# ── Step 4: Wordlist scan ─────────────────────────────────────────────────────

def scan_wordlists(console) -> dict[str, str | None]:
    console.rule("Wordlists")
    result = {}
    for name, candidates in WORDLIST_CANDIDATES.items():
        found = next((c for c in candidates if Path(c).exists()), None)
        sym = _status_symbol(bool(found), required=(name in ("seclists", "rockyou")))
        label = found or "[dim]not found[/]"
        console.print(f"  {sym} {name:<16} {label}")
        result[name] = found
    return result


# ── Step 5: .env file ─────────────────────────────────────────────────────────

def find_env_file(console) -> str | None:
    console.rule(".env File")
    candidates = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "mythos_engine" / ".env",
        Path.home() / ".mythosengine" / ".env",
    ]
    for c in candidates:
        if c.exists():
            console.print(f"  {_status_symbol(True)} {c}")
            return str(c)
    console.print(f"  {_status_symbol(False, required=False)} not found  [dim](optional — only needed for OpenAI API key)[/]")
    return None


# ── Step 6: Summary table ─────────────────────────────────────────────────────

def print_summary(state: dict, console) -> bool:
    console.rule("Summary")
    adapters  = state.get("adapters", {})
    tools     = state.get("tools", {})
    wordlists = state.get("wordlists", {})

    adapters_ok   = sum(1 for a in adapters.values() if a.get("exists"))
    tools_ok      = sum(1 for t in tools.values() if t.get("installed"))
    seclists_ok   = wordlists.get("seclists") is not None

    console.print(f"  Adapters  : [bold]{adapters_ok}/{len(adapters)}[/] present")
    console.print(f"  Tools     : [bold]{tools_ok}/{len(tools)}[/] installed")
    console.print(f"  SecLists  : {_status_symbol(seclists_ok, required=False)}")
    console.print(f"  State     : {STATE_FILE}")
    console.print(f"  Root      : {PROJECT_ROOT}")
    console.print()

    if adapters_ok == 0:
        console.print("[bold red]No adapters found.[/] The engine will not load models.")
        console.print("Run [bold]python3 bootstrap.py --pull[/] to download from HuggingFace.")
        return False

    console.print("[bold green]Bootstrap complete.[/] Run the engine with:")
    console.print(f"  [bold]cd {PROJECT_ROOT}/mythos_engine && python3 run.py --target <host> --bug-bounty --scope ../engagements/coupang-tw/scope.json --rag --execute[/]")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check",  action="store_true", help="Verify state only; exit 1 if required resources missing")
    parser.add_argument("--pull",   action="store_true", help="Pull missing adapters from HuggingFace Hub")
    parser.add_argument("--no-hf",  action="store_true", help="Skip HuggingFace download (offline)")
    parser.add_argument("--json",   action="store_true", help="Print state.json to stdout and exit")
    parser.add_argument("--quiet",  action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    if args.json:
        if STATE_FILE.exists():
            print(STATE_FILE.read_text())
        else:
            print("{}")
        return

    console = _get_console()
    if not args.quiet:
        console.print()
        console.print("[bold #f59e0b]⚡ Mythos Engine Bootstrap[/]")
        console.print(f"  Project root : [bold]{PROJECT_ROOT}[/]")
        console.print(f"  State file   : [bold]{STATE_FILE}[/]")
        console.print(f"  Host         : [bold]{os.uname().nodename if hasattr(os, 'uname') else 'unknown'}[/]")
        console.print(f"  Python       : [bold]{sys.version.split()[0]}[/]")
        console.print()

    state = load_existing_state()
    state["project_root"] = str(PROJECT_ROOT)

    if not args.check:
        state["runtime_dirs"] = {}
        for d in RUNTIME_DIRS:
            existed = d.exists()
            d.mkdir(parents=True, exist_ok=True)
            if not args.quiet:
                label = str(d).replace(str(Path.home()), "~")
                sym   = "[dim]·[/]" if existed else "[bold green]+[/]"
                if not existed:
                    console.print(f"  {sym} created {label}")
        if not args.quiet:
            console.print()

    state["adapters"]  = scan_adapters(console if not args.quiet else _FallbackConsole(), pull=args.pull, no_hf=args.no_hf)
    state["tools"]     = scan_tools(console if not args.quiet else _FallbackConsole())
    state["wordlists"] = scan_wordlists(console if not args.quiet else _FallbackConsole())
    state["env_file"]  = find_env_file(console if not args.quiet else _FallbackConsole())

    if not args.check:
        save_state(state)
        if not args.quiet:
            console.print(f"\n  [dim]State written to {STATE_FILE}[/]\n")

    ok = print_summary(state, console if not args.quiet else _FallbackConsole())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
