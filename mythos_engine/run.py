#!/usr/bin/env python3
"""Mythos Engine — entrypoint.

Usage examples:
    # Interactive TUI (requires GPU + trained adapters)
    python run.py --target 10.10.11.5

    # No-GPU mock mode for development / testing
    python run.py --target 10.10.11.5 --mock

    # Headless CLI streaming mode
    python run.py --target 10.10.11.5 --cli

    # Resume a previous session
    python run.py --target 10.10.11.5 --session <session_id>

    # Override specific adapter path
    python run.py --target 10.10.11.5 --adapter-path htb:/path/to/adapter

    # Show engagement state for a completed session
    python run.py --list-sessions
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make sure the mythos_engine package is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mythos",
        description="Mythos Engine — Adaptive Pentest AI with LoRA hot-swap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Target ────────────────────────────────────────────────────────────────
    p.add_argument("--target", "-t", required=False, default="",
                   help="Target IP / hostname / URL")

    # ── Mode ──────────────────────────────────────────────────────────────────
    p.add_argument("--mock", action="store_true",
                   help="No-GPU mode: return stub responses (for development / testing)")
    p.add_argument("--cli",  action="store_true",
                   help="Headless CLI mode (no TUI, prints to stdout)")

    # ── Execution ─────────────────────────────────────────────────────────────
    p.add_argument("--execute", action="store_true",
                   help="Enable live command execution (model commands run via subprocess). "
                        "Default: OFF (model advises only). "
                        "CAUTION: Only use on authorised targets.")
    p.add_argument("--execution-timeout", type=int, default=300, metavar="SECONDS",
                   help="Base timeout per command in seconds (default: 300). "
                        "Slow tools like nmap/ffuf/sqlmap use longer per-tool overrides.")

    # ── Bug bounty ────────────────────────────────────────────────────────────
    p.add_argument("--bug-bounty", action="store_true",
                   help="Enable bug bounty mode (web app phases, scope enforcement, "
                        "vulnerability report format instead of CTF flags)")
    p.add_argument("--scope", metavar="SCOPE_FILE",
                   help="Path to JSON scope file for bug bounty mode. "
                        "See docs for format. If omitted, target domain is used as scope.")
    p.add_argument("--program", metavar="HANDLE", default="",
                   help="Bug bounty program handle (e.g. coupang_tw). Used to load/save "
                        "the engagement knowledge store across sessions.")

    # ── Metasploit ────────────────────────────────────────────────────────────
    p.add_argument("--msf", action="store_true",
                   help="Enable Metasploit RPC integration (requires msfrpcd running)")
    p.add_argument("--msf-host", default="127.0.0.1")
    p.add_argument("--msf-port", type=int, default=55553)
    p.add_argument("--msf-password", default=None, metavar="PASS",
                   help="Metasploit RPC password (can also be set via MSF_PASSWORD env var)")

    # ── Session ───────────────────────────────────────────────────────────────
    p.add_argument("--session", metavar="ID",
                   help="Resume an existing session by ID")
    p.add_argument("--list-sessions", action="store_true",
                   help="List saved sessions and exit")

    # ── Instruction ───────────────────────────────────────────────────────────
    p.add_argument("--instruction", "-i", default=None,
                   help="Initial task / instruction sent to the model")

    # ── Model overrides ───────────────────────────────────────────────────────
    p.add_argument("--adapter-path", metavar="NAME:PATH", action="append", default=[],
                   help="Override an adapter path: htb:/new/path  (repeatable)")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature",    type=float, default=0.3)

    # ── RAG live intelligence ─────────────────────────────────────────────────
    p.add_argument("--rag", action="store_true", default=None,
                   help="Enable RAG: inject live NVD CVE + Exploit-DB + GHSA context "
                        "(enabled by default; use --no-rag to disable)")
    p.add_argument("--no-rag", action="store_true",
                   help="Disable live RAG intelligence injection")
    p.add_argument("--rag-max-age", type=int, default=24, metavar="HOURS",
                   help="Refresh RAG cache after this many hours (default: 24)")

    # ── Feedback loop ─────────────────────────────────────────────────────────
    p.add_argument("--no-feedback", action="store_true",
                   help="Disable engagement feedback loop (do not queue turns for retraining)")
    p.add_argument("--feedback-threshold", type=int, default=50, metavar="N",
                   help="Retrain threshold: trigger retraining when N new examples queued (default: 50)")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--debug",     action="store_true", help="Enable verbose debug logging")
    p.add_argument("--no-splash", action="store_true", help="Skip splash screen")

    return p


def _build_config(args: argparse.Namespace) -> "PentestGPTConfig":
    """Build PentestGPTConfig + MythosSettings from parsed args."""
    import os as _os
    from pentestgpt.core.config import (
        AdapterConfig, DEFAULT_ADAPTERS, MythosSettings, PentestGPTConfig,
    )

    # Start from defaults
    adapters = dict(DEFAULT_ADAPTERS)

    # Apply --adapter-path overrides
    for override in args.adapter_path:
        if ":" not in override:
            print(f"[warn] --adapter-path format is NAME:PATH, got: {override}", file=sys.stderr)
            continue
        name, path = override.split(":", 1)
        if name in adapters:
            adapters[name] = AdapterConfig(
                path=path, base=adapters[name].base,
                is_qwen3=adapters[name].is_qwen3, max_seq=adapters[name].max_seq,
            )
        else:
            adapters[name] = AdapterConfig(path=path, base="Qwen/Qwen3-14B", is_qwen3=True)

    # Metasploit password: arg > env var
    msf_pass = getattr(args, "msf_password", None) or _os.environ.get("MSF_PASSWORD")

    mythos = MythosSettings(
        adapters=adapters,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        mock=args.mock,
        # Execution
        allow_execution=getattr(args, "execute", False),
        execution_timeout=getattr(args, "execution_timeout", 30),
        # Metasploit
        msf_enabled=getattr(args, "msf", False),
        msf_host=getattr(args, "msf_host", "127.0.0.1"),
        msf_port=getattr(args, "msf_port", 55553),
        msf_password=msf_pass,
        # Bug bounty
        bug_bounty=getattr(args, "bug_bounty", False),
        scope_file=getattr(args, "scope", None),
        program_handle=getattr(args, "program", "") or "",
        # RAG — enabled by default; --no-rag explicitly disables
        rag_enabled=not getattr(args, "no_rag", False),
        rag_max_age_hours=getattr(args, "rag_max_age", 24),
        # Feedback loop
        feedback_enabled=not getattr(args, "no_feedback", False),
        feedback_threshold=getattr(args, "feedback_threshold", 50),
    )

    return PentestGPTConfig(
        target=args.target or "unknown",
        custom_instruction=args.instruction,
        mythos=mythos,
    )


def _build_backend(config: "PentestGPTConfig", mock: bool) -> "MythosBackend":
    from pentestgpt.core.mythos_backend import MythosBackend
    return MythosBackend.from_config(config, mock=mock)


def _list_sessions() -> None:
    from pentestgpt.core.session import SessionStore
    from rich.console import Console
    console = Console()
    store   = SessionStore()
    sessions = store.list_sessions()
    if not sessions:
        console.print("[dim]No sessions found.[/]")
        return
    console.print("[bold]Saved sessions:[/]\n")
    console.print(f"{'ID':<10} {'Date':<18} {'Target':<25} {'Status':<12}")
    console.print("-" * 68)
    for s in sessions:
        date_str   = s.created_at.strftime("%Y-%m-%d %H:%M")
        target_str = (s.target[:23] + "..") if len(s.target) > 25 else s.target
        console.print(f"{s.session_id:<10} {date_str:<18} {target_str:<25} {s.status.value:<12}")


async def _run_tui(args: argparse.Namespace, config: "PentestGPTConfig") -> None:
    backend = _build_backend(config, mock=args.mock)
    from pentestgpt.interface.mythos_tui import run_mythos_tui
    await run_mythos_tui(
        target=config.target,
        backend=backend,
        custom_instruction=args.instruction,
        debug=args.debug,
        resume_session=args.session,
    )


async def _run_cli(args: argparse.Namespace, config: "PentestGPTConfig") -> None:
    """Headless streaming CLI mode — styled with Rich to match TUI information density."""
    from pentestgpt.core.controller import AgentController
    from pentestgpt.core.events import EventBus, EventType
    from pentestgpt.interface.cli_renderer import MythosCLIRenderer

    backend  = _build_backend(config, mock=args.mock)
    events   = EventBus.get()

    renderer = MythosCLIRenderer(
        target      = config.target,
        backend     = backend,
        execute     = getattr(args, "execute",    False),
        bug_bounty  = getattr(args, "bug_bounty", False),
        scope_file  = getattr(args, "scope",      None),
        program     = getattr(args, "program",    "") or "",
        mock        = args.mock,
    )
    renderer.print_header()

    # Wire all event types — same set as the TUI
    events.subscribe(EventType.MESSAGE,       renderer.on_message)
    events.subscribe(EventType.STATE_CHANGED, renderer.on_state_changed)
    events.subscribe(EventType.TOOL,          renderer.on_tool)
    events.subscribe(EventType.FLAG_FOUND,    renderer.on_flag)

    controller = AgentController(config=config, backend=backend, events=events)
    task = args.instruction or f"Begin penetration testing against: {config.target}"

    result = await controller.run(task, resume_session_id=args.session)
    renderer.print_summary(result)

    if not result.get("success"):
        sys.exit(1)


def _check_bootstrap() -> None:
    """Warn (and optionally auto-run bootstrap) if state.json is missing."""
    state_file = Path.home() / ".mythosengine" / "state.json"
    if state_file.exists():
        return

    # Find bootstrap.py relative to this run.py (one level up from mythos_engine/)
    bootstrap = Path(__file__).resolve().parent.parent / "bootstrap.py"
    print("\n[!] ~/.mythosengine/state.json not found — environment not bootstrapped.")
    if bootstrap.exists():
        print(f"    Running bootstrap automatically: python3 {bootstrap}\n")
        import subprocess
        result = subprocess.run([sys.executable, str(bootstrap), "--quiet"], check=False)
        if result.returncode != 0:
            print("[!] Bootstrap reported missing required resources. See above.")
    else:
        print(f"    Run: python3 {bootstrap} (or python3 bootstrap.py from project root)")
    print()


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if args.list_sessions:
        _list_sessions()
        return

    if not args.target and not args.list_sessions:
        parser.error("--target is required unless --list-sessions is used")

    # Ensure machine is bootstrapped before loading the model
    _check_bootstrap()

    config = _build_config(args)

    try:
        if args.cli:
            asyncio.run(_run_cli(args, config))
        else:
            asyncio.run(_run_tui(args, config))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
