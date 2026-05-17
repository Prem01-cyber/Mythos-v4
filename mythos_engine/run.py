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

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument("--target", "-t", required=False, default="",
                   help="Target IP / hostname / URL")

    # ── Mode ──────────────────────────────────────────────────────────────────
    p.add_argument("--mock", action="store_true",
                   help="No-GPU mode: return stub responses (for development / testing)")
    p.add_argument("--cli",  action="store_true",
                   help="Headless CLI mode (no TUI, prints to stdout)")

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

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--debug",    action="store_true", help="Enable verbose debug logging")
    p.add_argument("--no-splash", action="store_true", help="Skip splash screen")

    return p


def _build_config(args: argparse.Namespace) -> "PentestGPTConfig":
    """Build PentestGPTConfig + MythosSettings from parsed args."""
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
            # Unknown adapter — assume Qwen3 base
            adapters[name] = AdapterConfig(path=path, base="Qwen/Qwen3-14B", is_qwen3=True)

    mythos = MythosSettings(
        adapters=adapters,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        mock=args.mock,
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
    """Headless streaming CLI mode."""
    from rich.console import Console
    from pentestgpt.core.controller import AgentController
    from pentestgpt.core.events import EventBus, EventType, Event

    console = Console()
    console.print(f"[bold #f59e0b]⚡ Mythos Engine[/] [dim]—[/] target: [bold]{config.target}[/]")
    if args.mock:
        console.print("[yellow]Mock mode (no GPU)[/]")
    console.print()

    backend = _build_backend(config, mock=args.mock)
    events  = EventBus.get()

    def on_msg(event: Event) -> None:
        text = event.data.get("text", "")
        if text and not event.data.get("routing"):
            console.print(text)

    def on_state(event: Event) -> None:
        state   = event.data.get("state", "")
        details = event.data.get("details", "")
        if details:
            console.print(f"[dim][{state}] {details}[/]")

    events.subscribe(EventType.MESSAGE, on_msg)
    events.subscribe(EventType.STATE_CHANGED, on_state)

    controller = AgentController(config=config, backend=backend, events=events)
    task = args.instruction or f"Begin penetration testing against: {config.target}"

    result = await controller.run(task, resume_session_id=args.session)
    if result.get("success"):
        console.print(f"\n[bold green]✓ Complete[/]  flags={result.get('flags_found', [])}")
    else:
        console.print(f"\n[bold red]✗ Error:[/] {result.get('error')}")
        sys.exit(1)


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
