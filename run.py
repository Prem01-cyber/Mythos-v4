#!/usr/bin/env python3
"""Mythos Engine — direct engagement runner.

Every adapter call, routing decision, command extraction, correction, and
analyst extraction is printed so nothing is hidden.

Usage:
    # Bug bounty with scope (--execute runs commands for real)
    python3 run.py --target supplier.meesho.com --bug-bounty \\
        --scope engagements/meesho/scope.json --program meesho \\
        --execute --execution-timeout 60

    # CTF / pentest box
    python3 run.py --target 10.10.11.5 --execute

    # Dry run — model advises only, no execution
    python3 run.py --target 10.10.11.5

    # Custom first instruction
    python3 run.py --target supplier.meesho.com --bug-bounty \\
        --scope engagements/meesho/scope.json \\
        --instruction "Focus on the /api/supplier/order endpoints for IDOR"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import warnings
from pathlib import Path

# ── Silence noisy transformers warnings before any import touches them ────────
warnings.filterwarnings("ignore", category=FutureWarning,  module="transformers")
warnings.filterwarnings("ignore", category=UserWarning,    module="transformers")
warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")

sys.path.insert(0, str(Path(__file__).parent / "mythos_engine"))

# ── ANSI colours ──────────────────────────────────────────────────────────────
R  = "\033[0m"
B  = "\033[1m"           # bold
DIM = "\033[2m"
RED  = "\033[31m"
GRN  = "\033[32m"
YEL  = "\033[33m"
CYN  = "\033[36m"
WHT  = "\033[37m"

ADAPTER_COLOUR: dict[str, str] = {
    "osint":      "\033[34m",    # blue
    "webapp":     "\033[35m",    # magenta
    "analyst":    "\033[36m",    # cyan
    "researcher": "\033[33m",    # yellow
    "htb":        "\033[32m",    # green
    "attack":     "\033[31m",    # red
    "vulhub":     "\033[91m",    # bright red
    "exploitdb":  "\033[95m",    # bright magenta
    "ad":         "\033[96m",    # bright cyan
    "cloud":      "\033[94m",    # bright blue
    "planner":    "\033[92m",    # bright green
    "executor":   "\033[93m",    # bright yellow
}

_LINE = "─" * 72


def _ac(adapter: str) -> str:
    return ADAPTER_COLOUR.get(adapter, WHT)


def _hdr(label: str, adapter: str = "") -> None:
    col = _ac(adapter) if adapter else CYN
    tag = f"  {B}{col}{label}{R}  "
    pad = (_LINE.__len__() - len(label) - 4) // 2
    print(f"\n{col}{'─'*pad}{tag}{col}{'─'*pad}{R}")


def _ok(msg: str)  -> None: print(f"  {GRN}✓{R} {msg}")
def _warn(msg: str)-> None: print(f"  {YEL}⚠{R} {msg}")
def _err(msg: str) -> None: print(f"  {RED}✗{R} {msg}")
def _info(msg: str)-> None: print(f"  {DIM}{msg}{R}")


# ── Load helpers ──────────────────────────────────────────────────────────────

def _load_model(project_root: str, max_tokens: int, temperature: float):
    """Load MultiAdapterModel — same path as test_adapter_flow."""
    from pentestgpt.core.config import DEFAULT_ADAPTERS, load_state
    from pentestgpt.core.multi_adapter_model import AdapterSpec, MultiAdapterModel

    state   = load_state()
    adapters: dict[str, AdapterSpec] = {}
    loaded_names: list[str] = []

    print(f"\n{B}Loading adapters…{R}")
    for name, cfg in DEFAULT_ADAPTERS.items():
        full = Path(project_root) / cfg.path
        if full.exists():
            adapters[name] = AdapterSpec(
                path=cfg.path, base=cfg.base, is_qwen3=cfg.is_qwen3, max_seq=cfg.max_seq
            )
            loaded_names.append(name)
            _ok(f"{_ac(name)}{B}{name}{R}  {DIM}{cfg.path}{R}")
        else:
            _warn(f"{name}  (not found: {cfg.path})")

    model = MultiAdapterModel(
        adapter_specs=adapters,
        project_root=project_root,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    print(f"\n{GRN}✓ {len(loaded_names)} adapters loaded{R}: {', '.join(loaded_names)}\n")
    return model


def _load_router(project_root: str):
    from pentestgpt.core.adapter_router import AdapterRouter
    router = AdapterRouter()
    clf_dir = str(Path(project_root) / "adapters" / "router_classifier")
    router.load_classifier(clf_dir)
    return router


def _load_scope(scope_file: str | None, target: str):
    if not scope_file:
        return None
    from pentestgpt.core.bug_bounty import BugBountyScope
    try:
        scope = BugBountyScope.from_file(scope_file)
        _ok(f"Scope loaded: {scope_file}  ({len(scope.in_scope)} in-scope entries)")
        return scope
    except Exception as exc:
        _warn(f"Could not load scope ({exc}) — proceeding without scope")
        return None


def _load_knowledge(program: str, target: str):
    from pentestgpt.core.engagement_knowledge import EngagementKnowledge
    k = EngagementKnowledge(program or target.replace(".", "_"))
    _ok(f"Knowledge store: {program or target}")
    return k


# ── Routing decision display ──────────────────────────────────────────────────

def _explain_route(router, message: str, state) -> tuple[str, str]:
    """Returns (adapter_name, human reason string)."""
    reason = router.explain(message, state)
    adapter = router.route(message, state)
    return adapter, reason


# ── Prompt assembly ───────────────────────────────────────────────────────────

def _build_prompt(
    adapter:    str,
    state,
    history:    list[dict],
    user_msg:   str,
    scope,
    knowledge,
    model,
    mode:       str,
) -> list[dict]:
    """Assemble context using ContextBudgetManager — same as mythos_backend."""
    from pentestgpt.core.context_manager import ContextBudgetManager, CONTEXT_LIMIT
    from pentestgpt.prompts.mythos_prompts import get_system_prompt, FORMAT_INSTRUCTION

    effective_limit = min(model.max_context_length(), CONTEXT_LIMIT)
    mgr = ContextBudgetManager(count_fn=model.count_tokens, model_limit=effective_limit)

    sys_prompt  = get_system_prompt(adapter, mode=mode) + FORMAT_INSTRUCTION
    scope_ctx   = scope.to_context_block() if scope else ""
    state_ctx   = state.summary_context() if hasattr(state, "summary_context") else ""
    know_ctx    = knowledge.context_block() if knowledge else ""

    result = mgr.assemble(
        system_base      = sys_prompt,
        scope_context    = scope_ctx,
        current_turn     = user_msg,
        engagement_state = state_ctx,
        history          = history[-8:],   # last 4 turns max passed to assembler
        knowledge        = know_ctx,
    )
    return result.messages


# ── Analyst extraction ────────────────────────────────────────────────────────

def _analyst_extract(model, tool_output: str, target: str, max_tokens: int) -> dict:
    """Call analyst adapter to extract structured findings from tool output."""
    from pentestgpt.prompts.mythos_prompts import SYSTEM_PROMPTS, FORMAT_INSTRUCTION

    prompt = (
        f"[Target: {target}]\n\n"
        f"Tool output:\n{tool_output[:3000]}\n\n"
        "Extract all intelligence as JSON with keys: "
        "subdomains, live_endpoints, open_ports, tech_stack, findings, anomalies, next_targets. "
        "Return only valid JSON."
    )
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPTS.get("analyst", "") + FORMAT_INSTRUCTION},
        {"role": "user",   "content": prompt},
    ]
    try:
        resp = model.generate("analyst", msgs, max_new_tokens=max_tokens)
        clean = resp.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:]).rsplit("```", 1)[0].strip()
        return json.loads(clean)
    except Exception:
        return {}


# ── Main engagement loop ──────────────────────────────────────────────────────

async def run_engagement(args: argparse.Namespace) -> None:
    project_root = str(Path(__file__).parent)

    # ── Banner ────────────────────────────────────────────────────────────────
    mode_label = "BUG BOUNTY" if args.bug_bounty else "PENTEST / CTF"
    print(f"""
{B}{CYN}{'━'*72}
  MYTHOS ENGINE
  target : {args.target}
  mode   : {mode_label}
  exec   : {'ON  ⚠ commands will run' if args.execute else 'OFF (dry run — advise only)'}
  scope  : {args.scope or '(none)'}
{'━'*72}{R}
""")

    # ── Load everything ───────────────────────────────────────────────────────
    model    = _load_model(project_root, args.max_tokens, args.temperature)
    router   = _load_router(project_root)
    scope    = _load_scope(args.scope, args.target)
    knowledge = _load_knowledge(args.program, args.target) if (args.program or args.bug_bounty) else None

    # Register all loaded adapters with the router
    for name in model.loaded_adapters():
        router.register(name)

    # ── Engagement state ──────────────────────────────────────────────────────
    from pentestgpt.core.engagement_state import EngagementState, Phase
    mode_str = "bug_bounty" if args.bug_bounty else "ctf"
    state    = EngagementState(
        target=args.target,
        mode=mode_str,
        phase=Phase.IDLE,
    )

    # ── Tool executor (if --execute) ──────────────────────────────────────────
    shell = None
    validator = None
    if args.execute:
        from pentestgpt.core.tool_executor  import ShellExecutor
        from pentestgpt.core.command_validator import CommandValidator
        workspace = str(Path(project_root) / "workspace")
        Path(workspace).mkdir(exist_ok=True)
        shell     = ShellExecutor(workspace=workspace, base_timeout=args.execution_timeout)
        validator = CommandValidator(workspace=workspace)

    # ── Initial prompt ────────────────────────────────────────────────────────
    history: list[dict] = []
    null_step_count = 0
    step = 0

    if args.instruction:
        current_msg = args.instruction
    elif args.bug_bounty:
        current_msg = (
            f"Begin bug bounty reconnaissance against {args.target}. "
            f"Start with passive subdomain discovery using subfinder, then probe "
            f"live hosts with httpx. Include mandatory scope headers in all requests."
        )
    else:
        current_msg = (
            f"Begin penetration test against {args.target}. "
            f"Start with port scanning using nmap to identify open services."
        )

    _ok(f"Initial instruction: {DIM}{current_msg[:100]}{R}\n")
    print(f"{DIM}Press Ctrl+C to stop the engagement.{R}\n")

    # ── Engagement loop ───────────────────────────────────────────────────────
    try:
        while True:
            step += 1
            findings: dict = {}    # reset each step
            _hdr(f"STEP {step}", "")

            # ── 1. ROUTING ────────────────────────────────────────────────────
            adapter, route_reason = _explain_route(router, current_msg, state)
            col = _ac(adapter)
            print(f"  {B}ROUTER{R}  {col}{B}{adapter.upper()}{R}  {DIM}← {route_reason}{R}")

            # ── 2. CONTEXT ASSEMBLY ───────────────────────────────────────────
            messages = _build_prompt(
                adapter=adapter, state=state, history=history,
                user_msg=current_msg, scope=scope, knowledge=knowledge,
                model=model, mode=mode_str,
            )
            total_ctx_tokens = sum(model.count_tokens(m.get("content","")) for m in messages)
            _info(f"Context: {len(messages)} messages, {total_ctx_tokens} tokens")
            _info(f"User turn preview: {current_msg[:120].replace(chr(10),' ')}…")

            # ── 3. GENERATION ─────────────────────────────────────────────────
            print(f"\n  {col}{B}Generating [{adapter}]…{R}")
            t0 = time.time()
            try:
                response = model.generate(adapter, messages, max_new_tokens=args.max_tokens)
            except Exception as exc:
                _err(f"Generation failed: {exc}")
                break
            elapsed = time.time() - t0
            out_tokens = model.count_tokens(response)
            print(f"  {DIM}({elapsed:.1f}s, {out_tokens} tokens){R}")

            # Print response (truncated if long)
            resp_preview = response[:600] + (f"\n  {DIM}[…{len(response)-600} chars truncated]{R}" if len(response) > 600 else "")
            print(f"\n{resp_preview}\n")

            # ── 4. COMMAND EXTRACTION ─────────────────────────────────────────
            from pentestgpt.core.tool_executor import ShellExecutor as _SE
            _tmp_shell = _SE(workspace=str(Path(project_root) / "workspace"))
            raw_cmds = _tmp_shell.extract_commands(response)

            if not raw_cmds:
                _warn("No <command> tag found in response — null step")
                null_step_count += 1
                # After 1 null step — ask planner for redirection
                if null_step_count == 1 and "planner" in model.loaded_adapters():
                    _hdr("PLANNER — redirecting after null step", "planner")
                    planner_msgs = _build_prompt(
                        adapter="planner", state=state, history=history,
                        user_msg=(
                            f"Standard approach found nothing new.\n"
                            f"Target: {args.target}\n"
                            f"Last adapter tried: {adapter}\n"
                            f"Analyse what has been tried and propose the single "
                            f"highest-value next action. Be specific: tool, target, flags, goal."
                        ),
                        scope=scope, knowledge=knowledge, model=model, mode=mode_str,
                    )
                    try:
                        plan_response = model.generate("planner", planner_msgs, max_new_tokens=args.max_tokens)
                        print(f"\n{plan_response[:400]}\n")
                        current_msg  = plan_response
                        null_step_count = 0
                    except Exception as exc:
                        _warn(f"Planner failed: {exc}")
                        current_msg = current_msg   # retry same
                elif null_step_count >= 3:
                    _warn("3 consecutive null steps — pausing. Check model output above.")
                    null_step_count = 0
                else:
                    # Model gave text but no command — feed its analysis back as context
                    current_msg = response[:600]
                history.append({"role": "user",      "content": current_msg})
                history.append({"role": "assistant", "content": response})
                continue

            null_step_count = 0
            cmd = raw_cmds[0]   # execute the first extracted command

            # ── 5. VALIDATION + CORRECTION ────────────────────────────────────
            final_cmd = cmd
            if validator:
                try:
                    vresult = validator.validate(cmd)
                    if not vresult.valid:
                        _warn(f"Validator flagged: {vresult.reason}")
                        # Correct via executor adapter
                        if "executor" in model.loaded_adapters():
                            _hdr("EXECUTOR — correcting command", "executor")
                            corr_prompt = (
                                f"The following command failed validation:\n"
                                f"  {cmd}\n\n"
                                f"Error: {vresult.reason}\n\n"
                                f"Generate the corrected command only, inside <command> tags."
                            )
                            exec_msgs = [
                                {"role": "system", "content":
                                 __import__("pentestgpt.prompts.mythos_prompts",
                                            fromlist=["SYSTEM_PROMPTS"]).SYSTEM_PROMPTS.get("executor","")},
                                {"role": "user", "content": corr_prompt},
                            ]
                            try:
                                corr_resp = model.generate("executor", exec_msgs, max_new_tokens=256)
                                corr_cmds = _tmp_shell.extract_commands(corr_resp)
                                if corr_cmds:
                                    final_cmd = corr_cmds[0]
                                    _ok(f"Executor corrected → {final_cmd}")
                                else:
                                    _warn("Executor returned no command — using original")
                            except Exception as exc:
                                _warn(f"Executor failed: {exc}")
                    else:
                        _ok(f"Command valid")
                except Exception as exc:
                    _warn(f"Validator error: {exc}")

            print(f"\n  {B}COMMAND{R}  {YEL}{final_cmd}{R}\n")

            # ── 6. EXECUTION ──────────────────────────────────────────────────
            tool_output = ""
            if shell:
                _hdr(f"EXECUTING", "")
                try:
                    loop = asyncio.get_event_loop()
                    exec_result = await loop.run_in_executor(
                        None, lambda: shell.run(final_cmd)
                    )
                    tool_output = exec_result.stdout or exec_result.stderr or ""
                    exit_ok = exec_result.returncode == 0
                    sym = _ok if exit_ok else _warn
                    sym(f"exit={exec_result.returncode}  ({len(tool_output)} chars output)")
                    if tool_output:
                        # Print first 40 lines
                        lines = tool_output.splitlines()
                        print("\n".join(f"    {l}" for l in lines[:40]))
                        if len(lines) > 40:
                            _info(f"[…{len(lines)-40} more lines]")
                except Exception as exc:
                    _err(f"Execution error: {exc}")
                    tool_output = f"ERROR: {exc}"
            else:
                _info("(--execute not set — skipping actual execution)")
                tool_output = f"[DRY RUN] Would execute: {final_cmd}"

            # ── 7. ANALYST EXTRACTION ─────────────────────────────────────────
            if "analyst" in model.loaded_adapters() and tool_output and len(tool_output) > 50:
                _hdr("ANALYST — extracting structured findings", "analyst")
                t0 = time.time()
                findings = _analyst_extract(model, tool_output, args.target, min(args.max_tokens, 350))
                elapsed = time.time() - t0
                if findings:
                    _ok(f"JSON extracted ({elapsed:.1f}s)  keys={list(findings.keys())}")
                    # Ingest into knowledge store
                    if knowledge:
                        try:
                            knowledge.ingest_structured(findings)
                            _info("Knowledge store updated")
                        except Exception:
                            pass
                    # Print summary
                    for key, val in findings.items():
                        if isinstance(val, list) and val:
                            print(f"    {DIM}{key}: {len(val)} items{R}")
                        elif val:
                            print(f"    {DIM}{key}: {str(val)[:80]}{R}")
                else:
                    _warn("Analyst returned no structured JSON — raw output kept")

            # ── 8. KNOWLEDGE UPDATE + NEXT PROMPT ────────────────────────────
            # If analyst extraction failed or wasn't run, fall back to raw ingestion
            if knowledge and not findings:
                try:
                    knowledge.ingest_tool_output(tool_output)
                except Exception:
                    pass

            # Add turn to history
            history.append({"role": "user",      "content": current_msg})
            history.append({"role": "assistant", "content": response})
            if tool_output:
                history.append({"role": "user",
                                 "content": f"[Tool output]\n{tool_output[:800]}"})

            # Build next prompt from tool output
            know_summary = knowledge.context_block()[:300] if knowledge else ""
            current_msg = (
                f"Previous command: {final_cmd}\n\n"
                f"Tool output (first 800 chars):\n{tool_output[:800]}\n\n"
                + (f"Current knowledge:\n{know_summary}\n\n" if know_summary else "")
                + f"Continue the engagement against {args.target}. "
                f"Propose and execute the next highest-value action."
            )

            # Advance phase based on findings
            _maybe_advance_phase(state, tool_output, mode_str)
            _info(f"Phase: {state.phase.value}")

    except KeyboardInterrupt:
        print(f"\n\n{YEL}Engagement stopped by user.{R}")

    # ── Final summary ─────────────────────────────────────────────────────────
    _hdr("ENGAGEMENT SUMMARY", "")
    _info(f"Steps completed : {step}")
    _info(f"Phase reached   : {state.phase.value}")
    if knowledge:
        ctx = knowledge.context_block()
        if ctx:
            print(f"\n{DIM}Knowledge collected:{R}")
            print(ctx[:800])


def _maybe_advance_phase(state, tool_output: str, mode: str) -> None:
    """Simple heuristic phase advancement — mirrors EngagementState.update()."""
    from pentestgpt.core.engagement_state import Phase
    out = tool_output.lower()
    if mode == "bug_bounty":
        if state.phase == Phase.IDLE and any(x in out for x in ["subdomain", "meesho.com"]):
            state.phase = Phase.RECON
        elif state.phase == Phase.RECON and any(x in out for x in ["200", "301", "nginx", "apache"]):
            state.phase = Phase.SUBDOMAIN_ENUM
        elif state.phase == Phase.SUBDOMAIN_ENUM and any(x in out for x in ["200", "react", "angular"]):
            state.phase = Phase.FINGERPRINT
    else:
        if state.phase == Phase.IDLE and "open" in out and "port" in out:
            state.phase = Phase.RECON
        elif state.phase == Phase.RECON and any(x in out for x in ["service", "version", "ssh", "http"]):
            state.phase = Phase.SERVICE_ENUM


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mythos",
        description="Mythos Engine — direct engagement runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--target", "-t", required=True,
                   help="Target IP / hostname / URL")
    p.add_argument("--bug-bounty", action="store_true",
                   help="Bug bounty mode (web phases, scope enforcement)")
    p.add_argument("--scope", metavar="FILE",
                   help="JSON scope file for bug bounty mode")
    p.add_argument("--program", metavar="HANDLE", default="",
                   help="Bug bounty program handle (for knowledge store)")
    p.add_argument("--execute", action="store_true",
                   help="Actually run commands (default: dry run, advise only)")
    p.add_argument("--execution-timeout", type=int, default=60, metavar="SECS",
                   help="Base command timeout in seconds (default: 60)")
    p.add_argument("--instruction", "-i", default=None,
                   help="Custom first instruction (overrides default recon prompt)")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Max new tokens per generation (default: 512)")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Sampling temperature (default: 0.3)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(run_engagement(args))


if __name__ == "__main__":
    main()
