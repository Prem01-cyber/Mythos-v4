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

# ── Tool-runtime imports (lazy — only materialise when needed) ────────────────
# These are resolved at runtime after sys.path is set (mythos_engine added below)

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
    tool_docs_ctx: str = "",
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
        history          = history[-8:],
        knowledge        = know_ctx,
        tool_docs        = tool_docs_ctx,
    )
    return result.messages


# ── Python fallback synthesis ─────────────────────────────────────────────────

async def _synthesise_fallback(
    model, binary: str, failed_cmd: str, error: str, target: str, shell
) -> str:
    """When a CLI tool keeps failing, tool_operator writes Python to do the same job."""
    from pentestgpt.prompts.mythos_prompts import SYSTEM_PROMPTS
    import shutil as _shutil

    synth_adapter = (
        "tool_operator" if "tool_operator" in model.loaded_adapters()
        else "executor"
    )
    _hdr(f"TOOL OPERATOR — synthesising Python fallback [{synth_adapter}]", synth_adapter)
    _info(f"'{binary}' failed twice — writing Python 3 replacement")

    installed = [t for t in [
        "requests","urllib","socket","json","re","concurrent.futures","subprocess"
    ]]  # always available in Python stdlib + venv

    synth_prompt = (
        f"Tool: {binary}\n"
        f"Reason unavailable: '{binary}' failed twice with this error:\n{error[:400]}\n\n"
        f"Goal: achieve what `{failed_cmd}` was trying to do, for target: {target}\n\n"
        f"Write a complete, self-contained Python 3 script that:\n"
        f"  - Accomplishes the same goal without the broken CLI tool\n"
        f"  - Uses only: {', '.join(installed)}\n"
        f"  - Prints all results to stdout\n"
        f"  - Handles errors gracefully (timeouts, connection errors)\n\n"
        f"Put the code inside a ```python ... ``` block."
    )
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPTS.get(synth_adapter, "")},
        {"role": "user",   "content": synth_prompt},
    ]
    try:
        resp = model.generate("executor", msgs, max_new_tokens=600)
        import re as _re, tempfile, os as _os, sys as _sys
        m = _re.search(r"```python\s*\n(.*?)```", resp, _re.DOTALL)
        if not m:
            return f"[Fallback synthesis failed — executor returned no Python block]\n{resp[:200]}"

        code = m.group(1).strip()
        workspace = str(Path(_sys.argv[0]).parent / "workspace")
        Path(workspace).mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=workspace,
            delete=False, prefix=f"mythos_{binary}_"
        ) as f:
            f.write(code)
            script_path = f.name

        _ok(f"Synthesised script: {script_path}")
        print(f"\n{DIM}--- script preview ---{R}")
        print("\n".join(f"    {l}" for l in code.splitlines()[:20]))
        print(f"{DIM}--- end preview ---{R}\n")

        exec_result = await shell.run(f"python3 {script_path}")
        out = exec_result.stdout or exec_result.stderr or ""
        if exec_result.exit_code == 0:
            _ok(f"Python fallback succeeded ({len(out)} chars)")
        else:
            _warn(f"Python fallback also failed (exit={exec_result.exit_code})")
        if out:
            print("\n".join(f"    {l}" for l in out.splitlines()[:30]))
        return out or f"[Python fallback for {binary} produced no output]"

    except Exception as exc:
        return f"[Fallback synthesis error: {exc}]"


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

    # ── Live tool introspection helper ────────────────────────────────────────
    async def _live_help(binary: str) -> str:
        """Run `binary --help` in the shell and return the output.

        This is what a human would do when a tool fails — run the help directly
        and read the actual installed version's flags. No caching, no guessing.
        """
        if not binary:
            return ""
        import shutil
        if not shutil.which(binary):
            return f"[{binary} is not installed on this system — cannot run {binary} --help]"
        try:
            proc = await asyncio.create_subprocess_shell(
                f"{binary} --help",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            output = (stdout or stderr or b"").decode("utf-8", errors="replace").strip()
            return output[:3000]   # cap at 3000 chars — enough for any help page
        except Exception as exc:
            return f"[Could not run {binary} --help: {exc}]"

    # Register all loaded adapters with the router
    for name in model.loaded_adapters():
        router.register(name)

    # ── Engagement state ──────────────────────────────────────────────────────
    from pentestgpt.core.engagement_state import EngagementState, Phase
    mode_str = "bug_bounty" if args.bug_bounty else "ctf"
    state    = EngagementState(mode=mode_str)
    state.set_target(args.target)

    # ── Tool executor (if --execute) ──────────────────────────────────────────
    shell = None
    validator = None
    workspace = str(Path(project_root) / "workspace")
    Path(workspace).mkdir(exist_ok=True)
    if args.execute:
        from pentestgpt.core.tool_executor    import ShellExecutor
        from pentestgpt.core.command_validator import CommandValidator
        shell     = ShellExecutor(workspace=workspace, timeout=args.execution_timeout)
        validator = CommandValidator()

    # ── Tool Runtime Layer initialisation ─────────────────────────────────────
    from pentestgpt.tools.env_manifest      import get_manifest
    from pentestgpt.tools.dispatcher        import ToolDispatcher, DispatchContext
    from pentestgpt.core.execution_policy   import ExecutionPolicy
    from pentestgpt.core.observation        import ObservationSummarizer

    _hdr("ENVIRONMENT MANIFEST", "")
    _manifest = get_manifest(rebuild=True)
    _info(_manifest.context_block())

    _dispatcher  = ToolDispatcher()
    _policy      = ExecutionPolicy()
    _obs         = ObservationSummarizer(workspace=workspace)

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

            # ── 2. ENV MANIFEST injection (tool collision awareness) ──────────
            # Compact manifest block (~150 tokens) tells the model which binaries
            # are installed and whether httpx is the Python client or PD binary.
            tool_docs_ctx = _manifest.context_block()

            # ── 3. CONTEXT ASSEMBLY ───────────────────────────────────────────
            messages = _build_prompt(
                adapter=adapter, state=state, history=history,
                user_msg=current_msg, scope=scope, knowledge=knowledge,
                model=model, mode=mode_str, tool_docs_ctx=tool_docs_ctx,
            )
            total_ctx_tokens = sum(model.count_tokens(m.get("content","")) for m in messages)
            _info(f"Context: {len(messages)} messages, {total_ctx_tokens} tokens")
            _info(f"User turn preview: {current_msg[:120].replace(chr(10),' ')}…")

            # ── 4. GENERATION ─────────────────────────────────────────────────
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

            # ── 5. COMMAND EXTRACTION (via ToolDispatcher) ────────────────────
            _disp_ctx = DispatchContext(
                model_response=response,
                target=args.target,
                workspace=workspace,
                model=model if args.execute else None,
            )

            # Extract commands to check for null step
            _preview_cmds = _dispatcher._extract_commands(response)
            _preview_py   = _dispatcher._extract_python(response)

            if not _preview_cmds and not _preview_py:
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
                        current_msg   = plan_response
                        null_step_count = 0
                    except Exception as exc:
                        _warn(f"Planner failed: {exc}")
                elif null_step_count >= 3:
                    _warn("3 consecutive null steps — pausing. Check model output above.")
                    null_step_count = 0
                else:
                    current_msg = response[:600]
                history.append({"role": "user",      "content": current_msg})
                history.append({"role": "assistant", "content": response})
                continue

            null_step_count = 0
            print(f"\n  {B}COMMANDS{R}  {YEL}{', '.join(_preview_cmds[:3])}{R}\n")

            # ── 6. DISPATCH + EXECUTE via Tool Runtime ────────────────────────
            if args.execute:
                _hdr("EXECUTING via ToolDispatcher", "")
                tool_results = await _dispatcher.execute_all(_disp_ctx)
            else:
                _info("(--execute not set — skipping actual execution)")
                from pentestgpt.tools.base import ToolResult as _TR
                tool_results = [
                    _TR(
                        success=True, exit_code=0,
                        stdout=f"[DRY RUN] Would execute: {c}",
                        stderr="", command_run=c, tool_name="shell_raw",
                    )
                    for c in (_preview_cmds or [f"# {response[:80]}"])
                ]

            # ── 7. POLICY + OBSERVATION ──────────────────────────────────────
            _next_turns = _obs.summarize_batch(tool_results, state, _policy)
            _combined_prompt = _obs.combined_prompt(_next_turns)

            for res, nt in zip(tool_results, _next_turns):
                sym = _ok if res.success else _warn
                sym(
                    f"[{res.tool_name}] exit={res.exit_code} "
                    f"elapsed={res.elapsed_s:.1f}s "
                    f"error_kind={res.error_kind or 'none'}"
                )
                if res.stdout.strip():
                    lines = res.stdout.splitlines()
                    print("\n".join(f"    {l}" for l in lines[:30]))
                    if len(lines) > 30:
                        _info(f"[…{len(lines)-30} more lines — full output: {nt.step_file}]")
                elif res.stderr.strip():
                    print(f"    {DIM}{res.stderr[:300]}{R}")

                if nt.circuit_break:
                    _warn(f"CIRCUIT BREAKER: {res.tool_name} — requesting pivot")

            # ── 8. ANALYST EXTRACTION (gated by policy) ──────────────────────
            _analyst_ran = False
            for res, nt in zip(tool_results, _next_turns):
                if nt.allow_analyst and "analyst" in model.loaded_adapters():
                    _hdr("ANALYST — extracting structured findings", "analyst")
                    t0 = time.time()
                    findings = _analyst_extract(
                        model, res.combined_output, args.target, min(args.max_tokens, 350)
                    )
                    elapsed = time.time() - t0
                    if findings:
                        _ok(f"JSON extracted ({elapsed:.1f}s)  keys={list(findings.keys())}")
                        if knowledge:
                            try:
                                knowledge.ingest_structured(findings)
                                _info("Knowledge store updated")
                            except Exception:
                                pass
                        for key, val in findings.items():
                            if isinstance(val, list) and val:
                                print(f"    {DIM}{key}: {len(val)} items{R}")
                            elif val:
                                print(f"    {DIM}{key}: {str(val)[:80]}{R}")
                        _analyst_ran = True
                    else:
                        _warn("Analyst returned no structured JSON — raw output kept")
                    break   # only call analyst once per step

            # Fall back to raw ingestion if analyst didn't run
            if not _analyst_ran and knowledge:
                for res in tool_results:
                    raw = res.combined_output
                    if raw and raw != "(no output)":
                        try:
                            knowledge.ingest_tool_output(raw)
                        except Exception:
                            pass
                        break

            # Add turn to history (summary only — not raw blob)
            history.append({"role": "user",      "content": current_msg})
            history.append({"role": "assistant", "content": response})
            if _combined_prompt:
                history.append({"role": "user",  "content": _combined_prompt})

            # Build next prompt using ObservationSummarizer output
            know_summary = knowledge.context_block()[:300] if knowledge else ""
            current_msg = (
                _combined_prompt
                + (f"\n\nCurrent knowledge:\n{know_summary}" if know_summary else "")
                + f"\n\nContinue the engagement against {args.target}. "
                f"Propose and execute the next highest-value action."
            )

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
