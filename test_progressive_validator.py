#!/usr/bin/env python3
"""Test progressive validation layers with controlled scenarios.

Run this before testing on live engagements to verify:
- Layer 1 catches invalid flags
- Layer 2 catches missing args and bad value types
- Layer 3 catches runtime errors with correct correction prompts

Usage:
    python3 test_progressive_validator.py
"""

import sys
from pathlib import Path

# Add mythos_engine to path
sys.path.insert(0, str(Path(__file__).parent / "mythos_engine"))

from pentestgpt.core.command_validator import (
    CommandValidator,
    parse_execution_errors,
    build_progressive_prompt,
    HelpParser,
)


# ── Mock ExecutionResult for layer 3 testing ─────────────────────────────────

class MockExecutionResult:
    """Simulates an ExecutionResult from tool_executor.py."""
    def __init__(self, command: str, exit_code: int, stderr: str, stdout: str = ""):
        self.command = command
        self.exit_code = exit_code
        self.stderr = stderr
        self.stdout = stdout
        self.combined_output = stdout + stderr


# ── Test scenarios ────────────────────────────────────────────────────────────

def test_layer1_valid_flags():
    """Layer 1: Valid command should pass."""
    print("\n" + "="*70)
    print("TEST 1: Layer 1 - Valid flags (curl with correct syntax)")
    print("="*70)
    
    validator = CommandValidator()
    commands = ["curl -o output.html -L https://example.com"]
    
    result = validator.validate_flags_sync(commands)
    
    print(f"Command: {commands[0]}")
    print(f"Needs correction: {result.needs_correction}")
    print(f"Expected: False")
    
    assert not result.needs_correction, "Valid command should pass layer 1"
    print("✓ PASS")


def test_layer1_invalid_flag():
    """Layer 1: Invalid flag should fail."""
    print("\n" + "="*70)
    print("TEST 2: Layer 1 - Invalid flag detection")
    print("="*70)
    
    validator = CommandValidator()
    commands = ["curl --totally-fake-flag https://example.com"]
    
    result = validator.validate_flags_sync(commands)
    
    print(f"Command: {commands[0]}")
    print(f"Needs correction: {result.needs_correction}")
    print(f"Correction level: {result.correction_level}")
    print(f"Errors: {[(e.flag, e.error_text[:50]) for e in result.errors]}")
    print(f"\nCorrection prompt (first 300 chars):\n{result.correction_prompt[:300]}")
    
    assert result.needs_correction, "Invalid flag should trigger correction"
    assert result.correction_level == 1, "Should be layer 1 error"
    print("✓ PASS")


def test_layer2_missing_required_arg():
    """Layer 2: Missing required argument should fail."""
    print("\n" + "="*70)
    print("TEST 3: Layer 2 - Missing required argument")
    print("="*70)
    
    validator = CommandValidator()
    commands = ["curl"]  # Missing required <url>
    
    result = validator.validate_semantic_sync(commands)
    
    print(f"Command: {commands[0]}")
    print(f"Needs correction: {result.needs_correction}")
    print(f"Correction level: {result.correction_level}")
    
    if result.needs_correction:
        for err in result.errors:
            print(f"Error kind: {err.kind}")
            print(f"Detail: {err.detail[:100]}")
        print(f"\nCorrection prompt (first 400 chars):\n{result.correction_prompt[:400]}")
    
    assert result.needs_correction, "Missing required arg should trigger correction"
    assert result.correction_level == 2, "Should be layer 2 error"
    print("✓ PASS")


def test_layer2_valid_with_required_arg():
    """Layer 2: Command with required arg should pass."""
    print("\n" + "="*70)
    print("TEST 4: Layer 2 - Valid command with required arg")
    print("="*70)
    
    validator = CommandValidator()
    commands = ["curl https://example.com"]
    
    result = validator.validate_semantic_sync(commands)
    
    print(f"Command: {commands[0]}")
    print(f"Needs correction: {result.needs_correction}")
    
    assert not result.needs_correction, "Valid command should pass layer 2"
    print("✓ PASS")


def test_layer3_runtime_bad_flag():
    """Layer 3: Runtime error from wrong binary version."""
    print("\n" + "="*70)
    print("TEST 5: Layer 3 - Runtime error (wrong httpx binary)")
    print("="*70)
    
    # Simulate the exact httpx error from screenshots
    mock_results = [
        MockExecutionResult(
            command="httpx -l output.txt -sc -title",
            exit_code=1,
            stderr="Error: No such option '-l'. (Did you mean '--help'?)",
        ),
    ]
    
    runtime_errors = parse_execution_errors(mock_results)
    
    print(f"Command: {mock_results[0].command}")
    print(f"Exit code: {mock_results[0].exit_code}")
    print(f"Stderr: {mock_results[0].stderr}")
    print(f"\nParsed errors: {len(runtime_errors)}")
    
    for err in runtime_errors:
        print(f"  Error type: {err.error_type}")
        print(f"  Detail: {err.detail}")
        print(f"  Correctable: {err.correctable}")
        print(f"  Layer: {err.layer}")
    
    # Build correction prompt for attempt 3 (runtime)
    if runtime_errors:
        prompt = build_progressive_prompt(
            runtime_errors,
            attempt=3,
            tool_doc_ctx="[TOOL DOCUMENTATION for httpx would be here]",
            max_attempts=3,
        )
        print(f"\nCorrection prompt for attempt 3:\n{prompt[:600]}...\n")
    
    assert len(runtime_errors) == 1, "Should detect 1 runtime error"
    assert runtime_errors[0].correctable, "Bad flag errors are correctable"
    assert runtime_errors[0].error_type == "bad_flag", "Should identify as bad_flag"
    print("✓ PASS")


def test_layer3_runtime_missing_file():
    """Layer 3: Runtime error from missing file."""
    print("\n" + "="*70)
    print("TEST 6: Layer 3 - Runtime error (missing file)")
    print("="*70)
    
    mock_results = [
        MockExecutionResult(
            command="ffuf -w /nonexistent/wordlist.txt -u http://target.com/FUZZ",
            exit_code=1,
            stderr="open /nonexistent/wordlist.txt: no such file or directory",
        ),
    ]
    
    runtime_errors = parse_execution_errors(mock_results)
    
    print(f"Command: {mock_results[0].command}")
    print(f"Parsed errors: {len(runtime_errors)}")
    
    for err in runtime_errors:
        print(f"  Error type: {err.error_type}")
        print(f"  Detail: {err.detail}")
        print(f"  Correctable: {err.correctable}")
    
    assert len(runtime_errors) == 1
    assert runtime_errors[0].error_type == "missing_file"
    assert runtime_errors[0].correctable
    print("✓ PASS")


def test_layer3_runtime_success():
    """Layer 3: Successful execution should not trigger errors."""
    print("\n" + "="*70)
    print("TEST 7: Layer 3 - Successful execution")
    print("="*70)
    
    mock_results = [
        MockExecutionResult(
            command="curl https://example.com",
            exit_code=0,
            stderr="",
            stdout="<html>...</html>",
        ),
    ]
    
    runtime_errors = parse_execution_errors(mock_results)
    
    print(f"Command: {mock_results[0].command}")
    print(f"Exit code: {mock_results[0].exit_code}")
    print(f"Parsed errors: {len(runtime_errors)}")
    
    assert len(runtime_errors) == 0, "Successful commands should have no errors"
    print("✓ PASS")


def test_progressive_prompts_escalation():
    """Test that correction prompts escalate in detail."""
    print("\n" + "="*70)
    print("TEST 8: Progressive prompt escalation (attempt 1 → 2 → 3)")
    print("="*70)
    
    validator = CommandValidator()
    commands = ["curl --fake-flag https://example.com"]
    result = validator.validate_flags_sync(commands)
    
    if not result.errors:
        print("No errors to test — skipping")
        return
    
    p1 = build_progressive_prompt(result.errors, attempt=1, max_attempts=3)
    p2 = build_progressive_prompt(result.errors, attempt=2, tool_doc_ctx="FULL DOCS HERE", max_attempts=3)
    p3 = build_progressive_prompt(result.errors, attempt=3, max_attempts=3)
    
    print("Attempt 1 prompt length:", len(p1))
    print("Attempt 2 prompt length:", len(p2))
    print("Attempt 3 prompt length:", len(p3))
    
    print(f"\nAttempt 1 (first 200 chars):\n{p1[:200]}...\n")
    print(f"\nAttempt 2 (first 200 chars):\n{p2[:200]}...\n")
    print(f"\nAttempt 3 (first 200 chars):\n{p3[:200]}...\n")
    
    assert "attempt 1/3" in p1.lower(), "Attempt 1 should mention 1/3"
    assert "attempt 2/3" in p2.lower(), "Attempt 2 should mention 2/3"
    assert "attempt 3/3" in p3.lower() or "final" in p3.lower(), "Attempt 3 should mention final"
    assert len(p2) > len(p1), "Attempt 2 should be more detailed"
    assert "FULL DOCS HERE" in p2, "Attempt 2 should include tool docs"
    
    print("✓ PASS - Prompts escalate correctly")


def test_help_parser():
    """Test HelpParser extraction of ToolSignature."""
    print("\n" + "="*70)
    print("TEST 9: HelpParser - Extract tool signature from --help")
    print("="*70)
    
    sig = HelpParser.parse("curl")
    
    if sig:
        print(f"Binary: {sig.binary}")
        print(f"Usage line: {sig.usage_line[:80]}")
        print(f"Required args: {sig.required_args}")
        print(f"Optional args: {sig.optional_args}")
        print(f"Flag types (sample): {dict(list(sig.flag_types.items())[:8])}")
        print(f"Mutually exclusive groups: {sig.mutually_exclusive}")
        
        assert "url" in sig.required_args or "URL" in sig.required_args, "curl should have url as required arg"
        assert "-o" in sig.flag_types or "--output" in sig.flag_types, "curl should have output flag typed"
        print("✓ PASS")
    else:
        print("curl not installed — skipping")


# ── Run all tests ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "#"*70)
    print("# PROGRESSIVE VALIDATOR TEST SUITE")
    print("#"*70)
    
    tests = [
        test_layer1_valid_flags,
        test_layer1_invalid_flag,
        test_layer2_missing_required_arg,
        test_layer2_valid_with_required_arg,
        test_layer3_runtime_bad_flag,
        test_layer3_runtime_missing_file,
        test_layer3_runtime_success,
        test_progressive_prompts_escalation,
        test_help_parser,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "#"*70)
    print(f"# RESULTS: {passed} passed, {failed} failed")
    print("#"*70)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed — progressive validator is ready for live testing")
        print("\nNext steps:")
        print("1. Push updated code to cloud machine")
        print("2. Clear tool_docs cache: rm -rf ~/.mythosengine/tool_docs/*.json")
        print("3. Run: cd mythos_engine && python3 run.py --target www.tw.coupang.com \\")
        print("         --bug-bounty --scope ../engagements/coupang-tw/scope.json \\")
        print("         --program coupang_tw --execute --execution-timeout 60")
        print("4. Watch for validation messages in TUI:")
        print("   - 'validating (layer 1 issue — correction 1/3)'")
        print("   - 'validating (layer 2 issue — correction 2/3)'")
        print("   - 'validating (runtime error — correction 3/3)'")


if __name__ == "__main__":
    main()
