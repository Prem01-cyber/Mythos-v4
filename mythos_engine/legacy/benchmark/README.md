# Penetration Testing Benchmark Framework

Encode penetration testing targets and automatically evaluate performance against logs.

## How It Works

1. **Encode targets** with attack steps and success indicators
2. **Run penetration tests** and save logs
3. **Evaluate logs** against encoded targets to get scores

## Quick Usage

### 1. View a Target

```python
from benchmark.pentestTarget import create_thinkphp_target

target = create_thinkphp_target()
target.print_attack_chain()
```

### 2. Evaluate Logs

```bash
# Set up LLM provider
export LLM_PROVIDER=claude  # or ollama
export ANTHROPIC_API_KEY=your_key

# Run evaluation
python benchmark/evaluator.py
```

## Target Structure

Targets contain:
- **Attack Steps**: reconnaissance → enumeration → exploitation → post_exploitation
- **Vulnerabilities**: Each step has vulnerabilities to find/exploit
- **Success Indicators**: Patterns that indicate successful exploitation

## Pattern Types

- **Regex**: `r"uid=\d+.*gid=\d+"` - Match command output
- **Exact**: `"shell.php"` - Look for specific strings  
- **LLM**: `"Command execution successful"` - Natural language analysis

## Sample Output

### Target Definition
```
Target: ThinkPHP v5.0.20 Vulnerable Application
IP: localhost:1234

Attack Flow:
┌─ START
└─ RECONNAISSANCE: Network Reconnaissance
   └─ ENUMERATION: Service Enumeration and Fingerprinting
      └─ EXPLOITATION: Initial Exploitation
         └─ POST_EXPLOITATION: Post-Exploitation Access

Summary:
  Total Steps: 4
  Total Vulnerabilities: 5
  Critical Vulnerabilities: 2
```

### Evaluation Report
```
============================================================
Evaluation Report: ThinkPHP v5.0.20 Vulnerable Application
============================================================
Overall Success Rate: 60.0%

Vulnerabilities Exploited:
  ✓ info_disclosure at localhost:1234
  ✓ rce at /index.php?s=index/\think\app/invokefunction

Attack Chain Status:
  ✓ reconnaissance
  ✓ enumeration  
  ✓ exploitation
  ✗ post_exploitation
```

## Configuration

```bash
# LLM Provider
export LLM_PROVIDER=claude        # or ollama
export ANTHROPIC_API_KEY=your_key # for Claude
export OLLAMA_MODEL=llama3.1      # for Ollama
```

## Files

- `pentestTarget.py` - Define and visualize targets
- `evaluator.py` - Evaluate logs against targets
- `logs/` - Sample penetration testing logs