# Mythos Engine v4

An autonomous penetration testing and bug bounty AI built on Qwen3-14B with a Mixture-of-LoRA-Adapters (MoLoRA) architecture. Each LoRA adapter is a domain specialist; the engine hot-swaps between them based on context, routing all command correction through a dedicated executor adapter.

---

## Architecture Overview

```
User Input
    │
    ▼
AdapterRouter  ──────────────────────────────────────────────────────────────
│  1. Forced override (/adapter cmd)                                         │
│  2. Learned MLP classifier (confidence ≥ 0.70)                            │
│  3. Keyword regex (CVE-, bloodhound, sqli, aws, HYPOTHESIS: …)            │
│  4. Phase-based routing (CTF phases or BB phases)                          │
│  5. Last-resort fallback                                                   │
└────────────────────────────────────────────────────────────────────────────
    │
    ▼
ContextBudgetManager.assemble()          ← token-measured, priority-ordered
│  P0  system_base   500t   adapter persona + format instruction
│  P0  scope         450t   WAF header, test accounts, in/out scope
│  P0  current_turn  400t   user message
│  P1  engagement    200t   phase, discovered ports/tech/creds
│  P2  history      1000t   recent turns (fitted, newest-first)
│  P3  tool_docs     300t   live --help docs for mentioned tools
│  P4  knowledge     250t   persistent findings, anomalies, endpoints
│  P5  rag           200t   live CVE/exploit snippets
└──────────────────────────────────────────────────────────────────────────
    │
    ▼
MultiAdapterModel.generate(adapter, messages)
│  Qwen3-14B base + hot-swapped LoRA via PEFT set_adapter()
│  Pool A: all 12 adapters in VRAM simultaneously
└──────────────────────────────────────────────────────────────────────────
    │
    ▼
ShellExecutor.extract_commands(response)
│  ─ <command>...</command>   primary format
│  ─ ```bash/sh ... ```       explicit fence
│  ─ <command>```bash...```</command>  HTB nested fence (unwrapped)
│  ─ <reasoning> with ```python code  exploitdb (saved → python3 /tmp/exploit.py)
└──────────────────────────────────────────────────────────────────────────
    │
    ▼
Validation Pipeline
│  L1  CommandValidator.validate_flags()    flag-set check  ──┐
│  L2  CommandValidator.validate_semantic() required args   ──┤→ executor adapter corrects
│  L3  parse_execution_errors()             runtime stderr  ──┘
│
│  Placeholders (YOUR_TOKEN) → blocked
│  Echo fabrication (echo "Found...") → blocked
│  Circuit breaker: ≥3 consecutive flag errors for same tool → skipped
└──────────────────────────────────────────────────────────────────────────
    │
    ▼
ShellExecutor.run_all()             async subprocess, per-tool timeouts
    │
    ▼
EngagementKnowledge.ingest_structured()  ← analyst adapter extracts JSON
│  Fallback: ingest_tool_output()         ← regex if analyst unavailable
│  Persists to ~/.mythosengine/engagements/<program>/knowledge.json
└──────────────────────────────────────────────────────────────────────────
    │
    ▼
EngagementState.update_from_output()   phase auto-advancement + finding extraction
    │
    ▼
Null-step tracking
│  null_step == 1  → planner adapter proposes next action
│  null_step >= 3  → hypothesis loop: researcher → executor → analyst (up to 4 rounds)
└──────────────────────────────────────────────────────────────────────────
```

---

## Adapters

### Domain Adapters (routing targets)

| Adapter | Purpose | CTF Phases | BB Phases | Keyword Triggers |
|---------|---------|-----------|----------|-----------------|
| **htb** | General pentest: service enum, post-exploitation, privesc chains | IDLE, SERVICE_ENUM, SHELL, PRIVESC, REPORT | report_draft (primary) | nmap, linpeas, sudo -l, suid |
| **vulhub** | Known CVE exploitation chains from Vulhub lab scenarios | VULN_ID (primary) | exploit_verify (fallback) | CVE-XXXX-XXXX |
| **attack** | MITRE ATT&CK technique implementation: lateral movement, credential access, C2, exfil | LOOT (primary) | idle/recon (fallback) | T1xxx, mimikatz, lsass, lateral movement |
| **exploitdb** | Write working exploit code for specific CVEs; saves Python scripts for execution | EXPLOITATION (primary) | exploit_verify (fallback) | write exploit, shellcode, msfvenom |
| **ad** | Active Directory attacks: Kerberoasting, DCSync, Pass-the-Hash, ADCS ESC1-8, ACL/GPO abuse | keyword-only | ad_recon (primary) | bloodhound, kerberoast, dcsync, rubeus, certipy |
| **webapp** | Web vulns: SQLi, XSS, SSRF, IDOR, SSTI, auth bypass, business logic | keyword-only | web_vuln_scan + exploit_verify (primary) | sqli, ssrf, idor, burp, jwt, ffuf |
| **osint** | Attack surface recon: subdomain enum, passive recon, leak discovery, fingerprinting | keyword-only | idle/recon/subdomain_enum/fingerprint (primary) | subfinder, amass, theHarvester, shodan |
| **cloud** | Cloud misconfigurations: AWS SSRF→IMDSv2, IAM escalation, S3 exposure, Lambda abuse | keyword-only | cloud_recon (primary) | aws, s3, iam, 169.254.169.254, pacu |

### Orchestration Adapters (backend-injected, never routed)

| Adapter | When Invoked | What It Does |
|---------|-------------|--------------|
| **executor** | L1/L2/L3 command validation failures | Takes wrong command + flag errors + tool docs → generates correct command |
| **analyst** | After every successful tool execution | Takes raw tool output → returns structured JSON: subdomains, endpoints, findings, anomalies |
| **planner** | After 1 null step (no new findings) | Takes current engagement state → proposes highest-value next action |
| **researcher** | After 3 null steps + anomalies detected | Forms hypothesis from anomalies → designs minimal probe → analyst evaluates → iterate (4 rounds max) |

---

## Routing Logic

### CTF Mode Phase Routing

```
IDLE/RECON      → htb (attack fallback)
SERVICE_ENUM    → htb (attack fallback)
VULN_ID         → vulhub (htb fallback)
EXPLOITATION    → exploitdb (vulhub fallback)
SHELL           → htb (attack fallback)
PRIVESC         → htb (attack fallback)
LOOT            → attack (htb fallback)
REPORT          → htb (attack fallback)
```

### Bug Bounty Mode Phase Routing

```
idle            → osint (webapp fallback)
recon           → osint (webapp fallback)
subdomain_enum  → osint (webapp fallback)
fingerprint     → osint (webapp fallback)
web_vuln_scan   → webapp (vulhub fallback)
exploit_verify  → webapp (exploitdb fallback)
cloud_recon     → cloud (osint fallback)
ad_recon        → ad (attack fallback)
report_draft    → htb (webapp fallback)
```

### Keyword Override Priorities (checked before phase routing)

```
11  HYPOTHESIS: / PROBE:         → researcher
10  standard tools found nothing → researcher
 9  CVE-XXXX-XXXX                → vulhub
 9  write/create/craft exploit   → exploitdb
 9  bloodhound / kerberoast      → ad
 8  aws / s3 / iam / 169.254…   → cloud
 8  sqli / ssrf / idor / burp    → webapp
 8  subfinder / amass / shodan   → osint
 7  T1xxx (ATT&CK technique ID)  → attack
 6  nmap / masscan               → htb
 5  linpeas / sudo -l            → htb
```

---

## Execution Pipeline Detail

### Command Correction (all 3 layers use `executor` adapter)

```
Domain adapter generates: gau -d supplier.meesho.com
                                     │
         CommandValidator.validate_flags()
         detects: "-d" not in gau's flag set
                                     │
         ContextBudgetManager assembles correction_messages with:
           - executor system prompt
           - wrong command
           - exact flag error
           - tool docs (⚠️ CORRECT USAGE: gau takes positional arg)
                                     │
         executor adapter generates: gau supplier.meesho.com
                                     │
         re-validate → execute
```

### Analyst Extraction (post-execution)

```
subfinder output: "api.supplier.meesho.com\nstatic.supplier.meesho.com"
                                     │
         analyst adapter called with structured extraction prompt
         returns JSON:
         {
           "subdomains": ["api.supplier.meesho.com", ...],
           "live_endpoints": [{"url": "...", "status_code": 200, "tech_stack": "nginx"}],
           "findings": [...],
           "anomalies": [...]
         }
                                     │
         engagement_knowledge.ingest_structured(extracted)
         persists to ~/.mythosengine/engagements/<program>/knowledge.json
```

### Hypothesis Loop (null_step >= 3 + anomalies)

```
researcher: "Response time 3x longer for format=pdf → possible SSTI"
                                     │
executor: curl https://target/export?format={{7*7}}
                                     │
analyst: "Response contains '49' — template evaluation confirmed"
                                     │
researcher: "Update hypothesis: Go template injection (not Jinja2)"
                                     │
executor: curl https://target/export?format={{.Env}}
... (up to 4 rounds)
```

---

## File Structure

```
Mythos-v4/
├── mythos_engine/
│   ├── run.py                          # Entry point
│   └── pentestgpt/
│       ├── core/
│       │   ├── mythos_backend.py       # Main orchestration loop
│       │   ├── multi_adapter_model.py  # Qwen3-14B + LoRA hot-swap
│       │   ├── adapter_router.py       # Routing: classifier → keyword → phase
│       │   ├── context_manager.py      # Token-budget-aware context assembly
│       │   ├── engagement_state.py     # Phase tracking, finding extraction
│       │   ├── engagement_knowledge.py # Persistent knowledge store + RAG index
│       │   ├── tool_executor.py        # Async subprocess executor + command extraction
│       │   ├── tool_docs.py            # Live tool docs: --help → tldr → cheat.sh
│       │   ├── command_validator.py    # 3-layer flag/semantic/runtime validation
│       │   ├── bug_bounty.py           # Scope loading + WAF header injection
│       │   ├── rag_retriever.py        # Live CVE/exploit feed + TF-IDF index
│       │   ├── feedback_loop.py        # Engagement feedback for future training
│       │   ├── tool_creator.py         # Custom tool scaffolding
│       │   └── config.py               # Pydantic settings + adapter path resolution
│       ├── prompts/
│       │   └── mythos_prompts.py       # All 12 adapter system prompts (CTF + BB)
│       └── interface/
│           ├── mythos_tui.py           # Textual TUI
│           └── mythos_cli.py           # Rich CLI streaming mode
├── adapters/
│   ├── mythos-v4-htb/best-adapter/     # LoRA weights
│   ├── mythos-v4-vulhub/best-adapter/
│   ├── mythos-v4-attack/best-adapter/
│   ├── mythos-v4-exploitdb/best-adapter/
│   ├── mythos-v4-ad/best-adapter/
│   ├── mythos-v4-webapp/best-adapter/
│   ├── mythos-v4-osint/best-adapter/
│   ├── mythos-v4-cloud/best-adapter/
│   ├── mythos-v4-executor/best-adapter/
│   ├── mythos-v4-analyst/best-adapter/
│   ├── mythos-v4-planner/best-adapter/
│   ├── mythos-v4-researcher/best-adapter/
│   └── router_classifier/              # Trained MLP routing head (optional)
├── data/
│   ├── raw/                            # Source data files per adapter
│   └── processed/                      # Training-ready JSONL files
├── bootstrap.py                        # One-time setup: detect GPU, resolve paths
├── train.py                            # LoRA fine-tuning script
├── prepare_data.py                     # Data processing pipeline
├── test_adapter_flow.py                # Adapter test: solo + chain + exec pipeline
├── install_tools.sh                    # Install all pentest tools
└── engagements/
    └── <program>/
        └── scope.json                  # Bug bounty scope definition
```

---

## Quick Start

### 1. Bootstrap (run once per machine)

```bash
python3 bootstrap.py
```

Detects GPU, resolves adapter paths, writes `~/.mythosengine/state.json`.

### 2. Install Tools

```bash
bash install_tools.sh
```

Installs: nmap, subfinder, amass, dnsx, httpx (projectdiscovery), katana, gau, nuclei, ffuf, sqlmap, etc.

### 3. Run

```bash
# CTF — interactive TUI
python3 mythos_engine/run.py --target 10.10.11.5 --execute

# CTF — headless CLI
python3 mythos_engine/run.py --target 10.10.11.5 --execute --cli

# Bug bounty with scope file
python3 mythos_engine/run.py \
  --target supplier.meesho.com \
  --bug-bounty \
  --scope engagements/meesho/scope.json \
  --program meesho \
  --execute --cli

# Advisory only (no execution)
python3 mythos_engine/run.py --target 10.10.11.5
```

### 4. Test Adapters (no GPU for pipeline test)

```bash
# Full solo + chain test (GPU)
python3 test_adapter_flow.py --target supplier.meesho.com --max-tokens 512

# Execution pipeline only — NO GPU required, tests all 6 validation layers
python3 test_adapter_flow.py --exec-pipeline

# With real shell execution of safe commands
python3 test_adapter_flow.py --exec-pipeline --execute
```

---

## Scope File Format (`scope.json`)

```json
{
  "program": "Meesho Bug Bounty",
  "platform": "HackerOne",
  "researcher": "your_handle",
  "test_header": {
    "header_name": "X-Hackerone",
    "header_value": "aquamarine_skeleton",
    "required": true
  },
  "test_accounts": [
    {
      "role": "supplier",
      "credentials": [
        {"username": "suppliertest-1@meeshoai.com", "password": "Hackerone@123$"}
      ]
    }
  ],
  "reward_table": [
    {"severity": "critical", "min": 5000, "max": 10000}
  ],
  "scope": [
    {"type": "web", "value": "*.meesho.com", "eligible": true},
    {"type": "web", "value": "supplier.meesho.com", "eligible": true}
  ],
  "out_of_scope": [
    {"type": "web", "value": "blog.meesho.com"}
  ]
}
```

The scope file injects:
- Mandatory WAF bypass header into every adapter's system prompt
- Test credentials for authenticated testing
- In/out-of-scope targets for scope enforcement

---

## Training New Adapters

### Data Generation

```bash
# Generate raw data for a source
python3 data/source10_executor.py --test    # preview first
python3 data/source10_executor.py           # generate

# Process raw → training JSONL
python3 prepare_data.py --source executor

# Check token distribution
python3 data/analyze_tokens.py processed/executor.jsonl
```

### Training

```bash
# Train a single adapter
python3 train.py \
  --source executor \
  --base unsloth/Qwen3-14B \
  --output outputs/mythos-v4-executor \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-4

# Best checkpoint is auto-selected and written to best-adapter/
# Copy to adapters/ for the engine to pick up:
cp -r outputs/mythos-v4-executor/best-adapter adapters/mythos-v4-executor/best-adapter
```

### Training Data Sources per Adapter

| Adapter | Source Files | Training Objective |
|---------|-------------|-------------------|
| htb | source1_htb.py | HTB writeup step-by-step exploitation |
| vulhub | source2_vulhub.py | CVE exploitation chains |
| attack | source3_attack.py | ATT&CK technique implementations |
| exploitdb | source4_exploitdb.py | Exploit-DB PoC code + analysis |
| ad | source5_ad.py | AD attack scenarios |
| webapp | source6_webapp.py | Web vuln exploitation |
| osint | source7_osint.py | Recon tool output processing |
| cloud | source8_cloud.py | Cloud attack scenarios |
| executor | source10_executor.py | Command correction: wrong+error → correct |
| analyst | source11_analyst.py | Tool output → structured JSON extraction |
| planner | source12_planner.py | Engagement state → next action plan |
| researcher | source13_researcher.py | Anomaly → hypothesis → probe chain |

---

## Persistent State

### Machine State
`~/.mythosengine/state.json` — written by `bootstrap.py`. Stores resolved tool paths, adapter paths, wordlist locations.

### Engagement Knowledge
`~/.mythosengine/engagements/<program>/knowledge.json` — persists across sessions:
- **ACQUIRED**: Subdomains, endpoints with status/tech/auth, API routes
- **FOUND**: VulnFindings with severity, PoC, CVSS, confirmation status
- **BUILT**: Custom tools, engagement wordlists, nuclei templates
- **test_queue**: Endpoints pending specific attack coverage
- **anomalies**: Unexplained behavioral signals for researcher hypothesis loop

### Tool Docs Cache
`~/.mythosengine/tool_docs/<tool>.json` — 7-day TTL. Forces local `--help` over online docs to catch binary version mismatches (e.g., Python httpx vs projectdiscovery httpx).

### RAG Cache
`~/.mythosengine/rag_cache/` — CVE and exploit feed, refreshed every 24h in background.

---

## Commands During a Session

| Command | Description |
|---------|-------------|
| `/adapter <name>` | Force-switch to specific adapter for one turn |
| `/adapters` | Show routing explanation + loaded adapters |
| `/scope` | Display current scope (BB mode) |
| `/knowledge` | Show engagement knowledge summary |
| `/report` | Generate a full engagement report |
| `/stop` | Stop the current execution loop |
| `/clear` | Clear conversation history |

---

## Known Gaps / Future Work

| Item | Severity | Notes |
|------|----------|-------|
| Router classifier needs training | Medium | `adapters/router_classifier/` must be populated; currently falls back to regex |
| `cloud` has no CTF-phase routing | Low | Relies on keyword triggers (aws, s3, iam) — works but not as clean as phase routing |
| `exploitdb` Python exploits run in `/tmp` | Low | Need cleanup policy; currently left on disk |
| Context budget fixed at 4096 | Low | `ContextBudgetManager` scales if model reports larger context; cloud GPU shows 40960 |
| Feedback loop training not automated | Medium | `FeedbackLoop` collects traces but `retrain.py` must be run manually |
| Scope enforcement in ShellExecutor | Low | `scope_checker=None` in `run_all()` — scope is enforced via prompt context only |

---

## Audit Status (May 2026)

All 12 adapters verified:
- ✅ All adapters load and produce domain-relevant output (test_adapter_flow.py)
- ✅ Chained flow works: osint → webapp → analyst → researcher
- ✅ X-Hackerone WAF header appears in cloud/webapp/osint commands
- ✅ `executor` adapter wired for all 3 correction layers
- ✅ `analyst` adapter called post-execution for structured extraction
- ✅ `planner` adapter wired at null_step==1
- ✅ `researcher` adapter fires in hypothesis loop (null_step>=3 + anomalies)
- ✅ `exploitdb` Python code saved and executed via python3
- ✅ HTB nested bash fence unwrapping (extract_commands)
- ✅ BB phase routing covers all 9 phases with correct adapter assignments
- ✅ Token-budget context assembly prevents context overflow
- ✅ Engagement knowledge persists and restores across sessions

---

## Native Tool Runtime Layer

The tool runtime layer sits between adapter output and the shell, enforcing correctness and truthfulness at the execution boundary.

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `ToolResult` | `tools/base.py` | Standard return shape for every tool execution |
| `EnvManifest` | `tools/env_manifest.py` | Session-start environment detection + binary collision resolution |
| Typed wrappers | `tools/pentest_tools.py` | 8 wrappers: `nmap_scan`, `curl_probe`, `subfinder_enum`, `gau_urls`, `probe_http`, `ffuf_dirs`, `gobuster_dir`, `run_python` |
| `ToolDispatcher` | `tools/dispatcher.py` | Translates model `<command>` output → typed wrapper calls; preflight tool_operator for ambiguous binaries |
| `ExecutionPolicy` | `core/execution_policy.py` | Fail-closed rules: flag errors, empty output, circuit breaker, analyst gating |
| `ObservationSummarizer` | `core/observation.py` | Compact per-step prompts (~150–300 tokens) + full output saved to `workspace/observations/` |

### Key properties

- **Binary collision detection**: `EnvManifest` distinguishes Python `httpx` from ProjectDiscovery `httpx`. The `probe_http` wrapper always uses the PD binary path, or falls back to a `curl` loop — it never calls bare `httpx`.
- **No invented flags**: `gau_urls` always passes the domain as a positional argument (not `-d`). `subfinder_enum` always uses `-d`. Every wrapper was built from tested flag patterns.
- **Fail-closed**: `ExecutionPolicy` ensures that `exit_code != 0`, empty stdout, or env errors (Python tracebacks) **cannot** advance the engagement phase or reach the analyst adapter.
- **Compact context**: `ObservationSummarizer` injects a 150–300 token summary into the next model turn instead of raw tool blobs. Full output is saved to disk for debugging.

### Training `tool_operator`

```bash
# Generate training data (uses real --help output from installed tools)
python3 src/source15_tool_operator.py

# Process into training format
python3 prepare_data.py --source tool_operator

# Train the adapter (requires GPU)
python3 train.py --source tool_operator
```

### CTF Validation Ladder

Use this progression to validate before returning to real-world bug bounty:

**Step 1 — Unit tests (no GPU)**
```bash
python3 test_adapter_flow.py --tool-runtime-test
```
Verifies: manifest detection, gau routing, policy gating, phase hold on empty output, compact summarizer prompts. All 5 checks must pass.

**Step 2 — Vulhub Docker (Log4Shell)**
```bash
# In one terminal: spin up the target
docker-compose -f vulhub/log4j/CVE-2021-44228/docker-compose.yml up -d

# In another: run Mythos
python3 run.py --target 127.0.0.1:8080 --execute
```
Success metric: Mythos identifies Log4Shell, proposes a valid `${jndi:ldap://…}` payload, and executes without hallucinating flags or fabricating output.

**Step 3 — HTB retired easy box (VPN required)**
```bash
python3 run.py --target <box-ip> --execute
```
Success metric: Score ≥ 3 — vulnerability correctly identified + at least one valid command executed per phase.

Only proceed to live bug bounty after scoring ≥ 3 on an HTB easy box.
