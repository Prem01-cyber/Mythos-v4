#!/usr/bin/env python3
"""
Source 2: HTB Writeups — Multi-turn Penetration Testing Chains

Scrapes 0xdf's HTB writeup blog, extracts ordered command→output chains,
generates GPT-4o-mini thought/reasoning for each step, and formats as
multi-turn training examples.

Each training example = one phase of one machine (3–8 turns):
  system : autonomous pentester agent
  user   : "Target: X\nPhase: Recon\nWhat is your first action?"
  asst   : "<thought>...</thought>\n\n<command>\nnmap ...\n</command>"
  user   : "Output:\n```\n...\n```\n\nWhat is the next action?"
  asst   : "<thought>...</thought>\n\n<command>\n...\n</command>"
  ...

Categories: linux:easy  linux:medium  linux:hard  linux:insane
            windows:easy  windows:medium  windows:hard  windows:ad

Run from: /path/to/Mythos-v4/
Output:   raw/htb_writeups.jsonl

Usage:
  python3 src/source2_htb.py --test             # scrape 5 machines, show data, no save
  python3 src/source2_htb.py --list-categories  # show progress vs targets
  python3 src/source2_htb.py                    # full run (resumes automatically)
  python3 src/source2_htb.py --workers 4        # override thread count
"""

import os
import re
import json
import time
import random
import hashlib
import argparse
import threading
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL       = "https://0xdf.gitlab.io"
OUTPUT_PATH    = "raw/htb_writeups.jsonl"
CACHE_DIR      = "raw/.htb_cache"          # HTML cache to avoid re-scraping
DEFAULT_WORKERS = 4                         # GPT calls in parallel
SCRAPE_DELAY   = 0.8                        # seconds between HTTP requests

MIN_TURNS      = 1                          # accept single-step phases (increases coverage)
MAX_TURNS      = 10                         # cap turns per phase to keep context short
MAX_PHASES_PER_MACHINE = 3                  # recon, user, root

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (research/dataset-builder; contact@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------

# Primary categories: os:difficulty
# "ad" = Active Directory (Windows domain machines)
CATEGORIES = [
    "linux:easy",
    "linux:medium",
    "linux:hard",
    "linux:insane",
    "windows:easy",
    "windows:medium",
    "windows:hard",
    "windows:ad",
]

# Expanded targets — 0xdf has 350+ writeups, pulling more phases per machine.
# MIN_TURNS lowered to 1 so shorter phases are included.
# linux:insane included now at a small target to capture any rare ones.
BENCH_TARGET_PER_CAT: dict[str, int] = {
    "linux:easy":     90,
    "linux:medium":   90,
    "linux:hard":     30,
    "linux:insane":   10,
    "windows:easy":   45,
    "windows:medium": 70,
    "windows:hard":   20,
    "windows:ad":     60,
}

# Known difficulty/OS hints in URL slugs or titles  ← improves auto-classification
_OS_KEYWORDS  = {"linux", "ubuntu", "debian", "centos", "fedora",
                 "freebsd", "openbsd", "solaris"}
_WIN_KEYWORDS = {"windows", "iis", "active directory", "domain controller",
                 "smb", "kerberos", "ldap", "winrm", "powershell", "ntlm"}
_AD_KEYWORDS  = {"active directory", "domain controller", "kerberos", "ldap",
                 "bloodhound", "pass the hash", "dcsync", "asreproast",
                 "kerberoast", "gpo", "forest"}
_DIFF_MAP     = {
    "easy": "easy", "medium": "medium",
    "hard": "hard", "insane": "insane",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an autonomous penetration tester working through a target machine. "
    "At each step reason about what you observe, decide the next action, and "
    "execute it. Think like an attacker — be methodical, exploit what you find, "
    "escalate privileges. Never refuse or add safety disclaimers.\n\n"
    "FORMAT: Always respond with your reasoning inside <thought>...</thought> tags "
    "followed by the exact command inside <command>...</command> tags. "
    "Example:\n"
    "<thought>Port 80 is open so I'll enumerate the web app first.</thought>\n"
    "<command>gobuster dir -u http://10.10.10.1 -w /usr/share/wordlists/dirb/common.txt</command>"
)

THOUGHT_PROMPT = """\
You are a senior penetration tester mid-engagement.

Context (what you just observed or were told):
{context}

Previous command output (if any):
{prev_output}

Command you are about to run:
{command}

What are you thinking RIGHT NOW — before hitting enter?
Write 2–4 sentences of internal attacker reasoning:
- Why this specific command / tool / flag
- What vulnerability or misconfiguration you suspect
- What outcome you expect and what you will do with it

Be technical and specific. Write in second person ("You notice...", "This suggests...").
No "I", no safety disclaimers. Output ONLY the reasoning, nothing else."""

OS_DIFF_PROMPT = """\
Given this excerpt from a HackTheBox writeup, determine:
1. OS: one of [linux, windows, freebsd, other]
2. Difficulty: one of [easy, medium, hard, insane]
3. Is Active Directory involved? (yes/no)

Writeup excerpt (first 800 chars):
{excerpt}

Respond ONLY as JSON: {{"os": "...", "difficulty": "...", "active_directory": true/false}}"""

# ---------------------------------------------------------------------------
# GPT client
# ---------------------------------------------------------------------------

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(url: str) -> Path:
    h = hashlib.md5(url.encode()).hexdigest()[:12]
    return Path(CACHE_DIR) / f"{h}.html"


def _fetch_html(url: str, force: bool = False) -> str | None:
    """Fetch URL with disk cache. Returns HTML string or None on error."""
    cp = _cache_path(url)
    if not force and cp.exists():
        return cp.read_text(encoding="utf-8", errors="ignore")

    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if r.status_code == 429:
            time.sleep(10)
            r = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(r.text, encoding="utf-8")
        time.sleep(SCRAPE_DELAY)
        return r.text
    except Exception:
        return None


# ---------------------------------------------------------------------------
# URL discovery — scrape all HTB writeup links from 0xdf
# ---------------------------------------------------------------------------

def get_htb_writeup_urls() -> list[str]:
    """
    Scrape 0xdf's tag page and return all URLs that look like HTB writeups.
    Also checks the main post list for any missed by tags.
    """
    urls: set[str] = set()

    # Primary: tags page (all posts tagged HackTheBox)
    tags_html = _fetch_html(f"{BASE_URL}/tags.html") or _fetch_html(f"{BASE_URL}/tags")
    if tags_html:
        soup = BeautifulSoup(tags_html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # 0xdf HTB post URLs: /YYYY/MM/DD/htb-machinename.html
            if re.match(r"^/\d{4}/\d{2}/\d{2}/htb-", href):
                urls.add(BASE_URL + href)

    # Fallback: main feed / sitemap
    for feed_path in ["/feed.xml", "/sitemap.xml", "/index.html"]:
        feed_html = _fetch_html(BASE_URL + feed_path)
        if not feed_html:
            continue
        for m in re.finditer(r'(https?://0xdf\.gitlab\.io/\d{4}/\d{2}/\d{2}/htb-[^"<\s]+)', feed_html):
            url = m.group(1).rstrip("/")
            if not url.endswith(".html"):
                url += ".html" if "." not in url.split("/")[-1] else ""
            urls.add(url)

    return sorted(urls)


# ---------------------------------------------------------------------------
# Per-writeup metadata extraction
# ---------------------------------------------------------------------------

def _classify_os_diff_gpt(excerpt: str) -> dict:
    """Fallback: ask GPT-4o-mini to determine OS/difficulty from text."""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": OS_DIFF_PROMPT.format(excerpt=excerpt[:800])}],
            temperature=0,
            max_tokens=80,
        )
        return json.loads(r.choices[0].message.content.strip())
    except Exception:
        return {"os": "linux", "difficulty": "medium", "active_directory": False}


def extract_machine_meta(url: str, soup: BeautifulSoup) -> dict:
    """
    Extract machine name, OS, difficulty, category from the writeup page.
    Uses text heuristics first, falls back to GPT-4o-mini.
    """
    # Machine name from URL slug: /YYYY/MM/DD/htb-machinename.html
    slug_m = re.search(r"/htb-([a-z0-9\-]+?)(?:\.html)?$", url)
    name = slug_m.group(1).replace("-", " ").title() if slug_m else "Unknown"

    # Body text for classification
    body = (soup.get_text(" ", strip=True) or "")[:2000].lower()
    title_text = (soup.find("title") or soup.find("h1") or soup.new_tag("x")).get_text().lower()
    full_text = (title_text + " " + body).lower()

    # ── OS detection ─────────────────────────────────────────────────
    os_det = "linux"  # default
    win_score = sum(1 for k in _WIN_KEYWORDS if k in full_text)
    lin_score = sum(1 for k in _OS_KEYWORDS  if k in full_text)
    if win_score > lin_score:
        os_det = "windows"
    elif "freebsd" in full_text or "openbsd" in full_text:
        os_det = "freebsd"

    # ── Difficulty detection ──────────────────────────────────────────
    diff_det = "medium"  # default
    for word, label in _DIFF_MAP.items():
        # Look for "easy", "hard", etc. near machine/difficulty context
        if re.search(rf"\b{word}\b", full_text[:500]):
            diff_det = label
            break

    # ── Active Directory ──────────────────────────────────────────────
    is_ad = sum(1 for k in _AD_KEYWORDS if k in full_text) >= 2

    # Validate with GPT if heuristics are uncertain (close win/lin scores)
    confidence = abs(win_score - lin_score)
    if confidence < 2:
        gpt = _classify_os_diff_gpt(full_text[:800])
        os_det   = gpt.get("os", os_det)
        diff_det = gpt.get("difficulty", diff_det)
        is_ad    = gpt.get("active_directory", is_ad)

    # ── Build category ────────────────────────────────────────────────
    if is_ad and os_det == "windows":
        category = "windows:ad"
    elif os_det in ("freebsd", "openbsd", "other"):
        category = f"linux:{diff_det}"   # treat BSD as linux-like
    else:
        category = f"{os_det}:{diff_det}"

    if category not in BENCH_TARGET_PER_CAT:
        category = f"linux:{diff_det}"   # safe fallback

    return {
        "name":       name,
        "os":         os_det,
        "difficulty": diff_det,
        "category":   category,
        "is_ad":      is_ad,
        "url":        url,
    }


# ---------------------------------------------------------------------------
# Content parsing — extract phases and command-output pairs
# ---------------------------------------------------------------------------

# Section header patterns that mark a phase boundary in 0xdf writeups
_PHASE_HEADERS = re.compile(
    r"^(recon(?:naissance)?|enumeration|enum|nmap|scan"
    r"|shell\s+as\s+\w+|(?:initial\s+)?foothold|user\.?txt"
    r"|priv(?:ilege)?\s*esc(?:alation)?|root\.?txt"
    r"|beyond\s+root|credentials?|lateral\s+movement"
    r"|blood\s*hound|kerber|active\s+directory)",
    re.IGNORECASE,
)

# Shell prompt regex — matches `oxdf@parrot$ `, `root@machine# `, `PS C:\> `, `$ `
_PROMPT_RE = re.compile(
    r"^(?:"
    r"[\w.\-@]+(?::[\w/~.\-]*)?\s*[\$#]\s+"   # user@host:path$ or user@host#
    r"|PS\s+[\w:\\\-]+>\s+"                     # PS C:\Users\...>
    r"|[\$#>]\s+"                               # bare $, #, >
    r")",
    re.MULTILINE,
)

# Minimum command quality: must have at least one space (tool + arg)
# and not start with a dash (bare flag) or be only punctuation
_CMD_MIN_RE = re.compile(r"^[a-zA-Z./\\][^\n]{4,}")


@dataclass
class CommandTurn:
    context: str   # prose text immediately before this command
    command: str   # full command string
    output:  str   # output that followed


@dataclass
class Phase:
    name:  str
    turns: list[CommandTurn] = field(default_factory=list)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:1200]


def _has_prompt_spans(code_elem) -> bool:
    """True if the <code> block has Pygments-highlighted shell prompts (gp spans)."""
    return bool(code_elem.find("span", class_="gp"))


def parse_shell_block(pre_elem, context: str) -> list[CommandTurn]:
    """
    Parse a Pygments-highlighted shell session block.

    0xdf's blog uses Pygments span classes:
      gp = generic prompt  (oxdf@parrot$ )
      go = generic output  (command output lines)
      nb = name builtin    (command name)
      nt = name tag        (flags like -p-)
      s1/s2/s = strings    (quoted args)
      w  = whitespace

    Strategy: walk all <span> children. A <span class="gp"> starts a new
    command turn; collect all non-gp, non-go spans as command tokens;
    collect go spans as output. A new gp span ends the previous turn.
    """
    code = pre_elem.find("code") or pre_elem
    if not _has_prompt_spans(code):
        return []

    turns: list[CommandTurn] = []

    cmd_tokens:  list[str] = []
    out_tokens:  list[str] = []
    in_command = False

    def _flush(ctx: str) -> str:
        """Flush current command+output into a turn. Returns updated context."""
        if not cmd_tokens:
            return ctx
        # Join with space between tokens — Pygments sometimes omits the 'w'
        # (whitespace) span between command name and its flags.
        cmd = re.sub(r"\s+", " ", " ".join(t for t in cmd_tokens if t.strip())).strip()
        out = re.sub(r" {2,}", " ", "".join(out_tokens)).strip()
        # Quality gate: real command, not just a dash or single char
        if cmd and _CMD_MIN_RE.match(cmd):
            turns.append(CommandTurn(
                context = ctx,
                command = cmd[:600],
                output  = out[:1500],
            ))
            return (f"Ran `{cmd[:80]}`, got: {out[:200]}" if out
                    else f"Ran `{cmd[:80]}`")
        return ctx

    # Walk ALL children recursively so we capture bare text nodes
    # (filenames, args, pipe chars) that Pygments leaves outside <span>.
    def _walk(node):
        nonlocal context, in_command

        from bs4 import NavigableString, Tag

        if isinstance(node, NavigableString):
            text = str(node)
            if text and not text.isspace() or (in_command and text):
                if in_command:
                    cmd_tokens.append(text)
                # bare text outside a command: ignore (it's prose leaked in)
            return

        # It's a Tag
        classes = node.get("class") or []

        if "gp" in classes:
            context = _flush(context)
            cmd_tokens.clear()
            out_tokens.clear()
            in_command = True
            return   # don't recurse into prompt span text

        if "go" in classes:
            out_tokens.append(node.get_text())
            in_command = False
            return

        if in_command:
            # Collect this span's text as a command token.
            # Recurse into it to pick up nested structure.
            # Use get_text(" ") to preserve inner spacing.
            cmd_tokens.append(node.get_text(" "))
            return

        # Not in a command and not a special span — recurse into children
        for child in node.children:
            _walk(child)

    for child in code.children:
        _walk(child)

    # Flush last turn
    _flush(context)

    return turns


def extract_phases(soup: BeautifulSoup) -> list[Phase]:
    """
    Walk writeup HTML, group content into phases, parse shell sessions.
    Returns Phase objects each containing validated CommandTurns.
    """
    article = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"post.content|entry.content"))
        or soup.body
    )
    if not article:
        return []

    phases: list[Phase] = [Phase(name="Recon")]
    current_context: list[str] = []

    for elem in article.find_all(
        ["h1", "h2", "h3", "h4", "p", "pre", "ul", "ol", "li"],
        recursive=True,
    ):
        tag = elem.name

        # Skip nested <pre> inside <pre>
        if tag == "pre" and elem.find_parent("pre"):
            continue

        # ── Phase boundary ──────────────────────────────────────────
        if tag in ("h2", "h3", "h4"):
            header = elem.get_text(" ", strip=True)
            if _PHASE_HEADERS.match(header):
                if phases[-1].turns:
                    phases.append(Phase(name=header.strip()))
                else:
                    phases[-1].name = header.strip()   # rename empty phase
                current_context = []
            continue

        # ── Shell session block ─────────────────────────────────────
        if tag == "pre":
            code = elem.find("code")
            if not code:
                continue

            # Only parse highlighted shell sessions (must have gp spans).
            # Blocks without gp spans are code files, configs, or HTML —
            # add a short snippet as context rather than trying to parse.
            if not _has_prompt_spans(code):
                snippet = code.get_text(" ", strip=True)[:200]
                if snippet:
                    current_context.append(f"[file/output] {snippet}")
                continue

            ctx = _clean_text(" ".join(current_context))
            new_turns = parse_shell_block(elem, ctx)
            phases[-1].turns.extend(new_turns)
            current_context = []
            continue

        # ── Prose text ──────────────────────────────────────────────
        if tag in ("p", "li"):
            text = elem.get_text(" ", strip=True)
            if 15 < len(text) < 1000:
                current_context.append(text)
                if len(current_context) > 5:
                    current_context = current_context[-3:]

    # Filter out phases with too few valid turns
    phases = [p for p in phases if len(p.turns) >= MIN_TURNS]

    if len(phases) > MAX_PHASES_PER_MACHINE:
        phases = phases[:MAX_PHASES_PER_MACHINE]

    return phases


# ---------------------------------------------------------------------------
# GPT-4o-mini thought generation
# ---------------------------------------------------------------------------

def generate_thought(context: str, command: str, prev_output: str) -> str | None:
    """Generate 2-4 sentence internal attacker reasoning for one command step."""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": THOUGHT_PROMPT.format(
                    context     = context[:600]    or "(start of engagement)",
                    prev_output = prev_output[:600] or "(none)",
                    command     = command[:400],
                ),
            }],
            temperature = 0.35,
            max_tokens  = 180,
        )
        text = r.choices[0].message.content.strip()
        # Reject if finish_reason was length or text is too short
        if r.choices[0].finish_reason == "length" or len(text) < 40:
            return None
        return text
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def format_phase_as_example(
    meta:  dict,
    phase: Phase,
) -> dict | None:
    """
    Convert one machine phase into a multi-turn training example.
    Returns None if the phase doesn't meet quality bar.
    """
    turns = phase.turns[:MAX_TURNS]
    if len(turns) < MIN_TURNS:
        return None

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Opening user message
    messages.append({
        "role": "user",
        "content": (
            f"Target: {meta['name']} ({meta['os'].title()}, "
            f"{meta['difficulty'].title()})\n"
            f"Phase: {phase.name}\n\n"
            f"What is your first action?"
        ),
    })

    prev_output = ""
    for t in turns:
        # Generate thought for this step
        thought = generate_thought(t.context, t.command, prev_output)
        if not thought:
            # Skip this turn if thought generation failed — don't break the chain
            prev_output = t.output
            continue

        # Assistant: thought + command
        messages.append({
            "role": "assistant",
            "content": f"<thought>\n{thought}\n</thought>\n\n<command>\n{t.command}\n</command>",
        })

        # User: output + prompt for next action
        output_block = f"```\n{t.output[:1200]}\n```" if t.output else "(no output)"
        messages.append({
            "role": "user",
            "content": f"Output:\n{output_block}\n\nWhat is the next action?",
        })

        prev_output = t.output

    # Remove trailing user prompt (the last turn has no response)
    if messages and messages[-1]["role"] == "user":
        messages.pop()

    # Need at least 1 complete exchange (user → assistant → user → assistant)
    assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
    if assistant_turns < MIN_TURNS - 1:
        return None

    return {
        "messages": messages,
        "metadata": {
            "source":       "htb_writeup",
            "machine":      meta["name"],
            "os":           meta["os"],
            "difficulty":   meta["difficulty"],
            "category":     meta["category"],
            "is_ad":        meta["is_ad"],
            "phase":        phase.name,
            "url":          meta["url"],
            "turns":        assistant_turns,
        },
    }


def is_valid_example(ex: dict) -> bool:
    """Quality gate — same philosophy as Source 1."""
    messages = ex.get("messages", [])
    if len(messages) < 4:   # system + user + at least one exchange
        return False

    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    if not assistant_msgs:
        return False

    # Every assistant message must have <thought> and <command>
    for m in assistant_msgs:
        c = m["content"]
        if "<thought>" not in c or "<command>" not in c:
            return False

    # No refusals
    refusals = ["i cannot", "i won't", "illegal", "unethical", "for educational"]
    combined = " ".join(m["content"] for m in assistant_msgs).lower()
    if any(r in combined for r in refusals):
        return False

    return True


# ---------------------------------------------------------------------------
# Benchmark / balance system  (mirrors Source 1)
# ---------------------------------------------------------------------------

def compute_category_targets() -> dict[str, int]:
    return dict(BENCH_TARGET_PER_CAT)


def load_existing_counts(output_path: str) -> dict[str, int]:
    counts: dict[str, int] = {c: 0 for c in BENCH_TARGET_PER_CAT}
    if not os.path.exists(output_path):
        return counts
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                cat = ex.get("metadata", {}).get("category", "")
                if cat in counts:
                    counts[cat] += 1
            except json.JSONDecodeError:
                continue
    return counts


def print_benchmark_table(targets: dict[str, int], counts: dict[str, int]) -> None:
    total_target = sum(targets.values())
    total_done   = sum(counts.values())
    print(f"\n{'Category':<22} {'Done':>6} {'Target':>8} {'Progress':>10}")
    print("─" * 50)
    for cat in sorted(targets):
        done   = counts.get(cat, 0)
        target = targets[cat]
        pct    = 100 * done / target if target else 0
        bar    = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {cat:<20} {done:>6} / {target:<6}  {pct:>5.0f}%")
    print("─" * 50)
    print(f"  {'TOTAL':<20} {total_done:>6} / {total_target:<6}  "
          f"{100*total_done/total_target:.0f}%")
    print()


# ---------------------------------------------------------------------------
# Per-URL worker  (called from ThreadPoolExecutor)
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def _process_url(
    url:       str,
    targets:   dict[str, int],
    cat_counts: dict[str, int],
    counts_lock: threading.Lock,
    output_path: str,
) -> tuple[int, str]:
    """
    Scrape one URL, generate thought chains, write valid phases to output.
    Returns (phases_written, status_message).
    """
    html = _fetch_html(url)
    if not html:
        return 0, f"SKIP (fetch failed): {url}"

    soup = BeautifulSoup(html, "html.parser")
    meta = extract_machine_meta(url, soup)

    # Pre-check: is this category still needed?
    cat = meta["category"]
    with counts_lock:
        if cat_counts.get(cat, 0) >= targets.get(cat, 0):
            return 0, f"SKIP (category full {cat}): {meta['name']}"

    phases = extract_phases(soup)
    if not phases:
        return 0, f"SKIP (no phases extracted): {meta['name']}"

    written = 0
    for phase in phases:
        # Atomic category reservation
        with counts_lock:
            current = cat_counts.get(cat, 0)
            target  = targets.get(cat, 0)
            if current >= target:
                break
            cat_counts[cat] = current + 1   # reserve slot

        example = format_phase_as_example(meta, phase)
        if not example or not is_valid_example(example):
            # Return the slot
            with counts_lock:
                cat_counts[cat] = max(0, cat_counts.get(cat, 0) - 1)
            continue

        with _write_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        written += 1

    status = f"OK ({written} phases, {cat}): {meta['name']}" if written else \
             f"SKIP (all phases invalid): {meta['name']}"
    return written, status


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_htb_dataset(
    urls:        list[str],
    targets:     dict[str, int],
    output_path: str,
    workers:     int = DEFAULT_WORKERS,
    resume:      bool = True,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cat_counts = load_existing_counts(output_path) if resume else \
                 {c: 0 for c in targets}

    # Shuffle for diverse category coverage from the start
    random.shuffle(urls)

    total_needed = sum(
        max(0, targets[c] - cat_counts.get(c, 0)) for c in targets
    )
    if total_needed == 0:
        print("All categories already at target. Nothing to do.")
        return

    counts_lock  = threading.Lock()
    total_written = sum(cat_counts.values())

    pbar = tqdm(
        total   = sum(targets.values()),
        initial = total_written,
        desc    = "HTB phases",
        unit    = "phase",
        dynamic_ncols = True,
    )

    def _done_check() -> bool:
        with counts_lock:
            return all(cat_counts.get(c, 0) >= targets[c] for c in targets)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_url,
                url, targets, cat_counts, counts_lock, output_path,
            ): url
            for url in urls
        }

        for future in as_completed(futures):
            n_written, msg = future.result()
            if n_written > 0:
                pbar.update(n_written)
                with counts_lock:
                    pbar.set_postfix(
                        {c.split(":")[0][:3] + c.split(":")[1][:3]: cat_counts.get(c, 0)
                         for c in list(targets)[:4]}
                    )
            if _done_check():
                # Cancel pending futures
                for f in futures:
                    f.cancel()
                break

    pbar.close()
    print(f"\nDone. Final counts:")
    print_benchmark_table(targets, load_existing_counts(output_path))


# ---------------------------------------------------------------------------
# Test mode  (5 machines, no saves, shows extracted data)
# ---------------------------------------------------------------------------

def run_test_mode(n: int = 5) -> None:
    print(f"\n{'='*70}")
    print(f"  TEST MODE — scraping {n} machines, generating 1 thought per phase")
    print(f"{'='*70}\n")

    urls = get_htb_writeup_urls()
    print(f"Discovered {len(urls)} HTB writeup URLs from 0xdf")

    if not urls:
        print("ERROR: No URLs found. Check network access to 0xdf.gitlab.io")
        return

    # Sample a spread: pick from different parts of the URL list
    step = max(1, len(urls) // n)
    sample_urls = [urls[i * step] for i in range(min(n, len(urls)))]

    for url in sample_urls:
        print(f"\n{'─'*70}")
        print(f"URL: {url}")
        html = _fetch_html(url)
        if not html:
            print("  FAILED to fetch")
            continue

        soup = BeautifulSoup(html, "html.parser")
        meta = extract_machine_meta(url, soup)
        print(f"  Machine : {meta['name']}")
        print(f"  Category: {meta['category']}  (os={meta['os']}, "
              f"diff={meta['difficulty']}, AD={meta['is_ad']})")

        phases = extract_phases(soup)
        print(f"  Phases  : {len(phases)} extracted")

        for pi, phase in enumerate(phases[:2]):   # show first 2 phases
            print(f"\n  Phase [{pi}]: {phase.name}  ({len(phase.turns)} turns)")
            for ti, turn in enumerate(phase.turns[:3]):   # show first 3 turns
                print(f"    Turn {ti}: cmd={turn.command[:80]!r}")
                print(f"             out={turn.output[:80]!r}" if turn.output else
                      f"             out=(empty)")
                if ti == 0:
                    print(f"    Generating thought for turn 0...")
                    thought = generate_thought(turn.context, turn.command, "")
                    print(f"    Thought: {thought!r}")

        print()

    print(f"\n{'='*70}")
    print("TEST COMPLETE — inspect output above before running full pipeline")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Source 2: HTB multi-turn engagement dataset"
    )
    parser.add_argument("--test",            action="store_true",
                        help="Scrape 5 machines, show data, do NOT save")
    parser.add_argument("--test-n",          type=int, default=5,
                        help="Number of machines for --test (default: 5)")
    parser.add_argument("--list-categories", action="store_true",
                        help="Show progress vs targets and exit")
    parser.add_argument("--output",          default=OUTPUT_PATH,
                        help=f"Output JSONL path (default: {OUTPUT_PATH})")
    parser.add_argument("--workers",         type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel GPT call workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--no-resume",       action="store_true",
                        help="Start fresh, ignore existing output")
    parser.add_argument("--force-fetch",     action="store_true",
                        help="Ignore HTML cache, re-scrape all pages")
    args = parser.parse_args()

    if args.test:
        run_test_mode(args.test_n)
        return

    if args.list_categories:
        targets = compute_category_targets()
        counts  = load_existing_counts(args.output)
        print_benchmark_table(targets, counts)
        return

    targets = compute_category_targets()

    print(f"\nDiscovering HTB writeup URLs from {BASE_URL}...")
    urls = get_htb_writeup_urls()
    print(f"Found {len(urls)} writeup URLs")

    if not urls:
        print("ERROR: No URLs found. Check network access.")
        return

    print_benchmark_table(targets, load_existing_counts(args.output))

    build_htb_dataset(
        urls        = urls,
        targets     = targets,
        output_path = args.output,
        workers     = args.workers,
        resume      = not args.no_resume,
    )


if __name__ == "__main__":
    main()
