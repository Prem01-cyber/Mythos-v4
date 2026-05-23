#!/usr/bin/env python3
"""
Mythos-v4 Fine-tuning Script
Model  : Qwen/Qwen3-14B (default — recommended)
Stack  : Unsloth + Flash Attention 2 + TRL SFTTrainer
Target : A100 SXM 80 GB

No quantization — full BF16 weights + LoRA adapters.
Flash Attention is mathematically identical to standard attention;
no precision loss, just memory efficiency and ~2x speed.

Model Selection:
  q3-8b   → Qwen3-8B,   18 GB BF16, single A100, fastest training
  q3-14b  → Qwen3-14B,  30 GB BF16, single A100, best 1-GPU choice ★
  q3-32b  → Qwen3-32B,  65 GB BF16, needs 2×A100 or --qlora
  Qwen3 has significantly better reasoning than Qwen2.5-Coder at the same size.
  Community consensus (2026): "Qwen2.5-Coder 32B is ancient vs Qwen3."
  The reasoning advantage is critical for <thought> quality and exploit chains.

Architecture: Specialized LoRA Adapters (one per source)
  Each adapter is trained independently from the same frozen base model.
  This eliminates catastrophic forgetting — base weights never change.
  Adapters are swapped at inference time based on task type:
    • Adapter 1 (exploitdb)   → exploit code generation, CVE analysis
    • Adapter 2 (htb)         → multi-turn pentest methodology, HTB chains
    • Adapter 3 (vulhub)      → CVE exploitation, RCE/SSRF/SSTI chains
    • Adapter 4 (attack)      → ATT&CK red team techniques
    • Adapter 5 (ad)          → Active Directory attack chains
    • Adapter 6 (webapp)      → Web application exploitation (OWASP Top 10)
    • Adapter 7 (osint)       → OSINT and external reconnaissance
    • Adapter 8 (cloud)       → Cloud security (AWS/Azure/GCP)
    • Adapter 9 (executor)    → Command correction + output filtering
    • Adapter 10 (analyst)    → HackerOne reports + tool interpretation
    • Adapter 11 (planner)    → Goal decomposition + adaptive replanning
    • Adapter 12 (researcher) → Novelty discovery via anomaly reasoning

  --source flag selects which dataset this adapter trains on.
  Output path automatically namespaced: outputs/mythos-v4-{source}/

Usage:
    python3 train.py --source exploitdb              # Adapter 1 — exploit code
    python3 train.py --source htb                    # Adapter 2 — pentest methodology
    python3 train.py --source vulhub                 # Adapter 3 — CVE exploitation
    python3 train.py --source attack                 # Adapter 4 — ATT&CK techniques
    python3 train.py --source ad                     # Adapter 5 — Active Directory
    python3 train.py --source webapp                 # Adapter 6 — Web app exploitation
    python3 train.py --source osint                  # Adapter 7 — OSINT recon
    python3 train.py --source cloud                  # Adapter 8 — Cloud security
    python3 train.py --source combined               # all sources merged (ablation only)
    python3 train.py --model q3-8b --source exploitdb   # fast 8B training
    python3 train.py --model q3-32b --qlora --source htb  # 32B on single A100
"""

import os
import sys
import json
import random
import argparse
import math
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Args — parse before any heavy imports so --help is instant
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",   default="q3-14b",
                        choices=["q3-8b", "q3-14b", "q3-32b", "7b", "14b", "32b"],
                        help="Model: q3-14b (default, recommended) → Qwen3-14B 30GB BF16 on single A100; "
                             "q3-8b → Qwen3-8B 18GB, fastest; "
                             "q3-32b → Qwen3-32B 65GB, needs 2×A100 or --qlora; "
                             "7b/14b/32b = legacy Qwen2.5-Coder variants")
parser.add_argument("--source",  default=None,
                    choices=["exploitdb", "htb", "vulhub", "attack",
                             "ad", "webapp", "osint", "cloud",
                             "executor", "analyst", "planner", "researcher",
                             "combined"],
                    help="Which dataset to train on. "
                         "exploitdb → exploit code adapter (3 epochs, seq=2048); "
                         "htb → pentest methodology adapter (3 epochs, seq=2560); "
                         "vulhub → CVE exploitation adapter (4 epochs, seq=2048); "  # overfit observed at epoch 3 in first run
                         "attack → ATT&CK red team adapter (5 epochs, seq=1280); "
                         "ad → Active Directory adapter (12 epochs, seq=1024); "
                         "webapp → web app exploitation adapter (10 epochs, seq=1792); "
                         "osint → OSINT recon adapter (12 epochs, seq=1024); "
                         "cloud → cloud security adapter (14 epochs, seq=1280); "
                         "executor → command correction + output filtering (10 epochs, seq=1024); "
                         "analyst → HackerOne reports + tool interpretation (10 epochs, seq=1024); "
                         "planner → goal decomposition + adaptive replanning (8 epochs, seq=1024); "
                         "researcher → novelty discovery via anomaly reasoning (4 epochs, seq=1024); "
                         "combined → all sources merged (ablation only)")
parser.add_argument("--data",    default=None,
                    help="Override dataset path (optional, --source sets this automatically)")
parser.add_argument("--output",  default=None,
                    help="Override output directory (optional, auto-named from --source)")
parser.add_argument("--epochs",  type=int, default=None,
                    help="Training epochs (auto-set per source: exploitdb=3, "
                         "htb=8, vulhub=6, attack=6; override with this flag)")
parser.add_argument("--lr",      type=float, default=1e-4,
                    help="Peak learning rate (default: 1e-4)")
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--qlora",   action="store_true",
                    help="Use 4-bit NF4 quantized base (QLoRA) — allows 32B on single A100. "
                         "Avoids precision loss on LoRA updates but base weights are quantized.")
args = parser.parse_args()

random.seed(args.seed)

MODEL_MAP = {
    # ── Qwen3 (recommended) ────────────────────────────────────────────
    # Qwen3 has significantly better reasoning than Qwen2.5-Coder at the same
    # size. Critical for exploit analysis, `<thought>` quality, multi-step chains.
    # Community: "Qwen2.5-Coder 32B is ancient/outdated vs Qwen3" (2026).
    "q3-8b":  "Qwen/Qwen3-8B",          #  18 GB BF16 — fits any A100, fast training
    "q3-14b": "Qwen/Qwen3-14B",         #  30 GB BF16 — fits A100 80GB, best 1-GPU choice ★
    "q3-32b": "Qwen/Qwen3-32B",         #  65 GB BF16 — needs 2×A100 or --qlora on 1×A100

    # ── Qwen2.5-Coder (legacy, kept for continuity) ────────────────────
    # Use only if re-running a previously started Source 1 adapter.
    "7b":  "Qwen/Qwen2.5-Coder-7B-Instruct",
    "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
}
MODEL_NAME = MODEL_MAP[args.model]

# Source → (dataset file, output directory, recommended_epochs, max_seq_len)
# All paths point to processed/ — run `python3 prepare_data.py` locally to generate them.
#
# Per-source max_seq_len set to P99.5 of actual token distribution, rounded to 256.
# Less padding = faster training, lower VRAM per step, larger effective batch.
#   exploitdb  2048  — max=2043, perfect fit, 0 truncations
#   htb        2560  — P97 coverage; tail-turn truncation for long chains; saves 37% vs 4096
#   vulhub     2048  — P99.5=1913; was 3000, saves 32%
#   attack     1280  — P99.5=1053; was 2048, saves 37%
#   ad         1024  — max=940 (!); was 4096, saves 75% compute
#   webapp     1792  — P99.5=1653; was 3000, saves 40%
#   osint      1024  — P99.5=848;  was 2048, saves 50%
#   cloud      1280  — P99.5=1052; was 2048, saves 37%
#
# Different max_seq_len per adapter is NOT an issue:
#   Each adapter trains on a frozen base; seq_len only affects activation memory
#   during that training run. Adapters are fully interoperable at inference time.
#
# Epoch targets — calibrated to ~350-420 gradient steps (eff_batch=16):
#   exploitdb: 2144 ex / 16 × 3 epochs = 402 steps ✓
#   htb:       1946 ex / 16 × 3 epochs = 365 steps ✓
#   vulhub:     644 ex / 16 × 4 epochs = 161 steps  (eval loss minimum at epoch 3; 8 was overfitting)
#   attack:    1230 ex / 16 × 5 epochs = 384 steps ✓
#   ad:         274 ex / 16 × 12 epochs = 205 steps (small dataset, more passes)
#   webapp:     326 ex / 16 × 10 epochs = 204 steps
#   osint:      287 ex / 16 × 12 epochs = 215 steps
#   cloud:      255 ex / 16 × 14 epochs = 222 steps (smallest)
#   executor:   631 ex / 16 × 10 epochs = ~395 steps  
#   analyst:    637 ex / 16 × 10 epochs = ~398 steps
#   planner:    914 ex / 16 × 8 epochs  = ~457 steps
#   researcher: 1992 ex / 16 × 4 epochs = ~498 steps
SOURCE_MAP = {
    # source   (data path,                        output dir,                    epochs, seq_len)
    "exploitdb": ("processed/exploitdb.jsonl",    "outputs/mythos-v4-exploitdb",  3,   2048),
    "htb":       ("processed/htb_writeups.jsonl", "outputs/mythos-v4-htb",        3,   2560),
    "vulhub":    ("processed/vulhub.jsonl",        "outputs/mythos-v4-vulhub",     4,   2048),  # best at epoch 3 (0.5236); 4 gives headroom
    "attack":    ("processed/attack.jsonl",        "outputs/mythos-v4-attack",     5,   1280),
    "ad":        ("processed/ad.jsonl",            "outputs/mythos-v4-ad",        12,   1024),
    "webapp":    ("processed/webapp.jsonl",        "outputs/mythos-v4-webapp",    10,   1792),
    "osint":     ("processed/osint.jsonl",         "outputs/mythos-v4-osint",     12,   1024),
    "cloud":     ("processed/cloud.jsonl",         "outputs/mythos-v4-cloud",     14,   1280),
    "executor":  ("processed/executor.jsonl",      "outputs/mythos-v4-executor",  10,   1024),  # command correction + output filtering
    "analyst":   ("processed/analyst.jsonl",       "outputs/mythos-v4-analyst",   10,   1024),  # HackerOne reports + tool interpretation
    "planner":   ("processed/planner.jsonl",       "outputs/mythos-v4-planner",    8,   1024),  # goal decomposition + replanning
    "researcher":   ("processed/researcher.jsonl",    "outputs/mythos-v4-researcher",    4,   1024),  # novelty discovery (3 subtypes merged)
    "tool_operator":("processed/tool_operator.jsonl", "outputs/mythos-v4-tool_operator", 8,   1280),  # first-principles tool use + flag correction
    "combined":     ("processed/combined.jsonl",      "outputs/mythos-v4-combined",      3,   2048),
}

# Resolve source: explicit --source wins; fall back to detecting from --data; else default
if args.source:
    _data_default, _out_default, _epoch_default, _seq_default = SOURCE_MAP[args.source]
elif args.data:
    _data_default  = args.data
    _out_default   = "outputs/mythos-v4-custom"
    _epoch_default = 3
    _seq_default   = 3000
else:
    parser.error("Specify --source (exploitdb | htb | vulhub | attack | ad | webapp | osint | cloud | executor | analyst | planner | researcher | tool_operator | combined).")

DATA_PATH   = args.data   or _data_default
OUTPUT_DIR  = args.output or _out_default

if args.epochs is None:
    EPOCHS = _epoch_default
else:
    EPOCHS = args.epochs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SEQ_LEN   = _seq_default   # per-source tuned value (see SOURCE_MAP comments)
LORA_R        = 64
LORA_ALPHA    = 128        # = 2 × r (standard scaling)
LORA_DROPOUT  = 0          # 0 = full Unsloth patching on LoRA matrices (faster).
                           # Non-zero dropout blocks Unsloth from patching those
                           # layers, causing a measurable performance hit.
                           # LoRA regularisation comes from rank constraint, not dropout.
EVAL_SPLIT    = 0.05       # 5% held-out eval
BATCH_SIZE    = 4          # per-device
GRAD_ACCUM    = 4          # effective batch = 16
WARMUP_STEPS  = 32         # fixed steps — warmup_ratio is deprecated in TRL v5

# Target all attention + MLP projection layers
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ---------------------------------------------------------------------------
# Imports (after arg parse)
# ---------------------------------------------------------------------------
import torch
from datasets import Dataset

print(f"\n{'═'*60}")
print(f"  Mythos-v4 Fine-tuning  —  Adapter: {args.source or 'custom'}")
print(f"  Model  : {MODEL_NAME}")
print(f"  Data   : {DATA_PATH}")
print(f"  Epochs : {EPOCHS}")
print(f"  LR     : {args.lr}")
print(f"  Output : {OUTPUT_DIR}")
print(f"{'═'*60}\n")

print("Loading Unsloth...")
from unsloth import FastLanguageModel

# ---------------------------------------------------------------------------
# Load base model — full BF16, no quantization
# ---------------------------------------------------------------------------
if args.qlora:
    print(f"Loading {MODEL_NAME} in 4-bit NF4 (QLoRA — single A100 mode)...")
    print("  Note: base weights quantized to 4-bit; LoRA updates remain in BF16.")
    print("  Use this only when 2×A100 is unavailable for 32B.\n")
else:
    print(f"Loading {MODEL_NAME} in BF16 (full precision)...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = torch.bfloat16,
    load_in_4bit    = args.qlora,  # True only for 32B on single A100
)

print(f"  Parameters : {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
print(f"  VRAM used  : {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# ---------------------------------------------------------------------------
# LoRA adapters
# ---------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r                        = LORA_R,
    target_modules           = LORA_TARGETS,
    lora_alpha               = LORA_ALPHA,
    lora_dropout             = LORA_DROPOUT,  # 0 = full Unsloth patching
    bias                     = "none",
    use_gradient_checkpointing = "unsloth",  # Unsloth's optimised GC (30% more VRAM headroom)
    random_state             = args.seed,
    use_rslora               = False,
    loftq_config             = None,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\n  LoRA rank     : {LORA_R}  alpha={LORA_ALPHA}")
print(f"  Trainable     : {trainable/1e6:.1f}M / {total/1e9:.1f}B  ({100*trainable/total:.2f}%)")
print(f"  VRAM after LoRA: {torch.cuda.memory_allocated() / 1e9:.1f} GB\n")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
print(f"Loading dataset from {DATA_PATH}...")

with open(DATA_PATH) as f:
    raw = [json.loads(l) for l in f if l.strip()]

# Shuffle with fixed seed for reproducibility
random.shuffle(raw)

# Log source + category distribution before split
from collections import defaultdict
sources = Counter(e["metadata"].get("source", "unknown") for e in raw)
cats    = Counter(e["metadata"].get("category", "unknown") for e in raw)

print(f"  Total examples : {len(raw)}")
print()
print("  Source breakdown:")
for src, n in sources.most_common():
    bar = "█" * int(20 * n / max(sources.values()))
    print(f"    {src:<25} {n:>5}  {bar}")

print()
print("  Category distribution:")
for cat, n in sorted(cats.items()):
    bar = "█" * int(20 * n / max(cats.values()))
    print(f"    {cat:<35} {n:>4}  {bar}")

# Train / eval split
split_idx  = int(len(raw) * (1 - EVAL_SPLIT))
train_data = raw[:split_idx]
eval_data  = raw[split_idx:]
print(f"\n  Train : {len(train_data)}  |  Eval : {len(eval_data)}\n")

# ---------------------------------------------------------------------------
# Format messages using the model's native chat template
# Loss is computed ONLY on the assistant turn (system + user tokens are masked).
# Unsloth's SFTTrainer handles this automatically when messages are provided.
# ---------------------------------------------------------------------------
def format_example(example: dict) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize           = False,
        add_generation_prompt = False,
    )
    return {"text": text}

train_dataset = Dataset.from_list(train_data).map(format_example, remove_columns=["messages", "metadata"])
eval_dataset  = Dataset.from_list(eval_data).map(format_example, remove_columns=["messages", "metadata"])

# Spot-check token lengths
sample_lens = [
    len(tokenizer(ex["text"], return_tensors="pt").input_ids[0])
    for ex in train_dataset.select(range(min(50, len(train_dataset))))
]
import statistics
print(f"Token length check (50 samples):")
print(f"  min={min(sample_lens)}  median={statistics.median(sample_lens):.0f}  max={max(sample_lens)}")
over_limit = sum(1 for l in sample_lens if l > MAX_SEQ_LEN)
print(f"  Over {MAX_SEQ_LEN}: {over_limit}/50\n")

# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------
steps_per_epoch = math.ceil(len(train_dataset) / (BATCH_SIZE * GRAD_ACCUM))
total_steps     = steps_per_epoch * EPOCHS

print(f"Training config:")
print(f"  steps/epoch      : {steps_per_epoch}")
print(f"  total steps      : {total_steps}")
print(f"  effective batch  : {BATCH_SIZE * GRAD_ACCUM}")
print(f"  warmup steps     : {WARMUP_STEPS}\n")

from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir               = OUTPUT_DIR,
    num_train_epochs         = EPOCHS,

    # Batch
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = 2,
    gradient_accumulation_steps = GRAD_ACCUM,

    # Optimiser — 8-bit AdamW keeps optimizer states in 8-bit without
    # touching the model weights (which remain BF16). No precision loss
    # on the forward/backward pass.
    optim                    = "adamw_8bit",
    learning_rate            = args.lr,
    lr_scheduler_type        = "cosine",
    warmup_steps             = WARMUP_STEPS,  # replaces deprecated warmup_ratio
    weight_decay             = 0.01,
    max_grad_norm            = 1.0,

    # Precision — BF16 throughout, no FP16 (BF16 has wider range, safer on A100)
    bf16                     = True,
    fp16                     = False,

    # Sequence
    max_seq_length           = MAX_SEQ_LEN,
    # Packing: fills each 2048-token window with multiple short examples.
    # Doubles effective throughput since median is ~1029 tokens.
    # Disabled here to keep per-example loss clean; enable for speed if eval
    # loss looks noisy.
    packing                  = False,

    # Eval & checkpointing
    # save_total_limit=1 + load_best_model_at_end=True: HuggingFace guarantees
    # the best checkpoint is never deleted, so at most 2 dirs exist on disk at
    # any time (latest + best if they differ). After training completes, we copy
    # the best adapter out and delete all checkpoint dirs entirely.
    eval_strategy            = "steps",
    eval_steps               = steps_per_epoch,   # eval once per epoch
    save_strategy            = "steps",
    save_steps               = steps_per_epoch,
    save_total_limit         = 1,                 # only keep 1 checkpoint; best is always preserved
    load_best_model_at_end   = True,
    metric_for_best_model    = "eval_loss",
    greater_is_better        = False,

    # Logging
    logging_steps            = 10,
    report_to                = "none",             # set "wandb" if you have it
    seed                     = args.seed,
    dataset_text_field       = "text",
)

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = train_dataset,
    eval_dataset    = eval_dataset,
    args            = training_args,
)

# ---------------------------------------------------------------------------
# Pre-training VRAM snapshot
# ---------------------------------------------------------------------------
torch.cuda.synchronize()
vram_before = torch.cuda.memory_allocated() / 1e9
vram_reserved = torch.cuda.memory_reserved() / 1e9
print(f"VRAM before training: {vram_before:.1f} GB allocated / {vram_reserved:.1f} GB reserved")
print(f"VRAM headroom: {(80 - vram_reserved):.1f} GB free\n")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print("Starting training...\n")
trainer_stats = trainer.train()

# ---------------------------------------------------------------------------
# Save best adapter — load_best_model_at_end already restored best weights
# ---------------------------------------------------------------------------
import shutil

adapter_path = os.path.join(OUTPUT_DIR, "best-adapter")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"\nBest adapter saved → {adapter_path}")

# Delete all intermediate checkpoint dirs — keep only best-adapter/
# Merged model is NOT saved here; create it on-demand with merge.py when needed.
ckpt_dirs = [
    d for d in Path(OUTPUT_DIR).iterdir()
    if d.is_dir() and d.name.startswith("checkpoint-")
]
for ckpt in ckpt_dirs:
    shutil.rmtree(ckpt)
    print(f"  Removed checkpoint dir: {ckpt.name}")
print(f"  Disk usage: {sum(f.stat().st_size for f in Path(adapter_path).rglob('*') if f.is_file()) / 1e6:.0f} MB")

# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------
elapsed_min = trainer_stats.metrics["train_runtime"] / 60
print(f"\n{'═'*60}")
print(f"  TRAINING COMPLETE")
print(f"{'═'*60}")
print(f"  Runtime        : {elapsed_min:.1f} min")
print(f"  Train loss     : {trainer_stats.metrics['train_loss']:.4f}")
print(f"  Samples/sec    : {trainer_stats.metrics['train_samples_per_second']:.1f}")
print(f"  Best adapter   : {adapter_path}")
print(f"  Cost estimate  : ~${elapsed_min / 60 * 1.807:.2f} at $1.807/hr")
print(f"{'═'*60}\n")
print("To merge into a single model when needed:")
print(f"  python3 merge.py --source {args.source or 'htb'}")
