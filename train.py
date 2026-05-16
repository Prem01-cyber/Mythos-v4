#!/usr/bin/env python3
"""
Mythos-v4 Fine-tuning Script
Model  : Qwen/Qwen2.5-Coder-14B-Instruct
Stack  : Unsloth + Flash Attention 2 + TRL SFTTrainer
Target : A100 SXM 80 GB

No quantization — full BF16 weights + LoRA adapters.
Flash Attention is mathematically identical to standard attention;
no precision loss, just memory efficiency and ~2x speed.

Dataset (combined):
  processed/exploitdb.jsonl   Source 1 — exploit code + reasoning (2143 examples)
  raw/htb_writeups.jsonl      Source 2 — HTB multi-turn methodology (174 examples)
  → combined into processed/combined.jsonl by prepare_data.py

Architecture: Specialized LoRA Adapters (one per source)
  Each adapter is trained independently from the same frozen base model.
  This eliminates catastrophic forgetting entirely — base weights never change.
  Adapters are swapped at inference time based on task type:
    • Adapter 1 (exploitdb)   → exploit code generation, CVE analysis
    • Adapter 2 (htb)         → multi-turn pentest methodology, HTB chains
    • Adapter N (future)      → additional specializations (Source 3, 4, ...)

  --source flag selects which dataset this adapter trains on.
  Output path automatically namespaced: outputs/mythos-v4-{source}/

Usage:
    python3 train.py --source exploitdb     # Adapter 1 — exploit code
    python3 train.py --source htb           # Adapter 2 — pentest methodology
    python3 train.py --source combined      # combined (for ablation comparison)
    python3 train.py --model 7b --source htb
    python3 train.py --epochs 5 --source exploitdb
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
parser.add_argument("--model",   default="14b",
                    choices=["7b", "14b"],
                    help="Model size (default: 14b)")
parser.add_argument("--source",  default=None,
                    choices=["exploitdb", "htb", "combined"],
                    help="Which dataset to train on. "
                         "exploitdb → exploit adapter; "
                         "htb → methodology adapter; "
                         "combined → ablation only")
parser.add_argument("--data",    default=None,
                    help="Override dataset path (optional, --source sets this automatically)")
parser.add_argument("--output",  default=None,
                    help="Override output directory (optional, auto-named from --source)")
parser.add_argument("--epochs",  type=int, default=3,
                    help="Training epochs (default: 3)")
parser.add_argument("--lr",      type=float, default=1e-4,
                    help="Peak learning rate (default: 1e-4)")
parser.add_argument("--seed",    type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

MODEL_MAP = {
    "7b":  "Qwen/Qwen2.5-Coder-7B-Instruct",
    "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
}
MODEL_NAME = MODEL_MAP[args.model]

# Source → dataset file + output directory
SOURCE_MAP = {
    "exploitdb": ("processed/exploitdb.jsonl",  "outputs/mythos-v4-exploitdb"),
    "htb":       ("raw/htb_writeups.jsonl",      "outputs/mythos-v4-htb"),
    "combined":  ("processed/combined.jsonl",    "outputs/mythos-v4-combined"),
}

# Resolve source: explicit --source wins; fall back to detecting from --data; else default
if args.source:
    _data_default, _out_default = SOURCE_MAP[args.source]
elif args.data:
    # Infer from data path for backwards compat
    _data_default = args.data
    _out_default  = "outputs/mythos-v4-custom"
else:
    # No --source and no --data → require --source
    parser.error("Specify --source (exploitdb | htb | combined) to select adapter target.")

DATA_PATH   = args.data   or _data_default
OUTPUT_DIR  = args.output or _out_default

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SEQ_LEN   = 2048
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
print(f"  Epochs : {args.epochs}")
print(f"  LR     : {args.lr}")
print(f"  Output : {OUTPUT_DIR}")
print(f"{'═'*60}\n")

print("Loading Unsloth...")
from unsloth import FastLanguageModel

# ---------------------------------------------------------------------------
# Load base model — full BF16, no quantization
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_NAME} in BF16 (no quantization)...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = torch.bfloat16,
    load_in_4bit    = False,   # full precision — A100 80GB has the VRAM
    # Flash Attention 2 is enabled automatically by Unsloth for supported models
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
total_steps     = steps_per_epoch * args.epochs

print(f"Training config:")
print(f"  steps/epoch      : {steps_per_epoch}")
print(f"  total steps      : {total_steps}")
print(f"  effective batch  : {BATCH_SIZE * GRAD_ACCUM}")
print(f"  warmup steps     : {WARMUP_STEPS}\n")

from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir               = OUTPUT_DIR,
    num_train_epochs         = args.epochs,

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

    # Eval
    eval_strategy            = "steps",
    eval_steps               = steps_per_epoch,   # eval once per epoch
    save_strategy            = "steps",
    save_steps               = steps_per_epoch,
    save_total_limit         = 3,                 # keep last 3 checkpoints
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
# Save final adapter
# ---------------------------------------------------------------------------
adapter_path = os.path.join(args.output, "final-adapter")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"\nAdapter saved to: {adapter_path}")

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
print(f"  Adapter        : {adapter_path}")
print(f"  Cost estimate  : ~${elapsed_min / 60 * 1.807:.2f} at $1.807/hr")
print(f"{'═'*60}\n")

# ---------------------------------------------------------------------------
# Optional: merge adapter into base for single-file deployment
# ---------------------------------------------------------------------------
print("Merging adapter into base weights (for single-file deployment)...")
merged_path = os.path.join(args.output, "merged-bf16")
model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
print(f"Merged model saved to: {merged_path}")
