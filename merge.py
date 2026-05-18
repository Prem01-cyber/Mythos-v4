#!/usr/bin/env python3
"""
merge.py — Merge a LoRA best-adapter into its base model on demand.

Why separate from training:
  - Merged models are large (14B = ~30 GB BF16) and rarely needed on disk.
  - During training and normal inference, the adapter is loaded on top of
    the frozen base — no merge required.
  - Only merge when you need: GGUF export, vLLM serving, or offline deployment.

Usage:
    python3 merge.py --source htb
    python3 merge.py --source exploitdb --output /mnt/fast/merged-exploitdb
    python3 merge.py --source htb --gguf q4_k_m    # quantize after merge
    python3 merge.py --list                          # show which adapters exist
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description="Merge best-adapter into base model")
parser.add_argument("--source",  choices=["exploitdb", "htb", "vulhub", "attack",
                                           "ad", "webapp", "osint", "cloud"],
                    help="Which adapter to merge")
parser.add_argument("--output",  default=None,
                    help="Output path for merged model (default: outputs/mythos-v4-{source}/merged)")
parser.add_argument("--model",   default="q3-14b",
                    choices=["q3-8b", "q3-14b", "q3-32b"],
                    help="Base model (must match what was used for training)")
parser.add_argument("--gguf",    default=None,
                    metavar="QUANT",
                    help="Also export to GGUF with this quantization, e.g. q4_k_m, q8_0, f16")
parser.add_argument("--list",    action="store_true",
                    help="List all saved adapters and their sizes, then exit")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Source map (must match train.py SOURCE_MAP)
# ---------------------------------------------------------------------------
SOURCE_MAP = {
    "exploitdb": "outputs/mythos-v4-exploitdb/best-adapter",
    "htb":       "outputs/mythos-v4-htb/best-adapter",
    "vulhub":    "outputs/mythos-v4-vulhub/best-adapter",
    "attack":    "outputs/mythos-v4-attack/best-adapter",
    "ad":        "outputs/mythos-v4-ad/best-adapter",
    "webapp":    "outputs/mythos-v4-webapp/best-adapter",
    "osint":     "outputs/mythos-v4-osint/best-adapter",
    "cloud":     "outputs/mythos-v4-cloud/best-adapter",
}

MODEL_MAP = {
    "q3-8b":  "Qwen/Qwen3-8B",
    "q3-14b": "Qwen/Qwen3-14B",
    "q3-32b": "Qwen/Qwen3-32B",
}

# ---------------------------------------------------------------------------
# --list mode
# ---------------------------------------------------------------------------
if args.list:
    print(f"\n{'Adapter':<12}  {'Path':<45}  {'Size':>8}  Status")
    print("-" * 80)
    for src, path in SOURCE_MAP.items():
        p = Path(path)
        if p.exists():
            size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
            print(f"  {src:<12}  {path:<45}  {size_mb:>6.0f}MB  ✓ ready")
        else:
            print(f"  {src:<12}  {path:<45}  {'—':>8}  not trained yet")
    print()
    sys.exit(0)

if not args.source:
    parser.error("Specify --source or --list")

adapter_path = SOURCE_MAP[args.source]
if not Path(adapter_path).exists():
    print(f"ERROR: No best-adapter found at {adapter_path}")
    print(f"  Run: python3 train.py --source {args.source}")
    sys.exit(1)

output_path = args.output or f"outputs/mythos-v4-{args.source}/merged"
base_model  = MODEL_MAP[args.model]

print(f"\n{'═'*60}")
print(f"  merge.py — {args.source} adapter → merged model")
print(f"{'═'*60}")
print(f"  Adapter   : {adapter_path}")
print(f"  Base      : {base_model}")
print(f"  Output    : {output_path}")
if args.gguf:
    print(f"  GGUF quant: {args.gguf}")
print()

# ---------------------------------------------------------------------------
# Imports (heavy — after arg parsing)
# ---------------------------------------------------------------------------
import torch
print("Loading Unsloth...")
from unsloth import FastLanguageModel

print(f"Loading base model {base_model} in BF16...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = base_model,
    max_seq_length = 4096,
    dtype          = torch.bfloat16,
    load_in_4bit   = False,
)

print(f"Applying LoRA adapter from {adapter_path}...")
model = FastLanguageModel.get_peft_model(model, adapter_path)

# ---------------------------------------------------------------------------
# Merge LoRA weights into base model
# ---------------------------------------------------------------------------
print("Merging LoRA weights into base (this modifies base weights in-place)...")
merged = model.merge_and_unload()

print(f"Saving merged model → {output_path}")
Path(output_path).mkdir(parents=True, exist_ok=True)
merged.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

size_gb = sum(f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()) / 1e9
print(f"  Merged model size: {size_gb:.1f} GB")

# ---------------------------------------------------------------------------
# Optional GGUF export
# ---------------------------------------------------------------------------
if args.gguf:
    gguf_path = f"{output_path}-{args.gguf}.gguf"
    print(f"\nExporting to GGUF ({args.gguf}) → {gguf_path}")
    merged.save_pretrained_gguf(
        gguf_path,
        tokenizer,
        quantization_method = args.gguf,
    )
    gguf_size = Path(gguf_path).stat().st_size / 1e9
    print(f"  GGUF size: {gguf_size:.1f} GB")

print(f"\n{'═'*60}")
print(f"  MERGE COMPLETE")
print(f"  Load with: AutoModelForCausalLM.from_pretrained('{output_path}')")
if args.gguf:
    print(f"  GGUF:      ollama create mythos-{args.source} -f {gguf_path}")
print(f"{'═'*60}\n")
