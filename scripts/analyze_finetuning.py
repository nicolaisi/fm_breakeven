"""
Fine-tuning analysis: full fine-tuning vs LoRA vs zero-shot.

Produces the numbers cited in Finding 3 of the paper:
  - Full fine-tuning beats LoRA on N/30 datasets (at full training data)
  - Full fine-tuning beats zero-shot on N/30 datasets (at full training data)
  - Catastrophic failure cases (MASE inflated 3-20x) at low training fractions
  - Seed-level variance (instability) by setting

Run from: ~/Development/fm_breakeven/
Usage: python scripts/analyze_finetuning.py
"""

import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from results_io import load_results

# ── Load data ──────────────────────────────────────────────────────────────────
_, fm = load_results()

print("Settings:", fm['setting'].unique().tolist())
print(f"Total rows: {len(fm)}")
print()

# ── 1. Best MASE per dataset at full training data ─────────────────────────────
full = fm[(fm['setting'] == 'fine-tuned-full') & (fm['train_fraction'] == 1.0)]
lora = fm[(fm['setting'] == 'fine-tuned-lora') & (fm['train_fraction'] == 1.0)]
zs   = fm[(fm['setting'] == 'zero-shot')       & (fm['train_fraction'] == 1.0)]

# Mean over seeds per model, then the best model (as in src/results_io.best_zero_shot)
best = lambda d: d.groupby(['dataset', 'model'])['mase_fixed'].mean().groupby('dataset').min()
full_best = best(full).rename('full_best')
lora_best = best(lora).rename('lora_best')
zs_best   = best(zs).rename('zs_best')

cmp = pd.concat([full_best, lora_best, zs_best], axis=1).round(3)
cmp['full_beats_lora'] = cmp['full_best'] < cmp['lora_best']
cmp['full_beats_zs']   = cmp['full_best'] < cmp['zs_best']

print("=" * 70)
print("1. PER-DATASET COMPARISON AT FULL TRAINING DATA (train_fraction=1.0)")
print("=" * 70)
print(cmp.to_string())
print()
print(f"Full fine-tuning beats LoRA:       {cmp['full_beats_lora'].sum()} / {len(cmp)} datasets")
print(f"Full fine-tuning beats zero-shot:  {cmp['full_beats_zs'].sum()} / {len(cmp)} datasets")
print()

# ── 2. Seed-level variance (instability proxy) ─────────────────────────────────
print("=" * 70)
print("2. SEED-LEVEL VARIANCE BY SETTING (instability proxy)")
print("=" * 70)
var = fm[fm['train_fraction'] == 1.0].groupby(
    ['dataset', 'model', 'setting']
)['mase'].std().reset_index(name='seed_std')

print("Mean seed std by setting:")
print(var.groupby('setting')['seed_std'].agg(['mean', 'median', 'max']).round(4))
print()

# ── 3. Catastrophic failures (MASE > 10) at low training fractions ─────────────
print("=" * 70)
print("3. CATASTROPHIC FAILURES (MASE > 10) AT LOW FRACTIONS (<=0.05)")
print("=" * 70)
low = fm[(fm['setting'] == 'fine-tuned-full') & (fm['train_fraction'] <= 0.05)]
failures = low[low['mase'] > 10].sort_values('mase', ascending=False)

print(f"Total catastrophic cases: {len(failures)}")
print()
print("By model family:")
failures['family'] = failures['model'].apply(
    lambda x: 'Moirai' if 'Moirai' in x else ('Chronos' if 'Chronos' in x else 'Lag-Llama')
)
print(failures.groupby('family')['mase'].agg(['count', 'mean', 'max']).round(3))
print()
print("Top 20 worst cases:")
print(failures[['dataset', 'model', 'train_fraction', 'mase']].head(20).to_string(index=False))
print()

# ── 4. Inflation ratio vs zero-shot ───────────────────────────────────────────
print("=" * 70)
print("4. MASE INFLATION RATIO (full fine-tune / zero-shot) AT LOW FRACTIONS")
print("=" * 70)
zs_ref = best(zs).rename('zs_mase')

# Best full fine-tune per dataset at low fraction (seed mean per model and
# fraction, fixed denominator so it is comparable with zero-shot)
low_best = low.groupby(['dataset', 'model', 'train_fraction'])['mase_fixed'].mean() \
              .groupby('dataset').min().rename('full_low_best')
inflation = pd.concat([low_best, zs_ref], axis=1).dropna()
inflation['inflation_ratio'] = (inflation['full_low_best'] / inflation['zs_mase']).round(2)
inflation = inflation.sort_values('inflation_ratio', ascending=False)

print(inflation.to_string())
print()
print(f"Datasets with >3x inflation: {(inflation['inflation_ratio'] > 3).sum()}")
print(f"Max inflation: {inflation['inflation_ratio'].max():.1f}x ({inflation['inflation_ratio'].idxmax()})")
