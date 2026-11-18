# Foundation Model Break-Even Analysis

**When Do Foundation Models Pay Off? A Break-Even Analysis of Pretrained Time Series Forecasters**

[LM4TED: Large Models for Time Series Data Mining](https://lm4ted.github.io/), IEEE ICDM 2026 Workshop, Shenyang, China

Code for measuring break-even points — the training data size at which foundation models (Chronos, Moirai, Lag-Llama) start outperforming classical methods (Naive, ETS, ARIMA, XGBoost) across 30 benchmark datasets.

## Quick Start

```bash
# Setup
git clone https://github.com/nicolaisi/fm_breakeven.git
cd fm_breakeven
pip install -r requirements.txt

# 0. Pre-fetch all 30 datasets (recommended before long runs)
./run_all.sh download

# 1. Run everything (classical + foundation + Lag-Llama, all 30 datasets)
./run_all.sh

# 2. Or run selectively
./run_all.sh classical      # CPU only, all 30 datasets
./run_all.sh foundation     # GPU: Chronos-Bolt + Moirai
./run_all.sh lagllama       # GPU: Lag-Llama zero-shot only
./run_all.sh new            # New 20 datasets only (classical + foundation)
./run_all.sh rq3            # RQ3 feature analysis (after experiments complete)
```

Foundation models require a GPU:

```bash
pip install -r requirements-gpu.txt
./run_all.sh foundation
```

## Datasets

30 datasets total: 10 original + 20 new.

### Original 10

| Dataset | Frequency | Horizon | Source |
|---------|-----------|---------|--------|
| ETTh1 | Hourly | 24 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| ETTh2 | Hourly | 24 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| ETTm1 | 15-min | 96 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| ETTm2 | 15-min | 96 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| Electricity | Hourly | 24 | [thuml/Time-Series-Library](https://huggingface.co/datasets/thuml/Time-Series-Library) |
| Traffic | Hourly | 24 | [thuml/Time-Series-Library](https://huggingface.co/datasets/thuml/Time-Series-Library) |
| Weather | 10-min | 720 | [thuml/Time-Series-Library](https://huggingface.co/datasets/thuml/Time-Series-Library) |
| Exchange | Daily | 30 | [thuml/Time-Series-Library](https://huggingface.co/datasets/thuml/Time-Series-Library) |
| M4-Hourly | Hourly | 24 | [M4 Competition](https://github.com/Mcompetitions/M4-methods) |
| ILI | Weekly | 24 | [thuml/Time-Series-Library](https://huggingface.co/datasets/thuml/Time-Series-Library) |

### New 20

| Dataset | Frequency | Horizon | Source |
|---------|-----------|---------|--------|
| M4-Daily | Daily | 14 | M4 Competition |
| M4-Weekly | Weekly | 13 | M4 Competition |
| M4-Monthly-Monash | Monthly | 18 | [Monash](https://forecastingdata.org) |
| M3-Monthly | Monthly | 18 | M3 Competition |
| M3-Quarterly | Quarterly | 8 | M3 Competition |
| Hospital | Monthly | 12 | Monash |
| COVID-Deaths | Daily | 30 | Monash |
| NN5-Daily | Daily | 56 | Monash |
| FRED-MD | Monthly | 12 | Monash |
| Dominick | Weekly | 8 | Monash |
| KDD-Cup-Hourly | Hourly | 48 | Monash |
| Solar-10min | 10-min | 60 | Monash |
| AusElectricity | 30-min | 48 | Monash |
| Wind-Farms | 1-min | 60 | Monash |
| Sunspot | Daily | 30 | Monash |
| Saugeen | Daily | 30 | Monash |
| Pedestrian | Hourly | 24 | Monash |
| Rideshare | Hourly | 24 | Monash |
| Tourism-Monthly | Monthly | 24 | Monash |
| USBirths | Daily | 30 | Monash |

All datasets download automatically on first use. Run `./run_all.sh download` to pre-fetch.

## Models

**Classical (CPU):**
- Naive (Seasonal Naive)
- ETS (AutoETS via statsforecast)
- ARIMA (AutoARIMA via statsforecast)
- XGBoost (with lag features)

**Foundation (GPU):**
- Chronos-Bolt S/B (Amazon) — zero-shot, fine-tuned-full, fine-tuned-LoRA
- Moirai S/B/L (Salesforce) — zero-shot, fine-tuned-full, fine-tuned-LoRA
- Lag-Llama — zero-shot only

## Experiment Configuration

```python
# Training fractions (% of full training set)
TRAIN_FRACTIONS = [0.02, 0.05, 0.10, 0.20, 0.50, 1.00]

# Random seeds for variance estimation
SEEDS = [0, 1, 2]

# Split: 60% train / 20% val / 20% test (chronological)
```

## Output Files

```
results/
├── classical_rerun.csv           # Classical results used in the paper (all 30 datasets)
├── foundation_rerun_*.csv        # FM results used in the paper (zero-shot, Lag-Llama, LoRA, full fine-tuning)
├── classical_results_<ts>.csv    # Raw output of a run_classical.py run (git-ignored)
├── foundation_raw_<ts>.csv       # Raw output of a run_foundation.py run (git-ignored)
├── dataset_features.csv          # Features: τ_acf, Hurst (R/S), noise ratio, seasonal strength, spectral entropy, |ρ|₅₀
├── summary_breakeven.csv         # Break-even category and n* per dataset
├── fig1_scaling_curves.*         # Fig. 1: scaling curves per dataset
├── fig2_feature_space.*          # Fig. 3 in the paper: feature space
├── fig3_finetuning.*             # Fig. 2 in the paper: zero-shot vs. LoRA
└── archive/                      # Superseded results (old Lag-Llama call, hourly Wind-Farms config, Mac/MPS rerun)

logs/                             # Run logs and run manifests (git-ignored)
```

`scripts/merge_rerun.py` swaps one dataset's rows in the `*_rerun.csv` files for a new run, archiving the old rows.

## Repository Structure

```
fm-breakeven/
├── run_all.sh                        # Main entry point
├── plot_fig1.py                      # Figure 1: scaling curves
├── plot_fig2.py                      # Figure 2: feature space
├── plot_fig3.py                      # Figure 3: fine-tuning
├── src/
│   ├── data_loader.py                # Dataset loading (auto-download, 30 datasets)
│   ├── results_io.py                 # Loads the rerun CSVs; break-even detection
│   ├── models.py                     # Classical + foundation model wrappers
│   ├── config.py                     # Experiment configuration
│   ├── features.py                   # Feature extraction (τ_acf, H, S, Φ)
│   ├── analysis.py                   # Scaling curve fitting, break-even computation
│   ├── experiments.py                # Experiment runner
│   └── visualization.py             # Plotting utilities
├── scripts/
│   ├── run_classical.py              # Run classical experiments (CPU)
│   ├── run_foundation.py             # Run foundation model experiments (GPU)
│   ├── run_quick.py                  # Minimal run for installation verification
│   ├── smoke_test.py                 # Sanity check all 30 datasets load and run
│   ├── analyze_results.py            # Aggregate results and generate figures
│   ├── analyze_finetuning.py         # Fine-tuning analysis: full vs LoRA vs zero-shot
│   ├── validate_findings.py          # Checks every result-dependent number in the paper tex
│   ├── compute_features.py           # Dataset feature extraction script
│   ├── download_datasets.py          # Pre-fetch all 30 datasets
│   ├── feature_analysis.py           # RQ3: random-forest LOO and feature correlations on the break-even categories
│   ├── merge_rerun.py                # Replace one dataset's rows with a new run
│   ├── check_alignment.py            # Sanity check: test windows start where their context ends
│   └── summarize_results.py          # Writes results/summary_breakeven.csv
├── preprocessing/README.md           # Every data and evaluation decision
├── checklist/                        # ML reproducibility checklist
├── requirements.txt                  # CPU dependencies
└── requirements-gpu.txt              # GPU dependencies (Chronos, Moirai, Lag-Llama)
```

## Reproducing Paper Findings

Run `scripts/validate_findings.py` to recompute every result-dependent number in the paper from the rerun CSVs and compare it with the reported value:

```bash
python scripts/validate_findings.py                 # 48 value checks (about 3 minutes)
python scripts/validate_findings.py --tex paper.tex # 63 checks with the paper source
```

The checks cover the categories and break-even ranges, Findings 1–5, seed variance, the RQ3 classifier (including its range over 20 forest seeds) and correlations, and the decision rule. Given the paper source, the script also confirms that each claiming sentence appears verbatim and checks Tables I–III row by row. Exits with code 0 if all pass, 1 if any fail.

## Fine-Tuning Analysis

Run `scripts/analyze_finetuning.py` to reproduce the numbers cited in **Finding 3** of the paper:

```bash
python scripts/analyze_finetuning.py
```

This script produces four sections:

| Section | What it computes |
|---|---|
| 1. Per-dataset comparison | Best MASE for full / LoRA / zero-shot at `train_fraction=1.0`; counts how often full beats LoRA and zero-shot |
| 2. Seed-level variance | Mean/median/max std across seeds per setting — proxy for instability |
| 3. Catastrophic failures | Cases with MASE > 10 at low fractions (≤5%), broken down by model family |
| 4. Inflation ratio | Per-dataset ratio of best full fine-tune MASE (at low fraction) vs zero-shot MASE |

**Key findings from the analysis:**
- Full fine-tuning beats LoRA and zero-shot on only **9/30** datasets each at full training data
- At low fractions (≤5%), **6 runs** with MASE > 10, all Moirai-L on Hospital (20.1–26.0 vs zero-shot 1.22)
- LoRA has lower seed std (0.058 vs 0.095 at frac≤5%) — more stable in the low-data regime

## Key Results

All numbers below are verified by `scripts/validate_findings.py`.

| Finding | Result | Command to reproduce |
|---|---|---|
| FM zero-shot beats classical (frac=1.0) | 18/30 datasets | `python scripts/validate_findings.py` |
| Categories: FM-dominant / Early / Late | 16 / 6 / 8 datasets | `python scripts/validate_findings.py` |
| LoRA improves over zero-shot at full data | 16/30 datasets (>5% better on 3, >5% worse on 11) | `python scripts/validate_findings.py` |
| Full fine-tuning beats LoRA / zero-shot at full data | 9/30 each | `python scripts/validate_findings.py` |
| Best classical at full data | ARIMA 11, Naive 7, ETS 6, XGBoost 6 | `python scripts/validate_findings.py` |
| Random forest LOO on 4 features | 40.0% (40.0–43.3% over 20 forest seeds; majority baseline 53.3%) | `python scripts/feature_analysis.py` |
| Data starvation rule settles the low-data choice | 10/30 datasets (final 5, provisional 5) | `python scripts/validate_findings.py` |

```bash
# Reproduce all findings in one command
python scripts/validate_findings.py
# Expected output: 48/48 checks passed
```

## GPU Requirements

Foundation models need:
- CUDA-enabled GPU (16+ GB VRAM recommended)
- ~4 GPU-hours for the full experiment suite on an RTX 5090 (Chronos/Moirai zero-shot, LoRA and full fine-tuning; Lag-Llama under a minute)

### Per-Model Runtime (RTX 5090, mean per dataset-fraction-seed)

| Model | Zero-shot inference | LoRA fine-tune | Total GPU-hours (all 30 datasets) |
|---|---|---|---|
| Chronos-Bolt-S | <1s | ~1.6s | 0.23h |
| Chronos-Bolt-B | <1s | ~2.9s | 0.44h |
| Moirai-S | ~0.1s | ~1.5s | 0.23h |
| Moirai-B | ~0.1s | ~2.5s | 0.37h |
| Moirai-L | ~0.2s | ~4.3s | 0.65h |
| Lag-Llama | ~0.2s | N/A | <0.01h |

Runtimes measured from `duration_s` column in raw results CSVs. Full fine-tuning adds ~50% over LoRA.

### VRAM Requirements

| Model | Inference | Fine-tuning |
|-------|-----------|-------------|
| Chronos-Bolt-S | ~1 GB | ~3 GB |
| Chronos-Bolt-B | ~2 GB | ~6 GB |
| Moirai-S | <1 GB | ~2 GB |
| Moirai-B | ~1 GB | ~4 GB |
| Moirai-L | ~2 GB | ~8 GB |
| Lag-Llama | <1 GB | N/A |

**Consumer GPUs (RTX 4090/5090 with 24-32GB) can run everything, including fine-tuning.**

### Suggested Workflow

```bash
# 1. Test single model + dataset first (~10 min)
python scripts/run_foundation.py --gpu \
    --datasets ETTh1 \
    --models Chronos-Bolt-S

# 2. Expand to more models
python scripts/run_foundation.py --gpu \
    --datasets ETTh1 Exchange \
    --models Chronos-Bolt-S Chronos-Bolt-B Moirai-S

# 3. Full run (~4 GPU-hours on an RTX 5090, plus ~1 CPU-hour for classical)
./run_all.sh
```

Only use a compute cluster if you need results faster. RTX 5090 ≈ 80% A100 speed for this workload.

## Citation

```bibtex
@inproceedings{jerome2026breakeven,
  title={When Do Foundation Models Pay Off?
         A Break-Even Analysis of Pretrained Time Series Forecasters},
  author={Nicholas Tan Jerome and Frank Simon},
  booktitle={IEEE International Conference on Data Mining Workshops (ICDMW),
             LM4TED: Large Models for Time Series Data Mining},
  address={Shenyang, China},
  year={2026},
  url={https://lm4ted.github.io/}
}
```

## License

MIT
