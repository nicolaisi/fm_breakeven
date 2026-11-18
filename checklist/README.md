# ML Reproducibility Checklist (v2.0)

Assessment of Jerome & Simon (2026), "When Do Foundation Models Pay Off? A Break-Even Analysis of Pretrained Time Series Forecasters" (LM4TED Workshop, IEEE ICDM 2026). The blank checklist is `ReproducibilityChecklist-v2.0.pdf`.

Legend: **Yes** | **Partial** | **No** | **N/A**

Last updated: 2026-11-18

---

## Models and Algorithms

- [x] Clear description of the mathematical setting, algorithm, and/or model — break-even point defined in Eq. 1 (Sec. III-A); fixed-denominator MASE (Sec. III-E3); model settings and LoRA configuration in Sec. III-D and III-E4
- [x] Clear explanation of any assumptions — consolidated in `preprocessing/README.md`: chronological split, middle 20% as forecast context, fixed-denominator MASE, subsampling order, seasonal period per dataset, minimum training sizes
- [ ] ~~Analysis of complexity (time, space, sample size)~~ — **N/A**: empirical benchmark study, no new algorithm proposed

---

## Theoretical Claims

- [ ] ~~Clear statement of the claim~~ — **N/A**: all findings are empirical
- [ ] ~~Complete proof of the claim~~ — **N/A**

---

## Datasets

- [x] Relevant statistics (number of examples) — Table I reports length, frequency, horizon and number of variables for all 30 datasets; Table II reports n_train
- [x] Details of train / validation / test splits — 60/20/20 chronological split with up to five test windows (Sec. III-E2, `preprocessing/README.md`)
- [x] Explanation of excluded data and all preprocessing steps — `preprocessing/README.md`: multi-series aggregation, Rideshare constant-stretch removal, max(m, 15) minimum for classical runs, skipped fine-tuning runs
- [x] Link to downloadable version of the dataset — `./run_all.sh download` fetches all 30; sources listed in the README
- [ ] ~~Description of new data collection process~~ — **N/A**: all datasets are existing public benchmarks

---

## Shared Code

- [x] Specification of dependencies — `requirements.txt` (CPU) and `requirements-gpu.txt` (GPU)
- [x] Training code — `scripts/run_classical.py`, `scripts/run_foundation.py` (entry point `run_all.sh`)
- [x] Evaluation code — `src/results_io.py` (break-even detection), `scripts/feature_analysis.py` (RQ3), `scripts/validate_findings.py` (recomputes every result-dependent number in the paper: 48 value checks, 63 with the paper source)
- [ ] ~~(Pre-)trained model(s)~~ — **N/A**: uses publicly available FMs (Chronos-Bolt, Moirai, Lag-Llama) from Hugging Face
- [x] README file includes results table + exact commands to reproduce — Key Results table in the README; `python scripts/validate_findings.py` reproduces all of them

---

## Experimental Results

- [x] Range of hyperparameters, selection method, and full specification — LoRA: r = 8, α = 16, dropout 0.05, AdamW lr 5·10⁻⁴ (10⁻⁴ for full fine-tuning), weight decay 0.01, steps = min(300, max(50, ⌊n/h⌋)); fractions 2, 5, 10, 20, 50, 100%; XGBoost features and settings in Sec. III-D; ETS/ARIMA use statsforecast AutoETS/AutoARIMA defaults (no tuning, stated as a limitation)
- [x] Exact number of training and evaluation runs — 8,100 runs: 30 datasets × 3 seeds × (4 classical × 6 fractions + 6 zero-shot FMs + 5 LoRA and 5 fully fine-tuned FMs × 6 fractions); 7,266 complete, the rest are configurations too short to fit (`preprocessing/README.md`)
- [x] Clear definition of the specific measure or statistics used — MASE with a fixed denominator (frac = 1.0); seasonal period m per dataset in `preprocessing/README.md`
- [x] Description of results with central tendency and variation — FM results are means over 3 seeds; seed standard deviations in Sec. III-E3; Fig. 1 LoRA curves show ±1 std bands; RQ3 classifier reported with its range over 20 forest seeds
- [x] Average runtime for each result or estimated energy cost — per-model runtime table in the README; about 4 GPU-hours in total (Sec. III-E4)
- [x] Description of computing infrastructure — NVIDIA RTX 5090 (32 GB GDDR7) for FMs in bfloat16; CPU for classical methods (Sec. III-E4)

---

## Summary

| Category | ✅ Yes | N/A |
|---|---|---|
| Models & Algorithms | 2 | 1 |
| Theoretical Claims | 0 | 2 |
| Datasets | 4 | 1 |
| Shared Code | 4 | 1 |
| Experimental Results | 6 | 0 |
| **Total** | **16 / 16** | **5** |

All applicable items checked. No outstanding items.
