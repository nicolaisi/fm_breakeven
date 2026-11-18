# Preprocessing and Evaluation Decisions

Every data and evaluation decision in the benchmark, as implemented in `src/data_loader.py`, `scripts/run_classical.py` and `scripts/run_foundation.py` (paper Sec. III).

---

## Multi-Series Datasets

The Monash collections and M4 contain many series. Each is reduced to one representative series:

1. Take up to N series spaced evenly through the collection: N = 50 by default, 100 for M4-Monthly, 200 for M4-Weekly. (For the M4 CSVs, series shorter than 30 points are dropped first.)
2. Drop series shorter than the 25th percentile of their lengths.
3. Truncate the rest to their most recent 25th-percentile length.
4. Z-score each series and average them.

Table I of the paper reports lengths after this aggregation.

### Rideshare

The averaged series contains about 200 constant hourly values (COVID-era zero padding in the source data). Constant stretches of at least two seasonal periods are removed before the split, leaving 329 of 541 points. Otherwise the smallest training windows are constant, which makes the MASE denominator zero and the classical fits degenerate.

---

## Split and Test Windows

All 30 datasets use a **chronological 60/20/20 split**:

- **First 60%:** training pool. Training sets are its most recent 2%, 5%, 10%, 20%, 50% or 100%.
- **Middle 20%:** never used for training or model selection, but provided as context at forecast time.
- **Final 20%:** test segment. Up to five evenly spaced, non-overlapping windows of length h (h capped at the test-segment length), each forecast from all history directly before it.

Each series is z-scored with the mean and standard deviation of the training pool. The nine shortest datasets get one or two test windows; the others get five.

**Forecast context.** Every model conditions on all history before the test window, up to its own context limit (Chronos-Bolt 2,048, Moirai 512, Lag-Llama its 32-step context plus lags up to 1,092). Classical methods are trained only on the current fraction. Zero-shot FMs are therefore independent of the fraction and run once per dataset.

---

## Minimum Training Sizes

- **Classical methods** run only if the training set has at least max(m, 15) samples. Smaller configurations are recorded as errors ("Insufficient data") and omitted; this removes the lowest fractions of the short datasets (528 of 2,160 classical runs).
- **ETS and ARIMA** use season length min(m, 24) and are fitted on the most recent min(n, 200) samples.
- **Fine-tuned FMs** use the most recent max(2m, ⌊f·n⌋) samples at fraction f. If that is too short for one context window plus the horizon, the run is recorded as an error rather than scored as zero-shot (306 runs).

---

## Seasonal Periods

| m | Datasets |
|---|---|
| 4 | M3-Quarterly |
| 5 | Exchange (business days) |
| 7 | COVID-Deaths, M4-Daily, NN5-Daily, USBirths |
| 12 | FRED-MD, Hospital, M3-Monthly, M4-Monthly-Monash, Tourism-Monthly |
| 24 | ETTh1, ETTh2, Electricity, KDD-Cup-Hourly, M4-Hourly, Pedestrian, Rideshare, Traffic |
| 48 | AusElectricity |
| 52 | Dominick, ILI, M4-Weekly |
| 96 | ETTm1, ETTm2 |
| 144 | Solar-10min, Weather |
| 365 | Saugeen, Sunspot |
| 1440 | Wind-Farms |

---

## MASE Denominator

The MASE denominator is **fixed to the full training pool** for every fraction, model and setting: the in-sample seasonal-naïve MAE with lag m at frac = 1.0. Only the numerator (forecast error) varies across fractions, so errors are comparable across training sizes. The analysis computes it as `mae / denom_full` (`src/results_io.py`).

Exchange has no meaningful seasonality (S = 0.00); its MASE uses m = 5 and should not be compared directly with other datasets.

---

## Dataset Features

Computed from the training pool only (`scripts/compute_features.py`):

| Feature | Description | Used in classifier |
|---|---|---|
| `n_train` | Training pool length | Yes |
| `seas_strength` (S) | STL seasonal strength with period m | Yes |
| `noise_ratio` (σ_r) | std(Δx) / std(x) | Yes |
| `acf_mean_50` (\|ρ\|₅₀) | Mean \|ρ(k)\| for k = 1..50 | Yes |
| `tau_acf` | First lag with ACF < 0.5, capped at 200 | No (saturates) |
| `hurst` (H) | R/S Hurst exponent | No (saturates near 1) |
| `spec_entropy` (Φ) | Spectral entropy of the normalised periodogram | No |
| `dfa_hurst`, `periodogram_sharpness` | Extra features for the 6-feature check in `feature_analysis.py` | No |

---

## Reproducibility

All preprocessing is deterministic given the raw dataset files:

```bash
./run_all.sh download                      # fetch all 30 datasets
python scripts/smoke_test.py               # every dataset loads and runs
python scripts/check_alignment.py .        # test windows start where their context ends
```
