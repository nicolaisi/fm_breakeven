#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Full fm_breakeven experiment pipeline (30 datasets)
#
# Usage:
#   ./run_all.sh              # run everything (30 datasets)
#   ./run_all.sh classical    # classical only
#   ./run_all.sh foundation   # foundation only
#   ./run_all.sh lagllama     # Lag-Llama zero-shot only
#   ./run_all.sh core         # classical + all FMs zero-shot (enough for every n*)
#   ./run_all.sh zeroshot     # all 6 FMs, zero-shot only
#   ./run_all.sh lora         # Chronos + Moirai, zero-shot + LoRA (no full FT)
#   ./run_all.sh new          # new 20 datasets only (classical + foundation)
#   ./run_all.sh download     # pre-fetch all 30 datasets (run first)
#   ./run_all.sh rq3          # RQ3 LOO analysis (after experiments complete)
#
# Device: FM_DEVICE=auto|cuda|mps|cpu ./run_all.sh ...
#   (default auto: CUDA, then Apple MPS, then CPU)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

FM_DEVICE="${FM_DEVICE:-auto}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Colours
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] $*${NC}"; }
fail() { echo -e "${RED}[$(date +%H:%M:%S)] $*${NC}"; exit 1; }

# ── Dataset lists ─────────────────────────────────────────────────────────────
ORIGINAL_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 Electricity Traffic Weather Exchange M4-Hourly ILI"

NEW_DATASETS="M4-Daily M4-Weekly M4-Monthly-Monash KDD-Cup-Hourly \
              Hospital COVID-Deaths \
              NN5-Daily FRED-MD Dominick \
              Solar-10min AusElectricity Wind-Farms Sunspot Saugeen \
              Pedestrian Rideshare \
              Tourism-Monthly M3-Quarterly M3-Monthly USBirths"

ALL_DATASETS="$ORIGINAL_DATASETS $NEW_DATASETS"

# ---------------------------------------------------------------------------
run_classical() {
    local datasets="${1:-$ALL_DATASETS}"
    log "=== CLASSICAL EXPERIMENTS ==="
    log "Datasets: $datasets"
    log "Models: Naive ETS ARIMA XGBoost"
    log "Fractions: 0.02 0.05 0.10 0.20 0.50 1.00"

    python scripts/run_classical.py \
        --datasets $datasets \
        --fractions 0.02 0.05 0.10 0.20 0.50 1.00 \
        2>&1 | tee "$LOG_DIR/classical_${TIMESTAMP}.log"

    log "Classical experiments complete."
}

# ---------------------------------------------------------------------------
run_foundation() {
    local datasets="${1:-$ALL_DATASETS}"
    log "=== FOUNDATION MODEL EXPERIMENTS (Chronos + Moirai) ==="
    log "Datasets: $datasets"
    log "Models: Chronos-Bolt-S Chronos-Bolt-B Moirai-S Moirai-B Moirai-L"
    log "Extra args: ${*:2}"
    log "Fractions: 0.02 0.05 0.10 0.20 0.50 1.00"

    python scripts/run_foundation.py --device "$FM_DEVICE" \
        --datasets $datasets \
        --models Chronos-Bolt-S Chronos-Bolt-B Moirai-S Moirai-B Moirai-L \
        "${@:2}" \
        --fractions 0.02 0.05 0.10 0.20 0.50 1.00 \
        2>&1 | tee "$LOG_DIR/foundation_${TIMESTAMP}.log"

    log "Foundation experiments complete."
}

# ---------------------------------------------------------------------------
run_lagllama() {
    local datasets="${1:-$ALL_DATASETS}"
    log "=== LAG-LLAMA ZERO-SHOT ==="

    python scripts/run_foundation.py --device "$FM_DEVICE" \
        --datasets $datasets \
        --models Lag-Llama \
        --zero-shot-only \
        --fractions 0.02 0.05 0.10 0.20 0.50 1.00 \
        2>&1 | tee "$LOG_DIR/lagllama_${TIMESTAMP}.log"
}

# ---------------------------------------------------------------------------
run_classical_ili() {
    log "=== CLASSICAL — ILI only (if not already in results) ==="

    python scripts/run_classical.py \
        --datasets ILI \
        --fractions 0.02 0.05 0.10 0.20 0.50 1.00 \
        2>&1 | tee "$LOG_DIR/classical_ili_${TIMESTAMP}.log"
}

# ---------------------------------------------------------------------------
run_download() {
    log "=== PRE-FETCHING ALL 30 DATASETS ==="
    python scripts/download_datasets.py \
        2>&1 | tee "$LOG_DIR/download_${TIMESTAMP}.log"
    log "Download complete. Check log: $LOG_DIR/download_${TIMESTAMP}.log"
}

# ---------------------------------------------------------------------------
run_rq3() {
    log "=== RQ3 FEATURE ANALYSIS (random-forest LOO, correlations) ==="
    python scripts/compute_features.py 2>&1 | tee "$LOG_DIR/features_${TIMESTAMP}.log" || \
        warn "compute_features.py failed — skipping feature step"
    python scripts/feature_analysis.py \
        2>&1 | tee "$LOG_DIR/rq3_${TIMESTAMP}.log"
    log "RQ3 analysis complete."
}

# ---------------------------------------------------------------------------
# Record exactly what produced this run's results (code version, packages, device)
write_manifest() {
    local f="$LOG_DIR/manifest_${TIMESTAMP}.txt"
    {
        echo "mode:      ${1:-all}"
        echo "started:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "commit:    $(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "uncommitted changes:"; git status --short 2>/dev/null | sed 's/^/  /'
        echo "FM_DEVICE: $FM_DEVICE"
        echo "python:    $(python --version 2>&1)"
        python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '| mps:', torch.backends.mps.is_available())" 2>/dev/null
        echo "--- pip freeze ---"; python -m pip freeze 2>/dev/null
    } > "$f"
    log "Run manifest: $f"
}

MODE="${1:-all}"
write_manifest "$MODE"

case "$MODE" in
    all)
        run_classical "$ALL_DATASETS"
        run_foundation "$ALL_DATASETS"
        run_lagllama "$ALL_DATASETS"
        log "=== ALL DONE (30 datasets) ==="
        ;;
    classical)
        run_classical "$ALL_DATASETS"
        ;;
    foundation)
        run_foundation "$ALL_DATASETS"
        ;;
    lagllama)
        run_lagllama "$ALL_DATASETS"
        ;;
    zeroshot)
        run_foundation "$ALL_DATASETS" --zero-shot-only
        run_lagllama "$ALL_DATASETS"
        ;;
    core)
        # CLASSICAL_DATASETS="..." limits the classical stage (e.g. to resume)
        run_classical "${CLASSICAL_DATASETS:-$ALL_DATASETS}"
        run_foundation "$ALL_DATASETS" --zero-shot-only
        run_lagllama "$ALL_DATASETS"
        log "=== CORE DONE: classical + zero-shot (all break-even points) ==="
        ;;
    lora)
        run_foundation "$ALL_DATASETS" --skip-full-ft
        ;;
    ili)
        run_classical_ili
        ;;
    new)
        log "=== NEW 20 DATASETS ONLY ==="
        run_classical "$NEW_DATASETS"
        run_foundation "$NEW_DATASETS"
        run_lagllama "$NEW_DATASETS"
        log "=== NEW DATASETS DONE ==="
        ;;
    download)
        run_download
        ;;
    rq3)
        run_rq3
        ;;
    *)
        fail "Unknown mode: $MODE. Use: all | classical | foundation | lagllama | zeroshot | core | lora | ili | new | download | rq3"
        ;;
esac
