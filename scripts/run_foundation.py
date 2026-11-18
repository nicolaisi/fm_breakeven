#!/usr/bin/env python3
"""
Run foundation model experiments (requires GPU).

This runs:
- 8 datasets
- 6 foundation models (Chronos-Bolt S/B, Moirai S/B/L, Lag-Llama)
- 2 settings per model (zero-shot, fine-tuned)
- 5 training fractions
- 3 random seeds

Prerequisites:
- NVIDIA GPU (CUDA, 16+ GB VRAM) recommended; Apple Silicon (MPS) and CPU
  also work (zero-shot is fast; fine-tuning is much slower)
- pip install chronos-forecasting
- pip install git+https://github.com/SalesforceAIResearch/uni2ts.git
- pip install git+https://github.com/time-series-foundation-models/lag-llama.git

Usage:
    python scripts/run_foundation.py
    python scripts/run_foundation.py --models Chronos-Bolt-S Chronos-Bolt-B
    python scripts/run_foundation.py --datasets ETTh1 Exchange --zero-shot-only
    python scripts/run_foundation.py --skip-full-ft          # zero-shot + LoRA
    python scripts/run_foundation.py --device mps            # force a device
"""

import os
# Apple Silicon: let ops that MPS does not implement fall back to the CPU
# instead of raising. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import argparse
import warnings
warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing")

# Patch lagged_sequence_values BEFORE lag_llama is imported.
# lag_llama/model/module.py does `from gluonts.torch.util import lagged_sequence_values`
# at module load time, so the patch must be in place before that import happens.
# Root cause: during autoregressive decoding, some lag slices are 1 element shorter
# because full_sequence doesn't have enough history. We pad short slices to max_len.
def _patch_lagged_sequence_values():
    import gluonts.torch.util as _gts_util
    import torch as _torch
    _orig = _gts_util.lagged_sequence_values

    def _patched(indices, prior_sequence, sequence, dim=-1):
        full_sequence = _torch.cat([prior_sequence, sequence], dim=dim)
        seq_len = sequence.shape[dim]
        lags_values = []
        for lag_index in indices:
            begin_index = -(lag_index + seq_len)
            end_index = -lag_index if lag_index > 0 else None
            lags_values.append(full_sequence[..., begin_index:end_index])
        # Pad any short slices to match the longest (always seq_len for well-formed input)
        target_len = max(v.shape[-1] for v in lags_values)
        lags_values = [
            _torch.nn.functional.pad(v, (target_len - v.shape[-1], 0))
            if v.shape[-1] < target_len else v
            for v in lags_values
        ]
        return _torch.stack(lags_values, dim=-1)

    _gts_util.lagged_sequence_values = _patched

_patch_lagged_sequence_values()


# Lag-Llama time features: no override. Lag-Llama builds its own 6 features
# (time_features_from_frequency_str("S")), which is what the checkpoint expects
# (feature_size = n_lags + 2 + 6). An earlier override forced 7 and broke the shapes.
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from config import DATA_DIR, RESULTS_DIR, FIGURES_DIR, TRAIN_FRACTIONS, SEEDS
from data_loader import load_and_prepare, AVAILABLE_DATASETS, pandas_freq


def pick_device(requested: str = "auto") -> str:
    """Return 'cuda', 'mps' or 'cpu'. 'auto' prefers CUDA, then Apple MPS, then CPU."""
    import torch
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("ERROR: --device cuda requested but no CUDA GPU is available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("ERROR: --device mps requested but MPS is not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_dtype(device: str):
    """bfloat16 on CUDA (as in the original runs); float32 on MPS / CPU,
    where bfloat16 support is incomplete or slow."""
    import torch
    return torch.bfloat16 if device == "cuda" else torch.float32


def report_device(device: str):
    import torch
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: CUDA — {name} ({memory:.1f} GB), dtype bfloat16")
    elif device == "mps":
        print("Device: Apple MPS, dtype float32 "
              "(Moirai and Lag-Llama inference run on the CPU)")
    else:
        print("Device: CPU, dtype float32 (fine-tuning will be slow)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FREQ_MAP = {
    "H": "h", "T": "min", "S": "s",
    "Q": "QE", "QS": "QS",
    "M": "ME", "MS": "MS",
    "A": "YE", "Y": "YE", "AS": "YS",
    "D": "D", "B": "B",
    "QE": "QE", "ME": "ME", "YE": "YE",
}

# GluonTS uses legacy pandas aliases for some frequencies.
# CONSERVATIVE: only fix aliases that GluonTS explicitly rejects.
_GLUONTS_UPPERCASE = {"me", "qe", "ye"}

def _normalize_freq(freq: str) -> str:
    """Normalise legacy pandas frequency aliases to modern pandas aliases."""
    return _FREQ_MAP.get(str(freq).upper(), str(freq).lower())

def _to_gluonts_freq(freq: str) -> str:
    """Return a GluonTS-compatible frequency string.

    GluonTS with pandas>=2.0 accepts ME, QE, H, D, W, 15min etc directly.
    Guard only against accidentally passing lowercase me/qe/ye.
    """
    norm = _normalize_freq(freq)
    norm = norm.upper() if norm.lower() in _GLUONTS_UPPERCASE else norm
    return pandas_freq(norm)   # ME/QE/YE -> M/Q/A on pandas < 2.2


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_chronos_model(model_path: str, device: str = "cuda"):
    """Load Chronos or Chronos-Bolt model from HuggingFace."""
    try:
        from chronos import ChronosPipeline, ChronosBoltPipeline
        cls = ChronosBoltPipeline if "bolt" in model_path.lower() else ChronosPipeline
        pipeline = cls.from_pretrained(
            model_path,
            device_map=device,
            dtype=model_dtype(device),
        )
        return pipeline
    except ImportError:
        print("Chronos not installed. Install with: pip install chronos-forecasting")
        return None


def load_moirai_model(model_path: str, device: str = "cuda"):
    """Load Moirai model from HuggingFace via uni2ts."""
    try:
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        module = MoiraiModule.from_pretrained(model_path)
        return {"module": module, "device": device, "model_path": model_path}
    except ImportError:
        print("Uni2TS/Moirai not installed. Install with: "
              "pip install git+https://github.com/SalesforceAIResearch/uni2ts.git")
        return None


def load_lagllama_model(device: str = "cuda"):
    """Load Lag-Llama model from HuggingFace, reading architecture from checkpoint."""
    try:
        import torch
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            repo_id="time-series-foundation-models/Lag-Llama",
            filename="lag-llama.ckpt",
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_kwargs = ckpt.get("hyper_parameters", {}).get("model_kwargs", {})
        return {"ckpt_path": ckpt_path, "device": device, "model_kwargs": model_kwargs}
    except ImportError:
        print("Lag-Llama not installed. Install with: "
              "pip install git+https://github.com/time-series-foundation-models/lag-llama.git")
        return None


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def _apply_lora(model, target_modules):
    """Add rank-8 LoRA adapters (alpha 16, dropout 0.05) to `target_modules`
    and freeze every other weight. Uses peft's inject_adapter_in_model, which
    keeps the model's own class (get_peft_model refuses ChronosBolt in recent
    peft versions)."""
    from peft import LoraConfig, inject_adapter_in_model
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                     target_modules=target_modules, bias="none")
    model = inject_adapter_in_model(cfg, model)
    n_lora = 0
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name
        n_lora += p.numel() if p.requires_grad else 0
    if n_lora == 0:
        raise RuntimeError(f"LoRA: no modules matched {target_modules}")
    if not getattr(_apply_lora, "_logged", set()) or type(model).__name__ not in _apply_lora._logged:
        _apply_lora._logged = getattr(_apply_lora, "_logged", set()) | {type(model).__name__}
        print(f"[LoRA] {type(model).__name__}: {n_lora:,} trainable parameters")
    return model


def _finetune_chronos_model(model, y_train, horizon, device, lora=False):
    """Fine-tune a ChronosBoltModelForForecasting on y_train.

    Returns a fine-tuned copy; original model is not mutated.
    Uses sliding-window quantile loss (built into model.forward when target is given).
    """
    import copy, torch
    from torch.optim import AdamW

    model_ft = copy.deepcopy(model).to(device)

    if lora:
        # peft asks for the input embeddings, which recent transformers no longer
        # finds automatically for ChronosBolt; it is the shared embedding.
        if hasattr(model_ft, "shared"):
            model_ft.get_input_embeddings = lambda m=model_ft: m.shared
        # T5 encoder+decoder attention uses "q" and "v" projection names
        model_ft = _apply_lora(model_ft, ["q", "v"])

    model_ft.train()
    lr = 5e-4 if lora else 1e-4
    optimizer = AdamW(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=lr)

    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    n = len(y_train)
    # Model's native prediction length (e.g. 64 for Bolt-Small) — target must not exceed this
    native_pred_len = getattr(model_ft, 'chronos_config', None)
    native_pred_len = native_pred_len.prediction_length if native_pred_len is not None else horizon
    train_horizon = min(horizon, native_pred_len)  # clip to model capacity

    # Context length: up to 512, leaving room for at least one target window
    ctx_len = min(512, n - train_horizon)
    if ctx_len < 4:
        # Fail the run rather than score zero-shot weights as fine-tuned
        raise ValueError(f"training data too short to fine-tune: "
                         f"{n} samples, horizon {train_horizon}")

    n_steps = min(300, max(50, n // max(train_horizon, 1)))

    for _ in range(n_steps):
        max_start = n - ctx_len - train_horizon
        start = int(torch.randint(0, max(1, max_start + 1), (1,)).item())
        ctx = y_t[start : start + ctx_len].unsqueeze(0)
        tgt = y_t[start + ctx_len : start + ctx_len + train_horizon].unsqueeze(0)

        optimizer.zero_grad()
        out = model_ft(context=ctx, target=tgt)
        loss = out.loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_ft.parameters(), 1.0)
        optimizer.step()

    model_ft.eval()
    return model_ft


def run_chronos_experiment(pipeline, y_train, contexts, horizon, fine_tune=False, lora=False):
    """Run Chronos / Chronos-Bolt prediction (zero-shot or fine-tuned).

    Fine-tuning (if any) uses y_train only. Forecasts are made from each
    context in `contexts` (all history before a test window). Returns a list
    of forecasts, one per context."""
    import torch, copy

    device = next(pipeline.model.parameters()).device

    if fine_tune:
        model_ft = _finetune_chronos_model(pipeline.model, y_train, horizon, device, lora=lora)
        # Temporarily swap model in a shallow pipeline copy for prediction
        pipeline_ft = copy.copy(pipeline)
        pipeline_ft.model = model_ft
    else:
        pipeline_ft = pipeline

    preds = []
    for ctx in contexts:
        context = torch.tensor(np.asarray(ctx), dtype=torch.float32).unsqueeze(0).to(device)
        forecast = pipeline_ft.predict(context, horizon)
        preds.append(np.median(forecast[0].cpu().numpy(), axis=0))
    return preds


def _finetune_moirai_module(module, y_train, horizon, freq, lora=False, device="cuda"):
    """Fine-tune a MoiraiModule via direct gradient steps.

    Bypasses MoiraiFinetune transform pipeline (too many internal fields required).
    Instead constructs batches manually and calls module.forward() with NLL loss.
    Returns fine-tuned deepcopy; original module not mutated.
    """
    import copy, torch
    import numpy as np

    module_ft = copy.deepcopy(module)
    if device == "mps":
        device = "cpu"   # uni2ts' scaler uses float64, which MPS does not support

    if lora:
        try:
            module_ft = _apply_lora(module_ft, ["q_proj", "k_proj", "v_proj", "out_proj"])
        except Exception as e:
            # Previously swallowed, which silently turned "LoRA" runs into full
            # fine-tuning. Fail the run instead (recorded as an error row).
            raise RuntimeError(f"Moirai LoRA could not be applied: {e}") from e

    try:
        n = len(y_train)
        ctx_len = min(512, n - horizon)
        if ctx_len < 4:
            # Previously returned the untrained module, so the run was scored as
            # fine-tuned although it was zero-shot. Fail the run instead.
            raise ValueError(f"training data too short to fine-tune: "
                             f"{n} samples, horizon {horizon}")

        num_steps = min(300, max(50, n // max(horizon, 1)))
        # Moirai always uses max_patch_size as the last tensor dimension
        # patch_size tensor tells the model actual values per patch; rest padded with zeros
        patch_size = int(min(module_ft.patch_sizes)) if hasattr(module_ft, "patch_sizes") else 8
        max_patch_size = int(max(module_ft.patch_sizes)) if hasattr(module_ft, "patch_sizes") else 128

        # Pad ctx_len and horizon to multiples of patch_size
        ctx_patches = (ctx_len + patch_size - 1) // patch_size
        pred_patches = (horizon + patch_size - 1) // patch_size
        ctx_len_pad = ctx_patches * patch_size
        horizon_pad = pred_patches * patch_size
        total_patches = ctx_patches + pred_patches

        module_ft = module_ft.to(device)
        module_ft.train()
        lr = 5e-4 if lora else 1e-4
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, module_ft.parameters()),
            lr=lr, weight_decay=1e-2
        )

        y_t = torch.tensor(y_train, dtype=torch.float32)
        rng = np.random.default_rng(42)

        for step in range(num_steps):
            # Sample a random window
            max_start = n - ctx_len - horizon
            start = int(rng.integers(0, max(1, max_start + 1)))
            ctx_raw = y_t[start : start + ctx_len].numpy()
            tgt_raw = y_t[start + ctx_len : start + ctx_len + horizon].numpy()

            # Pad to multiples of patch_size
            ctx_np = np.zeros(ctx_len_pad, dtype=np.float32)
            ctx_np[-ctx_len:] = ctx_raw
            tgt_np = np.zeros(horizon_pad, dtype=np.float32)
            tgt_np[:horizon] = tgt_raw

            # Patchify into [total_patches, patch_size], then pad last dim to max_patch_size
            ctx_patches_t = torch.tensor(ctx_np).view(ctx_patches, patch_size)
            tgt_patches_t = torch.tensor(tgt_np).view(pred_patches, patch_size)
            patches = torch.cat([ctx_patches_t, tgt_patches_t], dim=0)  # [total, patch_size]
            # Pad to max_patch_size along last dim
            target = torch.zeros(1, total_patches, max_patch_size)
            target[0, :, :patch_size] = patches

            # observed_mask: True where real values exist
            observed = torch.zeros(1, total_patches, max_patch_size, dtype=torch.bool)
            # context patches: real values in last ctx_len positions
            ctx_offset = ctx_len_pad - ctx_len
            for pi in range(ctx_patches):
                start_i = pi * patch_size
                end_i = (pi + 1) * patch_size
                real_start = max(0, ctx_offset - start_i)
                observed[0, pi, real_start:patch_size] = True
            # prediction patches: first horizon values real
            for pi in range(pred_patches):
                real_end = min(patch_size, horizon - pi * patch_size)
                if real_end > 0:
                    observed[0, ctx_patches + pi, :real_end] = True

            sample_id  = torch.zeros(1, total_patches, dtype=torch.long)
            time_id    = torch.arange(total_patches, dtype=torch.long).unsqueeze(0)
            variate_id = torch.zeros(1, total_patches, dtype=torch.long)
            pred_mask  = torch.zeros(1, total_patches, dtype=torch.bool)
            pred_mask[0, ctx_patches:] = True
            patch_size_t = torch.full((1, total_patches), patch_size, dtype=torch.long)

            # Move to the training device
            target     = target.to(device)
            observed   = observed.to(device)
            sample_id  = sample_id.to(device)
            time_id    = time_id.to(device)
            variate_id = variate_id.to(device)
            pred_mask  = pred_mask.to(device)
            patch_size_t = patch_size_t.to(device)

            optimizer.zero_grad()
            distr = module_ft(
                target=target,
                observed_mask=observed,
                sample_id=sample_id,
                time_id=time_id,
                variate_id=variate_id,
                prediction_mask=pred_mask,
                patch_size=patch_size_t,
            )
            # NLL loss: pass full target, then select prediction patches + real values
            log_prob = distr.log_prob(target)   # [1, total_patches, max_patch]
            # Mask to prediction patches and real (non-padded) values only
            pred_obs_full = observed.clone()
            pred_obs_full[0, :ctx_patches, :] = False  # zero out context
            real_log_prob = log_prob[pred_obs_full]
            loss = -real_log_prob.mean() if real_log_prob.numel() > 0 else log_prob.mean() * 0

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module_ft.parameters(), 1.0)
                optimizer.step()

        module_ft.eval()
        return module_ft

    except Exception as e:
        import traceback
        if not hasattr(_finetune_moirai_module, "_error_shown"):
            print(f"\n[Moirai finetune error]: {e}")
            traceback.print_exc()
            _finetune_moirai_module._error_shown = True
        # Previously returned the un-trained module, so a failed fine-tune was
        # scored as if it had succeeded. Fail the run instead.
        raise


def run_moirai_experiment(model_dict, y_train, contexts, horizon, fine_tune=False, lora=False):
    """Run Moirai prediction via GluonTS predictor (avoids manual tensor padding)."""
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai import MoiraiForecast

    freq = _to_gluonts_freq(model_dict.get("freq", "h"))

    if fine_tune:
        module = _finetune_moirai_module(
            model_dict["module"], y_train, horizon, freq, lora=lora,
            device=model_dict.get("device", "cuda"))
    else:
        module = model_dict["module"]

    # GluonTS' predictor picks CUDA or CPU ("auto"); on a Mac, bring a module
    # fine-tuned on MPS back to the CPU for sampling.
    if model_dict.get("device") == "mps":
        module = module.to("cpu")

    ctx_len = min(512, min(len(c) for c in contexts))

    # Use create_predictor so uni2ts handles all internal padding automatically
    model = MoiraiForecast(
        module=module,
        prediction_length=horizon,
        context_length=ctx_len,
        patch_size="auto",
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=32)

    ds = ListDataset(
        [{"start": pd.Timestamp("2020-01-01"), "target": np.asarray(c)} for c in contexts],
        freq=freq,
    )
    forecasts = list(predictor.predict(ds))
    return [np.median(f.samples, axis=0) for f in forecasts]


# Lag-Llama lag sets per base frequency — passed explicitly to bypass
# LagLlamaEstimator.__init__ calling get_lags_for_frequency("Q") internally.
_LAGLLAMA_LAGS = {
    "h":   [1, 2, 3, 6, 12, 24, 48, 168],
    "min": [1, 2, 3, 6, 12, 24, 48, 96, 144, 288],
    "d":   [1, 2, 3, 4, 5, 6, 7, 14, 28],
    "me":  [1, 2, 3, 6, 12, 24],
    "qe":  [1, 2, 3, 4, 8],
    "ye":  [1, 2, 3, 5],
}
_LAGLLAMA_LAGS_DEFAULT = [1, 2, 3, 4, 5, 6, 7]

# Strip numeric multiplier from freq strings like "15min" → "min", "10min" → "min"
import re as _re
def _base_freq(freq: str) -> str:
    return _re.sub(r"^[0-9]+", "", freq).lower() or freq.lower()


def run_lagllama_experiment(model_dict, y_train, contexts, horizon, fine_tune=False):
    """Run Lag-Llama prediction via GluonTS predictor."""
    import torch
    from gluonts.dataset.common import ListDataset
    from lag_llama.gluon.estimator import LagLlamaEstimator

    # Lag-Llama runs on CUDA if available, otherwise on the CPU, unless
    # --lagllama-device says otherwise (e.g. mps on Apple hardware).
    req = model_dict.get("lagllama_device", "auto")
    if req == "auto":
        req = "cuda" if model_dict["device"] == "cuda" else "cpu"
    device = torch.device(req)
    ckpt_path = model_dict["ckpt_path"]
    mk = model_dict.get("model_kwargs", {})
    freq = _to_gluonts_freq(model_dict.get("freq", "h"))
    base = _base_freq(freq)

    # Use lags_seq exactly as checkpoint was trained with (84 lags → wte second dim = 92).
    # The checkpoint stores 0-based lag indices; LagLlamaEstimator subtracts 1 from
    # whatever get_lags_for_frequency returns, so pass them +1 (patched lookup below).
    # Passing them unshifted gave a lag of -1 (the next value) and shifted every lag.
    lags_seq = [l + 1 for l in (mk.get("lags_seq") or _LAGLLAMA_LAGS.get(base, _LAGLLAMA_LAGS_DEFAULT))]

    # PyTorch 2.6+: patch torch.load so create_lightning_module can unpickle GluonTS classes
    _orig_load = torch.load
    def _patched_load(f, *args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(f, *args, **kwargs)
    torch.load = _patched_load

    # Monkeypatch get_lags_for_frequency - called unconditionally in __init__ with an int freq
    import lag_llama.gluon.estimator as _ll_mod
    import gluonts.time_feature.lag as _lag_mod
    _orig_get_lags = _lag_mod.get_lags_for_frequency
    def _safe_get_lags(freq_str, **kw):
        if not isinstance(freq_str, str):
            return lags_seq
        try:
            return _orig_get_lags(freq_str, **kw)
        except Exception:
            return lags_seq
    _lag_mod.get_lags_for_frequency = _safe_get_lags
    _ll_mod.get_lags_for_frequency = _safe_get_lags


    # Context length the checkpoint was trained with (32), as in the Lag-Llama
    # README; max_context_length (2048) is only an upper bound, and longer
    # contexts need RoPE scaling. Histories are clipped to context + largest lag;
    # shorter ones are left to GluonTS, which zero-pads and masks them.
    max_ctx = mk.get("context_length", 32)
    if not getattr(run_lagllama_experiment, "_ctx_logged", False):
        print(f"[Lag-Llama] context length {max_ctx}, max lag {max(lags_seq) - 1}")
        run_lagllama_experiment._ctx_logged = True
    needed = max_ctx + max(lags_seq)
    contexts_clipped = [np.asarray(c, dtype=float)[-needed:] for c in contexts]

    try:
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=horizon,
            context_length=max_ctx,
            n_layer=mk.get("n_layer", 8),
            n_embd_per_head=mk.get("n_embd_per_head", 16),
            n_head=mk.get("n_head", 9),
            lags_seq=lags_seq,
            rope_scaling=mk.get("rope_scaling", None),
            time_feat=True,  # must match checkpoint: wte shape = input + lags + time_feats
            scaling=mk.get("scaling", "robust"),
            batch_size=64,
            num_parallel_samples=100,
            # Must stay False: series are z-scored, so clipping samples at 0
            # would cap every forecast at the training-pool mean.
            nonnegative_pred_samples=False,
            device=device,
        )
        lightning_module = estimator.create_lightning_module(use_kv_cache=True)  # same forecasts, avoids recomputing the context each step
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(
            transformation=transformation,
            module=lightning_module,
        )
        ds = ListDataset(
            [{"start": pd.Timestamp("2020-01-01"), "target": c} for c in contexts_clipped],
            freq=freq,  # freq="ME"/"QE" (GluonTS-safe); base="me"/"qe" is only for lags lookup
        )
        forecasts = list(predictor.predict(ds))
        y_pred = [np.median(f.samples, axis=0) for f in forecasts]
    finally:
        torch.load = _orig_load
        _lag_mod.get_lags_for_frequency = _orig_get_lags
        _ll_mod.get_lags_for_frequency = _orig_get_lags

    return y_pred

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_mase(y_true, y_pred, y_train, seasonal_period):
    """Compute Mean Absolute Scaled Error."""
    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    mae = np.mean(np.abs(y_true - y_pred))
    if len(y_train) <= seasonal_period:
        naive_errors = np.abs(np.diff(y_train))
    else:
        naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    naive_mae = np.mean(naive_errors)
    return mae / naive_mae if naive_mae > 0 else mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run foundation model experiments")
    parser.add_argument('--gpu', action='store_true',
                        help='Deprecated, no effect (kept so existing commands still work); '
                             'use --device')
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto',
                        help='Compute device (default: auto = CUDA, then Apple MPS, then CPU)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to run (default: all)')
    parser.add_argument('--models', nargs='+',
                        default=[
                            'Chronos-Bolt-S', 'Chronos-Bolt-B',
                            'Moirai-S', 'Moirai-B', 'Moirai-L',
                            'Lag-Llama',
                        ],
                        help='Foundation models to run')
    parser.add_argument('--zero-shot-only', action='store_true',
                        help='Only run zero-shot (skip fine-tuning)')
    parser.add_argument('--lagllama-device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto',
                        help='Device for Lag-Llama inference (default: CUDA if available, else CPU)')
    parser.add_argument('--skip-full-ft', action='store_true',
                        help='Skip full fine-tuning (run zero-shot + LoRA only)')
    parser.add_argument('--fractions', nargs='+', type=float, default=None,
                        help='Training fractions for fine-tuning')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Random seeds (default: all seeds from config)')

    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("ERROR: PyTorch not installed. Install with: pip install -r requirements-gpu.txt")
    device = pick_device(args.device)
    report_device(device)

    datasets = args.datasets or AVAILABLE_DATASETS
    models = args.models
    fractions = args.fractions or [0.05, 0.1, 0.2, 0.5, 1.0]
    seeds = args.seeds if args.seeds is not None else SEEDS

    settings = ['zero-shot']
    if not args.zero_shot_only:
        if not args.skip_full_ft:
            settings.append('fine-tuned-full')
        settings.append('fine-tuned-lora')

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Foundation Model Experiments")
    print("=" * 60)
    print(f"Datasets:  {datasets}")
    print(f"Models:    {models}")
    print(f"Settings:  {settings}")
    print(f"Fractions: {fractions}")
    print()

    # Note: chronos-bolt-large does not exist on HuggingFace
    MODEL_PATHS = {
        'Chronos-Bolt-S': 'amazon/chronos-bolt-small',
        'Chronos-Bolt-B': 'amazon/chronos-bolt-base',
        'Moirai-S':       'Salesforce/moirai-1.0-R-small',
        'Moirai-B':       'Salesforce/moirai-1.0-R-base',
        'Moirai-L':       'Salesforce/moirai-1.0-R-large',
        'Lag-Llama':      'time-series-foundation-models/Lag-Llama',
    }

    # Load models
    print("Loading foundation models...")
    loaded_models = {}
    for model_name in models:
        print(f"  Loading {model_name}...")
        model_path = MODEL_PATHS.get(model_name)
        try:
            if 'Chronos' in model_name:
                loaded_models[model_name] = load_chronos_model(model_path, device)
            elif 'Moirai' in model_name:
                loaded_models[model_name] = load_moirai_model(model_path, device)
            elif 'Lag-Llama' in model_name:
                loaded_models[model_name] = load_lagllama_model(device)
                if loaded_models[model_name] is not None:
                    loaded_models[model_name]["lagllama_device"] = args.lagllama_device
            else:
                print(f"    Unknown model type: {model_name}")
                loaded_models[model_name] = None
        except Exception as e:
            print(f"    ERROR loading {model_name}: {e}")
            loaded_models[model_name] = None

        if loaded_models.get(model_name) is None:
            print(f"    WARNING: Failed to load {model_name}")

    # Run experiments
    print("\nRunning experiments...")
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"foundation_raw_{timestamp}.csv")
    partial_file = output_file.replace(".csv", "_partial.csv")

    for ds_name in tqdm(datasets, desc="Datasets"):
        if results:   # checkpoint after each dataset, so a crash loses at most one
            pd.DataFrame(results).to_csv(partial_file, index=False)
        try:
            data_full = load_and_prepare(ds_name, train_fraction=1.0)
        except Exception as e:
            print(f"\nSkipping {ds_name}: failed to load — {e}")
            continue

        config       = data_full['config']
        y_train_full = data_full['y_train']
        windows      = data_full['windows']          # (context, target) pairs
        contexts     = [c for c, _ in windows]
        targets      = [t for _, t in windows]

        horizon = config.horizon
        sp      = config.seasonal_period

        for model_name in models:
            model = loaded_models.get(model_name)
            if model is None:
                continue

            for setting in settings:
                if 'Lag-Llama' in model_name and setting != 'zero-shot':
                    continue   # Lag-Llama is evaluated zero-shot only
                # Zero-shot does not train, and its context is the full history
                # before each test window, so it does not depend on the fraction.
                fracs = [1.0] if setting == 'zero-shot' else fractions

                for frac in fracs:
                    for seed in seeds:
                        np.random.seed(seed)
                        import torch, random
                        random.seed(seed)
                        torch.manual_seed(seed)   # FM sampling + fine-tuning (also seeds CUDA/MPS)

                        n_sub = max(sp * 2, int(len(y_train_full) * frac))
                        y_train = y_train_full[-n_sub:]

                        try:
                            fine_tune = setting in ('fine-tuned-full', 'fine-tuned-lora')
                            lora = (setting == 'fine-tuned-lora')

                            t0 = time.time()
                            if 'Chronos' in model_name:
                                y_pred = run_chronos_experiment(
                                    model, y_train, contexts, horizon, fine_tune, lora)
                            elif 'Moirai' in model_name:
                                model['freq'] = _to_gluonts_freq(config.freq)
                                y_pred = run_moirai_experiment(
                                    model, y_train, contexts, horizon, fine_tune, lora)
                            elif 'Lag-Llama' in model_name:
                                # Lag-Llama: fine-tuning not implemented — always zero-shot
                                model['freq'] = _to_gluonts_freq(config.freq)
                                y_pred = run_lagllama_experiment(
                                    model, y_train, contexts, horizon, fine_tune=False)
                            else:
                                raise ValueError(f"No inference function for {model_name}")
                            duration_s = time.time() - t0

                            # Mean MAE over test windows; MASE uses this run's
                            # training data (analysis rescales via mae/denom_full).
                            maes = []
                            for tgt, pred in zip(targets, y_pred):
                                n = min(len(tgt), len(pred))
                                maes.append(float(np.mean(np.abs(tgt[:n] - pred[:n]))))
                            mae = float(np.mean(maes))
                            mase = compute_mase(np.zeros(1), np.full(1, mae), y_train, sp)

                            results.append({
                                'dataset': ds_name,
                                'model': model_name,
                                'setting': setting,
                                'train_fraction': frac,
                                'n_train': n_sub,
                                'seed': seed,
                                'mase': mase,
                                'mae': mae,
                                'n_windows': len(maes),
                                'duration_s': round(duration_s, 3),
                            })

                        except Exception as e:
                            import traceback
                            tb = traceback.format_exc()
                            if 'Lag-Llama' in model_name:
                                print(f"\n=== LAG-LLAMA ERROR ({ds_name} frac={frac} seed={seed}) ===")
                                print(tb)
                                print("===")
                            results.append({
                                'dataset': ds_name,
                                'model': model_name,
                                'setting': setting,
                                'train_fraction': frac,
                                'n_train': n_sub,
                                'seed': seed,
                                'mase': np.nan,
                                'error': str(e),
                            })

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    if os.path.exists(partial_file):
        os.remove(partial_file)

    print(f"\nResults saved to {output_file}")
    print(f"Total experiments: {len(results_df)}")
    print(f"Successful: {results_df['mase'].notna().sum()}")

    if len(results_df) > 0 and 'mase' in results_df.columns:
        print("\nSummary by model:")
        summary = results_df.groupby(['model', 'setting'])['mase'].agg(
            ['mean', 'std', 'count'])
        print(summary.round(3))


if __name__ == "__main__":
    main()
