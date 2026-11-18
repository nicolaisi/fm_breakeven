"""Sanity check: with a trending seasonal series, forecasts made from the
right origin should be accurate; the old origin (end of training pool) should not."""
import sys, types, numpy as np
repo = sys.argv[1]
sys.path.insert(0, f"{repo}/src")
import data_loader as dl
from models import get_model

n, sp, h = 3000, 24, 24
t = np.arange(n)
rng = np.random.default_rng(0)
y = 10 + 0.01 * t + np.sin(2 * np.pi * t / sp) + 0.05 * rng.standard_normal(n)
cfg = types.SimpleNamespace(seasonal_period=sp, horizon=h, freq="1H")
dl.load_dataset = lambda name, data_dir=None: (None, cfg)
dl.get_target_series = lambda df, config: y

for frac in (0.05, 1.0):
    d = dl.load_and_prepare("synthetic", train_fraction=frac)
    # check: each window's context ends exactly where its target begins
    for ctx, tgt in d["windows"]:
        assert len(ctx) >= int(n * 0.8)
        k = len(ctx)
        full = (y - y[: int(n * 0.6)].mean()) / y[: int(n * 0.6)].std()
        assert np.allclose(full[k:k + len(tgt)], tgt), "context/target misaligned"
    print(f"frac={frac}: n_train={d['n_train']}, windows={len(d['windows'])}, origins={[len(c) for c,_ in d['windows']]}")
    for name in ["Naive", "ETS", "ARIMA", "XGBoost"]:
        kw = {"t0": d["train_start"]} if name == "XGBoost" else {}
        m = get_model(name, horizon=h, seasonal_period=sp, freq=cfg.freq).fit(d["y_train"], **kw)
        old = np.mean(np.abs(m.predict() - d["windows"][0][1]))           # old behaviour
        new = np.mean([np.mean(np.abs(m.predict_from(c) - tg)) for c, tg in d["windows"]])
        print(f"   {name:8s} MAE old-origin {old:6.3f}   new-origin {new:6.3f}")
print("OK: every window's context ends exactly where its target begins")
