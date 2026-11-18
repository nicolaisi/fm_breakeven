"""
Figure 3: Zero-Shot vs LoRA Fine-Tuned — M4-Hourly and M4-Weekly
Loads from the rerun results (src/results_io.py) — no hardcoded MASE values.

Run from: ~/Development/fm_breakeven/
Output: results/fig3_finetuning.pdf, results/fig3_finetuning.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

os.makedirs('results', exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from results_io import load_results, detect_breakeven, best_zero_shot

cl, fm = load_results()
FRACS = np.array([0.02, 0.05, 0.10, 0.20, 0.50, 1.00])

def detect(ds):
    return detect_breakeven(cl, fm, ds)[:2]

def get_fm_zs(ds):
    return best_zero_shot(fm, ds)[1]

def get_lora_curve(ds):
    """Best LoRA model per fraction, mean over seeds (fixed-denominator MASE)."""
    ys = []
    for frac in FRACS:
        sub = fm[(fm.dataset == ds) & (fm.setting == 'fine-tuned-lora') &
                 np.isclose(fm.train_fraction, frac)]
        ys.append(sub.groupby('model').mase_fixed.mean().min() if len(sub) else np.nan)
    return np.array(ys)

def get_cl_curve(ds, model=None):
    """Best classical (or a specific model) per fraction, fixed-denominator MASE."""
    ys = []
    for frac in FRACS:
        sub = cl[(cl.dataset == ds) & np.isclose(cl.train_fraction, frac) & cl.mase_fixed.notna()]
        if model:
            sub = sub[sub.model == model]
        ys.append(sub.mase_fixed.min() if len(sub) else np.nan)
    return np.array(ys)

def get_x(ds, lora):
    """Actual training-set size per fraction (LoRA uses at least 2m samples)."""
    d = fm[(fm.dataset == ds) & (fm.setting == 'fine-tuned-lora')] if lora else cl[cl.dataset == ds]
    return np.array([d[np.isclose(d.train_fraction, f)].n_train.median() for f in FRACS])

def get_n_full(ds):
    return int(cl[(cl.dataset == ds) & (cl.train_fraction == 1.0)].n_train.median())

# ── Data ──────────────────────────────────────────────────────────────────────
M4_N    = get_n_full('M4-Hourly')
M4_ZS   = get_fm_zs('M4-Hourly')
M4_LORA = get_lora_curve('M4-Hourly')
M4_CL   = get_cl_curve('M4-Hourly')        # best classical per fraction
M4_X    = get_x('M4-Hourly', True)
M4_XC   = get_x('M4-Hourly', False)

WK_N    = get_n_full('M4-Weekly')
WK_ZS   = get_fm_zs('M4-Weekly')
WK_LORA = get_lora_curve('M4-Weekly')
WK_CL   = get_cl_curve('M4-Weekly')               # best classical per fraction
WK_X    = get_x('M4-Weekly', True)
WK_XC   = get_x('M4-Weekly', False)

print(f"M4-Hourly: ZS={M4_ZS:.3f}, n_full={M4_N}")
print(f"  LoRA: {np.round(M4_LORA,3)}")
print(f"  CL:   {np.round(M4_CL,3)}")
print(f"M4-Weekly: ZS={WK_ZS:.3f}, n_full={WK_N}")
print(f"  LoRA: {np.round(WK_LORA,3)}")
print(f"  CL:   {np.round(WK_CL,3)}")

# ── Colours ───────────────────────────────────────────────────────────────────
ZS_COL   = '#B71C1C'
LORA_COL = '#E91E63'
CL_COL   = '#2E7D32'

FONT    = 18
FONT_LG = 20
TICK    = 16

fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))
plt.rcParams.update({'font.family': 'sans-serif'})

def draw_panel(ax, xs, xc, zs_val, lora_ys, cl_ys, cl_label,
               title_main, ymin, ymax):
    ax.axhline(zs_val, color=ZS_COL, lw=3.0, ls='--', zorder=4,
               label='FM zero-shot')
    valid_l = ~np.isnan(lora_ys)
    ax.plot(xs[valid_l], lora_ys[valid_l], color=LORA_COL, lw=2.8,
            marker='D', markersize=9, zorder=5, label='FM LoRA fine-tuned')
    valid_c = ~np.isnan(cl_ys)
    if valid_c.any():
        ax.plot(xc[valid_c], cl_ys[valid_c], color=CL_COL, lw=2.6,
                ls='-.', marker='s', markersize=8, zorder=3, label=cl_label)
    ax.set_xscale('log')
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    lo, hi = np.nanmin(np.r_[xs, xc]), np.nanmax(np.r_[xs, xc])
    ticks = [t for t in (10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000, 5000) if lo * 0.8 <= t <= hi * 1.25]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Training samples', fontsize=FONT_LG)
    ax.set_ylabel('MASE', fontsize=FONT_LG)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(labelsize=TICK)
    ax.grid(True, which='major', ls='--', alpha=0.22, color='#aaa')
    ax.grid(True, which='minor', ls=':', alpha=0.10, color='#ccc')
    ax.set_facecolor('#F9F9F9')
    ax.set_title(title_main, fontsize=FONT_LG, fontweight='bold', pad=6)

def _ylim(zs, lora, cl):
    """Axis range from the data (all curves visible, with headroom)."""
    vals = np.concatenate([[zs], lora[np.isfinite(lora)], cl[np.isfinite(cl)]])
    lo, hi = vals.min(), vals.max()
    pad = 0.12 * (hi - lo)
    return dict(ymin=max(0, lo - pad), ymax=hi + pad)

def _panel_title(prefix, ds):
    pat, n_star = detect(ds)
    if pat == 'FM-dominant':
        tag = 'FM-dominant'
    elif n_star is not None:
        tag = f'{pat} break-even, n*={n_star:,}'
    else:
        tag = f'{pat} break-even'
    return f'{prefix}  [{tag}]'

draw_panel(axes[0], M4_X, M4_XC, M4_ZS, M4_LORA, M4_CL,
           cl_label='Best classical',
           title_main=_panel_title('(a) M4-Hourly', 'M4-Hourly'),
           **_ylim(M4_ZS, M4_LORA, M4_CL))

draw_panel(axes[1], WK_X, WK_XC, WK_ZS, WK_LORA, WK_CL,
           cl_label='Best classical',
           title_main=_panel_title('(b) M4-Weekly', 'M4-Weekly'),
           **_ylim(WK_ZS, WK_LORA, WK_CL))

legend_elements = [
    Line2D([0],[0], color=ZS_COL,   lw=3.0, ls='--',
           label='FM zero-shot (flat; data-independent)'),
    Line2D([0],[0], color=LORA_COL, lw=2.8, marker='D', ms=6,
           label='FM LoRA fine-tuned'),
    Line2D([0],[0], color=CL_COL,   lw=2.6, ls='-.', marker='s', ms=5,
           label='Classical (best model)'),
]
fig.legend(handles=legend_elements, fontsize=FONT, loc='lower center',
           bbox_to_anchor=(0.5, -0.03), ncol=3,
           framealpha=0.95, edgecolor='#ccc', columnspacing=1.3)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig('results/fig3_finetuning.pdf', bbox_inches='tight', dpi=200)
plt.savefig('results/fig3_finetuning.png', bbox_inches='tight', dpi=200)
print("Saved: results/fig3_finetuning.pdf / .png")
