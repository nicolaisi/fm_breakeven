"""
Figure 1: Scaling Curves — Foundation Models vs. Classical Methods (30 datasets)
Loads all data from the rerun results (src/results_io.py) — no hardcoded values.

Usage:
    python plot_fig1.py
Outputs: results/fig1_scaling_curves.{pdf,png}
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from results_io import load_results, detect_breakeven, best_zero_shot, FRACS

cl, fm = load_results()
ALL_DS = sorted(set(cl.dataset) & set(fm.dataset))
cl = cl[cl.dataset.isin(ALL_DS)]
be_info = {ds: detect_breakeven(cl, fm, ds)[:2] for ds in ALL_DS}
HAS_LORA = (fm.setting == 'fine-tuned-lora').any()

PAT_ORDER = {'FM-dominant': 0, 'Early': 1, 'Mid': 2, 'Late': 3}
DS_ORDER = sorted(ALL_DS, key=lambda d: (PAT_ORDER[be_info[d][0]], d))

PATTERN_COLOR = {
    'FM-dominant': '#1565C0',
    'Early':       '#C62828',
    'Mid':         '#6A1B9A',
    'Late':        '#2E7D32',
}
PATTERN_LABEL = {
    'FM-dominant': 'FM-dominant',
    'Early':       'Early break-even',
    'Mid':         'Mid break-even',
    'Late':        'Late break-even',
}
PATTERN_LEGEND_LABEL = {
    'FM-dominant': 'FM-dominant',
    'Early':       r'Early break-even ($\leq$5% data)',
    'Mid':         r'Mid break-even (10% data)',
    'Late':        r'Late break-even ($\geq$20% data)',
}
CL_COLORS = {'XGBoost': '#1565C0', 'ETS': '#00838F',
             'ARIMA': '#6A1B9A',   'Naive': '#546E7A'}
CL_LS     = {'XGBoost': '-', 'ETS': '-', 'ARIMA': '--', 'Naive': '-.'}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size':   18,
    'axes.linewidth': 1.0,
})


def draw_panel(ax, ds):
    pattern, n_star = be_info[ds]
    border_col = PATTERN_COLOR[pattern]
    n_full = int(cl[(cl.dataset == ds) & (cl.train_fraction == 1.0)].n_train.median())
    all_y = []

    ds_cl = cl[(cl.dataset == ds) & cl.mase_fixed.notna()]
    for model, color in CL_COLORS.items():
        df_m = ds_cl[ds_cl.model == model].groupby('train_fraction').mase_fixed.mean()
        if len(df_m) == 0:
            continue
        xs = df_m.index.values * n_full
        ys = df_m.values
        ax.plot(xs, ys, color=color, lw=3.4, ls=CL_LS[model],
                marker='o', markersize=6, zorder=3, alpha=0.85)
        all_y.extend(ys.tolist())

    fm_zs_val = best_zero_shot(fm, ds)[1]
    ax.axhline(fm_zs_val, color='#B71C1C', lw=3.2, ls='--', zorder=4)
    all_y.append(fm_zs_val)

    # LoRA: best model per fraction (mean over seeds), fixed denominator
    lora_xs, lora_ys, lora_sd = [], [], []
    lora_ds = fm[(fm.dataset == ds) & (fm.setting == 'fine-tuned-lora') & fm.mase_fixed.notna()]
    for frac in FRACS:
        sub = lora_ds[lora_ds.train_fraction == frac]
        if len(sub):
            g = sub.groupby('model').mase_fixed
            best = g.mean().idxmin()
            lora_xs.append(sub.n_train.median())   # actual size: LoRA uses >= 2m samples
            lora_ys.append(g.mean()[best])
            lora_sd.append(g.std(ddof=1).fillna(0)[best])
    lora_xs, lora_ys, lora_sd = map(np.array, (lora_xs, lora_ys, lora_sd))
    all_y.extend(lora_ys.tolist())

    finite = [v for v in all_y if np.isfinite(v)]
    ymin = max(0, min(finite) * 0.80)
    ymax = max(max(finite) * 1.22, fm_zs_val * 1.15)
    ymax = min(ymax, fm_zs_val * 2.5)          # keep the crossing readable
    ax.set_ylim(ymin, ymax)
    off = [m for m in CL_COLORS
           if (ds_cl[ds_cl.model == m].groupby('train_fraction').mase_fixed.mean() > ymax).any()]
    if len(lora_ys) and (lora_ys > ymax).any():
        off.append('LoRA')
    if off:
        ax.text(0.02, 0.98, '↑ off scale: ' + ', '.join(off), transform=ax.transAxes,
                fontsize=11, color='#555', va='top', ha='left')

    if len(lora_xs):
        ax.fill_between(lora_xs, np.maximum(lora_ys - lora_sd, ymin),
                        np.minimum(lora_ys + lora_sd, ymax),
                        alpha=0.15, color='#E53935', zorder=2)
        ax.plot(lora_xs, np.minimum(lora_ys, ymax), color='#E53935', lw=2.6, zorder=4, alpha=0.9)
        ax.scatter(lora_xs, np.minimum(lora_ys, ymax), color='#E53935', marker='D',
                   s=50, zorder=5, alpha=0.9)

    x_lo = min(ds_cl.n_train.min(), lora_xs.min() if len(lora_xs) else np.inf)
    x_hi = n_full
    if n_star is not None:
        ax.axvline(n_star, color='#424242', lw=2.0, ls=':', zorder=5)
        ytext = ymin + (ymax - ymin) * 0.86
        # label on the side of the line with more room (log scale)
        right = np.log(n_star / x_lo) > np.log(x_hi / n_star)
        ax.text(n_star * (0.88 if right else 1.12), ytext, f'$n^*$={n_star:,}',
                fontsize=16, color='#222', va='top', ha='right' if right else 'left',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                          edgecolor='none', alpha=0.7))
    else:
        ax.fill_between([FRACS[0] * n_full, FRACS[-1] * n_full], ymin, fm_zs_val,
                        alpha=0.06, color='#1565C0', zorder=1)

    ax.set_xscale('log')
    if x_hi / x_lo < 30:   # short range: plain numbers, no overlapping minor labels
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Training samples', fontsize=16)
    ax.set_ylabel('MASE', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, which='major', ls='--', alpha=0.22, color='#999')
    ax.grid(True, which='minor', ls=':', alpha=0.10, color='#bbb')
    ax.set_facecolor('#F9F9F9')
    for spine in ax.spines.values():
        spine.set_edgecolor(border_col)
        spine.set_linewidth(2.2)
    ds_title = ds.replace('M4-Monthly-Monash', 'M4-M-Monash').replace('-', '\u2010')  # hyphen, not mathtext minus
    ax.set_title(f'$\\mathbf{{{ds_title}}}$\n{PATTERN_LABEL[pattern]}',
                 fontsize=17, color=border_col, pad=3)


def legend_handles():
    """Legend entries; LoRA only if there are LoRA results, and only
    categories that occur among the datasets."""
    has_lora = (fm.setting == 'fine-tuned-lora').any()
    present = {be_info[d][0] for d in ALL_DS}
    h = [Line2D([0], [0], color='#B71C1C', lw=3.2, ls='--', label='FM zero-shot (best)')]
    if has_lora:
        h.append(Line2D([0], [0], color='#E53935', lw=2.6, marker='D', ms=4, label='FM LoRA (best)'))
    h += [
        Line2D([0], [0], color='#1565C0', lw=3.4, marker='o', ms=3, label='XGBoost'),
        Line2D([0], [0], color='#00838F', lw=3.4, marker='o', ms=3, label='ETS'),
        Line2D([0], [0], color='#6A1B9A', lw=3.4, ls='--', marker='o', ms=3, label='ARIMA'),
        Line2D([0], [0], color='#546E7A', lw=3.4, ls='-.', marker='o', ms=3, label='Naive'),
        Line2D([0], [0], color='#424242', lw=2.0, ls=':', label='Break-even $n^*$'),
    ]
    for pat in [p for p in ['FM-dominant', 'Early', 'Mid', 'Late'] if p in present]:
        h.append(mpatches.Patch(facecolor=PATTERN_COLOR[pat], alpha=0.35,
                                edgecolor=PATTERN_COLOR[pat], linewidth=2.0,
                                label=PATTERN_LEGEND_LABEL[pat]))
    return h


def main():
    os.makedirs('results', exist_ok=True)

    NROWS, NCOLS = 5, 6
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(26, 22))
    for ax, ds in zip(axes.flatten(), DS_ORDER):
        draw_panel(ax, ds)
    for ax in axes.flatten()[len(DS_ORDER):]:
        ax.axis('off')
    fig.legend(handles=legend_handles(), loc='lower center', bbox_to_anchor=(0.5, -0.02),
               fontsize=18, ncol=6, framealpha=0.95, edgecolor='#bbb',
               columnspacing=1.0, handlelength=2.4)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.subplots_adjust(hspace=0.55, wspace=0.38)
    for ext in ('pdf', 'png'):
        plt.savefig(f'results/fig1_scaling_curves.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("Saved: results/fig1_scaling_curves.{pdf,png}")

    print(f"\nDataset order ({len(DS_ORDER)} datasets):")
    for ds in DS_ORDER:
        pat, n = be_info[ds]
        print(f"  {ds:<22} {pat}  n*={n}")


if __name__ == '__main__':
    main()
