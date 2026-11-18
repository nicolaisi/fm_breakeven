"""
Figure 2: Feature Space and Break-Even Patterns (30 datasets)
Loads from dataset_features.csv and the rerun results (src/results_io.py) — no hardcoded values.

Run from: ~/Development/fm_breakeven/
Output: results/fig2_feature_space.pdf, results/fig2_feature_space.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

# ── Load data ─────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from results_io import load_results, detect_breakeven

feat = pd.read_csv('results/dataset_features.csv')
cl, fm = load_results()
feat['pattern'] = feat['dataset'].apply(lambda ds: detect_breakeven(cl, fm, ds)[0])

PATTERN_COLOR  = {'FM-dominant':'#1565C0','Early':'#C62828','Mid':'#E65100','Late':'#2E7D32'}
PATTERN_MARKER = {'FM-dominant':'D','Early':'o','Mid':'P','Late':'s'}
PATTERN_LABEL  = {'FM-dominant':'FM-dominant','Early':'Early break-even',
                  'Mid':'Mid break-even','Late':'Late break-even'}

noise_vals = feat['noise_ratio'].values
norm = Normalize(vmin=noise_vals.min(), vmax=noise_vals.max())
cmap = plt.cm.YlOrRd

fig, ax = plt.subplots(figsize=(22, 17))
plt.rcParams.update({'font.family': 'sans-serif'})

# Small jitter to separate overlapping points (Traffic/ETTh2, etc.)
import hashlib

# Manual overrides: (dS, dn_log) — used when hash jitter would cross a reference line
_JITTER_OVERRIDE = {
    'ILI': (0.0, 0.5),   # hash jitter pushes ILI to n≈1146; true n=579, must stay below n=700 line
    # hash jitter put these short low-S series on top of each other
    'FRED-MD': (0.0, 0.0), 'Rideshare': (0.0, 0.0), 'M4-Daily': (0.0, 0.0),
}

def jitter_offset(ds):
    if ds in _JITTER_OVERRIDE:
        return _JITTER_OVERRIDE[ds]
    # Deterministic tiny offsets based on dataset name hash
    h = int(hashlib.md5(ds.encode()).hexdigest(), 16)
    return ((h % 100) - 50) / 5000, ((h % 97) - 48) / 2.0

# Per-dataset label offsets: (dx, log10_dy, ha)
# xytext = (plot_S + dx, plot_n * 10**log10_dy)
_OFFSETS = {
    'Wind-Farms':        (0.02,  0.15, 'left'),
    'AusElectricity':    ( 0.03,  0.15, 'left'),
    'Sunspot':           ( 0.02,  0.15, 'left'),
    'ETTm1':             (0.05,  0.15, 'right'),
    'ETTm2':             (-0.01,  0.18, 'right'),
    'Solar-10min':       (-0.05, -0.28, 'right'),
    'Weather':           ( 0.02,  0.10, 'left'),
    'Electricity':       (-0.04,  -0.09, 'right'),
    'Pedestrian':        ( 0.03,  0.00, 'left'),
    'Saugeen':           ( 0.00,  0.18, 'left'),
    'ETTh1':             (-0.04,  0.10, 'right'),
    'ETTh2':             ( 0.03,  0.10, 'left'),
    'Traffic':           (0.00, 0.18, 'left'),
    'KDD-Cup-Hourly':    (0.05,  -0.35, 'left'),
    'Exchange':          (0.00, 0.45, 'left'),
    'USBirths':          ( 0.04,  0.0, 'left'),
    'ILI':               ( 0.02,  0.35, 'left'),
    'NN5-Daily':         (-0.06,  0.35, 'right'),
    'M4-Hourly':         ( 0.00, -0.4, 'left'),
    'Rideshare':         ( 0.13, -0.15, 'left'),
    'FRED-MD':           ( 0.04,  0.15, 'left'),
    'M4-Daily':          ( 0.10,  0.08, 'left'),
    'M4-Weekly':         (-0.03,  0.15, 'right'),
    'COVID-Deaths':      ( 0.05, -0.35, 'left'),
    'Tourism-Monthly':   ( -0.02, -0.50, 'left'),
    'Dominick':          ( 0.05, -0.55, 'left'),
    'Hospital':          (-0.04,  0.15, 'right'),
    'M3-Quarterly':      ( 0.07,  0.38, 'left'),
    'M4-Monthly-Monash': ( 0.03,  -0.5, 'left'),
    'M3-Monthly':        (-0.03,  0.15, 'right'),
}

for _, row in feat.iterrows():
    ds      = row['dataset']
    n       = row['n_train']
    S       = row['seas_strength']
    sigma_r = row['noise_ratio']
    tau     = row['tau_acf']
    pattern = row['pattern']

    color  = cmap(norm(sigma_r))
    edge   = PATTERN_COLOR[pattern]
    marker = PATTERN_MARKER[pattern]

    dS, dn_log = jitter_offset(ds)
    plot_S = S + dS
    plot_n = n * (1.05 ** dn_log)

    ax.scatter(plot_S, plot_n, s=640, marker=marker, facecolor=color,
               edgecolors=edge, linewidths=3.5, zorder=4)

    tau_note = f'\n(τ={int(tau)})' if tau < 30 else ''
    dx, log_dy, ha = _OFFSETS.get(ds, (0.04 if S < 0.5 else -0.04, 0.15, 'left' if S < 0.5 else 'right'))
    text_S = plot_S + dx
    text_n = plot_n * (10 ** log_dy)
    ax.annotate(f'{ds}{tau_note}', xy=(plot_S, plot_n),
                xytext=(text_S, text_n),
                fontsize=28, ha=ha, va='bottom',
                color=PATTERN_COLOR[pattern], zorder=5,
                arrowprops=dict(arrowstyle='-', color=edge, lw=1.2, alpha=0.5))

# Reference lines (illustrative thresholds from paper hypotheses)
ax.axhline(700,  color='#555', lw=2.6, ls='--', alpha=0.6, zorder=2)
ax.axvline(0.05, color='#888', lw=2.2, ls=':',  alpha=0.5, zorder=2)
ax.text(0.06, 1.1e6, 'S=0.05', fontsize=32, color='#666')
ax.text(0.3, 750, 'n=700', fontsize=32, color='#444', va='bottom')

ax.set_yscale('log')
ax.set_xlabel('Seasonal Strength  $S$', fontsize=52)
ax.set_ylabel('Training Samples  $n_{\\mathrm{train}}$  (log scale)', fontsize=52)
ax.tick_params(labelsize=40)
ax.set_xlim(-0.08, 1.12)
ax.set_ylim(20, 1e6)
ax.grid(True, which='major', ls='--', alpha=0.2, color='#aaa')
ax.grid(True, which='minor', ls=':', alpha=0.10, color='#bbb')
ax.set_facecolor('#F9F9F9')

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.10)
cbar.set_label('Noise Ratio  $\\sigma_r$', fontsize=44)
cbar.ax.tick_params(labelsize=36)

legend_elements = [
    Line2D([0],[0], marker=PATTERN_MARKER[p], color='w',
           markerfacecolor='#bbb', markeredgecolor=PATTERN_COLOR[p],
           markeredgewidth=2.0, markersize=28, label=PATTERN_LABEL[p])
    for p in ['FM-dominant', 'Early', 'Mid', 'Late'] if (feat['pattern'] == p).any()
]
legend_elements.append(
    Line2D([0],[0], color='#555', lw=2.6, ls='--', label='$n=700$ (illustrative)')
)
fig.legend(handles=legend_elements, fontsize=36, loc='lower center',
           bbox_to_anchor=(0.44, -0.04), ncol=3,
           framealpha=0.95, edgecolor='#ccc',
           title='Break-Even Pattern', title_fontsize=40)

plt.tight_layout(rect=[0, 0.10, 1, 1])
os.makedirs('results', exist_ok=True)
plt.savefig('results/fig2_feature_space.pdf', bbox_inches='tight', dpi=200)
plt.savefig('results/fig2_feature_space.png', bbox_inches='tight', dpi=200)
print(f"Saved: results/fig2_feature_space.pdf / .png  ({len(feat)} datasets)")
