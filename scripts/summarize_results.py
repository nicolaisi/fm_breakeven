#!/usr/bin/env python3
"""Every number the paper and the slides quote, from the rerun results.

Usage:  python scripts/summarize_results.py
Writes: results/summary_breakeven.csv (one row per dataset)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from results_io import load_results, detect_breakeven, best_zero_shot  # noqa: E402

ORDER = ['FM-dominant', 'Early', 'Mid', 'Late']
short = lambda m: m.replace('Chronos-Bolt', 'Chr-Bolt')

cl, fm = load_results()
datasets = sorted(set(cl.dataset) & set(fm.dataset))
zs = fm[(fm.setting == 'zero-shot') & fm.mase_fixed.notna()]
lora = fm[(fm.setting == 'fine-tuned-lora') & fm.mase_fixed.notna()]
feat_path = os.path.join(ROOT, 'results', 'dataset_features.csv')
feat = pd.read_csv(feat_path).set_index('dataset') if os.path.exists(feat_path) else None

rows = []
for ds in datasets:
    pat, nstar, frac = detect_breakeven(cl, fm, ds)
    z = zs[zs.dataset == ds]
    zmodel, zmase = best_zero_shot(fm, ds)                      # seed mean, best model
    c_full = cl[(cl.dataset == ds) & (cl.train_fraction == 1.0) & cl.mase_fixed.notna()]
    cbest = c_full.loc[c_full.mase_fixed.idxmin()]
    r = dict(dataset=ds, pattern=pat, n_star=nstar, frac=frac,
             n_train=int(c_full.n_train.median()),
             best_FM=short(zmodel), FM_mase=zmase,
             best_CL=cbest.model, CL_mase=cbest.mase_fixed)
    r['CL_wins_full'] = cbest.mase_fixed <= zmase
    r['reversal'] = pat != 'FM-dominant' and not r['CL_wins_full']
    r['gap_full'] = cbest.mase_fixed - zmase
    l = lora[(lora.dataset == ds) & (lora.train_fraction == 1.0)]
    if len(l):
        # LoRA varies across seeds, zero-shot hardly: compare seed means, not best-of-3
        lb = l.groupby('model').mase_fixed.mean()
        r['best_LoRA'], r['LoRA_mase'] = short(lb.idxmin()), lb.min()
        r['LoRA_mase_bestseed'] = l.groupby('model').mase_fixed.min().min()
        r['FM_mase_seedmean'] = z.groupby('model').mase_fixed.mean().min()
    if feat is not None and ds in feat.index:
        r['S'] = feat.loc[ds, 'seas_strength'] if 'seas_strength' in feat else np.nan
    rows.append(r)
df = pd.DataFrame(rows)
df['pattern'] = pd.Categorical(df.pattern, ORDER, ordered=True)
df = df.sort_values(['pattern', 'n_star', 'dataset'])
pd.set_option('display.width', 250)
cols = [c for c in ['dataset', 'pattern', 'n_star', 'frac', 'n_train', 'best_FM', 'FM_mase',
                    'best_CL', 'CL_mase', 'CL_wins_full', 'reversal', 'best_LoRA', 'LoRA_mase', 'S']
        if c in df]
print(df[cols].round(3).to_string(index=False))

print('\n== Categories ==')
print(df.pattern.value_counts().reindex(ORDER).to_dict())
fin = df[df.n_star.notna()]
for name, sub in [('all', fin), ('Early', fin[fin.pattern == 'Early']),
                  ('Mid+Late', fin[fin.pattern.isin(['Mid', 'Late'])])]:
    if len(sub):
        print(f"n* range {name}: {int(sub.n_star.min()):,} – {int(sub.n_star.max()):,}"
              f"  (ratio {sub.n_star.max() / sub.n_star.min():.0f}x)")
print('Reversals (classical wins first, FM ahead at full data):', df[df.reversal].dataset.tolist())
fmd = df[df.pattern == 'FM-dominant']
close = fmd[fmd.gap_full < 0.1 * fmd.FM_mase]
print(f"FM-dominant with full-data margin < 10% of FM MASE: {len(close)} {close.dataset.tolist()}")

print('\n== Findings ==')
print(f"Finding 1: zero-shot FM beats best classical at full data on {(~df.CL_wins_full).sum()}/30")
print('Finding 4: best zero-shot FM (all datasets):', df.best_FM.value_counts().to_dict())
print('           best zero-shot FM (FM-dominant):', fmd.best_FM.value_counts().to_dict())
print('Finding 5: best classical at full data:', df.best_CL.value_counts().to_dict())

if feat is not None and 'S' in df:
    rule = df[(df.n_train < 700) & (df.S >= 0.05)]
    early_win = rule.frac.fillna(9)
    print(f"\n== Decision rule (n_train < 700 and S >= 0.05): {len(rule)} datasets ==")
    print('  classical first matches FM below 20%:', rule[early_win < 0.2].dataset.tolist() or 'none')
    print('  final (FM-dominant):', rule[rule.pattern == 'FM-dominant'].dataset.tolist())
    print('  provisional (classical overtakes at >=20%):', rule[rule.pattern != 'FM-dominant'].dataset.tolist())
    exc = df[(df.n_train < 700) & (df.S < 0.05)]
    print('  exceptions S < 0.05:', exc[['dataset', 'pattern', 'n_star']].to_dict('records'))
else:
    print('\n(no results/dataset_features.csv: run scripts/compute_features.py for the rule check)')

print('\n== Table III: mean MASE at full data over datasets and seeds; k = best within class ==')
full_fm = fm[(fm.train_fraction == 1.0) & fm.mase_fixed.notna()]
t = full_fm.groupby(['model', 'setting']).mase_fixed.mean().unstack().round(3)
fm_best = full_fm.groupby(['dataset', 'model', 'setting']).mase_fixed.mean().reset_index()
# k counted within each setting: best zero-shot model, best LoRA model per dataset
winners = fm_best.loc[fm_best.groupby(['dataset', 'setting']).mase_fixed.idxmin()]
t_k = winners.groupby(['model', 'setting']).size().unstack(fill_value=0)
print(t.join(t_k, rsuffix='_k').fillna('').to_string())
cfull = cl[(cl.train_fraction == 1.0) & cl.mase_fixed.notna()]
ct = cfull.groupby('model').mase_fixed.mean().round(3).to_frame('mean MASE')
ct['k'] = df.best_CL.value_counts()
print(ct.fillna(0).to_string())

if len(lora):
    have = df[df.LoRA_mase.notna()] if 'LoRA_mase' in df else df.iloc[:0]
    print(f"\n== LoRA (datasets with LoRA results: {len(have)}) ==")
    have = have.assign(rel=have.LoRA_mase / have.FM_mase_seedmean - 1)
    print(f"Best LoRA model (seed mean, full data) better than best zero-shot (seed mean): "
          f"{(have.rel < 0).sum()}/{len(have)}; median change {have.rel.median():+.1%}; "
          f">5% better {(have.rel < -0.05).sum()}, >5% worse {(have.rel > 0.05).sum()}")
    print(f"(best-of-3-seeds comparison, biased towards LoRA: "
          f"{(have.LoRA_mase_bestseed < have.FM_mase).sum()}/{len(have)})")
    pair = pd.concat([zs.groupby(['dataset', 'model']).mase_fixed.mean().rename('zs'),
                      lora[lora.train_fraction == 1.0].groupby(['dataset', 'model']).mase_fixed.mean().rename('lora')],
                     axis=1).dropna()
    pr = pair.lora / pair.zs - 1
    print(f"Paired (same model, seed mean): LoRA better in {(pr < 0).sum()}/{len(pr)}, median change {pr.median():+.1%}")
    print(have.sort_values('rel')[['dataset', 'best_FM', 'FM_mase_seedmean', 'best_LoRA', 'LoRA_mase', 'rel']]
          .round(3).to_string(index=False))

print('\nModels included:', fm.groupby('setting').model.unique().apply(list).to_dict())
out = os.path.join(os.environ.get('FM_RESULTS_DIR') or os.path.join(ROOT, 'results'), 'summary_breakeven.csv')
df.to_csv(out, index=False)
print('Saved', out)
