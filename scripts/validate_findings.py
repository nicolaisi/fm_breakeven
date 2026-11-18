"""
Checks every result-dependent number in the paper against the rerun CSVs.

Each check recomputes a value from results/ (classical_rerun.csv,
foundation_rerun_*.csv, dataset_features.csv; fixed-denominator MASE, FM seed
means) and then confirms that the text claiming it appears verbatim in the tex,
so both a changed result and an edited sentence are caught.

Run from: ~/Development/fm_breakeven/
Usage: python scripts/validate_findings.py [--tex PATH]

The paper source is not part of the public repository. Without it, the
recomputed values are still checked against the numbers reported in the
paper, but the text checks (and Tables I-III, which are only checked
against the tex) are skipped.

Exits with code 0 if all checks pass, 1 if any fail.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from results_io import load_results, detect_breakeven, best_zero_shot, FRACS  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--tex', default=os.path.join(ROOT, 'icdm_workshop_final', 'conference_101719.tex'))
args = parser.parse_args()
TEX = open(args.tex).read() if os.path.exists(args.tex) else None
if TEX is None:
    print(f'Paper tex not found ({args.tex}): checking values only, text and table checks skipped.')

errors = []
n_checks = 0


def check(label, ok, detail=''):
    global n_checks
    n_checks += 1
    print(f"  {'✓' if ok else '✗'}  {label}" + ('' if ok else f'  <-- FAILED  {detail}'))
    if not ok:
        errors.append(label)


def claim(label, value, expected, *snippets):
    """value (recomputed) must equal expected, and every snippet must be in the tex."""
    missing = [s for s in snippets if s not in TEX] if TEX is not None else []
    check(label, value == expected and not missing,
          f'computed {value!r}, paper {expected!r}' + (f'; not in tex: {missing}' if missing else ''))


def pct(x):
    return int(round(x))


# ── Data ──────────────────────────────────────────────────────────────────────
cl, fm = load_results()
feat = pd.read_csv(os.path.join(ROOT, 'results', 'dataset_features.csv')).set_index('dataset')
DATASETS = sorted(fm.dataset.unique())
zs = fm[fm.setting == 'zero-shot']
lora = fm[fm.setting == 'fine-tuned-lora']
full = fm[fm.setting == 'fine-tuned-full']

rows = []
for ds in DATASETS:
    pat, n_star, frac = detect_breakeven(cl, fm, ds)
    zm, zv = best_zero_shot(fm, ds)
    zm = zm.replace('Chronos-Bolt', 'Chr-Bolt')
    c_full = cl[(cl.dataset == ds) & (cl.train_fraction == 1.0) & cl.mase_fixed.notna()]
    best_cl = c_full.loc[c_full.mase_fixed.idxmin()]
    rows.append(dict(dataset=ds, pattern=pat, n_star=n_star, frac=frac, zs_model=zm, zs=zv,
                     cl_model=best_cl.model, cl=best_cl.mase_fixed,
                     n_train=int(c_full.n_train.median())))
B = pd.DataFrame(rows).set_index('dataset').join(feat.drop(columns='n_train'))
B['cl_wins_full'] = B.cl <= B.zs
B['margin'] = (B.zs - B.cl).abs() / B.zs * 100


def best_per_dataset(d):
    return d.groupby(['dataset', 'model']).mase_fixed.mean().groupby('dataset').min()


lora_full = best_per_dataset(lora[lora.train_fraction == 1.0])
full_full = best_per_dataset(full[full.train_fraction == 1.0])
lora_rel = (lora_full / B.zs - 1) * 100

# ── RQ1: categories and break-even ranges ─────────────────────────────────────
print('\nRQ1: categories and break-even points')
cnt = B.pattern.value_counts().to_dict()
claim('FM-dominant 16', cnt.get('FM-dominant'), 16, r'\textbf{FM-dominant (16 datasets):}',
      'on 16 of 30 datasets')
claim('Early 6', cnt.get('Early'), 6, r'\textbf{Early break-even (6):}')
claim('Late 8', cnt.get('Late'), 8, r'\textbf{Late (8):}')
claim('no Mid', cnt.get('Mid', 0), 0, 'No dataset breaks even at 10\\%.')
early, late = B[B.pattern == 'Early'], B[B.pattern == 'Late']
claim('Early n* range', (int(early.n_star.min()), int(early.n_star.max())), (21, 6324), '21--6,324 samples')
claim('Late n* range', (int(late.n_star.min()), int(late.n_star.max())), (24, 22177), '24--22,177 samples')
claim('Early within 5%', bool((early.frac <= 0.05).all()), True, 'at most 5\\% of the data')
claim('all Early persist to 100%', bool(early.cl_wins_full.all()), True, 'the advantage persists to 100\\%')
claim('non-monotonic Late', sorted(late[~late.cl_wins_full].index), ['M4-Daily', 'Sunspot'],
      'M4-Daily and Sunspot are non-monotonic')
claim('FM ahead at full data', int((~B.cl_wins_full).sum()), 18, 'on 18 of 30 datasets')
fragile_fm = sorted(B[(B.pattern == 'FM-dominant') & (B.margin < 5)].index)
claim('fragile FM-dominant (<5%)', fragile_fm, ['Dominick', 'NN5-Daily', 'USBirths'],
      'USBirths 0.4\\%, Dominick 2.2\\%, NN5-Daily 4.8\\%')
claim('fragile margins', [round(B.margin[d], 1) for d in ['USBirths', 'Dominick', 'NN5-Daily', 'M4-Monthly-Monash']],
      [0.4, 2.2, 4.8, 1.5], 'M4-Monthly-M.\\ 1.5\\%')

# ── Table II rows ─────────────────────────────────────────────────────────────
print('\nTable II rows')
short = {'M4-Monthly-Monash': 'M4-Monthly-M.'}
bad = []
for ds, r in B.iterrows():
    ns = '---' if pd.isna(r.n_star) else f'{int(r.n_star):,}'
    row = (f"{short.get(ds, ds)} & {r.zs_model} & {r.zs:.3f} & {r.cl_model} & {r.cl:.3f} & {ns} & "
           f"{r.n_train:,} & {int(r.tau_acf)} & {r.hurst:.2f} & {r.noise_ratio:.2f} & "
           f"{r.seas_strength:.2f} & {r.spec_entropy:.2f}")
    if TEX is not None and row not in TEX:
        bad.append(ds)
if TEX is not None:
    check('all 30 rows match the results', not bad, f'rows differ: {bad}')

# ── Table III ─────────────────────────────────────────────────────────────────
print('\nTable III')
zs_mean = zs.groupby('model').mase_fixed.mean()
lora_mean = lora[lora.train_fraction == 1.0].groupby('model').mase_fixed.mean()
zs_k = Counter(B.zs_model)
lora_k = Counter(lora[lora.train_fraction == 1.0].groupby(['dataset', 'model']).mase_fixed.mean()
                 .unstack().idxmin(axis=1))
cl_mean = cl[cl.train_fraction == 1.0].groupby('model').mase_fixed.mean()
cl_k = Counter(B.cl_model)
short_fm = {'Chronos-Bolt-S': 'Chr-Bolt-S', 'Chronos-Bolt-B': 'Chr-Bolt-B'}
for m in ['Chronos-Bolt-S', 'Chronos-Bolt-B', 'Moirai-S', 'Moirai-B', 'Moirai-L', 'Lag-Llama']:
    k = short_fm.get(m, m)
    cells = f"{m:<14} & {zs_mean[m]:.2f} & {zs_k.get(k, 0):<2} & "
    cells += (f"{lora_mean[m]:.2f} & {lora_k.get(m, 0):<2} &" if m in lora_mean else '--   & -- &')
    if TEX is not None:
        check(f'{m} row', cells in TEX, f'expected "{cells}"')
for m in ['ARIMA', 'Naive', 'ETS', 'XGBoost']:
    cells = f"& {m:<7} & {cl_mean[m]:.2f} & {cl_k.get(m, 0)}"
    if TEX is not None:
        check(f'{m} cells', cells in TEX, f'expected "{cells}"')

# ── Findings ──────────────────────────────────────────────────────────────────
print('\nFindings 2-5')
claim('LoRA better than ZS', int((lora_rel < 0).sum()), 16, 'LoRA improves over zero-shot on 16 of 30')
claim('LoRA >5% better / worse', (int((lora_rel < -5).sum()), int((lora_rel > 5).sum())), (3, 11),
      'more than 5\\% better on 3 datasets and more than 5\\% worse on 11')
claim('LoRA median improvement', round(-lora_rel.median(), 1), 0.6, 'the median improvement is only 0.6\\%')
claim('LoRA degradation examples',
      [pct(lora_rel[d]) for d in ['M4-Weekly', 'M4-Monthly-Monash', 'M3-Quarterly', 'Tourism-Monthly',
                                  'M4-Hourly', 'Rideshare']],
      [52, 51, 28, 23, 15, 10],
      '52\\% worse than the best zero-shot model on M4-Weekly, 51\\% on M4-Monthly-M., 28\\% on M3-Quarterly, '
      '23\\% on Tourism-Monthly, 15\\% on M4-Hourly', 'and 10\\% on Rideshare')
m4h = lora[(lora.dataset == 'M4-Hourly') & (lora.train_fraction == 1.0)].groupby('model').mase_fixed.mean()
claim('M4-Hourly LoRA vs ZS', (m4h.idxmin(), round(m4h.min(), 3), B.zs_model['M4-Hourly'], round(B.zs['M4-Hourly'], 3)),
      ('Chronos-Bolt-B', 0.852, 'Chr-Bolt-B', 0.738), '(Chr-Bolt-B, 0.852 vs.\\ 0.738)')
m4w = lora[(lora.dataset == 'M4-Weekly') & (lora.train_fraction == 1.0)].groupby('model').mase_fixed.mean()
claim('Fig. 3 M4-Weekly', (B.zs_model['M4-Weekly'], round(B.zs['M4-Weekly'], 3), round(m4w.min(), 3)),
      ('Chr-Bolt-S', 0.509, 0.772), 'best ZS Chr-Bolt-S 0.509 vs.\\ best LoRA 0.772')
claim('Moirai-S LoRA vs ZS', (round(lora_mean['Moirai-S'], 2), round(zs_mean['Moirai-S'], 2)), (1.25, 1.24),
      'Moirai-S LoRA (1.25) is no better than zero-shot (1.24)')
claim('full FT beats LoRA / ZS', (int((full_full < lora_full).sum()), int((full_full < B.zs).sum())), (9, 9),
      'beats LoRA and zero-shot on only 9 of 30 datasets each')
cat = full[(full.train_fraction <= 0.05) & (full.mase_fixed > 10)]
claim('full FT failures', (len(cat), sorted(cat.dataset.unique()), sorted(cat.model.unique()),
                           round(cat.mase_fixed.min(), 1), round(cat.mase_fixed.max(), 1), round(B.zs['Hospital'], 2)),
      (6, ['Hospital'], ['Moirai-L'], 20.1, 26.0, 1.22),
      'Moirai-L reaches MASE 20.1--26.0 on Hospital versus zero-shot 1.22 (6 runs with MASE $>$ 10)')
fam = B.zs_model.str.split('-').str[0].map({'Chr': 'Chronos-Bolt', 'Moirai': 'Moirai', 'Lag': 'Lag-Llama'})
claim('best ZS family counts', (int((fam == 'Chronos-Bolt').sum()), zs_k['Chr-Bolt-B'], zs_k['Chr-Bolt-S'],
                                int((fam == 'Moirai').sum()), sorted(B[fam == 'Lag-Llama'].index)),
      (20, 12, 8, 10, []),
      'best zero-shot FM on 20 of 30 datasets (Base 12, Small 8); Moirai variants win on 10',
      'Lag-Llama is best on none')
claim('best classical counts', (cl_k['ARIMA'], cl_k['Naive'], cl_k['ETS'], cl_k['XGBoost']), (11, 7, 6, 6),
      'ARIMA is best on 11 datasets, Naive on 7, and ETS and XGBoost on 6 each')
claim('XGBoost wins', sorted(B[B.cl_model == 'XGBoost'].index),
      sorted(['AusElectricity', 'ETTh2', 'KDD-Cup-Hourly', 'Pedestrian', 'Sunspot', 'NN5-Daily']),
      '(AusElectricity, ETTh2, KDD-Cup-Hourly, Pedestrian, Sunspot) and NN5-Daily')
claim('XGBoost worst mean', (cl_mean.idxmax(), round(cl_mean.max(), 2)), ('XGBoost', 2.43),
      'has the worst mean MASE (2.43)')

# ── Seed variance (Sec. III) ──────────────────────────────────────────────────
print('\nSeed variance')
zsd = pd.Series({ds: zs[(zs.dataset == ds) & (zs.model == best_zero_shot(fm, ds)[0])].mase_fixed.std()
                 for ds in DATASETS})
claim('ZS seed std', (round(zsd.mean(), 3), round(zsd.median(), 3), round(zsd.max(), 3), zsd.idxmax()),
      (0.004, 0.0, 0.024, 'M4-Daily'), 'average 0.004 (median 0.000', 'maximum 0.024, Moirai-S on M4-Daily')
sd = fm.groupby(['dataset', 'model', 'setting', 'train_fraction']).mase_fixed.std().reset_index()
lo_low = sd[(sd.setting == 'fine-tuned-lora') & (sd.train_fraction <= 0.05)].mase_fixed
lo_full = sd[(sd.setting == 'fine-tuned-lora') & (sd.train_fraction == 1.0)].mase_fixed
ft_low = sd[(sd.setting == 'fine-tuned-full') & (sd.train_fraction <= 0.05)].mase_fixed
claim('LoRA / full FT seed std', (round(lo_low.mean(), 3), round(lo_low.max(), 2), round(lo_full.mean(), 3),
                                  round(ft_low.mean(), 3)), (0.058, 0.97, 0.066, 0.095),
      'averages 0.058 at low fractions ($\\leq$5\\%, max 0.97) and 0.066 at full data',
      '(mean 0.095)', '(seed std 0.058 vs.\\ 0.095)')

# ── RQ2 / RQ3 ─────────────────────────────────────────────────────────────────
print('\nRQ2 / RQ3')
claim('Early noise <= 0.11 on 5 of 6', int((early.noise_ratio <= 0.115).sum()), 5, 'five of six have $\\sigma_r \\leq 0.11$')
claim('Early best classical simple', sorted(early.cl_model.unique()), ['ARIMA', 'ETS', 'Naive'],
      'simple extrapolator (Naive, ETS, or ARIMA)')
claim('Late short', int((late.n_train < 500).sum()), 7, '$n_{\\text{train}} < 500$ for 7 of 8')
med = B.groupby('pattern').noise_ratio.median().round(2)
claim('median noise by category', (med['Early'], med['FM-dominant'], med['Late']), (0.05, 0.25, 0.44),
      'median $\\sigma_r = 0.05$, vs.\\ 0.25 for FM-dominant and 0.44 for Late')
long_ = B[B.n_train > 4000].pattern.value_counts()
claim('long series split', (long_.get('FM-dominant'), long_.get('Early'), long_.get('Late')), (10, 5, 1),
      '10 FM-dominant, 5 Early, 1 Late')
claim('short datasets', (int((B.n_train < 700).sum()), sorted(B[(B.n_train < 700) & (B.pattern == 'Early')].index)),
      (14, ['FRED-MD']), 'of the 14 short datasets only FRED-MD')

feats4 = ['n_train', 'seas_strength', 'noise_ratio', 'acf_mean_50']
X, y = B[feats4].values, B.pattern.values
pred = []
for tr, te in LeaveOneOut().split(X):
    clf = RandomForestClassifier(n_estimators=500, max_depth=4, random_state=42).fit(X[tr], y[tr])
    pred.append(clf.predict(X[te])[0])
pred = np.array(pred)
claim('RF LOO accuracy / baseline', (round((pred == y).mean() * 100, 1), round(max(Counter(y).values()) / len(y) * 100, 1)),
      (40.0, 53.3), 'reaches 40.0\\% accuracy', 'below the 53.3\\% majority-class baseline')
rf_range = []
for rs in range(20):
    p_rs = [RandomForestClassifier(n_estimators=500, max_depth=4, random_state=rs, n_jobs=-1).fit(X[tr], y[tr]).predict(X[te])[0]
            for tr, te in LeaveOneOut().split(X)]
    rf_range.append(round((np.array(p_rs) == y).mean() * 100, 1))
claim('RF LOO accuracy over 20 forest seeds', (min(rf_range), max(rf_range)), (40.0, 43.3),
      '(40.0--43.3\\% over 20 forest seeds)')
claim('RF recovers Early / Late', (int(((pred == y) & (y == 'Early')).sum()), int(((pred == y) & (y == 'Late')).sum())),
      (0, 2), 'it recovers none of the 6 Early and only 2 of 8 Late')
corr = {}
for f in feats4 + ['dfa_hurst', 'periodogram_sharpness']:
    for p in ['FM-dominant', 'Early', 'Late']:
        corr[f, p] = pearsonr(B[f].values, (B.pattern == p).astype(float).values)
claim('correlations', [(round(corr[k][0], 2), round(corr[k][1], 3)) for k in
                       [('acf_mean_50', 'Early'), ('acf_mean_50', 'Late'), ('noise_ratio', 'Early')]],
      [(0.40, 0.030), (-0.43, 0.019), (-0.37, 0.045)],
      '($r = 0.40$, $p = 0.030$)', '($r = -0.43$, $p = 0.019$)', '($r = -0.37$, $p = 0.045$)')
claim('none survives Bonferroni (18 tests)', (len(corr), min(p for _, p in corr.values()) < 0.05 / len(corr)), (18, False),
      'Bonferroni correction over the 18 feature--category tests')

# ── Decision rule (Sec. V) ────────────────────────────────────────────────────
print('\nDecision rule')
rule = B[(B.n_train < 700) & (B.seas_strength >= 0.05)]
claim('rule covers 10', len(rule), 10, 'On all 10 such datasets')
claim('no classical match below 20%', bool((rule.frac.isna() | (rule.frac >= 0.2)).all()), True,
      'FM zero-shot is unmatched below 20\\% of the history')
first_frac = cl[cl.mase_fixed.notna()].groupby('dataset').train_fraction.min()
tested = sorted(rule.index[first_frac.reindex(rule.index) < 0.2])
claim('tested below 20% on 4, unfittable on 6', (tested, len(rule) - len(tested)),
      (['ILI', 'M4-Daily', 'M4-Hourly', 'NN5-Daily'], 6),
      'on 4 (M4-Hourly, NN5-Daily, ILI, M4-Daily) classical methods are fitted there and lose',
      'classical methods either lose to it (4 of these 10 datasets) or are too short to fit at all (6)')
claim('n* is an upper bound on 10 of 14', (int((B.frac.notna() & (B.frac == first_frac.reindex(B.index))).sum()), int(B.frac.notna().sum())),
      (10, 14), '(10 of the 14 datasets with a break-even)')
claim('classical first fittable at n* / only at 100%',
      (sorted(B.index[B.frac.notna() & (B.frac == first_frac.reindex(B.index)) & (B.pattern == 'Late')]),
       sorted(B.index[first_frac.reindex(B.index) == 1.0])),
      (['COVID-Deaths', 'Hospital', 'M3-Monthly', 'Tourism-Monthly'], ['Dominick', 'M3-Quarterly']),
      'M3-Quarterly and Dominick are FM-dominant on the 100\\% fraction alone',
      'on M3-Monthly, Hospital, Tourism-Monthly, and COVID-Deaths $n^*$ is the first fraction')
claim('final / provisional', (int((rule.pattern == 'FM-dominant').sum()), int((rule.pattern == 'Late').sum())), (5, 5),
      'final for the 5 FM-dominant cases; for the 5 Late cases', 'final on 5, provisional on 5')
claim('FM best at full on 6 of the 10', int((~rule.cl_wins_full).sum()), 6, 'stays best at full history on 6 of them')
claim('S < 0.05 exceptions', [(d, int(B.n_star[d])) for d in ['FRED-MD', 'COVID-Deaths']], [('FRED-MD', 21), ('COVID-Deaths', 25)],
      'FRED-MD: $n^*=21$; COVID-Deaths: $n^*=25$')
claim('M3-Monthly example', (B.cl_model['M3-Monthly'], round(B.cl['M3-Monthly'], 3), round(B.zs['M3-Monthly'], 3)),
      ('ARIMA', 0.602, 2.061), 'M3-Monthly: ARIMA 0.602 vs.\\ FM 2.061')
claim('M4-Hourly example', (B.zs_model['M4-Hourly'], round(B.zs['M4-Hourly'], 3), B.cl_model['M4-Hourly'],
                            round(B.cl['M4-Hourly'], 3)), ('Chr-Bolt-B', 0.738, 'ARIMA', 0.996),
      'Chr-Bolt-B zero-shot (0.738) beats ARIMA (0.996)')

# ── Tables I and III ──────────────────────────────────────────────────────────
print('\nTables I and III')
from data_loader import DATASETS as CONFIGS  # noqa: E402
ABBREV = {'M4-Mon.-M.': 'M4-Monthly-Monash', 'Tourism-M.': 'Tourism-Monthly',
          'KDD-Cup-H.': 'KDD-Cup-Hourly', 'AusElec.': 'AusElectricity'}


def table3(d, models):
    m = d.groupby(['dataset', 'model']).mase_fixed.mean().unstack()
    k = m.idxmin(axis=1).value_counts()
    return {mo: (round(m[mo].mean(), 2), int(k.get(mo, 0))) for mo in models}


if TEX is not None:
    t1 = TEX[TEX.index('label{tab:datasets}'):]
    t1 = t1[t1.index('\\midrule'):t1.index('\\bottomrule')]
    rows = {}
    for line in t1.splitlines()[1:]:
        cells = [c.strip() for c in line.rstrip('\\ ').split('&')]
        for name, length, h in [(cells[0], cells[3], cells[5]), (cells[6], cells[9], cells[11])]:
            rows[ABBREV.get(name, name)] = (int(length.replace(',', '')), int(h))
    n_obs = (cl[cl.train_fraction == 1.0].groupby('dataset').n_train.median() / 0.6).round()
    check('Table I lists all 30 datasets', sorted(rows) == DATASETS, sorted(set(rows) ^ set(DATASETS)))
    check('Table I horizons match configs', all(rows[d][1] == CONFIGS[d].horizon for d in rows),
          [(d, rows[d][1], CONFIGS[d].horizon) for d in rows if rows[d][1] != CONFIGS[d].horizon])
    check('Table I lengths match n_train / 0.6', all(abs(rows[d][0] - n_obs[d]) <= 2 for d in rows),
          [(d, rows[d][0], n_obs[d]) for d in rows if abs(rows[d][0] - n_obs[d]) > 2])

    zs_t = table3(zs, ['Chronos-Bolt-S', 'Chronos-Bolt-B', 'Moirai-S', 'Moirai-B', 'Moirai-L', 'Lag-Llama'])
    lo_t = table3(lora[lora.train_fraction == 1.0],
                  ['Chronos-Bolt-S', 'Chronos-Bolt-B', 'Moirai-S', 'Moirai-B', 'Moirai-L'])
    cl_t = table3(cl[cl.train_fraction == 1.0], ['ARIMA', 'Naive', 'ETS', 'XGBoost'])
    t3 = TEX[TEX.index('label{tab:model_comparison}'):]
    t3 = t3[t3.index('\\midrule'):t3.index('\\bottomrule')]
    cells = [[c.strip() for c in line.rstrip('\\ ').split('&')] for line in t3.splitlines()[1:]]
    tex_t3 = {}
    for c in cells:
        tex_t3[('ZS', c[0])] = (float(c[1]), int(c[2]))
        if c[3] != '--':
            tex_t3[('LoRA', c[0])] = (float(c[3]), int(c[4]))
        if c[5]:
            tex_t3[('CL', c[5])] = (float(c[6]), int(c[7]))
    want = {**{('ZS', k): v for k, v in zs_t.items()}, **{('LoRA', k): v for k, v in lo_t.items()},
            **{('CL', k): v for k, v in cl_t.items()}}
    check('Table III mean MASE and k', tex_t3 == want,
          [(k, tex_t3.get(k), v) for k, v in want.items() if tex_t3.get(k) != v])

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{n_checks - len(errors)}/{n_checks} checks passed")
if errors:
    print('FAILED:', errors)
sys.exit(1 if errors else 0)
