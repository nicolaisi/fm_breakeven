#!/usr/bin/env python3
"""RQ2/RQ3 feature analysis on the rerun categories.

Random forest (500 trees, depth 4, random_state 42) with leave-one-out CV on the
four paper features (n_train, S, noise ratio, |rho|_50), plus the six-feature
variant for reference; point-biserial correlations of each feature with each
category. Categories come from src/results_io.detect_breakeven.

Usage: python scripts/feature_analysis.py
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from results_io import load_results, detect_breakeven  # noqa: E402

FEATS4 = ['n_train', 'seas_strength', 'noise_ratio', 'acf_mean_50']
FEATS6 = FEATS4 + ['dfa_hurst', 'periodogram_sharpness']
THREE_WAY = {'FM-dominant': 'FM-dominant', 'Early': 'Early+Mid', 'Mid': 'Early+Mid', 'Late': 'Late'}

cl, fm = load_results()
feat = pd.read_csv(os.path.join(os.environ.get('FM_RESULTS_DIR') or os.path.join(ROOT, 'results'),
                                'dataset_features.csv'))
feat = feat.sort_values('dataset').reset_index(drop=True)   # the forest depends on row order
feat['pattern'] = feat.dataset.apply(lambda d: detect_breakeven(cl, fm, d)[0])
print('categories:', feat.pattern.value_counts().to_dict(), f'({len(feat)} datasets)')


def loo(cols, labels):
    X, y = feat[cols].values, np.asarray(labels)
    pred = []
    for tr, te in LeaveOneOut().split(X):
        clf = RandomForestClassifier(n_estimators=500, max_depth=4, random_state=42)
        clf.fit(X[tr], y[tr])
        pred.append(clf.predict(X[te])[0])
    pred = np.array(pred)
    acc = (pred == y).mean()
    base = max(Counter(y).values()) / len(y)
    per_class = {c: f"{((pred == c) & (y == c)).sum()}/{(y == c).sum()}" for c in sorted(set(y))}
    return acc, base, per_class


for name, cols in [('4 features (paper)', FEATS4), ('6 features', FEATS6)]:
    for lab_name, labels in [('four-class', feat.pattern), ('three-class (Early+Mid)', feat.pattern.map(THREE_WAY))]:
        acc, base, pc = loo(cols, labels)
        print(f"{name:20s} {lab_name:24s} LOO accuracy {acc:.1%}  (majority baseline {base:.1%})  correct per class {pc}")

clf = RandomForestClassifier(n_estimators=500, max_depth=4, random_state=42).fit(feat[FEATS4].values, feat.pattern.values)
print('\nfeature importances (4 features, all data):',
      dict(zip(FEATS4, np.round(clf.feature_importances_, 3))))

print('\npoint-biserial correlation of each feature with each category (r, p):')
rows = []
for f in FEATS6:
    for pat in ['FM-dominant', 'Early', 'Late']:
        b = (feat.pattern == pat).astype(float)
        if b.sum() == 0:
            continue
        r, p = pearsonr(feat[f].values, b.values)
        rows.append(dict(feature=f, category=pat, r=round(r, 3), p=round(p, 3)))
corr = pd.DataFrame(rows)
print(corr.pivot(index='feature', columns='category', values='r').to_string())
sig = corr[corr.p < 0.05]
print('\nsignificant at p < 0.05:', sig.to_dict('records') or 'none')
