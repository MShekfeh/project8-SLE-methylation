# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 12:49:52 2026

@author: marwa
"""

"""
Project 8, Stage 2b -- Linear ceiling check (does ANY model find signal?).
Same outer folds as Stage 2 (StratifiedKFold, 5 splits, seed 42).
  (a) elastic-net logistic regression on Project 6's 5,000-probe vocabulary
  (b) same model on ALL probes, top-K chosen by t-test INSIDE each training fold
  (c) single a-priori probe cg06872964 (chosen from literature before seeing labels)
"""
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, TOP_K, IFN_PROBE = 42, 1000, "cg06872964"
FULL = "../data/gse82218_beta_matrix_full.parquet"
IMPUTED = "../data/gse82218_zeroshot_input_imputed.parquet"
PROBES = "../data/project6_cpg_probe_list.txt"

# ---- Load full matrix (wide parquet needs the raised thrift limits) ----
pf = pq.ParquetFile(FULL, thrift_string_size_limit=10**9, thrift_container_size_limit=10**9)
table = pf.read()
probe_names = [n for n in table.schema.names if n != "gsm_id"]
X_full = np.stack([table.column(n).to_numpy() for n in probe_names], axis=1).astype(np.float32)
del table  # free the Arrow table; only the numpy copy is needed
print("full matrix:", X_full.shape, flush=True)

# ---- Labels come from the imputed file; verify alignment on ALL 5,000 probes (Rule #12) ----
imp = pd.read_parquet(IMPUTED)
vocab = [l.strip() for l in open(PROBES)]
col_idx = {n: i for i, n in enumerate(probe_names)}
sub = X_full[:, [col_idx[p] for p in vocab]]
X5 = imp[vocab].values.astype(np.float32)
valid = ~np.isnan(sub)  # imputed cells legitimately differ from raw NaNs
assert np.allclose(sub[valid], X5[valid], atol=1e-5), "Row order mismatch!"
print(f"alignment verified on {valid.sum()} values across 5,000 probes", flush=True)

y = (imp["group"].values != "NC").astype(int)  # SLE-combined = 1
print("labels: SLE-combined", y.sum(), "| NC", (1 - y).sum(), flush=True)

# ---- Drop probes with ANY NaN (label-independent, so no leakage) ----
keep = ~np.isnan(X_full).any(axis=0)
X_all = X_full[:, keep]
names_all = np.array(probe_names)[keep]
print(f"probes without NaN: {keep.sum()} of {len(keep)}", flush=True)


def elastic_net():  # name kept for a minimal diff; model is now L2 (ridge) + lbfgs
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(solver="lbfgs", Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], cv=3,
                             scoring="roc_auc", max_iter=2000, random_state=SEED),
    )


def single_probe_lr():
    return make_pipeline(StandardScaler(), LogisticRegression())


def run(label, X, make_model, select_in_fold=False, names=None):
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores, oof = [], np.zeros(len(y))
    selected_sets = []
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        Xtr, Xte = X[tr], X[te]
        if select_in_fold:
            # Welch t-test per probe, TRAINING samples only
            t, _ = ttest_ind(Xtr[y[tr] == 1], Xtr[y[tr] == 0], axis=0, equal_var=False)
            top = np.argsort(np.nan_to_num(np.abs(t)))[-TOP_K:]
            selected_sets.append(set(names[top]))
            Xtr, Xte = Xtr[:, top], Xte[:, top]
        model = make_model().fit(Xtr, y[tr])
        p = model.predict_proba(Xte)[:, 1]
        oof[te] = p
        scores.append(roc_auc_score(y[te], p))
        extra = f" | IFN probe selected: {IFN_PROBE in selected_sets[-1]}" if select_in_fold else ""
        print(f"  {label} fold {fold}: AUROC={scores[-1]:.3f}{extra}", flush=True)
    s = np.array(scores)
    print(f"{label}: fold-mean AUROC {s.mean():.3f} (+/- {s.std():.3f}) | "
          f"pooled out-of-fold AUROC {roc_auc_score(y, oof):.3f}\n", flush=True)
    out = {"fold_scores": scores, "mean": float(s.mean()), "std": float(s.std()),
           "pooled_oof_auroc": float(roc_auc_score(y, oof))}
    if select_in_fold:
        out["probes_selected_in_all_5_folds"] = sorted(set.intersection(*selected_sets))
        print(f"  probes selected in all 5 folds: {len(out['probes_selected_in_all_5_folds'])}\n")
    return out


results = {}
vocab_cols = [list(names_all).index(p) for p in vocab if p in set(names_all)]
results["a_vocab5000"] = run("(a) 5,000-probe vocab", X_all[:, vocab_cols], elastic_net)
results["b_infold_topk"] = run(f"(b) all probes, in-fold top-{TOP_K}", X_all, elastic_net,
                               select_in_fold=True, names=names_all)
ifn_col = [list(names_all).index(IFN_PROBE)]
results["c_single_probe"] = run(f"(c) single probe {IFN_PROBE}", X_all[:, ifn_col], single_probe_lr)

with open("../data/stage2b_linear_baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved ../data/stage2b_linear_baseline_results.json")
