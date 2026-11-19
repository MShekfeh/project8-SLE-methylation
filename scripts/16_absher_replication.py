"""
Project 8, Stage 3 -- Replication of Absher 2013 CpGs in GSE82218.

Declared BEFORE running:
  PRIMARY   = CD4 (T-cell) '**' CpGs with no NaN in GSE82218: fraction whose direction
              (SLE-combined minus NC) agrees with Absher's (SLE minus control means),
              vs a null from 10,000 permutations of our SLE/NC labels; one-sided, 0.05.
  SECONDARY = rank correlation of delta; IFN-near vs other; excluding CpGs in our stable
              set (selection-free); CD19 (B-cell) '**'; descriptive direction agreement
              among CpGs shared with our stable set. CD14 EXCLUDED: its '**' count (41)
              does not match the abstract (97), cause unresolved.
Naive binomial p vs 50% is reported only as a reference (neighboring CpGs are not independent).
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import binomtest, rankdata

SEED, N_PERM = 42, 10_000
S3 = "../data/absher_s3_clean.tsv"
FULL = "../data/gse82218_beta_matrix_full.parquet"
IMPUTED = "../data/gse82218_zeroshot_input_imputed.parquet"
VOCAB = "../data/project6_cpg_probe_list.txt"
DIRECTION = "../data/stable_probes_direction.tsv"
OUT = "../data/stage3_absher_replication_results.json"

s3 = pd.read_csv(S3, sep="\t")
s3["IFN"] = s3["IFN"].fillna("")
s3["is_ifn"] = s3["IFN"].str.upper().eq("IFN")

pf = pq.ParquetFile(FULL, thrift_string_size_limit=10**9, thrift_container_size_limit=10**9)
names = set(pf.schema_arrow.names)
imp = pd.read_parquet(IMPUTED)

# Rule #12: verify row alignment between the full matrix and the labelled file
vocab = [l.strip() for l in open(VOCAB)]
chk = vocab[:300]
a = pf.read(columns=chk).to_pandas()[chk].values.astype(float)
b = imp[chk].values.astype(float)
ok = ~np.isnan(a)
assert np.allclose(a[ok], b[ok], atol=1e-5), "row order mismatch"
print("alignment verified on", int(ok.sum()), "values")
y = (imp["group"].values != "NC").astype(int)
print(f"samples: SLE-combined {int(y.sum())}, NC {int((1 - y).sum())}")

present = [c for c in s3["CpG"] if c in names]
X = pf.read(columns=present).to_pandas()[present]
nan_cols = X.columns[X.isna().any(axis=0)]
print(f"S3 CpGs: {len(s3)} | in GSE82218 matrix: {len(present)} | with NaN (dropped): {len(nan_cols)}")
X = X.drop(columns=nan_cols)
m = s3[s3["CpG"].isin(X.columns)].set_index("CpG").loc[list(X.columns)].rename_axis("CpG").reset_index()
assert list(m["CpG"]) == list(X.columns), "Absher rows are not aligned with matrix columns"
Xv = X.values.astype(float)
our_delta_all = Xv[y == 1].mean(0) - Xv[y == 0].mean(0)

stable = pd.read_csv(DIRECTION, sep="\t")
in_stable = m["CpG"].isin(set(stable["probeID"])).values

rng = np.random.default_rng(SEED)
perms = np.array([rng.permutation(y) for _ in range(N_PERM)], dtype=np.int8)


def rank_corr(u, v_ranks_centered):
    ru = rankdata(u)
    ru = ru - ru.mean()
    return float((ru @ v_ranks_centered) / np.sqrt((ru @ ru) * (v_ranks_centered @ v_ranks_centered)))


def run(name, mask, cell, primary=False):
    cols = np.where(mask)[0]
    n = len(cols)
    d_abs = m.loc[mask, f"{cell}_delta"].values
    Xs = Xv[:, cols]
    obs_d = our_delta_all[cols]
    sgn = np.sign(d_abs)
    conc_obs = float(np.mean(np.sign(obs_d) == sgn))
    ra = rankdata(d_abs)
    ra = ra - ra.mean()
    rho_obs = rank_corr(obs_d, ra)
    conc_null = np.empty(N_PERM)
    rho_null = np.empty(N_PERM)
    for i, p in enumerate(perms):
        d = Xs[p == 1].mean(0) - Xs[p == 0].mean(0)
        conc_null[i] = np.mean(np.sign(d) == sgn)
        rho_null[i] = rank_corr(d, ra)
    p_conc = (1 + np.sum(conc_null >= conc_obs)) / (1 + N_PERM)
    p_rho = (1 + np.sum(rho_null >= rho_obs)) / (1 + N_PERM)
    k = int(round(conc_obs * n))
    p_bin = binomtest(k, n, 0.5, alternative="greater").pvalue
    print(f"{name:42s} n={n:4d} | concordant {k:4d} ({100 * conc_obs:5.1f}%) | null mean "
          f"{100 * conc_null.mean():4.1f}%, 95th pct {100 * np.percentile(conc_null, 95):4.1f}% | "
          f"perm p = {p_conc:.4f} | rank rho {rho_obs:+.3f} (perm p = {p_rho:.4f}) | "
          f"naive binomial p = {p_bin:.2g} {'[PRIMARY]' if primary else ''}", flush=True)
    return {"name": name, "n": n, "concordant": k, "concordant_frac": conc_obs,
            "null_mean": float(conc_null.mean()), "null_p95": float(np.percentile(conc_null, 95)),
            "perm_p": float(p_conc), "rank_rho": rho_obs, "rho_perm_p": float(p_rho),
            "naive_binomial_p": float(p_bin), "primary": primary}


cd4 = (m["CD4_tier"] == "**").values
cd19 = (m["CD19_tier"] == "**").values
ifn = m["is_ifn"].values
print(f"\nCD4 '**' CpGs usable: {int(cd4.sum())} of 1033 | CD19 '**' usable: {int(cd19.sum())} of 166\n")
print("=== Replication in GSE82218 (direction agreement with Absher) ===")
results = [
    run("CD4 ** all", cd4, "CD4", primary=True),
    run("CD4 ** IFN-near", cd4 & ifn, "CD4"),
    run("CD4 ** non-IFN", cd4 & ~ifn, "CD4"),
    run("CD4 ** all, excluding our stable-set CpGs", cd4 & ~in_stable, "CD4"),
    run("CD4 ** non-IFN, excluding our stable set", cd4 & ~ifn & ~in_stable, "CD4"),
    run("CD19 ** all (secondary)", cd19, "CD19"),
]

print("\n=== Descriptive: direction agreement among CpGs shared with our stable set (all of S3) ===")
sh = s3.merge(stable[["probeID", "delta_beta", "direction"]], left_on="CpG", right_on="probeID")
for d in ("hypo_in_SLE", "hyper_in_SLE"):
    s = sh[sh["direction"] == d]
    agree = int((np.sign(s["CD4_delta"]) == np.sign(s["delta_beta"])).sum())
    sig = int((s["CD4_tier"] == "**").sum())
    print(f"  {d:13s} shared n={len(s):3d} | Absher CD4 delta same sign as ours: {agree:3d} | "
          f"of which CD4 '**': {sig:3d} | IFN-near: {int(s['IFN'].fillna('').str.upper().eq('IFN').sum())}")

with open(OUT, "w") as f:
    json.dump({"n_perm": N_PERM, "seed": SEED, "tests": results}, f, indent=2)
print("\nsaved", OUT)
