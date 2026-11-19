"""
Project 8, Stage 3 -- Exploratory composition probe (intermediate-beta enrichment in controls).
EXPLORATORY. Idea comes from Absher 2013 Figure S1's caption (non-IFN T-cell CpGs enriched for
intermediate beta in controls), not from our data. No primary test is declared.
Caveat: bounded range (floor/ceiling) makes ANY differential set enriched for intermediate beta
vs random probes; the informative comparisons are BETWEEN sets, and they are not context-adjusted.
Intermediate = control-mean beta in [0.2, 0.8] (declared; sensitivity [0.3, 0.7]).
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED, N_DRAW, N_PERM = 42, 10_000, 10_000
LO, HI, LO2, HI2 = 0.2, 0.8, 0.3, 0.7
FULL = "../data/gse82218_beta_matrix_full.parquet"
IMPUTED = "../data/gse82218_zeroshot_input_imputed.parquet"
VOCAB = "../data/project6_cpg_probe_list.txt"
ELIG = "../data/eligible_probes_nonan.txt"
ANNOT = "../data/HM450.hg38.manifest.gencode.v41.tsv.gz"
S3 = "../data/absher_s3_clean.tsv"
DIRECTION = "../data/stable_probes_direction.tsv"
OUT = "../data/stage3_composition_probe_results.json"

pf = pq.ParquetFile(FULL, thrift_string_size_limit=10**9, thrift_container_size_limit=10**9)
imp = pd.read_parquet(IMPUTED)
vocab = [l.strip() for l in open(VOCAB)]
chk = vocab[:300]
a = pf.read(columns=chk).to_pandas()[chk].values.astype(float)
b = imp[chk].values.astype(float)
ok = ~np.isnan(a)
assert np.allclose(a[ok], b[ok], atol=1e-5), "row order mismatch"
print("alignment verified on", int(ok.sum()), "values", flush=True)
nc = np.where(imp["group"].values == "NC")[0]
print("control samples:", len(nc), flush=True)
assert len(nc) == 25

# ---- control-mean beta for every probe, read in column chunks ----
names = [n for n in pf.schema_arrow.names if n != "gsm_id"]
nc_mean = np.full(len(names), np.nan)
for start in range(0, len(names), 20000):
    chunk = names[start:start + 20000]
    df = pf.read(columns=chunk).to_pandas()[chunk]
    nc_mean[start:start + len(chunk)] = df.values[nc].astype(float).mean(axis=0)
    print(f"  scan {min(start + 20000, len(names))}/{len(names)}", flush=True)
eligible = [l.strip() for l in open(ELIG)]
assert len(eligible) == 471758
pool = pd.Series(nc_mean, index=names).loc[eligible]
assert not pool.isna().any(), "NaN control means in the eligible pool"
pv = pool.values
pos = {p: i for i, p in enumerate(eligible)}

# ---- genomic context categories (as in the enrichment test) ----
ann = pd.read_csv(ANNOT, sep="\t", usecols=["probeID", "genesUniq", "distToTSS"], dtype=str)
ann = ann.drop_duplicates("probeID").set_index("probeID").reindex(eligible)


def min_abs_tss(s):
    vals = []
    for v in str(s).split(";"):
        try:
            vals.append(abs(int(v)))
        except ValueError:
            pass
    return min(vals) if vals else np.inf


dist = np.array([min_abs_tss(s) for s in ann["distToTSS"]])
sym = np.array([isinstance(s, str) and any(g and not g.startswith("ENSG") and g not in (".", "NA")
                                           for g in s.split(";")) for s in ann["genesUniq"]])
cat = np.where(dist <= 1500, "promoter", np.where(sym, "other_gene", "no_gene"))
cat_idx = {c: np.where(cat == c)[0] for c in ("promoter", "other_gene", "no_gene")}

# ---- the four sets ----
dr = pd.read_csv(DIRECTION, sep="\t")
s3 = pd.read_csv(S3, sep="\t")
s3["IFN"] = s3["IFN"].fillna("")
ifn = s3["IFN"].str.upper().eq("IFN")
cd4 = s3["CD4_tier"].eq("**")
raw_sets = {
    "our hypo stable": dr.loc[dr["direction"] == "hypo_in_SLE", "probeID"],
    "our hyper stable": dr.loc[dr["direction"] == "hyper_in_SLE", "probeID"],
    "Absher T-cell non-IFN": s3.loc[cd4 & ~ifn, "CpG"],
    "Absher T-cell IFN": s3.loc[cd4 & ifn, "CpG"],
}
sets = {}
for k, v in raw_sets.items():
    ids = [p for p in v if p in pos]
    sets[k] = np.array([pos[p] for p in ids])
    print(f"{k}: {len(v)} probes -> {len(ids)} in the eligible pool", flush=True)

rng = np.random.default_rng(SEED)


def frac(v, lo=LO, hi=HI):
    return float(np.mean((v >= lo) & (v <= hi)))


results = {}
print(f"\n=== Control-mean beta of each set (intermediate = [{LO}, {HI}]) ===")
for k, idx in sets.items():
    v = pv[idx]
    counts = {c: int((cat[idx] == c).sum()) for c in cat_idx}
    null = np.empty(N_DRAW)
    for d in range(N_DRAW):
        draw = np.concatenate([rng.choice(cat_idx[c], size=n, replace=False)
                               for c, n in counts.items() if n > 0])
        null[d] = frac(pv[draw])
    lo_p, hi_p = np.percentile(null, [2.5, 97.5])
    print(f"{k:24s} n={len(idx):4d} ctx={counts} | median {np.median(v):.2f} | <{LO}: {np.mean(v < LO):.2f} "
          f"| intermediate: {frac(v):.2f} | [{LO2},{HI2}]: {frac(v, LO2, HI2):.2f} "
          f"| >{HI}: {np.mean(v > HI):.2f} | matched null {null.mean():.2f} (95% range {lo_p:.2f}-{hi_p:.2f})",
          flush=True)
    results[k] = {"n": len(idx), "context": counts, "median": float(np.median(v)),
                  "frac_intermediate": frac(v), "frac_intermediate_30_70": frac(v, LO2, HI2),
                  "matched_null_mean": float(null.mean()),
                  "matched_null_95": [float(lo_p), float(hi_p)]}


def contrast(a_name, b_name):
    va, vb = pv[sets[a_name]], pv[sets[b_name]]
    obs = frac(va) - frac(vb)
    both = np.concatenate([va, vb])
    na = len(va)
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        s = rng.permutation(both)
        perm[i] = frac(s[:na]) - frac(s[na:])
    p = (1 + np.sum(np.abs(perm) >= abs(obs))) / (1 + N_PERM)
    print(f"{a_name} minus {b_name}: difference in intermediate fraction {obs:+.3f} "
          f"(shuffle p, two-sided = {p:.4f}) [exploratory; context not adjusted]", flush=True)
    return {"difference": obs, "p_two_sided": float(p)}


print("\n=== Contrasts between sets (exploratory) ===")
results["contrast_hyper_vs_hypo"] = contrast("our hyper stable", "our hypo stable")
results["contrast_absher_nonIFN_vs_IFN"] = contrast("Absher T-cell non-IFN", "Absher T-cell IFN")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved", OUT)
