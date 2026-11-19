"""
Project 8, Stage 3 -- Interferon enrichment among the stable probes.

Question: are Hallmark IFN-alpha-response genes over-represented among the
hypomethylated (116) and hypermethylated (344) stable probes, relative to
random probe sets of the same size?

Declared BEFORE running:
  statistic (primary)  = number of DISTINCT Hallmark IFN genes covered by >=1 probe
  statistic (secondary)= number of probes mapping to >=1 Hallmark IFN gene
  PRIMARY tests        = hypo and hyper subsets, context-matched null, distinct genes,
                         one-sided, alpha = 0.025 (Bonferroni over the two tests)
  everything else (unmatched null, all-460, probe-count statistic) = secondary
Context-matched null: same number of promoter-proximal / other-gene / no-gene probes
as the observed set, drawn from the eligible pool (probes with no NaN).
Limitation: the null draws independent probes, while real differential probes cluster
in correlated blocks; this makes the distinct-gene test conservative.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED, N_DRAWS, ALPHA_PRIMARY, PROMOTER_BP = 42, 10_000, 0.025, 1500
FULL = "../data/gse82218_beta_matrix_full.parquet"
ANNOT = "../data/HM450.hg38.manifest.gencode.v41.tsv.gz"
HALLMARK = "../data/HALLMARK_INTERFERON_ALPHA_RESPONSE.grp"
DIRECTION = "../data/stable_probes_direction.tsv"
ELIGIBLE_CACHE = "../data/eligible_probes_nonan.txt"
OUT = "../data/stage3_ifn_enrichment_results.json"
# Older MSigDB source identifiers -> current symbols (listed on the MSigDB page)
ALIASES = {"FAM125A": "MVB12A", "FAM46A": "TENT5A", "FTSJD2": "CMTR1",
           "PRIC285": "HELZ2", "WARS": "WARS1"}


def load_eligible() -> list:
    """Probes with no NaN in any of the 55 samples (the pool eligible for selection).
    Read in column chunks to keep memory low; cached after the first run."""
    if Path(ELIGIBLE_CACHE).exists():
        return [l.strip() for l in open(ELIGIBLE_CACHE)]
    pf = pq.ParquetFile(FULL, thrift_string_size_limit=10**9,
                        thrift_container_size_limit=10**9)
    names = [n for n in pf.schema_arrow.names if n != "gsm_id"]
    keep = []
    for start in range(0, len(names), 20000):
        chunk = names[start:start + 20000]
        df = pf.read(columns=chunk).to_pandas()
        keep.extend(df.columns[~df.isna().any(axis=0)])
        del df
        print(f"  eligibility scan: {min(start + 20000, len(names))}/{len(names)}", flush=True)
    Path(ELIGIBLE_CACHE).write_text("\n".join(keep) + "\n")
    return keep


def split_genes(s) -> list:
    if not isinstance(s, str):
        return []
    return [g for g in s.split(";") if g and g not in (".", "NA")]


def min_abs_tss(s) -> float:
    vals = []
    for v in str(s).split(";"):
        try:
            vals.append(abs(int(v)))
        except ValueError:
            pass
    return min(vals) if vals else np.inf


eligible = load_eligible()
print("eligible probes:", len(eligible), flush=True)
assert len(eligible) == 471758, f"expected 471758 NaN-free probes (earlier run), got {len(eligible)}"
pos = {p: i for i, p in enumerate(eligible)}

# ---- Hallmark gene set ----
tokens = [l.strip() for l in open(HALLMARK)]
raw = [t for t in tokens if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", t) and not t.startswith("HALLMARK_")]
assert len(set(raw)) == 97, f"expected 97 genes, got {len(set(raw))}; first lines: {tokens[:4]}"
hall = sorted({ALIASES.get(g, g) for g in raw})
gene_col = {g: j for j, g in enumerate(hall)}
print("Hallmark genes:", len(hall), flush=True)

# ---- Annotation for the eligible pool ----
ann = pd.read_csv(ANNOT, sep="\t", usecols=["probeID", "genesUniq", "distToTSS"], dtype=str)
ann = ann.drop_duplicates("probeID").set_index("probeID")
print("eligible probes absent from annotation (treated as no-gene):",
      int((~pd.Index(eligible).isin(ann.index)).sum()), flush=True)
ann = ann.reindex(eligible)
pool_genes = [split_genes(s) for s in ann["genesUniq"]]
dist = np.array([min_abs_tss(s) for s in ann["distToTSS"]])
sym = np.array([any(not g.startswith("ENSG") for g in gl) for gl in pool_genes])
cat = np.where(dist <= PROMOTER_BP, "promoter", np.where(sym, "other_gene", "no_gene"))
print("pool context counts:", {c: int((cat == c).sum()) for c in ("promoter", "other_gene", "no_gene")}, flush=True)

# ---- Coverage matrix: probe x Hallmark gene ----
M = np.zeros((len(eligible), len(hall)), dtype=bool)
for i, gl in enumerate(pool_genes):
    for g in gl:
        j = gene_col.get(g)
        if j is not None:
            M[i, j] = True
no_probe = [hall[j] for j in np.where(~M.any(axis=0))[0]]
print(f"Hallmark genes with >=1 eligible probe: {len(hall) - len(no_probe)}/{len(hall)}")
print("Hallmark genes with NO eligible probe (check for naming mismatches):", no_probe, flush=True)

# ---- Observed subsets ----
dr = pd.read_csv(DIRECTION, sep="\t")
subsets = {
    "hypo_in_SLE": dr.loc[dr["direction"] == "hypo_in_SLE", "probeID"].tolist(),
    "hyper_in_SLE": dr.loc[dr["direction"] == "hyper_in_SLE", "probeID"].tolist(),
    "all_460": dr["probeID"].tolist(),
}
all_idx = np.array([pos[p] for p in subsets["all_460"]])
assert int((cat[all_idx] == "promoter").sum()) == 243, "promoter count differs from earlier output"
print("sanity: 243 promoter-proximal stable probes reproduced\n", flush=True)

rng = np.random.default_rng(SEED)
cat_idx = {c: np.where(cat == c)[0] for c in ("promoter", "other_gene", "no_gene")}


def stats(idx):
    sub = M[idx]
    return int(sub.any(axis=0).sum()), int(sub.any(axis=1).sum())


def run_test(name, probes, primary):
    idx = np.array([pos[p] for p in probes])
    obs_genes, obs_probes = stats(idx)
    covered = [hall[j] for j in np.where(M[idx].any(axis=0))[0]]
    k = {c: int((cat[idx] == c).sum()) for c in cat_idx}
    res = {"n_probes": len(idx), "context": k, "observed_distinct_genes": obs_genes,
           "observed_probes_in_IFN_genes": obs_probes, "covered_genes": covered,
           "primary": primary}
    print(f"=== {name} (n={len(idx)}; context {k}) {'[PRIMARY]' if primary else '[secondary]'} ===")
    print(f"observed: {obs_genes} distinct Hallmark genes, {obs_probes} probes in them")
    for null_name in ("matched", "unmatched"):
        g = np.empty(N_DRAWS, dtype=int)
        p = np.empty(N_DRAWS, dtype=int)
        for d in range(N_DRAWS):
            if null_name == "matched":
                draw = np.concatenate([rng.choice(cat_idx[c], size=k[c], replace=False)
                                       for c in cat_idx if k[c] > 0])
            else:
                draw = rng.choice(len(eligible), size=len(idx), replace=False)
            g[d], p[d] = stats(draw)
        p_g = (1 + np.sum(g >= obs_genes)) / (1 + N_DRAWS)
        p_p = (1 + np.sum(p >= obs_probes)) / (1 + N_DRAWS)
        res[null_name] = {"null_mean_genes": float(g.mean()), "null_p95_genes": float(np.percentile(g, 95)),
                          "fold_enrichment_genes": float(obs_genes / max(g.mean(), 1e-9)),
                          "p_distinct_genes": float(p_g),
                          "null_mean_probes": float(p.mean()), "p_probe_count": float(p_p)}
        print(f"  {null_name:9s} null: genes mean {g.mean():.2f} (95th pct {np.percentile(g, 95):.0f}) "
              f"-> fold {obs_genes / max(g.mean(), 1e-9):.1f}x, p(genes) = {p_g:.4f} | "
              f"p(probe count) = {p_p:.4f}")
    print(f"  genes covered: {covered}\n", flush=True)
    return res


results = {}
for name, primary in (("hypo_in_SLE", True), ("hyper_in_SLE", True), ("all_460", False)):
    results[name] = run_test(name, subsets[name], primary)

print(f"PRIMARY decisions (matched null, distinct genes, alpha = {ALPHA_PRIMARY}):")
for name in ("hypo_in_SLE", "hyper_in_SLE"):
    p = results[name]["matched"]["p_distinct_genes"]
    print(f"  {name}: p = {p:.4f} -> {'significant' if p < ALPHA_PRIMARY else 'not significant'}")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print("saved", OUT)
