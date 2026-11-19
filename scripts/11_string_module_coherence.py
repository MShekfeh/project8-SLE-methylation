"""
Project 8, Stage 3 -- Do genes near hypomethylated stable probes form a connected
module in STRING (v12.0, human)?

Declared BEFORE running:
  statistic  = size of the largest connected component (LCC) of the induced subgraph
  null       = 10,000 degree-matched random gene sets (equal-size degree-rank bins,
               computed on the SAME network being tested)
  PRIMARY    = hypo_all gene set, STRING combined_score >= 700, one-sided, alpha = 0.05
  SECONDARY  = thresholds 400 / 900; ExpDB400 (experimental>=400 OR database>=400,
               strips co-expression and text mining); hypo genes minus the 97 Hallmark
               genes (non-circular: do the OTHER genes cluster?); hyper set; and the 25
               Hallmark hypo genes as a REFERENCE showing the expected clustering of
               interferon genes (circularity check, not a test)
Caveat: interferon genes are heavily co-studied, so STRING has many edges among them.
"""
import json
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SEED, N_DRAWS, N_BINS = 42, 10_000, 100
LINKS = "../data/9606.protein.links.detailed.v12.0.txt.gz"
INFO = "../data/9606.protein.info.v12.0.txt.gz"
HALLMARK = "../data/HALLMARK_INTERFERON_ALPHA_RESPONSE.grp"
DIRECTION = "../data/stable_probes_direction.tsv"
OUT = "../data/stage3_string_module_results.json"
ALIASES = {"FAM125A": "MVB12A", "FAM46A": "TENT5A", "FTSJD2": "CMTR1",
           "PRIC285": "HELZ2", "WARS": "WARS1"}
rng = np.random.default_rng(SEED)
t0 = time.time()

# ---- STRING protein -> gene symbol ----
info = pd.read_csv(INFO, sep="\t", usecols=["#string_protein_id", "preferred_name"], dtype=str)
genes = sorted(info["preferred_name"].dropna().unique())
G = len(genes)
gene_idx = {g: i for i, g in enumerate(genes)}
pid2g = dict(zip(info["#string_protein_id"], info["preferred_name"].map(gene_idx)))
print(f"STRING proteins: {len(info)} | distinct gene symbols: {G}", flush=True)

# ---- Edges, read in chunks; keep any edge that could matter for a threshold ----
parts, n_rows, n_unmapped = [], 0, 0
reader = pd.read_csv(LINKS, sep=" ", chunksize=1_000_000,
                     usecols=["protein1", "protein2", "experimental", "database", "combined_score"])
for chunk in reader:
    n_rows += len(chunk)
    chunk = chunk[(chunk["combined_score"] >= 400) | (chunk["experimental"] >= 400)
                  | (chunk["database"] >= 400)]
    g1, g2 = chunk["protein1"].map(pid2g), chunk["protein2"].map(pid2g)
    ok = (g1.notna() & g2.notna()).values
    n_unmapped += int((~ok).sum())
    a, b = g1.values[ok].astype(np.int32), g2.values[ok].astype(np.int32)
    keep = a < b  # each undirected edge appears in both directions; also drops self-loops
    sub = chunk[ok]
    parts.append(pd.DataFrame({
        "a": a[keep], "b": b[keep],
        "cs": sub["combined_score"].values[keep].astype(np.int16),
        "ex": sub["experimental"].values[keep].astype(np.int16),
        "db": sub["database"].values[keep].astype(np.int16)}))
    print(f"  rows read: {n_rows:,}", flush=True)
edges = pd.concat(parts, ignore_index=True)
edges = edges.groupby(["a", "b"], as_index=False).max()  # merge isoform duplicates
print(f"edges kept (undirected, any threshold): {len(edges):,} | "
      f"rows with unmapped proteins: {n_unmapped} | {time.time() - t0:.0f}s", flush=True)

masks = {"N400": edges["cs"] >= 400, "N700": edges["cs"] >= 700, "N900": edges["cs"] >= 900,
         "ExpDB400": (edges["ex"] >= 400) | (edges["db"] >= 400)}


def build(m):
    e = edges[m.values]
    A = coo_matrix((np.ones(len(e), dtype=np.int8), (e["a"].values, e["b"].values)), shape=(G, G))
    A = (A + A.T).tocsr()
    A.data[:] = 1
    return A


nets = {}
for name, m in masks.items():
    A = build(m)
    deg = np.asarray(A.sum(axis=1)).ravel()
    order = np.argsort(deg, kind="stable")
    bins = np.empty(G, dtype=int)
    bins[order] = (np.arange(G) * N_BINS) // G  # equal-size degree-rank bins
    members = [np.where(bins == b)[0] for b in range(N_BINS)]
    nets[name] = (A, deg, bins, members)
    print(f"{name}: {int(m.sum()):,} edges | genes with >=1 edge: {int((deg > 0).sum())} "
          f"| median degree {np.median(deg):.0f}", flush=True)

# ---- Gene sets ----
tokens = [l.strip() for l in open(HALLMARK)]
raw = [t for t in tokens if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", t) and not t.startswith("HALLMARK_")]
assert len(set(raw)) == 97, f"expected 97 Hallmark genes, got {len(set(raw))}"
hall = {ALIASES.get(g, g) for g in raw}

dr = pd.read_csv(DIRECTION, sep="\t")
dr["genesUniq"] = dr["genesUniq"].fillna("")


def gene_set(direction):
    out = set()
    for s in dr.loc[dr["direction"] == direction, "genesUniq"]:
        for g in s.split(";"):
            if g and g not in (".", "NA") and not g.startswith("ENSG"):
                out.add(g)
    return out


hypo, hyper = gene_set("hypo_in_SLE"), gene_set("hyper_in_SLE")
assert len(hall & hypo) == 25, f"expected 25 Hallmark genes among hypo genes, got {len(hall & hypo)}"
print(f"\ngenes with a symbol: hypo {len(hypo)}, hyper {len(hyper)}, overlap {len(hypo & hyper)}")
print("sanity: 25 Hallmark genes among hypo genes reproduced")
print("Hallmark genes absent from STRING names:", sorted(g for g in hall if g not in gene_idx))


def to_idx(s, label):
    miss = sorted(g for g in s if g not in gene_idx)
    print(f"{label}: {len(s)} genes -> {len(s) - len(miss)} found in STRING; "
          f"not found ({len(miss)}): {miss[:25]}{' ...' if len(miss) > 25 else ''}")
    return {gene_idx[g] for g in s if g in gene_idx}


sets = {"hypo_all": to_idx(hypo, "hypo_all"),
        "hypo_nonhallmark": to_idx(hypo - hall, "hypo_nonhallmark"),
        "hyper_all": to_idx(hyper, "hyper_all"),
        "hallmark_in_hypo (reference)": to_idx(hypo & hall, "hallmark_in_hypo")}


def lcc_size(A, idx):
    _, lab = connected_components(A[idx][:, idx], directed=False)
    return int(np.bincount(lab).max())


def run_test(set_name, members_idx, net_name, primary):
    A, deg, bins, bin_members = nets[net_name]
    idx = np.array(sorted(members_idx))
    obs = lcc_size(A, idx)
    n_linked = int((np.asarray(A[idx][:, idx].sum(axis=1)).ravel() > 0).sum())
    ub, cnt = np.unique(bins[idx], return_counts=True)
    null = np.empty(N_DRAWS, dtype=int)
    for d in range(N_DRAWS):
        draw = np.concatenate([rng.choice(bin_members[b], size=k, replace=False)
                               for b, k in zip(ub, cnt)])
        null[d] = lcc_size(A, draw)
    p = (1 + np.sum(null >= obs)) / (1 + N_DRAWS)
    print(f"{set_name:30s} {net_name:9s} n={len(idx):4d} | LCC observed {obs:3d} "
          f"({n_linked} genes have >=1 link inside the set) | null mean {null.mean():.1f}, "
          f"95th pct {np.percentile(null, 95):.0f} | p = {p:.4f} "
          f"{'[PRIMARY]' if primary else ''}", flush=True)
    return {"set": set_name, "network": net_name, "n_genes": len(idx), "lcc_observed": obs,
            "genes_with_link": n_linked, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "p": float(p), "primary": primary}


plan = [("hypo_all", "N700", True), ("hypo_all", "N400", False), ("hypo_all", "N900", False),
        ("hypo_all", "ExpDB400", False), ("hypo_nonhallmark", "N700", False),
        ("hypo_nonhallmark", "ExpDB400", False), ("hyper_all", "N700", False),
        ("hyper_all", "ExpDB400", False), ("hallmark_in_hypo (reference)", "N700", False)]
print("\n=== Module coherence (LCC of induced STRING subgraph vs degree-matched null) ===")
results = [run_test(s, sets[s], n, pr) for s, n, pr in plan]

# ---- Descriptive: LCC members and non-Hallmark genes attached to the Hallmark core ----
A700 = nets["N700"][0]
idx = np.array(sorted(sets["hypo_all"]))
_, lab = connected_components(A700[idx][:, idx], directed=False)
big = np.argmax(np.bincount(lab))
lcc_genes = sorted(genes[i] for i in idx[lab == big])
print(f"\nLCC members at N700 ({len(lcc_genes)} genes): {lcc_genes}")

core = np.array(sorted(sets["hallmark_in_hypo (reference)"]))
others = np.array(sorted(sets["hypo_nonhallmark"]))
links_to_core = np.asarray(A700[others][:, core].sum(axis=1)).ravel()
rank = np.argsort(-links_to_core)[:15]
print("\nnon-Hallmark hypo genes with most N700 links to the 25 Hallmark hypo genes "
      "(DESCRIPTIVE; genes were selected by the data):")
for r in rank:
    if links_to_core[r] > 0:
        print(f"  {genes[others[r]]:10s} {int(links_to_core[r])} links")

with open(OUT, "w") as f:
    json.dump({"string_version": "12.0", "n_draws": N_DRAWS, "seed": SEED, "tests": results,
               "lcc_genes_hypo_all_N700": lcc_genes}, f, indent=2)
print(f"\nsaved {OUT} | total {time.time() - t0:.0f}s")
