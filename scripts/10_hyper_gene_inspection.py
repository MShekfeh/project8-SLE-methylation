"""
Project 8, Stage 3 -- probe-level look at the six genes covered by the
hypermethylated stable probes. DESCRIPTIVE ONLY: genes were chosen because the
enrichment test flagged them, so no p-values are drawn from this.
"""
import pandas as pd

d = pd.read_csv("../data/stable_probes_direction.tsv", sep="\t")
d["genesUniq"] = d["genesUniq"].fillna("")
genes = ["EPSTI1", "HELZ2", "LGALS3BP", "CNP", "CSF1", "TRIM14"]

for g in genes:
    hit = d[d["genesUniq"].str.split(";").map(lambda L: g in L)]
    n_hypo = int((hit["direction"] == "hypo_in_SLE").sum())
    n_hyper = int((hit["direction"] == "hyper_in_SLE").sum())
    print(f"\n{g}: {len(hit)} stable probes ({n_hypo} hypo, {n_hyper} hyper)")
    print(hit[["probeID", "genesUniq", "delta_beta", "direction", "promoter_proximal"]]
          .sort_values("delta_beta").round(3).to_string(index=False))
