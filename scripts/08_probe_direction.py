"""
Project 8, Stage 3 prep -- direction of effect for the 460 stable probes.
delta = mean beta (SLE-combined) - mean beta (NC), all 55 samples.
Descriptive of a discovery set; no performance claim is made from this.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ttest_ind

FULL = "../data/gse82218_beta_matrix_full.parquet"
ann = pd.read_csv("../data/stable_probes_annotated.tsv", sep="\t")
ann["genesUniq"] = ann["genesUniq"].fillna("")
stable = ann["probeID"].tolist()

pf = pq.ParquetFile(FULL, thrift_string_size_limit=10**9, thrift_container_size_limit=10**9)
imp = pd.read_parquet("../data/gse82218_zeroshot_input_imputed.parquet")

# Rule #12: verify row alignment on 300 probes present in both files
vocab = [l.strip() for l in open("../data/project6_cpg_probe_list.txt")]
check = vocab[:300]
a = pf.read(columns=check).to_pandas()[check].values.astype(float)
b = imp[check].values.astype(float)
ok = ~np.isnan(a)
assert np.allclose(a[ok], b[ok], atol=1e-5), "row order mismatch"
print("alignment verified on", int(ok.sum()), "values")

X = pf.read(columns=stable).to_pandas()[stable]
assert not X.isna().any().any(), "unexpected NaN in stable probes"
sle = imp["group"].values != "NC"
Xv = X.values.astype(float)
delta = Xv[sle].mean(axis=0) - Xv[~sle].mean(axis=0)
t, _ = ttest_ind(Xv[sle], Xv[~sle], axis=0, equal_var=False)

ann["delta_beta"] = pd.Series(delta, index=stable).reindex(ann["probeID"]).values
ann["welch_t"] = pd.Series(t, index=stable).reindex(ann["probeID"]).values
ann["direction"] = np.where(ann["delta_beta"] < 0, "hypo_in_SLE", "hyper_in_SLE")

print("\nall 460 probes:\n", ann["direction"].value_counts().to_string())
print("median |delta beta|:", round(ann["delta_beta"].abs().median(), 3))
pp = ann[ann["promoter_proximal"]]
print("promoter-proximal (n=%d):\n" % len(pp), pp["direction"].value_counts().to_string())

# Probes with at least one HGNC-style symbol (not an Ensembl ID)
def has_symbol(s):
    return any(g and g != "." and not g.startswith("ENSG") for g in s.split(";"))
ann["has_symbol"] = ann["genesUniq"].map(has_symbol)
print("\nprobes with >=1 gene symbol:", int(ann["has_symbol"].sum()),
      "| Ensembl-only or no gene:", int((~ann["has_symbol"]).sum()))

# Declared before looking: interferon pathway genes, positive and negative arms
for g in ["IRF7", "USP18", "IFI44L", "MX1", "RSAD2", "OAS1", "IFIT1"]:
    hit = ann[ann["genesUniq"].str.split(";").map(lambda L: g in L)]
    print(f"\n{g}: {len(hit)} probes")
    if len(hit):
        print(hit[["probeID", "delta_beta", "welch_t", "promoter_proximal"]].round(3).to_string(index=False))

print("\nlargest |delta beta|:")
top = ann.reindex(ann["delta_beta"].abs().sort_values(ascending=False).index).head(10)
print(top[["probeID", "genesUniq", "delta_beta"]].round(3).to_string(index=False))

ann.to_csv("../data/stable_probes_direction.tsv", sep="\t", index=False)
print("\nsaved ../data/stable_probes_direction.tsv")
