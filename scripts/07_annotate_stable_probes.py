"""
Project 8, Stage 3 prep -- attach genes to the 460 probes selected in all 5
folds of Stage 2b (part b). These are a DISCOVERY set: selection effectively
used every sample, so no performance figure should be computed on them.
Annotation: Zhou lab HM450 hg38 gene table (GENCODE v41).
"""
import json
import pandas as pd

with open("../data/stage2b_linear_baseline_results.json") as f:
    stable = json.load(f)["b_infold_topk"]["probes_selected_in_all_5_folds"]
print("stable probes:", len(stable), "| includes cg06872964:", "cg06872964" in stable)

ann = pd.read_csv("../data/HM450.hg38.manifest.gencode.v41.tsv.gz", sep="\t",
                  usecols=["probeID", "genesUniq", "distToTSS"], dtype=str)
sub = pd.DataFrame({"probeID": stable}).merge(ann, on="probeID", how="left", indicator=True)
print("found in annotation:", (sub["_merge"] == "both").sum(), "of", len(sub))
sub = sub.drop(columns="_merge")
sub["genesUniq"] = sub["genesUniq"].fillna("")

no_gene = sub["genesUniq"].isin(["", ".", "NA"])
multi = sub["genesUniq"].str.contains(r"[;,|]")
print("probes with no gene assigned:", int(no_gene.sum()))
print("probes with several genes:", int(multi.sum()))
print("raw multi-gene examples (check the delimiter):", sub.loc[multi, "genesUniq"].head(8).tolist())


def min_abs_tss(s):
    """Smallest |distance to TSS| across transcripts (semicolon-separated)."""
    vals = []
    for v in str(s).split(";"):
        try:
            vals.append(abs(int(v)))
        except ValueError:
            pass
    return min(vals) if vals else float("nan")


sub["min_abs_dist_to_TSS"] = pd.to_numeric(sub["distToTSS"].map(min_abs_tss), errors="coerce")
sub["promoter_proximal"] = sub["min_abs_dist_to_TSS"] <= 1500  # NaN compares as False
print("promoter-proximal (<=1.5 kb of a TSS):", int(sub["promoter_proximal"].sum()))

genes = sorted({g for s in sub["genesUniq"] for g in pd.Series(s).str.split(r"[;,|]").iloc[0] if g and g not in (".", "NA")})
print("unique genes:", len(genes))

per_gene = (sub.assign(gene=sub["genesUniq"].str.split(r"[;,|]")).explode("gene")
            .query("gene != '' and gene != '.' and gene != 'NA'")["gene"].value_counts())
print("genes with the most stable probes:\n", per_gene.head(10).to_string())

# Declared BEFORE looking at the output: well-known interferon pathway genes
ifn = ["IFI44L", "IFI44", "IFIT1", "IFIT3", "MX1", "OAS1", "ISG15", "RSAD2", "USP18", "STAT1", "IRF7", "SOCS1"]
print("declared IFN-list genes present:", [g for g in ifn if g in set(genes)])

sub.to_csv("../data/stable_probes_annotated.tsv", sep="\t", index=False)
print("saved ../data/stable_probes_annotated.tsv")
