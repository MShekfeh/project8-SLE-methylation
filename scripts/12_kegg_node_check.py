"""
Project 8, Stage 3 -- Which of our hypomethylated genes appear as NODES in KEGG
interferon-relevant pathways? (Cheap check before building any directed graph.)
"""
import json
import re
import time
import urllib.request

import pandas as pd

PATHWAYS = {"hsa04630": "Jak-STAT signaling", "hsa04622": "RIG-I-like receptor signaling",
            "hsa04623": "Cytosolic DNA-sensing", "hsa05322": "Systemic lupus erythematosus"}
lcc = set(json.load(open("../data/stage3_string_module_results.json"))["lcc_genes_hypo_all_N700"])
dr = pd.read_csv("../data/stable_probes_direction.tsv", sep="\t")
dr["genesUniq"] = dr["genesUniq"].fillna("")
hypo = {g for s in dr.loc[dr["direction"] == "hypo_in_SLE", "genesUniq"] for g in s.split(";")
        if g and g not in (".", "NA") and not g.startswith("ENSG")}
print(f"our sets: {len(lcc)} module genes, {len(hypo)} hypomethylated genes with a symbol\n")


def kegg_genes(pid):
    req = urllib.request.Request(f"https://rest.kegg.jp/get/{pid}",
                                 headers={"User-Agent": "Mozilla/5.0"})
    text = urllib.request.urlopen(req, timeout=30).read().decode()
    name = re.search(r"^NAME\s+(.+)$", text, re.M)
    genes, in_gene = set(), False
    for line in text.splitlines():
        if line.startswith("GENE"):
            in_gene = True
        elif line and not line[0].isspace():
            in_gene = False
        if in_gene:
            m = re.match(r"^(?:GENE)?\s+\d+\s+([^;\s]+);", line)
            if m:
                genes.add(m.group(1))
    return (name.group(1) if name else pid), genes


for pid, label in PATHWAYS.items():
    try:
        name, genes = kegg_genes(pid)
    except Exception as e:
        print(f"{pid} ({label}): FAILED -> {e}\n")
        continue
    print(f"{pid} | {name} | {len(genes)} genes in pathway")
    print(f"   module genes present: {sorted(lcc & genes)}")
    print(f"   other hypo genes present: {sorted((hypo - lcc) & genes)}\n")
    time.sleep(0.5)
