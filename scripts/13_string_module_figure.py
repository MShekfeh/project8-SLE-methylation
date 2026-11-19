"""
Project 8, Stage 3 -- STRING module figure (descriptive).
Nodes: 34 genes in the LCC of the hypomethylated gene set (STRING v12.0, score >= 700).
Color = signed delta-beta of the gene's MOST EXTREME stable probe (all stable probes mapping
to the gene, either direction); thick black outline = gene has stable probes in BOTH directions.
Size = number of stable probes mapping to the gene (a probe near several genes counts for each).
Shape: circle = Hallmark IFN-alpha gene, square = other gene.
Edges: STRING combined_score >= 700, width scaled to score.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

LINKS = "../data/9606.protein.links.detailed.v12.0.txt.gz"
INFO = "../data/9606.protein.info.v12.0.txt.gz"
HALLMARK = "../data/HALLMARK_INTERFERON_ALPHA_RESPONSE.grp"
DIRECTION = "../data/stable_probes_direction.tsv"
RESULTS = "../data/stage3_string_module_results.json"
FIG = "../figures/stage3_string_module.png"
TABLE = "../data/stage3_string_module_genes.tsv"
ALIASES = {"FAM125A": "MVB12A", "FAM46A": "TENT5A", "FTSJD2": "CMTR1",
           "PRIC285": "HELZ2", "WARS": "WARS1"}
Path("../figures").mkdir(exist_ok=True)

module = set(json.load(open(RESULTS))["lcc_genes_hypo_all_N700"])
assert len(module) == 34, f"expected 34 module genes, got {len(module)}"

tokens = [l.strip() for l in open(HALLMARK)]
raw = [t for t in tokens if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", t) and not t.startswith("HALLMARK_")]
hall = {ALIASES.get(g, g) for g in raw}

# ---- gene-level summary from the stable probes ----
d = pd.read_csv(DIRECTION, sep="\t")
d["genesUniq"] = d["genesUniq"].fillna("")
rows = [(g, r.probeID, r.delta_beta) for r in d.itertuples()
        for g in r.genesUniq.split(";") if g in module]
gp = pd.DataFrame(rows, columns=["gene", "probe", "delta_beta"])
summ = gp.groupby("gene").apply(lambda x: pd.Series({
    "n_probes": len(x),
    "extreme_delta": x.loc[x["delta_beta"].abs().idxmax(), "delta_beta"],
    "mixed": bool((x["delta_beta"] < 0).any() and (x["delta_beta"] > 0).any())}),
    include_groups=False)
assert set(summ.index) == module, f"module genes lacking stable probes: {module - set(summ.index)}"
summ["hallmark"] = [g in hall for g in summ.index]

# ---- STRING edges among the module genes ----
info = pd.read_csv(INFO, sep="\t", usecols=["#string_protein_id", "preferred_name"], dtype=str)
pid2g = dict(zip(info["#string_protein_id"], info["preferred_name"]))
edges = {}
for chunk in pd.read_csv(LINKS, sep=" ", chunksize=1_000_000,
                         usecols=["protein1", "protein2", "combined_score"]):
    chunk = chunk[chunk["combined_score"] >= 700]
    g1, g2 = chunk["protein1"].map(pid2g), chunk["protein2"].map(pid2g)
    m = g1.isin(module) & g2.isin(module) & (g1 != g2)
    for a, b, s in zip(g1[m], g2[m], chunk["combined_score"][m]):
        key = tuple(sorted((a, b)))
        edges[key] = max(edges.get(key, 0), int(s))

G = nx.Graph()
G.add_nodes_from(sorted(module))
for (a, b), s in edges.items():
    G.add_edge(a, b, score=s)
assert nx.is_connected(G), "module is not one connected component; does not reproduce LCC = 34"
print(f"module: {G.number_of_nodes()} genes, {G.number_of_edges()} edges (connected, as expected)")
print(f"Hallmark genes: {int(summ['hallmark'].sum())} | other genes: {int((~summ['hallmark']).sum())} "
      f"| mixed-direction genes: {int(summ['mixed'].sum())}")

# ---- table ----
tab = summ.reset_index().rename(columns={"index": "gene"})
tab["degree_in_module"] = tab["gene"].map(dict(G.degree()))
tab = tab.sort_values("extreme_delta")
tab.to_csv(TABLE, sep="\t", index=False)
print("\n", tab.round(3).to_string(index=False))

# ---- figure ----
pos = nx.spring_layout(G, seed=42, k=1.8 / np.sqrt(len(G)), iterations=300)
fig, ax = plt.subplots(figsize=(13, 10))
widths = [0.3 + 1.7 * (G[u][v]["score"] - 700) / 300 for u, v in G.edges]
nx.draw_networkx_edges(G, pos, width=widths, edge_color="#999999", alpha=0.2, ax=ax)
cmap, vmin, vmax = plt.get_cmap("RdBu_r"), -0.5, 0.5
for shape, flag in (("o", True), ("s", False)):
    nodes = [g for g in G.nodes if bool(summ.loc[g, "hallmark"]) == flag]
    if not nodes:
        continue
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodes, node_shape=shape,
        node_color=[summ.loc[g, "extreme_delta"] for g in nodes], cmap=cmap, vmin=vmin, vmax=vmax,
        node_size=[150 + 120 * summ.loc[g, "n_probes"] for g in nodes],
        edgecolors=["black" if summ.loc[g, "mixed"] else "#666666" for g in nodes],
        linewidths=[2.6 if summ.loc[g, "mixed"] else 0.6 for g in nodes], ax=ax)
for g, (x, y) in pos.items():
    t = ax.text(x, y, g, ha='center', va='center', fontsize=8, zorder=5, color='black')
    t.set_path_effects([pe.withStroke(linewidth=2.8, foreground='white')])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
cb.set_label("Δβ (SLE − NC), most extreme stable probe per gene\nblue = hypomethylated in SLE")
ax.legend(handles=[
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb", markeredgecolor="#666666",
           markersize=11, label="Hallmark IFN-α response gene"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#bbbbbb", markeredgecolor="#666666",
           markersize=11, label="Other gene in the module"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb", markeredgecolor="black",
           markeredgewidth=2.6, markersize=11, label="Stable probes move in both directions")],
    loc="lower left", fontsize=9, frameon=True)
ax.set_title("Hypomethylated genes in SLE form one connected interferon module\n"
             "(34 of 64 genes; STRING v12.0, combined score ≥ 700)", fontsize=12)
ax.text(0.5, -0.01, "Descriptive figure. 24 of 34 genes are in the Hallmark IFN-α response set. "
        "Node size = number of stable probes.\nEdges include co-expression and text-mining evidence "
        "and reflect co-study bias. Layout is arbitrary.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555555")
ax.axis("off")
fig.savefig(FIG, dpi=200, bbox_inches="tight")
print(f"\nsaved {FIG} and {TABLE}")
