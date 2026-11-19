"""
Project 8 -- three-panel summary figure.
A: model comparison (AUROC per CV fold). Stage 2a / 2c fold scores are HARD-CODED from the Colab
   outputs (Drive: stage2_finetune_cv_results.json, stage2c_transformer_diagnostic_results.json);
   linear-model scores are loaded from data/stage2b_linear_baseline_results.json.
B: cg06872964 beta by group.
C: Absher T-cell CpGs: Absher delta vs our delta; statistics recomputed and asserted against
   data/stage3_absher_replication_results.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

FULL = "../data/gse82218_beta_matrix_full.parquet"
IMPUTED = "../data/gse82218_zeroshot_input_imputed.parquet"
VOCAB = "../data/project6_cpg_probe_list.txt"
S3 = "../data/absher_s3_clean.tsv"
BASELINE = "../data/stage2b_linear_baseline_results.json"
REPL = "../data/stage3_absher_replication_results.json"
FIG = "../figures/stage3_summary_panels.png"
Path("../figures").mkdir(exist_ok=True)
rng = np.random.default_rng(42)

# ---------- data + alignment check (Rule #12) ----------
pf = pq.ParquetFile(FULL, thrift_string_size_limit=10**9, thrift_container_size_limit=10**9)
names = set(pf.schema_arrow.names)
imp = pd.read_parquet(IMPUTED)
vocab = [l.strip() for l in open(VOCAB)]
chk = vocab[:300]
a = pf.read(columns=chk).to_pandas()[chk].values.astype(float)
b = imp[chk].values.astype(float)
ok = ~np.isnan(a)
assert np.allclose(a[ok], b[ok], atol=1e-5), "row order mismatch"
print("alignment verified on", int(ok.sum()), "values")
group = imp["group"].values
y = (group != "NC").astype(int)

# ---------- panel A data ----------
base = json.load(open(BASELINE))
models = [
    ("Transformer\nearly stopping\n(Stage 2a)", [0.43333333, 0.46666667, 0.76666667, 0.63333333, 0.63333333], "T"),
    ("Transformer\nbackbone LR 1e-5\n(Stage 2c)", [0.867, 0.733, 0.833, 0.433, 0.533], "T"),
    ("Transformer\nbackbone LR 1e-4\n(Stage 2c)", [0.800, 0.867, 0.700, 0.500, 0.567], "T"),
    ("Ridge\n5,000-probe\nvocabulary", base["a_vocab5000"]["fold_scores"], "L"),
    ("Ridge\nall probes,\nin-fold top-1000", base["b_infold_topk"]["fold_scores"], "L"),
    ("Single probe\ncg06872964", base["c_single_probe"]["fold_scores"], "L"),
]
for _, s, _ in models:
    assert len(s) == 5
assert np.isclose(np.mean(base["a_vocab5000"]["fold_scores"]), 0.967, atol=0.002)
assert np.isclose(np.mean(base["c_single_probe"]["fold_scores"]), 0.987, atol=0.002)

# ---------- panel B data ----------
probe = pf.read(columns=["cg06872964"]).to_pandas()["cg06872964"].values.astype(float)
assert not np.isnan(probe).any()

# ---------- panel C data ----------
s3 = pd.read_csv(S3, sep="\t")
s3["IFN"] = s3["IFN"].fillna("")
present = [c for c in s3["CpG"] if c in names]
X = pf.read(columns=present).to_pandas()[present]
X = X.drop(columns=X.columns[X.isna().any(axis=0)])
m = s3[s3["CpG"].isin(X.columns)].set_index("CpG").loc[list(X.columns)].rename_axis("CpG").reset_index()
assert list(m["CpG"]) == list(X.columns)
Xv = X.values.astype(float)
our_delta = Xv[y == 1].mean(0) - Xv[y == 0].mean(0)
sel = (m["CD4_tier"] == "**").values
ab = m.loc[sel, "CD4_delta"].values
ours = our_delta[sel]
is_ifn = m.loc[sel, "IFN"].str.upper().eq("IFN").values
n = int(sel.sum())
conc = int(np.sum(np.sign(ours) == np.sign(ab)))
rho = float(spearmanr(ours, ab)[0])
saved = json.load(open(REPL))["tests"][0]
assert saved["n"] == n and saved["concordant"] == conc, (n, conc, saved["n"], saved["concordant"])
assert np.isclose(saved["rank_rho"], rho, atol=1e-6), (rho, saved["rank_rho"])
perm_p = saved["perm_p"]
print(f"panel C: n={n}, concordant={conc} ({100 * conc / n:.1f}%), rho={rho:.3f}, perm p={perm_p:.4f}")

# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), gridspec_kw={"width_ratios": [1.4, 0.8, 1.05]})
cT, cL = "#d95f02", "#1b9e77"

ax = axes[0]
for i, (label, scores, kind) in enumerate(models):
    col = cT if kind == "T" else cL
    ax.scatter(i + rng.uniform(-0.1, 0.1, 5), scores, s=46, color=col, alpha=0.8, zorder=3)
    ax.hlines(np.mean(scores), i - 0.28, i + 0.28, color="black", lw=2.2, zorder=4)
    ax.text(i, 1.045, f"{np.mean(scores):.2f}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(0.5, color="gray", ls="--", lw=1)
ax.text(5.45, 0.505, "chance", ha="right", va="bottom", fontsize=8, color="gray")
ax.axvline(2.5, color="#cccccc", lw=1)
ax.set_xticks(range(len(models)))
ax.set_xticklabels([mm[0] for mm in models], fontsize=8)
ax.set_ylim(0.3, 1.09)
ax.set_ylabel("AUROC (one dot per CV fold; bar = mean)")
ax.set_title("A   Fine-tuned transformer vs linear models\nSLE-combined vs NC, 5-fold CV", fontsize=11, loc="left")

ax = axes[1]
order = [("NC", "#7f7f7f"), ("SLE", "#d95f02"), ("SLE+LN", "#b2182b")]
for i, (g, col) in enumerate(order):
    v = probe[group == g]
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=30, color=col, alpha=0.75, zorder=3)
    ax.hlines(v.mean(), i - 0.3, i + 0.3, color="black", lw=2.2, zorder=4)
    ax.text(i, 0.985, f"{v.mean():.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(3))
ax.set_xticklabels([f"{g}\n(n={int((group == g).sum())})" for g, _ in order], fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Beta value (methylation)")
ax.set_title("B   One probe: cg06872964\n(IFI44L promoter region)", fontsize=11, loc="left")

ax = axes[2]
ax.scatter(ab[~is_ifn], ours[~is_ifn], s=14, color="#7f7f7f", alpha=0.45, label=f"Other CpGs (n={int((~is_ifn).sum())})")
ax.scatter(ab[is_ifn], ours[is_ifn], s=22, color="#b2182b", alpha=0.8, label=f"Interferon-near CpGs (n={int(is_ifn.sum())})")
lim = float(max(np.abs(ab).max(), np.abs(ours).max())) * 1.08
ax.plot([-lim, lim], [-lim, lim], color="black", ls="--", lw=0.8)
ax.axhline(0, color="#aaaaaa", lw=0.8)
ax.axvline(0, color="#aaaaaa", lw=0.8)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_xlabel("Absher et al. T-cell Δβ (SLE − control), sorted CD4+")
ax.set_ylabel("Our Δβ (SLE − NC), bulk PBMC, GSE82218")
ax.text(0.03, 0.97, f"n = {n} T-cell CpGs (p < 1e-8 in Absher)\nsame direction: {100 * conc / n:.1f}%\n"
        f"Spearman ρ = {rho:.2f}\nlabel-permutation p = {perm_p:.4f}",
        transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
ax.legend(loc="lower right", fontsize=8)
ax.set_title("C   Absher T-cell CpGs: same direction in bulk PBMC", fontsize=11, loc="left")

fig.suptitle("Lupus signal in GSE82218: linear models recover it, a fine-tuned transformer recovers little of it, "
             "and an independent cohort's CpGs replicate", fontsize=13)
fig.text(0.5, 0.005,
         "A: ~11 test samples per fold, so single-fold AUROCs are noisy. "
         "C: neighboring CpGs move together, so 'same direction' is not 1,009 independent confirmations; "
         "the permutation p accounts for this.\nAbsher: Absher et al. 2013, PLoS Genet 9(8):e1003678 (Table S3). "
         "Cells were sorted in Absher et al. and bulk in GSE82218.",
         ha="center", fontsize=8, color="#555555")
fig.tight_layout(rect=[0, 0.07, 1, 0.94])
fig.savefig(FIG, dpi=200, bbox_inches="tight")
print("saved", FIG)
