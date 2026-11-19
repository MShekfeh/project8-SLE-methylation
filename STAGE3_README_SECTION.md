## Stage 3 — What Is the Signal Made Of? Interferon Enrichment, an Interaction Module, and Independent Replication

### Biological/ML Question

Stage 2 showed that a linear model recovers the lupus signal in GSE82218 (AUROC ~0.97)
while a fine-tuned transformer recovers little of it. Stage 3 asks what that signal
consists of, whether it is biologically coherent, and whether it replicates in an
independent cohort.

### What Was Run

All permutation tests use 10,000 draws with seed 42. Primary tests were declared in the
script docstrings before any result was seen; everything else is labelled secondary or
exploratory.

| Script | Purpose |
|---|---|
| `07_annotate_stable_probes.py` | Attach genes (Zhou lab HM450 hg38 table, GENCODE v41) to the 460 probes selected in the top 1,000 by in-fold t-test in all five Stage 2b folds |
| `08_probe_direction.py` | Direction and size of the group difference per stable probe |
| `09_ifn_enrichment_permutation.py` | Hallmark IFN-alpha gene-set enrichment, context-matched probe permutation |
| `10_hyper_gene_inspection.py` | Probe-level look at genes covered by the hypermethylated probes (descriptive) |
| `11_string_module_coherence.py` | Largest connected component of the STRING v12.0 subgraph, degree-matched null |
| `12_kegg_node_check.py` | Which module genes are nodes in four KEGG pathways |
| `13_string_module_figure.py` | Module figure |
| `14`–`16_*absher*.py` | Parse Absher et al. 2013 Tables S3/S4 and test replication in GSE82218 |
| `17_composition_probe.py` | Exploratory intermediate-beta probe |
| `18_summary_figure.py` | Three-panel summary figure |

### Results

**The stable set.** 460 probes: 116 hypomethylated and 344 hypermethylated in
SLE-combined vs. NC. The median |delta beta| is 0.085. The largest shifts are all
hypomethylation at interferon-pathway loci (MX1 -0.489; DTX3L/PARP9 -0.414; IFI44L -0.366;
CMPK2/RSAD2 -0.344). Probes were ranked by delta beta, not t, because the t-statistics are
circular (probes were chosen for large |t|). Of the hypomethylated probes, 91 of 116 (78%)
are promoter-proximal (within 1.5 kb of a transcription start site), against 152 of 344
(44%) of the hypermethylated probes and 57% of the eligible pool; this contrast was
noticed after looking and was not tested.

**Interferon enrichment (primary tests, matched null, alpha = 0.025).**

| Subset | Probes | Distinct Hallmark IFN genes observed | Null mean (95th percentile) | p |
|---|---|---|---|---|
| Hypomethylated | 116 | 25 of 97 | 0.63 (2) | < 0.0001 (floor of the test) |
| Hypermethylated | 344 | 6 | 1.46 (4) | 0.0063 |
| All 460 (secondary) | 460 | 28 | 2.04 (5) | < 0.0001 |

The null draws random probe sets matched to the observed numbers of promoter-proximal,
other-gene and no-gene probes. The hypermethylated signal is weak: 3 of its 6 genes
(EPSTI1, HELZ2, LGALS3BP) are also covered by hypomethylated probes. Probe-level
inspection shows mixed-direction loci: EPSTI1, OAS2 and LGALS3BP have a
promoter-proximal hypomethylated probe plus an opposite-direction probe outside
promoters (a post hoc observation in four genes); HELZ2 has four promoter-flagged probes
that split 1 hypo to 3 hyper (the flag means near any transcript's start site, so isoforms
may explain this; untested).

**The two-arm hypothesis from the plan.** All four IRF7 probes are hypomethylated and
promoter-proximal. The single USP18 probe is also hypomethylated (-0.247, not
promoter-proximal), so "epigenetic silencing of the negative regulator" is not supported
by this probe. SOCS1 is not in the stable set, which does not mean it shows no difference.

**Interaction module (STRING v12.0).** The hypomethylated genes (72 symbols, 64 in STRING)
form one connected component of 34 genes at combined score >= 700, against a
degree-matched null mean of 2.6 (95th percentile 4), p < 0.0001. It holds at score >= 400
(41 vs. 6.5, 95th percentile 12), >= 900 (26 vs. 1.8, 95th percentile 3) and with
experimental or curated-database evidence only (9 vs. 2.2, 95th percentile 3). The module
contains 24 of the 25 Hallmark genes among the hypomethylated set plus 10 others (IFIT1,
DDX58, OAS2, OAS3, XAF1, IFIT5, TRIM22, DTX3L, DDX60L, NLRC5), and includes IRF7 and
USP18. Without the 97 Hallmark genes, the remaining 39 still cluster at score >= 700 (9
vs. 1.7, 95th percentile 3, p < 0.0001) but only marginally under experimental or curated
evidence (3 vs. 1.6, 95th percentile 3, p = 0.053). The hypermethylated set (251 genes in
STRING) gives 43 vs. 13.7 (95th percentile 26, p = 0.0006) at >= 700 and 38 vs. 20.9
(95th percentile 39, p = 0.066) under stricter evidence; it was not examined further.
The 10 extra module genes are interferon-stimulated genes by background knowledge; this
was not tested against another gene set.

**Directed-pathway analysis (dropped).** The plan called for betweenness centrality in a
directed interferon network. Four KEGG pathways were checked for module-gene membership:
JAK-STAT (hsa04630, 168 genes) contains none; RIG-I-like receptor signaling (hsa04622, 72
genes) contains IRF7 and TRIM25; cytosolic DNA-sensing (hsa04623, 83 genes) contains IRF7;
SLE (hsa05322, 141 genes) contains none. The module consists mostly of end-of-cascade
effector genes that these pathways do not encode as nodes, so betweenness could not be
computed meaningfully. Reactome was not checked. Integrated gradients and TOMTOM motif
analysis were dropped along with the transformer.

**Independent replication (Absher et al. 2013, Table S3).** Absher et al. profiled sorted
CD4+ T-cells, CD19+ B-cells and CD14+ monocytes (49 SLE, 58 controls, 450K array). The parsed
table reproduces the abstract's counts for T-cells (1,033 CpGs at p < 1e-8) and B-cells
(166), **but not for monocytes (41 parsed vs. 97 reported; cause unresolved), so monocyte
analyses were excluded.** Direction of effect was compared in GSE82218 (SLE-combined minus
NC) against a null from 10,000 permutations of the SLE/NC labels (a naive binomial test is
anti-conservative because neighboring CpGs are not independent):

| Absher CpG set | Usable CpGs | Same direction | Null 95th percentile | Permutation p | Rank rho |
|---|---|---|---|---|---|
| T-cell, all (primary) | 1,009 | 982 (97.3%) | 91.1% | 0.0007 | +0.86 |
| T-cell, non-interferon | 947 | 923 (97.5%) | 91.3% | 0.0005 | +0.85 |
| T-cell, interferon-near | 62 | 59 (95.2%) | 96.8% | 0.090 | +0.84 (p = 0.006) |
| T-cell, excluding our stable-set CpGs | 918 | 891 (97.1%) | 90.7% | 0.0007 | +0.84 |
| B-cell (secondary) | 162 | 146 (90.1%) | 84.6% | 0.005 | +0.74 |

The interferon-near sign test lacks power because these CpGs share one axis of variation
(the null's 95th percentile is 96.8%); the rank correlation is the more informative
statistic there. The 1,009 concordant CpGs are not independent confirmations. Of our stable
probes, 83 of 116 hypomethylated and 18 of 344 hypermethylated ones appear in Table S3, and
Absher's T-cell direction matches ours for all 101. Probe cg06872964 is listed in Table S3
(IFI44L, interferon-flagged, T-cell delta -0.204), which resolves the earlier citation
TODO.

**Composition probe (exploratory).** Absher et al.'s Figure S1 caption reports that their
significant non-interferon T-cell CpGs are enriched for intermediate beta values. In our
controls, the fraction of CpGs with mean beta in [0.2, 0.8] is 0.82 for Absher's
non-interferon T-cell CpGs vs. 0.63 for their interferon-near ones (difference +0.196,
shuffle p = 0.0008), and 0.77 for our hypermethylated vs. 0.68 for our hypomethylated
stable probes (difference +0.092, p = 0.065). Every set is far above its matched null,
which is expected from the bounded beta range alone. This is consistent with, but not
evidence for, a composition-driven component: bulk PBMC mixing and the bounded range both
predict the pattern. **Cell-type deconvolution was not performed.**

### Interpretation

The lupus methylation signal in GSE82218 is concentrated at interferon-stimulated genes:
promoter-proximal hypomethylation at 25 of the 97 Hallmark interferon-response genes, which
form one connected STRING module and which agree in direction with an independent cohort of
sorted cells. A larger, weaker, more mixed hypermethylated remainder also agrees in
direction with Absher et al., is only modestly interferon-enriched, and may reflect cell
composition (untested). These data are correlational and cannot say whether the
hypomethylation causes or follows interferon exposure.

For liquid-biopsy work, the relevant point is that a methylation signature driven by
interferon-pathway inflammation exists in blood cells and is large; whether it affects
cfDNA classifiers in patients with autoimmune or inflammatory disease is a hypothesis this
project did not test.

### Limitations

* n = 55 samples with ~11 per test fold. The 460 stable probes are a discovery set
  (selection effectively used all samples); no performance figure is claimed from them.
* Gene counts double-count shared probes: PARP9/DTX3L, TRIM5/TRIM22 and CMPK2/RSAD2 have
  fully shared stable probes, so the 34-gene module is about 31 independent loci.
* STRING edges include co-expression and text mining and reflect co-study bias; degree is
  not importance.
* GSE82218 is bulk PBMC; Absher et al. used sorted cells and a different cohort.
* No SNP or cross-reactive probe masking was applied in the full-matrix analyses
  (Stages 2b-3); the Zhou lab masks exist but were not used.
* Many analyses were run; primary tests are declared, everything else is secondary or
  exploratory, and there is no global multiplicity correction.
* Monocyte counts in Table S3 do not match the abstract (unresolved); Reactome and
  cell-type deconvolution were not examined.

### Figures

* `figures/stage3_summary_panels.png` — model comparison, cg06872964 by group, Absher replication
* `figures/stage3_string_module.png` — the 34-gene STRING module (color = delta beta of the most extreme stable probe; descriptive; layout is arbitrary)

### Data and Citations

GSE82218 (this project); STRING v12.0 (human); MSigDB Hallmark gene sets (Liberzon et al.
2015, PubMed 26771021, CC-BY-4.0); Zhou, Laird and Shen 2017, *Nucleic Acids Research*
(HM450 annotation); Absher et al. 2013, *PLoS Genetics* 9(8):e1003678 (Tables S3/S4);
KEGG REST (pathway membership check only).
