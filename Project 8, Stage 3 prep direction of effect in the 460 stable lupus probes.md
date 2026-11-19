# Project 8, Stage 3 prep: direction of effect in the 460 stable lupus probes

Oct 10, 2026 · @Marwa

## What was run

The 460 probes selected in all five folds of the Stage 2b in-fold t-test were annotated with genes and compared between SLE-combined (n=30) and NC (n=25) across all 55 samples. They are a discovery set: selection effectively used every sample, so no performance claim is made from them.

Row alignment between the full 485,577-probe matrix and the imputed model-input file was re-verified on 300 shared probes (16,491 values) before any comparison. Delta beta (Δβ) is the mean beta in SLE-combined minus the mean beta in NC. Genes come from the Zhou lab HM450 hg38 gene table (GENCODE v41), and a probe counts as promoter-proximal if it lies within 1.5 kb of a transcription start site.

Scripts: `07_annotate_stable_probes.py` and `08_probe_direction.py`.

## Three quarters of the stable probes are hypermethylated in SLE

344 of the 460 stable probes (75%) have higher methylation in SLE and 116 have lower. The split also holds among promoter-proximal probes.

| Subset | Probes | Hypermethylated in SLE | Hypomethylated in SLE |
| --- | --- | --- | --- |
| All stable probes | 460 | 344 | 116 |
| Promoter-proximal (within 1.5 kb of a TSS) | 243 | 152 | 91 |

This does not match a simple "interferon hypomethylation" story, so the stable set holds at least two kinds of signal. 41 probes have no gene symbol (25 have no gene assigned, 16 carry only Ensembl IDs), so they cannot be placed on a pathway network.

## The largest effects are hypomethylated interferon-pathway loci

Nine of the ten largest shifts sit in genes I recognize as interferon-pathway genes, and all ten are hypomethylated in SLE. The top probe is at MX1 (Δβ = -0.489). The others sit at DTX3L/PARP9, IFI44L, CMPK2/RSAD2, IFI44 and IFIT1/LIPA. The exception, a probe at PDE7A (Δβ = -0.336), has not been checked against an interferon gene set.

The typical stable probe moves far less: the median |Δβ| across all 460 is 0.085. The single probe cg06872964, in the IFI44L promoter, has Δβ = -0.280. The interferon label here comes from background knowledge and is not yet tested against a curated gene set.

## IRF7 behaves as predicted, but USP18 is hypomethylated, not silenced

All 4 IRF7 probes are hypomethylated in SLE and promoter-proximal. The plan expected the negative regulators (USP18, SOCS1) to be epigenetically silenced, meaning hypermethylated. The single USP18 probe goes the other way (Δβ = -0.247) and lies more than 1.5 kb from a start site.

| Gene | Role | Stable probes | Δβ range | Promoter-proximal |
| --- | --- | --- | --- | --- |
| IRF7 | Positive arm | 4 | -0.030 to -0.196 | 4 of 4 |
| USP18 | Negative arm | 1 | -0.247 | 0 of 1 |
| IFI44L | Interferon-stimulated | 7 | -0.085 to -0.366 | 5 of 7 |
| MX1 | Interferon-stimulated | 6 | -0.016 to -0.489 | 6 of 6 |
| RSAD2 | Interferon-stimulated | 6 | -0.072 to -0.344 | 6 of 6 |
| IFIT1 | Interferon-stimulated | 1 | -0.316 | 1 of 1 |
| OAS1 | Interferon-stimulated | 1 | -0.038 | 1 of 1 |

USP18 is itself interferon-induced, so hypomethylation is plausible. SOCS1 is not in the stable set, which does not mean it shows no difference. One probe cannot settle this, so the write-up should not repeat the "silenced negative arm" wording without more evidence.

## Cautions before using these probes for anything beyond description

- **The t-statistics are circular.** Probes were chosen for large |t| in training folds, then t was computed on all 55 samples. Use t for ranking only, never for p-values.
- **Effect size and t are different.** MX1 probe cg08244262 has t = -8.4 but Δβ = -0.016. Weight probes by Δβ in the network analysis.
- **The hypermethylated majority is unexplained.** One untested hypothesis: SLE shifts PBMC cell composition (the neutrophil/lymphocyte balance), and many probes carry cell-type-specific methylation. That would produce many small, consistent shifts with no interferon content. These data cannot test it.
- **Gene counts double-count probes.** C2 and ZBTB12 each show 11 probes because the same probes map to both genes. 118 of the 460 probes map to several genes.
- **A citation is still unverified.** The link between cg06872964 and IFI44L in the lupus literature was recalled from memory and must be checked before it appears in a README.

## Next: test interferon enrichment separately in hypo and hyper probes

The next analysis asks whether interferon-pathway genes are enriched among the hypomethylated stable probes, the hypermethylated ones, or both. A standard gene-level hypergeometric test would be biased, because 450K probes are unevenly spread across genes and genes with many probes are picked up more easily by chance.

Instead, use a probe-level permutation: draw random probe sets of the same size from the array and compare the observed interferon-gene overlap against that null. The planned gene set is the MSigDB Hallmark Interferon Alpha Response set, and its download URL needs confirming before use.

After that comes the IFN signaling network figure, with betweenness centrality and node colour set by Δβ.

- [x] Decide: run the enrichment test before building the network figure (recommended), or build the figure first
