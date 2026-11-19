# Project 8, Stage 3: interferon enrichment among the stable lupus probes

Oct 11, 2026 · @Marwa

## Hypomethylated probes are strongly enriched for interferon genes

The 116 hypomethylated stable probes cover 25 of the 97 Hallmark interferon-alpha-response genes. Random context-matched probe sets cover about 0.6 on average, with a 95th percentile of 2, and none of the 10,000 draws reached 25. The p-value therefore sits at the floor of the test (below 0.0001).

| Subset | Probes | Distinct IFN genes observed | Matched null: mean (95th percentile) | p (threshold 0.025) |
| --- | --- | --- | --- | --- |
| Hypomethylated in SLE (primary) | 116 | 25 | 0.63 (2) | < 0.0001 |
| Hypermethylated in SLE (primary) | 344 | 6 | 1.46 (4) | 0.0063 |
| All 460 stable probes (secondary) | 460 | 28 | 2.04 (5) | < 0.0001 |

Quote "25 observed against a 95th percentile of 2", not the 40-fold ratio, because the ratio divides by a number close to zero. The context-matched and unmatched nulls agree, so promoter context is not driving the result.

The covered genes include IRF7 and USP18, the two arms of the plan's feedback-loop hypothesis, and both are hypomethylated. Others include IFI44L, MX1, RSAD2, IFIT3, OAS1, CMPK2 and PARP9.

## Hypermethylated probes pass the threshold, but the evidence is weak

The 344 hypermethylated stable probes cover 6 interferon genes against a matched-null 95th percentile of 4 (p = 0.0063, below the declared 0.025). The margin over the null is small.

Three of the six genes (EPSTI1, HELZ2, LGALS3BP) are also covered by hypomethylated probes, so these are loci where different CpGs move in opposite directions. Only CNP, CSF1 and TRIM14 are unique to the hyper set. This is not an independent hypermethylation signal.

The probe-count p-values of 0.0001 are the secondary statistic. Correlated neighboring probes inflate them, so they should not be quoted as the headline.

## What the enrichment does not show

The enrichment is correlational and covers one gene set chosen in advance, so three claims stay out of reach.

- **Causation.** Hypomethylation at interferon-response genes fits either a cause or a consequence of interferon exposure. These data cannot separate the two.
- **Specificity.** Only one gene set was tested. The IFN-gamma Hallmark set overlaps heavily with this one, and other immune sets were not examined, so the signal cannot be called specific to alpha-interferon biology.
- **Cell composition.** A neutrophil/lymphocyte shift is not an obvious source of hypomethylation at 25 specific interferon genes, but this test does not rule it out. The safe wording is "not explained by composition alone, not excluded."

## How the test was run

The design was fixed before any enrichment number was seen. The gene set is MSigDB HALLMARK\_INTERFERON\_ALPHA\_RESPONSE (M5911, 97 genes; Liberzon et al. 2015, PubMed 26771021, CC-BY-4.0). Five genes carry older source identifiers on the MSigDB page and were mapped to current symbols: FAM125A to MVB12A, FAM46A to TENT5A, FTSJD2 to CMTR1, PRIC285 to HELZ2, and WARS to WARS1.

- **Statistic:** the number of distinct Hallmark genes covered by at least one probe in the set. Distinct genes were chosen over probe counts because neighboring CpGs move together and one locus, such as IFI44L with 7 probes, would otherwise inflate the count.
- **Primary null:** 10,000 random probe sets of the observed size, drawn from the 471,758 probes eligible for selection (no NaN in any sample). Each draw matches the observed numbers of promoter-proximal, other-gene and no-gene probes.
- **Sensitivity null:** the same draws without context matching.
- **Primary tests:** hypo and hyper subsets, one-sided, with a threshold of 0.025 (Bonferroni across two tests). Everything else is secondary.
- **Sanity checks passed:** the pool size matched the earlier run, all 97 genes have at least one eligible probe, and the 243 promoter-proximal stable probes were reproduced.

The null draws independent probes, while real differential probes cluster in correlated blocks. For the distinct-gene statistic this makes the test conservative.

Script: `09_ifn_enrichment_permutation.py`. Results: `stage3_ifn_enrichment_results.json`.

## Next steps

Two analyses follow, and the first is quick and uses files already on hand.

1. **Inspect the six hyper-set genes probe by probe.** For EPSTI1, HELZ2, LGALS3BP, CNP, CSF1 and TRIM14, list each stable probe's direction and whether it is promoter-proximal. This would show whether the mixed-direction-loci reading is right.
2. **Choose the interferon network for the betweenness figure.** KEGG and Reactome give somewhat different graphs, and betweenness centrality depends on the graph. The plan is to declare one network in advance and run the other as a sensitivity check.

- [x] Draft the Stage 3 README section from these results
- [x] Check that KEGG and Reactome network data are both accessible before choosing
