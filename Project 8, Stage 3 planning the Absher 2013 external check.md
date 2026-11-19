# Project 8, Stage 3: planning the Absher 2013 external check

Nov 11, 2026 · @Marwa

## Absher et al. found interferon hypomethylation in all three sorted cell types

[Absher et al. 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3738443) studied 49 SLE patients and 58 controls on the Illumina 450K array, in sorted CD4+ T-cells, CD19+ B-cells and CD14+ monocytes. Monocytes came from about half of the subjects (27 SLE, 27 controls). These figures come from the abstract as quoted in search results, because the full text could not be opened.

- **Significant CpGs (p < 1×10⁻⁸):** 166 in B-cells, 97 in monocytes and 1,033 in T-cells.
- **Shared signal:** hypomethylation near type I interferon genes appeared in all three cell types, in both active and quiescent patients.
- **Composition effect:** the authors describe shifts in CD4+ subtype proportions, plus a lower-amplitude shift at more than 16,000 CpGs (FDR < 1%) near cell-division and MAPK genes.

A second, independent study ([Ann Rheum Dis 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5909746)) reports differential methylation of |Δβ| > 0.1 almost exclusively at interferon-regulated genes. Only its search snippet was read, so it needs reading before it is cited.

## Sorted cells versus bulk PBMC: agreement is meaningful, a miss proves little

GSE82218 is bulk PBMC, while Absher et al. sorted their cells. Overlap between the two would be real corroboration, because the sources, cohorts and filters are independent.

A miss would be far weaker evidence. A cell-type-specific signal can be diluted in a bulk sample, and the two studies applied different significance filters. The check should therefore be framed as corroboration when it overlaps, and as inconclusive when it does not.

## The "Tables S3/S4" in the kickoff note are unverified

The supplementary listing could not be read. PMC answered with a CAPTCHA, and the publisher's DOI page was not an allowed fetch, so no workaround was attempted.

The article text seen so far says only that Tables S1 and S2 describe the cohort demographics. It is unknown what S3 and S4 contain, and whether they list CpG IDs or only genes. That decides whether a probe-level comparison is possible.

## The check, declared before any comparison is run

The primary comparison asks what fraction of our hypomethylated stable probes appear in Absher's significant-CpG lists, taking the union across cell types. It uses the same context-matched permutation design as the interferon enrichment test.

- **Primary:** probe overlap against random probe sets of the same size from the eligible pool, matched on promoter-proximal, other-gene and no-gene counts.
- **Direction concordance:** among overlapping probes, do both datasets show SLE hypomethylated? Agreement in direction is stronger validation than overlap alone.
- **Secondary:** gene-level overlap, and a per-cell-type breakdown.
- **Fallback:** if the supplement lists genes and not CpG IDs, only the gene-level comparison is possible. This will be stated before anything is run.

Limits stated up front: different cell types, different cohorts and a different significance filter.

## Next: read the supplement list from the article in a browser

The supplementary files have to be listed and downloaded by hand, because they could not be reached from here.

- [ ] Open the article ([PMC3738443](https://pmc.ncbi.nlm.nih.gov/articles/PMC3738443) or the PLOS Genetics page for doi 10.1371/journal.pgen.1003678) and find the Supporting Information section
- [ ] Paste the titles of every supplementary item, with file formats
- [ ] Download the tables that list significant CpGs or genes into `data/`
- [ ] If the files are Excel, check that `openpyxl` imports
- [ ] Run the `Zone.Identifier` cleanup after copying from Windows, before any `git add`

Once the table titles are known, the check above is either run at probe level or reduced to genes.
