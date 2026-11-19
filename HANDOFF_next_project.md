# Handoff: Project 8 (SLE methylation, GSE82218) -> next project

## Status
Project 8 is complete through Stage 3. Remaining admin: GitHub push, Blog Post 5, LinkedIn post.

## Results to carry forward (one line each)
- Stage 1: Project 6 transformer zero-shot does not separate SLE vs NC (LR acc 0.600 vs 0.545 baseline). Interpretation revised: the CLS embedding does not preserve the signal; the signal IS in the input.
- Stage 2: fine-tuned transformer 0.587 (early stop), 0.680/0.687 (30 epochs, A/B; within noise of each other, under-fitting). L2 logistic regression 0.967 (5,000 probes), 0.967 (in-fold top 1,000 of all probes), single probe cg06872964 0.987. Same 5-fold stratified CV, seed 42, ~11 test samples per fold.
- Stage 3: 460 stable probes (116 hypo / 344 hyper). Hypo set: 25 of 97 Hallmark IFN-alpha genes (null mean 0.63, p < 0.0001); one 34-gene STRING module (null mean 2.6, p < 0.0001, robust across thresholds). Replication vs Absher 2013 sorted T-cells: 97.3% same direction of 1,009 CpGs (permutation p = 0.0007). Composition hypothesis: exploratory only, deconvolution not done.

## Lessons / reusable method (carry into next project)
1. Always test a simple linear baseline BEFORE a transformer; it changed the whole story here.
2. Declare primary tests before running; label everything else secondary/exploratory.
3. Feature selection inside the CV fold only; t-stats of selected probes are circular, rank by effect size.
4. Use permutation nulls matched to the confound (probe context, degree, label shuffle); naive binomial is anti-conservative for correlated CpGs.
5. Row-alignment asserts after every merge/reindex (Rule #12).
6. Attention over 5,001 tokens needs GPU for training (Colab); wide parquet needs raised thrift limits.
7. Document dropped analyses (betweenness on KEGG, integrated gradients, TOMTOM, saga elastic-net).

## Open items (honest list)
- Monocyte count mismatch in Absher Table S3: 41 parsed vs 97 reported; unresolved, monocytes excluded.
- No cell-type deconvolution (EpiDISH/Houseman): composition remains untested.
- No SNP / cross-reactive probe masking in full-matrix analyses (Zhou lab masks exist).
- Reactome not checked; Absher planning Claude Doc has a stale "unverified" section.
- The 34-gene module is ~31 independent loci (shared probes).
- Transformer failure mechanism (CLS attention dilution) is an untested hypothesis.
- 3 transformer configs tried; no further tuning by design.

## Housekeeping before starting the next project
- [ ] Create requirements.txt and .gitignore (exclude data/; STRING file is 133 MB)
- [ ] Remove *:Zone.Identifier files, git status, root commit + push (user MShekfeh)
- [ ] Copy small result JSONs into results/ (README paths point there)
- [ ] Update Absher planning doc
- [ ] Blog Post 5 (Medium) + LinkedIn post for Project 8
- [ ] Overdue: Blog Posts 3 and 4, LinkedIn posts for Projects 6 and 7

## Key files (compbio, ~/projects/project8-SLE-methylation)
scripts/01-18, figures/{gse82218_zeroshot_umap, stage3_string_module, stage3_summary_panels}.png,
data/ (not committed): gse82218_beta_matrix_full.parquet, stable_probes_annotated.tsv, absher_s3_clean.tsv, result JSONs.

## Instruction for the next chat
Say which week/project you are on. Read the master plan and teaching style guide, then explain biology and ML from scratch, explain code, one command at a time (`which python`, `pwd` first, `python -u`), remind about LinkedIn/Medium posts.
