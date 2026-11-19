## Stage 1 — Zero-Shot Probing: Does a Cancer-Trained Model Already Know About Lupus?

### Biological/ML Question

Does a methylation transformer trained exclusively on cancer epigenetic
dysregulation (5 TCGA classes: BRCA, PRAD, LUAD, COAD, Normal) already
separate systemic lupus erythematosus (SLE) from healthy PBMC methylation,
with **zero lupus-specific training**? If the pretrained embedding space
already encodes something disease-relevant, that's a strong "foundation
models generalize across disease domains" finding. If not, that's a real
result too -- it motivates fine-tuning as a genuine next step rather than
an assumed one.

### Dataset

**GSE82218** (UT Southwestern, PBMC DNA methylation, Illumina
HumanMethylation450 -- same platform as the Project 6/7 TCGA cohort):

| Group | n |
|---|---|
| Normal Control (NC) | 25 |
| SLE (no nephritis) | 15 |
| SLE + Lupus Nephritis (SLE+LN) | 15 |
| **Total** | **55** |

Age range 10-46 years -- note this cohort includes pediatric patients
(ages 10, 12, 13, 17, 18 all present), unlike the adult-only TCGA cohort
this project's zero-shot model was trained on. Worth keeping in mind as a
population difference, not just a disease-domain difference.

**Primary comparison, decided in advance (not chosen post-hoc):**
SLE-combined (SLE + SLE+LN, n=30) vs. NC (n=25). The 3-way split is
reported as exploratory only, per the small-N framing decided at project
kickoff.

### What I Built

1. Parsed GSE82218's Series Matrix File directly (processed beta values
   are included in the sample data table -- no separate normalization
   pipeline needed, matching the approach used for TCGA's SeSAMe files
   in Project 6).
2. Subset and reordered GSE82218's 485,577 probes to the **exact 5,000-probe
   vocabulary** the Project 6 transformer was trained on, preserving
   column order (order matters -- the model's input positions are
   probe-specific, not just probe-agnostic features). **Result: 5,000/5,000
   probes found, full overlap** -- no platform-compatibility confound
   going into the zero-shot test.
3. Found and mean-imputed 192 NaN beta values (0.07% of the matrix, 144
   probes, 54/55 samples affected, spread proportionally across all three
   groups) before running the model -- see Known Issues below.
4. Loaded the verified Project 6 `MethylationTransformer` checkpoint
   (`best_transformer.pt`), ran all 55 samples through in `return_embedding=True`
   mode to get 128-dim CLS embeddings, no fine-tuning.
5. UMAP visualization of the embeddings, colored by group.
6. **Quantified the separation formally** rather than relying on the UMAP
   plot alone: 5-fold stratified cross-validated logistic regression on
   the raw embeddings, predicting group, compared against a majority-class
   baseline.

### Known Issues Encountered This Stage

**Full self-attention over 5,001 tokens (5000 CpG + CLS) is memory-expensive
even at inference, even with only 55 samples.** A single forward pass over
all 55 samples attempted to allocate ~22GB (`55 x 4 heads x 5001^2 x 4 bytes`),
failing on `compbio`'s CPU-only, ~6.2GB budget -- the same constraint
`methylation_model.py`'s own docstring warns about for *training*, which
turned out to apply to inference too, since attention memory cost is driven
by sequence length, not batch size. **Fix: batch size 4** (roughly
4 x 400MB = 1.6GB peak), matching the batching approach Project 6's own
`09_attractor_analysis.py` used (`DataLoader(..., batch_size=32)` on GPU;
4 was chosen here for a smaller CPU budget).

**192 NaN values in the source beta matrix** (0.07% of cells, 144/5000
probes, 54/55 samples) -- individual probes occasionally fail QC on
individual array chips. Confirmed the missingness was sparse and spread
proportionally across all three groups (not concentrated in one group,
which would have suggested a batch/quality confound correlated with
disease status) before choosing per-probe mean imputation as a low-risk
fix over dropping the 144 affected probes entirely.

### Key Result — Clean Negative

| Comparison | Logistic regression CV accuracy | Majority-class baseline | Difference |
|---|---|---|---|
| **SLE (combined) vs. NC** (primary) | 0.600 (+/- 0.136) | 0.545 | +0.055 |
| NC / SLE / SLE+LN (exploratory) | 0.455 (+/- 0.057) | 0.455 | +0.000 |

**A model trained exclusively on cancer epigenetic dysregulation does not
zero-shot separate lupus from healthy PBMC methylation.** The primary
comparison's +0.055 improvement over baseline is well within its own
cross-validation noise (+/- 0.136 std, more than double the effect size)
-- not distinguishable from the classifier having learned nothing. The
3-class comparison shows literally zero improvement over always guessing
the majority class.

This is consistent with the UMAP visualization: NC, SLE, and SLE+LN
samples are thoroughly intermixed throughout the main embedding structure,
with no visually apparent grouping by disease status.

**One structural feature in the UMAP plot -- a small, tight 6-point
subcluster -- does NOT correspond to disease group** (it contains a mix
of all three labels). Checked against the two other variables available
(age, sex): neither showed a clear association (mean age 29.4 vs. 31.5
across the split; no obvious sex skew). **Honestly unexplained** -- most
plausibly a technical/batch signal not captured in this dataset's
available metadata, but that is a plausible guess, not a demonstrated
finding.

### Interpretation

This is a real, informative negative result, not a stalled analysis.
Cancer and autoimmune disease involve different epigenetic mechanisms at
the level of *which* pathways are dysregulated (oncogene/tumor-suppressor
methylation vs. interferon-pathway methylation), even though both involve
"aberrant" methylation broadly. A model that learned to distinguish tumor
types from normal tissue apparently did not incidentally learn a
representation that separates immune activation states -- these appear to
require genuinely different features, not just a different decision
boundary in the same feature space. **This motivates Stage 2 (fine-tuning)
as a real, open question** -- not "will fine-tuning obviously help" but
"can this architecture learn the lupus-relevant signal at all, given only
55 samples," which is itself an interesting test given Project 7 Stage 4's
prior finding that this same architecture family struggled to generalize
from small labeled sets.

### Figure

`figures/gse82218_zeroshot_umap.png`

### Files

- `scripts/01_parse_series_matrix.py` -- metadata parsing + alignment check
- `scripts/02_parse_beta_matrix_and_subset.py` -- full beta matrix parse,
  subset to Project 6's 5000-probe vocabulary
- `scripts/02b_impute_missing_values.py` -- per-probe mean imputation
- `scripts/03_zeroshot_embed_and_umap.py` -- embedding extraction + UMAP
- `scripts/04_quantify_separation.py` -- cross-validated separation test
- `data/gse82218_zeroshot_input_imputed.parquet` -- final model-ready input
- `data/gse82218_zeroshot_embeddings.npz` -- raw 128-dim embeddings + metadata

### What's Next

Stage 2: fine-tune on GSE82218 for lupus-vs-healthy classification, using
the same primary (SLE-combined vs. NC) / exploratory (3-way) framing
established here.
