# Session Handoff: Project 8 Stage 1 (Zero-Shot Probing) COMPLETE → Stage 2 (Fine-Tuning) Up Next
### Marwa Shekfeh · Genomics ML + Complex Systems Plan · August/September 2026

---

## Context for the Next Claude — Read This First

> Read `teaching_style_guide_UPDATED.md` first. Then read
> `genomics_ml_master_plan_MERGED.md` — single source of truth for pacing
> and week numbering. Then read `project8_kickoff_addendum.md` (documents
> the zero-shot definition decision, the dataset-search detour, and the
> small-N framing — all binding decisions from the kickoff session, not
> open questions). Then read this handoff. Say: "I am continuing the plan
> — Project 8, Stage 2, fine-tuning GSE82218 for lupus-vs-healthy
> classification."

**Just completed:** Project 8, Stage 1 — Zero-shot probing of GSE82218
through the Project 6 methylation transformer, with a genuine, clean
**negative result**: the cancer-trained model does not zero-shot separate
lupus from healthy PBMC methylation. Full writeup already drafted:
`STAGE1_README_SECTION.md` (ready to fold into the Project 8 README when
that gets built — do not re-derive this section, it's done).

**Up next:** Stage 2 — fine-tune on GSE82218 for lupus-vs-healthy
classification (per the merged plan's Weeks 23–25 structure).

**GitHub username:** MShekfeh
**Conda environment:** genomics-ml (compbio, WSL)
**Project folder:** `~/projects/project8-SLE-methylation/`
(`scripts/`, `data/`, `figures/` subfolders — same convention as every
prior project)

---

## Files That Exist Now — Know These Before Writing Any New Script

| File | What it is |
|---|---|
| `scripts/01_parse_series_matrix.py` | Parses GSE82218's Series Matrix File header — extracts group (NC/SLE/SLE+LN), age, sex per sample. Verifies alignment between metadata and data table (Rule #12) before saving anything. |
| `scripts/02_parse_beta_matrix_and_subset.py` | Parses the full 485,577-probe beta table, subsets + **reorders** to Project 6's exact 5,000-probe vocabulary (order matters — model input positions are probe-specific). Confirmed **5,000/5,000 probe overlap, zero missing** — full platform compatibility, no confound. |
| `scripts/02b_impute_missing_values.py` | Mean-imputes 192 NaN beta values (0.07% of matrix, 144/5000 probes, 54/55 samples, spread proportionally across all 3 groups — confirmed non-group-correlated before choosing imputation over dropping probes). |
| `scripts/03_zeroshot_embed_and_umap.py` | Loads Project 6's `best_transformer.pt` into the (verified-matching) `MethylationTransformer` class, embeds all 55 samples in `return_embedding=True` mode, batched at **batch_size=4** (see Known Issues below for why), UMAP visualization. |
| `scripts/04_quantify_separation.py` | Cross-validated logistic regression on the embeddings vs. majority-class baseline — the actual quantification behind the "no separation" conclusion, not just an eyeballed UMAP plot. |
| `scripts/methylation_model.py` | **Copied from Project 6** into this project's folder (see Known Issues — do not import across project folders, copy the file in). Contains `MethylationTransformer` class definition (verified via `strict=True` state_dict load, zero missing/unexpected keys). |
| `data/best_transformer.pt` | **Copied from Project 6.** Verified checkpoint: 5,000 probes, d_model=128, 4 layers, 5-class classifier head. **NOT** the same as Project 7's `pretrained_backbone.pt` / `transformer_clock_finetuned.pt` — those are `MethylationTransformerV2` (2,000 probes, regression head) and were mistakenly located first this session before the correct file was found. If a checkpoint file's provenance is ever unclear again, inspect the state_dict directly (`n_probes`, presence/absence of `regression_head`/`mask_token`/`classifier`) rather than trusting the filename alone. |
| `data/project6_cpg_probe_list.txt` | The exact 5,000 CpG probe IDs, in order, extracted from `beta_matrix_with_ids.parquet`'s column names. |
| `data/gse82218_zeroshot_input_imputed.parquet` | Final, model-ready input: 55 samples × 5,000 probes (imputed) + group/age/sex. **This is the file to load for Stage 2**, not the raw series matrix or the pre-imputation version. |
| `data/gse82218_zeroshot_embeddings.npz` | Raw 128-dim zero-shot CLS embeddings + metadata, for any future reference. |
| `figures/gse82218_zeroshot_umap.png` | The zero-shot UMAP figure — NC/SLE/SLE+LN thoroughly intermixed, no visual separation, plus one unexplained 6-point subcluster (doesn't correspond to disease group, age, or sex). |
| `STAGE1_README_SECTION.md` | Full Stage 1 writeup, ready for the eventual Project 8 README. |

---

## Stage 1 Result — Do Not Re-Derive, Just Cite

**Primary comparison (SLE-combined vs. NC, 30 vs. 25):** logistic regression
CV accuracy 0.600 (±0.136) vs. majority-class baseline 0.545 — a +0.055
difference well within the cross-validation noise. **Not distinguishable
from the model having learned nothing.**

**Exploratory 3-class (NC/SLE/SLE+LN):** CV accuracy 0.455 vs. baseline
0.455 — **exactly zero improvement.**

**Conclusion, stated plainly:** a methylation transformer trained
exclusively on cancer epigenetic dysregulation does not zero-shot separate
lupus from healthy PBMC methylation. This is a genuine, clean negative
result (in the same honest-negative-result tradition as Project 7 Stage 4),
and it motivates Stage 2 as a real open question — not "will fine-tuning
obviously help" but "can this architecture learn the lupus-relevant signal
at all, given only 55 samples," which is itself uncertain given Project 7
Stage 4's prior finding that this architecture family struggled to
generalize from small labeled sets (139–164 samples/fold there; here we'd
have even fewer per fold).

**The small UMAP subcluster (6 points) remains genuinely unexplained** —
checked against age and sex, neither showed a clear association. Don't
force an explanation if this comes up again; it's an honest open item, not
a solved one.

---

## Known Issues From This Session — Read Before Repeating Any of These

**Full self-attention over 5,001 tokens is memory-expensive even at
INFERENCE, even with a tiny sample count.** A single forward pass of all
55 samples tried to allocate ~22GB on CPU (`55 × 4 heads × 5001² × 4 bytes`)
and failed on `compbio`'s ~6.2GB budget. The lesson from `methylation_model.py`'s
own docstring (which warns about this for *training*) applies to inference
too, because attention memory cost scales with **sequence length**, not
batch size — 55 samples was never the problem, 5,001 tokens per sample was.
**Fix: batch_size=4** (≈1.6GB peak). **General rule going forward: any time
this model (or `MethylationTransformerV2`) is run on CPU, always batch,
regardless of how small the sample count looks — check the sequence-length
math first, not just the sample count.**

**GEO Series Matrix beta values can contain NaN (failed probes on
individual chips) — verify before modeling, don't assume clean data.**
Found 192 NaN values (0.07% of the matrix) only because we checked
explicitly; would have silently produced NaN embeddings otherwise (NaN
propagates through `value_projection`'s linear layer and poisons that
entire sample's embedding). Confirmed the missingness was sparse and
spread proportionally across all three groups (not concentrated in one
group, which would suggest a quality confound correlated with disease
status) before choosing per-probe mean imputation as the fix. **New
standing rule candidate for the teaching guide: any time a beta-value
matrix comes from a new external source (not your own pipeline's output),
explicitly check for NaN before running it through any model — do not
assume GEO/array data arrives clean.**

**Checkpoint/architecture confusion cost real session time — verify
provenance from the object itself, not the filename.** Both
`pretrained_backbone.pt` and `transformer_clock_finetuned.pt` (Project 7
Stage 4 artifacts) were located and nearly used before the correct
`best_transformer.pt` (Project 6) was found. All three have plausible-
sounding names. **What actually resolved it:** loading each checkpoint's
state_dict directly and checking `n_probes` (2000 vs 5000) and the
presence/absence of `regression_head`/`mask_token`/`classifier` keys —
concrete, checkable facts, not filename pattern-matching. Same category of
lesson as Project 6's "verify a downloaded tool's actual contents before
trusting file extensions or names."

**pyarrow missing from the active environment, twice, same session —
reinforces the standing `which python` habit.** First hit when writing
the full beta matrix to parquet; resolved by confirming `genomics-ml` was
active. Exact same category as the long-standing "after WSL restart, env
isn't auto-activated" issue — worth treating pyarrow/parquet failures
specifically as a first-suspect case of "check which environment is
active" before assuming a package needs installing.

**Files uploaded to a Claude chat for inspection are NOT thereby present
on compbio.** When methylation_model.py was uploaded here for verification,
it existed only in the chat session until explicitly copied onto
`compbio`'s filesystem — this is the same lesson as the long-standing
"files Claude creates in chat don't exist on the user's machine until
transferred" rule, just encountered from the opposite direction (a file
the user uploaded to Claude, not one Claude created for the user).

---

## What Stage 2 (Fine-Tuning) Covers, Per the Plan

Fine-tune on GSE82218 for lupus-vs-healthy classification. Concrete
decisions already made that Stage 2 should inherit, not re-litigate:

1. **Primary comparison: SLE-combined (n=30) vs. NC (n=25).** 3-way
   (NC/SLE/SLE+LN) is exploratory only — 15 vs. 15 for the SLE+LN split
   is underpowered for a confident claim (Decision 3, kickoff addendum).
2. **Use the same 5,000-probe input vocabulary and the same imputed data
   file** (`gse82218_zeroshot_input_imputed.parquet`) — don't re-derive
   the probe subsetting or re-run imputation from scratch.
3. **Apply Rule #11 (patient/sample grouping) explicitly before designing
   any train/test split**, even though GSE82218 appears to be
   one-sample-per-patient by construction (no dedicated patient ID field
   exists in this dataset, unlike TCGA's `case_id`) — this was checked at
   Stage 1 and found clean, but Stage 2 should note this was checked
   rather than silently assume it.
4. **Given only 55 samples total (30/25 for the primary split), think
   carefully about whether a simple train/test split is even viable, or
   whether cross-validation (the same `StratifiedKFold` pattern already
   used in `04_quantify_separation.py`) is the more honest evaluation
   approach** — this is the same class of small-N decision Project 7
   Stage 3 faced with its 205-sample Normal-only clock training set, just
   more acute here. Discuss with Marwa explicitly rather than defaulting
   to an 80/20 split that would leave a test set of ~11 samples.
5. **Architecture starting point:** the plan doesn't pin whether Stage 2
   fine-tunes the full Project 6 `MethylationTransformer` end-to-end,
   fine-tunes only a new classification head on frozen embeddings, or
   pretrains fresh on GSE82218's own data first (the
   `MethylationTransformerV2` pretrain+fine-tune pattern from Project 7
   Stage 4 is a precedent, though that project's negative result at
   139–164 samples/fold is a real cautionary note for what to expect at
   an even smaller N here). **Decide this explicitly with Marwa at the
   start of Stage 2, the same way the zero-shot definition was decided
   explicitly at kickoff** — don't default silently.
6. After fine-tuning: **Stage 3 (attribution analysis via integrated
   gradients + TOMTOM motif enrichment)** and **Stage 4 (IFN network
   mapping + betweenness centrality)** follow, per the original plan —
   not this session's job, but worth keeping in view when deciding Stage
   2's architecture, since Stage 3 needs a model whose predictions/
   gradients are interpretable at the level of individual CpGs.

**Paper of the week reminder:** Absher et al. 2013 should already have
been read per the kickoff handoff's instruction — confirm this happened,
and remember the addendum's Decision 5: Absher's data isn't downloadable
per-sample, but its **Table S3/S4 gene lists are a planned external
validation check for Stage 3/4's attribution results**, not something
needed yet in Stage 2.

---

## Standing Rules Still Applying (Unchanged From Kickoff)

- Rule #11 (patient/sample grouping) — checked at Stage 1, revisit if
  Stage 2's split design raises any new question.
- Rule #12 (row-order alignment) — applied throughout Stage 1's parsing;
  no new data sources expected in Stage 2, but if any external file is
  joined in, verify explicitly.
- Re-derive any control/ablation logic from the actual question, not old
  planning-doc pseudocode (Rule from Project 5) — relevant if a
  randomization null or held-out check gets added to Stage 2.
- `which python` / `pwd` checks remain standing habits every session.

---

## Outstanding Non-Blocking Content Items (Carried Forward, Unchanged)

Per Sustainability Rule 5 — raise gently, don't block Stage 2 on these:
- Blog Post 3 (Project 6): drafted, not published.
- Blog Post 4 (Project 7): drafted, not published.
- LinkedIn posts for Projects 6 and 7: not posted.
- **New from this session, not yet drafted:** no blog post or LinkedIn
  content has been drafted for Project 8 Stage 1 yet — the plan's Blog
  Post 5 ("Do genomic LLMs already know about lupus?") is intended to
  cover the *whole* Project 8 arc (zero-shot through network mapping), so
  this likely shouldn't be drafted until Stage 4 is done, per Sustainability
  Rule 5 ("content follows results, never the other way around") — but
  worth flagging to Marwa now that Stage 1's honest-negative framing is
  exactly the kind of "even though it wasn't the result I expected" moment
  the plan's Interview Narrative section already values.

---

## Quick-Start Checklist for the Next Claude

1. Confirm `which python` shows `.../envs/genomics-ml/...` on `compbio`.
2. Confirm `pwd` is `~/projects/project8-SLE-methylation/scripts/` before
   running anything with relative paths.
3. Read `STAGE1_README_SECTION.md` in full before writing any Stage 2 code
   — the primary/exploratory comparison framing and the exact input file
   to use are both decided there, not open questions.
4. First real decision for Stage 2: architecture starting point (see point
   5 above) — ask Marwa explicitly, don't default.
5. Second real decision: train/test split vs. cross-validation approach,
   given the small N (see point 4 above) — ask explicitly.
6. Gently raise the outstanding content items above, without blocking.

---

*Handoff written: end of Project 8 Stage 1 session (zero-shot probing
complete, clean negative result, README section drafted). Bring
`teaching_style_guide_UPDATED.md` + `genomics_ml_master_plan_MERGED.md` +
`project8_kickoff_addendum.md` + `STAGE1_README_SECTION.md` + this handoff
to the next session.*
