# Project 8 Kickoff Addendum — SLE/Lupus Methylation
## Append to `genomics_ml_master_plan_MERGED.md` (Weeks 23–25 section) and note in `teaching_style_guide_UPDATED.md`'s Progress Tracker
### Session: Project 8 kickoff, following the Project 7→8 handoff · August 2026

---

## Why This Addendum Exists

The original Weeks 23–25 plan text said to "load a pretrained Nucleotide
Transformer (or similar genomic LLM)" for zero-shot probing, without
pinning a specific checkpoint or resolving what "zero-shot" means for
methylation array data specifically. This was worked through explicitly
this session and needs to be locked into the planning docs so a future
Claude session doesn't re-litigate it or default to the wrong
interpretation.

---

## Decision 1 — What "Zero-Shot Probing" Means for This Project

**Resolved, binding decision:** Zero-shot probing = embedding GSE82218
samples through the **already-trained Project 6 methylation transformer**
(trained only on TCGA pan-cancer methylation, never seen lupus or any
autoimmune data), with no fine-tuning. This is *not* a public pretrained
sequence LLM (Nucleotide Transformer, DNABERT-2) — those take raw DNA
sequence as input, and GSE82218 (like all methylation array data) is
beta values at fixed CpG probes, not sequence. Neither Nucleotide
Transformer nor DNABERT-2 has ever seen a beta value as input, so
"zero-shot" on them would be a category error, not a meaningful test.

**The actual scientific question this tests:** does a model trained
exclusively on cancer epigenetic dysregulation already separate lupus
from healthy PBMC methylation, zero-shot? This is a more interesting and
more honest framing than the original plan text implied, and it directly
reuses infrastructure Marwa already has and trusts (rather than
onboarding a new pretrained checkpoint under time pressure).

**Two other options were considered and explicitly not chosen (for the
record, in case revisited later):**
- DNABERT-2 on flanking sequence context around each CpG (conflates
  sequence signal with methylation signal — more complex, not chosen)
- A methylation-specific pretrained model (CpGPT/MethylGPT — genuinely
  the most literal match to the original plan's intent, but adds a new
  untested dependency; deferred, not ruled out permanently)

**Update needed to master plan text:** replace "Load a pretrained
Nucleotide Transformer (or similar genomic LLM)" with the resolved
framing above, so this doesn't need to be re-decided next session.

---

## Decision 2 — Real GSE82218 Sample Counts (Confirmed via GEO)

The handoff document had GSE82218 as "not yet downloaded, structure
unknown." Now confirmed directly from the GEO Series page:

| Group | n |
|---|---|
| Normal Control (NC) | 25 |
| SLE (no nephritis) | 15 |
| SLE + Lupus Nephritis (SLE+LN) | 15 |
| **Total** | **55** |

- **Platform: GPL13534 (Illumina HumanMethylation450)** — same platform
  as the TCGA data used in Projects 6 and 7. This is good news for the
  zero-shot test: probe ID space (`cgXXXXXXXX`) substantially overlaps
  with what the Project 6 transformer was trained on, so there's no
  platform-mismatch confound going into the zero-shot embedding step.
- Tissue: PBMC (whole peripheral blood mononuclear cells, not a sorted
  cell subtype)
- Part of SuperSeries GSE82221; matched expression data exists as a
  separate subseries (GSE81622) — not needed now, but worth remembering
  it exists if a future validation step wants to cross-check methylation
  findings against expression in the same cohort.
- Submitting group: UT Southwestern (Zhu, Mi, Luo, Chen, Liu, Zuo, Li);
  no associated publication currently linked in GEO's citation field.

---

## Decision 3 — Small-N Framing, Decided Now Rather Than Discovered Later

**55 total samples is smaller than Project 7's Normal-only clock training
set (205)**, which was itself already flagged as small. Binding decision
for this project's statistics, stated now per the teaching guide's
pattern of front-loading known limitations:

- **Primary, adequately-powered comparison: SLE vs. NC** (30 vs. 25 —
  note this combines SLE and SLE+LN into one "SLE" group of 30 for this
  framing, vs. NC's 25). This is the comparison the fine-tuning
  classifier (Stage 2) and the main attribution/network analysis
  (Stages 3–4) should be built around.
- **Exploratory only: SLE vs. SLE+LN** (15 vs. 15). Any subgroup finding
  here should be reported as hypothesis-generating, not as a validated
  result — 15 vs. 15 is underpowered for a confident claim, and this
  should be stated plainly in the eventual README/blog post rather than
  discovered after the fact when a p-value disappoints.

---

## Decision 4 — Data Source: Series Matrix File, Not Raw Intensities

GEO's page for GSE82218 explicitly notes **"Processed data included
within Sample table"** — meaning the beta values are already present as
a data table inside the **Series Matrix File** itself
(`GSE82218_series_matrix.txt.gz`), not in a separate raw file.

**Decision: use the Series Matrix File.** Do NOT use
`GSE82218_non-normalized.txt.gz` (134MB raw signal intensities) — using
that would require implementing our own 450K normalization pipeline
(background correction, dye-bias correction, beta-value computation),
which this project doesn't need to own. This mirrors the same
"use the processed format, don't reinvent a preprocessing pipeline"
choice made for TCGA's SeSAMe beta files in Project 6, Stage 1.

**First technical task, in progress as of this addendum:** download and
inspect the Series Matrix File header to confirm exactly how NC/SLE/
SLE+LN group labels are encoded in the `!Sample_characteristics_ch1`
rows, before writing any parsing code — per the "verify actual file
structure before writing a parser" habit from Project 6's Known Issues.

---

## Decision 5 — Considered and Rejected a Larger Dataset; Confirmed GSE82218 as Primary

Before committing to GSE82218's 55 samples, we explicitly checked for a
larger alternative, rather than defaulting to the first dataset the plan
named. Two real candidates were evaluated:

- **CLUES (California Lupus Epidemiology Study), n=101** — larger, but
  run on the **Illumina EPIC array**, not 450K. Rejected: this would
  break the clean zero-shot platform match between GSE82218 and the
  Project 6/7 transformer's 450K-trained probe space (see Decision 2).
- **Absher et al. 2013 (the paper already assigned as this project's
  "Paper of the Week"), n=107, 450K platform** — the most attractive
  candidate on paper: larger sample size, correct platform, and literally
  the same source as our required biology reading. **Checked directly
  against the published paper's full text and Supporting Information
  list — this dataset has no public per-sample GEO deposit.** Only
  aggregate Supporting Information tables (Table S3: significant CpGs
  with per-group mean beta values; Table S4: shared genes across cell
  types) are available, not a downloadable sample-by-sample beta matrix.
  This was standard practice for a 2013 paper, predating near-universal
  journal GEO-deposition requirements. **Conclusion: not usable as
  training/embedding data — there is nothing to load per-sample.**

**Final decision: GSE82218 (55 samples) remains the primary dataset**,
per the original plan and Decision 2 above.

**Consolation, not nothing — a genuine external validation step gained
at zero extra data-wrangling cost:** Absher's Table S3/S4 gene lists
(CpGs/genes found significantly hypomethylated near IFN-regulated genes,
independently derived from the entirely separate 107-sample UAB cohort)
give us a real, published reference list. Once Stage 3's attribution
analysis (integrated gradients) surfaces GSE82218's own top CpGs/genes,
**cross-check for overlap with Absher's list** as an external replication
check — the same "discovery cohort vs. independent validation" logic used
in the 2025 childhood-lupus paper found during this same search, except
here the "validation cohort" is a different published study rather than
a held-out sample split. This should be built into Stage 4 (network
mapping), not treated as optional: report the overlap percentage plainly,
whatever it turns out to be, the same way Jaccard overlaps and other
honest comparison numbers have been reported throughout this plan.

---

## Standing Rules Still To Be Applied (Not Yet Checked for GSE82218)

Both explicitly called out in the Project 8 handoff as needing to be
re-applied to this new dataset — **not yet done as of this addendum**,
should be the first checks once the Series Matrix data table is parsed:

- **Rule #11 (patient/sample grouping):** check for repeated patients or
  technical replicates in GSE82218 before assuming one-sample-per-row.
  TCGA's 239-duplicated-patient finding in Project 7 is the cautionary
  precedent — don't assume a GEO cohort is cleaner just because it's
  smaller.
- **Rule #12 (row-order alignment):** if group labels and beta values
  end up needing separate parsing steps from the Series Matrix File,
  verify alignment against a shared field (e.g. GSM accession ID)
  explicitly rather than trusting column order.

---

## Progress Tracker Update

No status change yet — Project 8 remains **🔲 In Progress** (moved from
🔲 Next), Stage: dataset located and download strategy decided, Series
Matrix File download in progress. Update both
`teaching_style_guide_UPDATED.md` and `genomics_ml_master_plan_MERGED.md`
trackers to reflect "In Progress" rather than "Next" once this addendum
is filed.

---

*Addendum written and updated: August 2026, Project 8 kickoff session.
Updated once more after a deliberate detour to check for a
larger-sample-size alternative dataset before committing engineering
time — CLUES (EPIC platform mismatch) and Absher et al. 2013 (no public
per-sample data) were both evaluated and ruled out; GSE82218 confirmed
as primary. First data download command not yet run as of this update.
Bring alongside `teaching_style_guide_UPDATED.md` +
`genomics_ml_master_plan_MERGED.md` +
`session_summary_project8_sle_kickoff_handoff.md` to the next session.*
