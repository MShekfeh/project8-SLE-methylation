## Stage 2 — Fine-Tuning and Linear Baselines: The Signal Is in the Input; the Transformer Does Not Extract It

### Biological/ML Question

Stage 1 found that the Project 6 methylation transformer, used zero-shot,
does not separate SLE from healthy PBMC methylation. Stage 2 asks two
things: (1) can full end-to-end fine-tuning on GSE82218 recover the signal,
and (2) is the signal actually present in the model's 5,000-probe input in
the first place? Question (2) turned out to be the more informative one.

### What Was Run

All experiments use the same outer evaluation: 5-fold `StratifiedKFold`
(seed 42) on the 55 GSE82218 samples, primary comparison SLE-combined (n=30)
vs. NC (n=25), metric AUROC. Each outer test fold holds about 11 samples.

1. **2a — Fine-tuning with early stopping** (`05_finetune_transformer_cv_colab.py`).
   Full fine-tune of the Project 6 transformer (new 2-class head; backbone LR
   1e-5, head LR 1e-3), nested CV with an inner validation split for early
   stopping. A first attempt on CPU was killed by the OS (out of memory):
   backpropagation keeps every layer's 5,001 x 5,001 attention activations,
   which exceeds the ~6 GB budget, so this ran on a Colab GPU.
2. **2b — Linear baselines** (`06_linear_baseline_cv.py`), same folds:
   (a) L2-regularized logistic regression on the 5,000-probe vocabulary;
   (b) the same on all probes, with the top 1,000 chosen by t-test **inside
   each training fold only**; (c) a single probe, cg06872964 (an IFI44L
   promoter probe chosen from prior literature before looking at any
   labels). A first version used an elastic-net penalty with the `saga`
   solver and was abandoned after about 30 minutes without finishing a fold;
   it was replaced with L2 + `lbfgs`. (The function is still named
   `elastic_net` in the code for minimal diff; the model is L2.)
3. **2c — Transformer diagnostic retry** (`06b_transformer_diagnostic_colab.py`).
   No early stopping, fixed 30 epochs, no inner split, only final-epoch test
   AUROC reported, training loss and AUROC logged. Two configurations
   declared in advance and both reported (backbone LR 1e-5 and 1e-4).

### Results

| Model | Fold-mean AUROC | Notes |
|---|---|---|
| 2a Transformer, fine-tuned with early stopping | 0.587 +/- 0.122 | best epoch was often epoch 1-2 |
| 2c Transformer, config A (backbone LR 1e-5) | 0.680 +/- 0.169 | train loss 0.67-0.74 at epoch 30 |
| 2c Transformer, config B (backbone LR 1e-4) | 0.687 +/- 0.138 | train loss 0.65-0.77 at epoch 30 |
| 2b(a) L2 logistic regression, 5,000-probe vocabulary | **0.967 +/- 0.052** | pooled out-of-fold AUROC 0.959 |
| 2b(b) L2 logistic regression, all probes, in-fold top-1000 | 0.967 +/- 0.042 | pooled out-of-fold AUROC 0.969 |
| 2b(c) single probe cg06872964 | 0.987 +/- 0.027 | pooled out-of-fold AUROC 0.991 |

3-class (NC / SLE / SLE+LN) fine-tuning collapsed to predicting the
majority class (NC) in all five folds (accuracy 0.4545 = 25/55). It is
reported as exploratory only and was not pursued further.

### Findings

* **The lupus signal is present in the model's input.** A linear model on
  the same 5,000 probes reaches ~0.97 AUROC, and in-fold selection from all
  485,577 probes adds nothing. Probe cg06872964 alone separates the groups
  (mean beta 0.49 in NC vs. 0.19 in SLE and 0.23 in SLE+LN; AUROC 0.009 on
  raw beta, i.e. 0.991 with the direction flipped), in the direction
  predicted by interferon-pathway hypomethylation.
* **The transformer does not extract it.** The zero-shot CLS embedding
  discards the signal (Stage 1), and fine-tuning recovers only a weak part
  of it (~0.68). The transformer scored below ridge on every one of the 10
  fold comparisons (5 folds x 2 configs).
* **The diagnostic points to under-fitting, not memorization.** Training
  loss stayed near ln 2 (0.693, the loss of a guessing model) and training
  AUROC after 30 epochs ranged only from 0.70 to 0.89.
* **Stage 2a's score was uninformative.** Early stopping on ~9 validation
  samples frequently selected epoch 1-2, i.e. a nearly untrained model, and
  inner-validation and outer-test scores did not track each other.
* **Do not read 0.68 vs. 0.587 as an improvement.** With ~11-sample test
  folds, the null SD of a single fold's AUROC is ~0.18 (~0.08 for the
  five-fold mean); the difference is within noise, and configs A and B are
  indistinguishable.

**Untested hypothesis for the mechanism:** the lupus signal is concentrated
in a few probes among 5,001 tokens; the CLS token's attention starts near
uniform, diluting any single probe's contribution, and the Project 6
weights were shaped to extract tissue-identity features. This is consistent
with the results but was not directly tested.

### Confound Checks and Remaining Limitations

* **Sex:** NC 20F/5M, SLE-combined 26F/4M. Too balanced to explain the effect.
* **Age:** mean 32.2 (NC) vs. 29.1 (SLE-combined) years, SD 8-9, overlapping
  ranges; 1 of 25 NC and 3 of 30 SLE samples are under 18. Too small a gap
  to explain the effect.
* **Not checkable with available metadata:** PBMC cell-type composition
  (SLE shifts neutrophil/lymphocyte proportions, which alone can change bulk
  methylation) and batch (the unexplained 6-sample cluster from Stage 1 may
  be a batch effect). The separation should not be attributed purely to
  within-cell interferon biology until this is addressed; Absher et al. 2013
  gene lists are the planned external check in Stage 3.
* **n = 55** with ~11-sample test folds: all AUROCs are noisy and the
  between-fold SDs are wide.
* **Number of transformer configurations tried: 3** (Stage 2a, 2c-A, 2c-B),
  all on the same folds. No further tuning was done, to avoid best-of-N
  selection on 55 samples.
* **TODO before publishing:** verify the cg06872964 / IFI44L literature
  citation. The probe ID was recalled from memory, not from a checked source.

### Revised Stage 1 Interpretation

The Stage 1 README section attributed the zero-shot negative to the
cancer-trained model not learning a representation that separates immune
activation states, and implied the lupus signal might not be capturable
from this input. Stage 2b shows that reading was too strong. The signal is
present in the raw 5,000-probe input; what the Stage 1 result supports is
that **the Project 6 CLS embedding does not preserve it**. Replace the
"Interpretation" paragraph in Stage 1 with that statement.

### Files

* `scripts/05_finetune_transformer_cv_colab.py` and `data/stage2_finetune_cv_results.json`
* `scripts/06_linear_baseline_cv.py` and `data/stage2b_linear_baseline_results.json`
* `scripts/06b_transformer_diagnostic_colab.py` and `data/stage2c_transformer_diagnostic_results.json`

### What's Next

Stage 3: attribution on the linear model, which has real signal and
CpG-level coefficients. The 460 probes selected by the in-fold t-test in all
five folds are a discovery set (any performance figure computed on them
would be circular). Map probes to genes, then to the IFN-alpha/beta network,
and compute betweenness centrality.
