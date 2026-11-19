"""
Project 8, Stage 1 (continued) — Impute missing beta values.

Decision, made explicitly (not defaulted into): 192 NaN values (0.07% of
the 55x5000 matrix) were found scattered across 144 probes and 54/55
samples, with missingness spread proportionally across all three groups
(25/25 NC, 15/15 SLE, 14/15 SLE+LN affected) - i.e. sparse, universal,
non-group-correlated technical noise (individual probes occasionally
failing QC on individual array chips), not a disease-associated pattern.

Given this pattern, per-probe mean imputation (fill each missing value
with that probe's average beta value across the samples that DO have
data) is standard, low-risk practice - it won't introduce group-specific
bias given how evenly the missingness is spread, and preserves all 5000
probes rather than dropping 144 of them to fix a 0.07% problem.

Operates directly on the already-parsed gse82218_zeroshot_input.parquet
rather than re-running the full 485,577-row source parse.
"""

import pandas as pd

INPUT_FILE = "../data/gse82218_zeroshot_input.parquet"
OUTPUT_FILE = "../data/gse82218_zeroshot_input_imputed.parquet"


def main():
    df = pd.read_parquet(INPUT_FILE)
    metadata_cols = ["group", "age_years", "sex"]
    cpg_cols = [c for c in df.columns if c not in metadata_cols]

    n_nan_before = df[cpg_cols].isna().values.sum()
    print(f"NaN values before imputation: {n_nan_before}")

    # Per-probe mean, computed only from samples that HAVE a value for
    # that probe (pandas .mean() skips NaN by default) - then fillna
    # broadcasts that mean into the gaps for that specific column.
    probe_means = df[cpg_cols].mean(axis=0, skipna=True)
    df[cpg_cols] = df[cpg_cols].fillna(probe_means)

    n_nan_after = df[cpg_cols].isna().values.sum()
    print(f"NaN values after imputation: {n_nan_after}")
    assert n_nan_after == 0, "Imputation did not resolve all NaN values - investigate before proceeding"

    df.to_parquet(OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
