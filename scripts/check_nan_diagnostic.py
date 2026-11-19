"""
Project 8 diagnostic — check gse82218_zeroshot_input.parquet for NaN values.
Run this before re-running 03_zeroshot_embed_and_umap.py.
"""

import pandas as pd

df = pd.read_parquet("../data/gse82218_zeroshot_input.parquet")
cpg_cols = [c for c in df.columns if c not in ["group", "age_years", "sex"]]

nan_mask = df[cpg_cols].isna()
n_nan_total = nan_mask.values.sum()
print(f"Total NaN values in the 5000-probe matrix: {n_nan_total}")

if n_nan_total > 0:
    samples_with_nan = nan_mask.any(axis=1)
    print(f"\nSamples with at least one NaN: {samples_with_nan.sum()} / {len(df)}")
    print(df.loc[samples_with_nan, "group"].value_counts())

    probes_with_nan = nan_mask.any(axis=0)
    print(f"\nProbes with at least one NaN: {probes_with_nan.sum()} / {len(cpg_cols)}")

    print("\nNaN count per affected sample:")
    print(nan_mask.loc[samples_with_nan].sum(axis=1))
else:
    print("No NaNs found in the CpG matrix - the issue must be elsewhere "
          "(e.g. inside the model's own computation, not the input data).")
