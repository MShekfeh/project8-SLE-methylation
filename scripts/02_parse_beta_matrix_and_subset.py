"""
Project 8, Stage 1 (continued) — Parse full beta matrix, subset to
Project 6's 5,000-probe vocabulary, save zero-shot-ready input.

Depends on:
  ../data/GSE82218_series_matrix.txt   (raw file, from script 01)
  ../data/sample_metadata.tsv          (output of script 01)
  ../data/project6_cpg_probe_list.txt  (the exact 5000 probe IDs the
                                         Project 6 transformer was trained
                                         on, extracted from
                                         beta_matrix_with_ids.parquet)

Produces:
  ../data/gse82218_beta_matrix_full.parquet   (55 samples x 485,577 probes,
                                                 full matrix, kept for
                                                 reference/future use)
  ../data/gse82218_zeroshot_input.parquet     (55 samples x [overlap] probes,
                                                 in Project 6's exact column
                                                 order, with group/age/sex
                                                 attached - this is what
                                                 actually gets fed to the
                                                 model)

Why we save the full matrix too, not just the subset: if a probe is
missing from the 5000-list overlap, we may want to investigate why
(different manifest version, probe retired, etc.) without re-downloading
and re-parsing the 107MB source file again.
"""

import io
import pandas as pd

MATRIX_FILE = "../data/GSE82218_series_matrix.txt"
PROBE_LIST_FILE = "../data/project6_cpg_probe_list.txt"
METADATA_FILE = "../data/sample_metadata.tsv"


def main():
    with open(MATRIX_FILE, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    table_start = raw.index("!series_matrix_table_begin")
    table_end = raw.index("!series_matrix_table_end") if "!series_matrix_table_end" in raw else len(raw)
    table_text = raw[table_start:table_end]
    table_text = "\n".join(table_text.split("\n")[1:])  # drop the "!series_matrix_table_begin" line

    # pd.read_csv on the in-memory text is far faster and more robust
    # than manual line-splitting for a table this size (485,577 rows).
    # index_col=0 makes the probe ID (ID_REF) the row index directly.
    beta_full = pd.read_csv(io.StringIO(table_text), sep="\t", index_col=0, quotechar='"')
    print(f"Raw parsed table: {beta_full.shape[0]} probes x {beta_full.shape[1]} samples")

    # Transpose so rows = samples, columns = probes - matches the
    # orientation of Project 6's beta_matrix_with_ids.parquet, and is
    # the orientation the transformer expects (one row per sample).
    beta_full = beta_full.T
    beta_full.index.name = "gsm_id"
    print(f"After transpose: {beta_full.shape[0]} samples x {beta_full.shape[1]} probes")

    beta_full.to_parquet("../data/gse82218_beta_matrix_full.parquet")
    print("Saved full matrix to ../data/gse82218_beta_matrix_full.parquet")

    # --- Subset to Project 6's exact 5000-probe vocabulary ---
    with open(PROBE_LIST_FILE) as f:
        project6_probes = [line.strip() for line in f if line.strip()]
    print(f"\nProject 6 probe list: {len(project6_probes)} probes")

    available = [p for p in project6_probes if p in beta_full.columns]
    missing = [p for p in project6_probes if p not in beta_full.columns]
    print(f"Found in GSE82218: {len(available)} / {len(project6_probes)}")
    if missing:
        print(f"MISSING probes ({len(missing)}): {missing[:20]}"
              f"{' ... (truncated)' if len(missing) > 20 else ''}")
    else:
        print("No missing probes - full overlap.")

    # Subset AND reorder columns to match Project 6's exact column order.
    # Order matters here: the transformer's input layer expects probes
    # in a fixed, specific order, the same order it was trained on -
    # a plain set-intersection subset without reordering would silently
    # feed the model mismatched columns.
    zeroshot_input = beta_full[available].copy()

    # Attach group/age/sex metadata, joining on gsm_id
    metadata_df = pd.read_csv(METADATA_FILE, sep="\t").set_index("gsm_id")
    zeroshot_input = zeroshot_input.join(metadata_df[["group", "age_years", "sex"]])

    print(f"\nFinal zero-shot input shape: {zeroshot_input.shape}")
    print("\nGroup counts in final matrix:")
    print(zeroshot_input["group"].value_counts())

    zeroshot_input.to_parquet("../data/gse82218_zeroshot_input.parquet")
    print("\nSaved ../data/gse82218_zeroshot_input.parquet")


if __name__ == "__main__":
    main()
