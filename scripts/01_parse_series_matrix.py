"""
Project 8, Stage 1 — Parse GSE82218 Series Matrix File.

Reads the raw Series Matrix File and produces two clean, aligned outputs:
  - sample_metadata.tsv   (one row per sample: gsm_id, group, age, sex)
  - beta_matrix.parquet   (rows = samples [gsm_id as index], columns = CpG probes)

Per teaching guide Rule #12: metadata and the beta-value table are parsed
from two separate blocks of the same file. We do NOT trust that they're
in the same order just because they look that way — we verify explicitly
before saving anything.

Run from the project's scripts/ folder:
    python -u 01_parse_series_matrix.py
"""

import re
import pandas as pd

MATRIX_FILE = "../data/GSE82218_series_matrix.txt"


def parse_metadata(lines):
    """Pull the three metadata rows we need out of the header block.
    Each of these lines is tab-separated: the row label, then one
    quoted value per sample, in GSM column order."""
    meta = {}
    for line in lines:
        if line.startswith("!Sample_title"):
            meta["title"] = [x.strip('"') for x in line.strip().split("\t")[1:]]
        elif line.startswith("!Sample_geo_accession"):
            meta["gsm_id"] = [x.strip('"') for x in line.strip().split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            # This label repeats 3x (tissue, Sex, age) - keep the raw
            # values in the order encountered, we'll assign names below
            # by inspecting the content itself, not just line order,
            # since relying on line order alone is exactly the kind of
            # unverified assumption Rule #12 warns about.
            values = [x.strip('"') for x in line.strip().split("\t")[1:]]
            if values[0].startswith("tissue:"):
                meta["tissue"] = values
            elif values[0].startswith("Sex:"):
                meta["sex"] = [v.replace("Sex: ", "") for v in values]
            elif values[0].startswith("age:"):
                meta["age_years"] = [
                    int(re.search(r"(\d+)y", v).group(1)) for v in values
                ]
    return meta


def parse_group(title):
    """Extract the group label from the bracketed tag in Sample_title,
    e.g. 'normal control-01 [NC01]' -> 'NC', 'SLEpatient with lupus
    nephritis-01 [SLE+LN01]' -> 'SLE+LN'. We strip trailing digits from
    the bracket tag rather than matching on the free-text prefix, since
    the free text has inconsistent spacing ('SLEpatient' vs 'SLE patient')
    but the bracket tags are clean and consistent."""
    bracket = re.search(r"\[([A-Za-z+]+)\d+\]", title)
    if not bracket:
        raise ValueError(f"Could not parse group from title: {title}")
    return bracket.group(1)


def main():
    with open(MATRIX_FILE, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    header_lines = [l for l in raw.split("\n") if l.startswith("!Sample")]
    meta = parse_metadata(header_lines)

    metadata_df = pd.DataFrame({
        "gsm_id": meta["gsm_id"],
        "title": meta["title"],
        "age_years": meta["age_years"],
        "sex": meta["sex"],
    })
    metadata_df["group"] = metadata_df["title"].apply(parse_group)

    print("Parsed metadata shape:", metadata_df.shape)
    print("\nGroup counts:")
    print(metadata_df["group"].value_counts())
    print("\nFirst few rows:")
    print(metadata_df.head())

    # --- Now parse the actual beta-value data table ---
    table_start = raw.index("!series_matrix_table_begin")
    table_end = raw.index("!series_matrix_table_end") if "!series_matrix_table_end" in raw else len(raw)
    table_text = raw[table_start:table_end]

    # Skip the "!series_matrix_table_begin" line itself, keep the rest
    table_lines = table_text.split("\n")[1:]
    table_lines = [l for l in table_lines if l.strip()]

    header = [x.strip('"') for x in table_lines[0].split("\t")]
    data_gsm_ids = header[1:]  # first column is "ID_REF"

    print(f"\nBeta table: {len(table_lines) - 1} probes x {len(data_gsm_ids)} samples")

    # --- Rule #12 check: is the data table's sample order identical
    # to the metadata's sample order? Do NOT proceed if this fails. ---
    assert data_gsm_ids == metadata_df["gsm_id"].tolist(), (
        "Sample order mismatch between metadata block and data table! "
        "Do not proceed - alignment must be fixed before any further step."
    )
    print("\n✅ Alignment verified: data table sample order matches metadata order exactly.")

    metadata_df.to_csv("../data/sample_metadata.tsv", sep="\t", index=False)
    print("\nSaved ../data/sample_metadata.tsv")
    print("\n(Beta matrix parsing itself deferred to the next script, once this "
          "metadata step is confirmed correct - one step at a time.)")


if __name__ == "__main__":
    main()
