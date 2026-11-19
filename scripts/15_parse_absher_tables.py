"""
Project 8, Stage 3 -- Parse Absher 2013 Tables S3/S4 (DOCX), validate against the
abstract's counts, and describe overlap with our probes. DESCRIPTIVE; no tests here.
Cell format: 'control_mean/SLE_mean **'; ** = p<1e-8, * = FDR<1%, none = not significant.
Columns: CD4 = T-cells, CD19 = B-cells, CD14 = monocytes.
"""
import re

import docx
import numpy as np
import pandas as pd

S3, S4 = "../data/pgen.1003678.s007.docx", "../data/pgen.1003678.s008.docx"
ELIGIBLE = "../data/eligible_probes_nonan.txt"
DIRECTION = "../data/stable_probes_direction.tsv"
PAT = re.compile(r"^([0-9.]+)\s*/\s*([0-9.]+)\s*(\*{0,2})$")
CELLS = (("CD4", "T-cells", 1033), ("CD19", "B-cells", 166), ("CD14", "monocytes", 97))


def read_table(path):
    d = docx.Document(path)
    assert len(d.tables) == 1, f"expected 1 table in {path}, found {len(d.tables)}"
    rows = [[c.text.strip() for c in r.cells] for r in d.tables[0].rows]
    return pd.DataFrame(rows[1:], columns=rows[0])


def clean(df, name):
    assert list(df.columns) == ["CpG", "Chr", "Pos", "Genes", "IFN", "CD4", "CD19", "CD14"], list(df.columns)
    n_raw = len(df)
    df = df[df["CpG"] != "CpG"].copy()  # repeated header rows from page breaks
    print(f"{name}: {n_raw} body rows, {n_raw - len(df)} repeated header rows dropped")
    bad = ~df["CpG"].str.match(r"^(cg|ch|rs)\d+")
    print(f"  rows with unexpected CpG id: {int(bad.sum())} {df.loc[bad, 'CpG'].head(5).tolist()}")
    print(f"  duplicated CpG ids: {int(df['CpG'].duplicated().sum())}")
    print(f"  IFN column values: {df['IFN'].value_counts(dropna=False).to_dict()}")
    for c, label, _ in CELLS:
        m = df[c].map(lambda s: PAT.match(s))
        fail = m.isna()
        print(f"  {c} ({label}): cells not matching 'ctrl/SLE [*]': {int(fail.sum())} "
              f"{df.loc[fail, c].head(3).tolist()}")
        df[f"{c}_ctrl"] = m.map(lambda x: float(x.group(1)) if x else np.nan)
        df[f"{c}_sle"] = m.map(lambda x: float(x.group(2)) if x else np.nan)
        df[f"{c}_tier"] = m.map(lambda x: x.group(3) if x else None)
        df[f"{c}_delta"] = df[f"{c}_sle"] - df[f"{c}_ctrl"]
    return df


s3 = clean(read_table(S3), "S3")
s4 = clean(read_table(S4), "S4")
print("\nS4 CpGs all contained in S3:", set(s4["CpG"]) <= set(s3["CpG"]))
print("distinct 'Genes' strings in S4:", s4["Genes"].nunique(), "| in S3:", s3["Genes"].nunique())
print("multi-gene examples (check the delimiter):",
      s3.loc[s3["Genes"].str.contains(r"[;,/ ]"), "Genes"].head(5).tolist())

print("\n=== Validation against the abstract, and direction of the significant CpGs (S3) ===")
ifn = s3["IFN"].str.upper().eq("IFN")
for c, label, expected in CELLS:
    sig = s3[f"{c}_tier"] == "**"
    print(f"{c} ({label}): '**' rows = {int(sig.sum())} (abstract reports {expected}) | "
          f"'*' rows = {int((s3[f'{c}_tier'] == '*').sum())}")
    for flag, lab in ((True, "IFN-near"), (False, "other")):
        m = sig & (ifn == flag)
        d = s3.loc[m, f"{c}_delta"]
        print(f"    {lab:9s} n={int(m.sum()):4d} | SLE lower: {int((d < 0).sum()):4d} | "
              f"SLE higher: {int((d > 0).sum()):4d} | median delta {d.median():+.3f}")

print("\n=== Descriptive overlap with our probes ===")
eligible = {l.strip() for l in open(ELIGIBLE)}
dr = pd.read_csv(DIRECTION, sep="\t")
ids = set(s3["CpG"])
print(f"S3 CpGs: {len(ids)} | in our eligible (no-NaN) pool: {len(ids & eligible)} "
      f"({100 * len(ids & eligible) / len(ids):.0f}%)")
for lab, sub in (("all 460 stable", dr), ("hypo 116", dr[dr["direction"] == "hypo_in_SLE"]),
                 ("hyper 344", dr[dr["direction"] == "hyper_in_SLE"])):
    print(f"  overlap with {lab}: {len(set(sub['probeID']) & ids)}")
ov = s3.merge(dr[["probeID", "genesUniq", "delta_beta", "direction"]],
              left_on="CpG", right_on="probeID")[
    ["CpG", "Genes", "genesUniq", "delta_beta", "direction", "CD4_tier", "CD4_delta"]]
print(ov.round(3).sort_values("delta_beta").head(60).to_string(index=False))

s3.to_csv("../data/absher_s3_clean.tsv", sep="\t", index=False)
s4.to_csv("../data/absher_s4_clean.tsv", sep="\t", index=False)
print("\nsaved ../data/absher_s3_clean.tsv and ../data/absher_s4_clean.tsv")
