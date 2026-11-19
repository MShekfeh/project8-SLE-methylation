"""Inspect the structure of an Absher 2013 supplementary DOCX table (read-only)."""
import sys

import docx

d = docx.Document(sys.argv[1])
print("tables:", len(d.tables), "| paragraphs:", len(d.paragraphs))
for i, t in enumerate(d.tables):
    print(f"table {i}: {len(t.rows)} rows x {len(t.columns)} cols")
    for r in t.rows[:5]:
        print("  ", [c.text.strip()[:28] for c in r.cells])
for p in d.paragraphs[:8]:
    if p.text.strip():
        print("para:", p.text.strip()[:150])
