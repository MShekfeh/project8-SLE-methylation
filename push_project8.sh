#!/usr/bin/env bash
# Run on compbio from the project root. Usage: bash push_project8.sh
set -euo pipefail
cd ~/projects/project8-SLE-methylation
pwd

# 1. Put the new files in place (copy README.md, requirements.txt, .gitignore, HANDOFF here first)
for f in README.md requirements.txt .gitignore; do test -f "$f" || { echo "MISSING $f"; exit 1; }; done

# 2. Results folder (README paths point to results/)
mkdir -p results figures
cp data/stage*_results.json results/ 2>/dev/null || true
cp data/stable_probes_annotated.tsv data/stable_probes_direction.tsv data/stage3_string_module_genes.tsv results/ 2>/dev/null || true
ls -lh results/

# 3. Clean download artifacts
find . -name "*:Zone.Identifier" -delete

# 4. Init + inspect BEFORE adding
git init -b main 2>/dev/null || true
git add .
echo "---- files staged ----"
git status --short
echo "---- largest staged files (bytes) ----"
git ls-files -z | xargs -0 du -b 2>/dev/null | sort -n | tail -5
echo "Review the lists above. Nothing over ~5 MB should appear. Press Enter to commit, Ctrl-C to abort."
read -r _

git commit -m "Project 8: SLE methylation (GSE82218), Stages 1-3

Zero-shot probing, fine-tuning vs linear baselines, and interferon
enrichment / STRING module / Absher replication analysis.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XZXp9S6komuv7VgEY65ZiQ"

echo "Now create the empty repo on GitHub (no README/license), then run:"
echo "  git remote add origin https://github.com/MShekfeh/project8-SLE-methylation.git"
echo "  git push -u origin main"
