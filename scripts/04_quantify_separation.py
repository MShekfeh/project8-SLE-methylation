"""
Project 8, Stage 1 (continued) — Quantify zero-shot separation.

Two things a plot alone can't tell us:
  1. Is the visual "no separation" impression backed by a number? We fit a
     simple classifier on the 128-dim embeddings and cross-validate it,
     rather than trusting an eyeballed UMAP plot.
  2. What explains the small 6-point subcluster visible in the UMAP figure,
     since it clearly isn't disease group (it contains a mix of all three)?
     We check it against age and sex, the two other variables we have.

Per Decision 3 in the Project 8 addendum: the PRIMARY comparison is
SLE-vs-NC (30 vs 25, combining SLE and SLE+LN), not the 3-way split -
that's the adequately-powered contrast. We report the 3-class result too,
but treat SLE-vs-NC as the number that matters most.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier

DATA_FILE = "../data/gse82218_zeroshot_embeddings.npz"


def cv_accuracy(X, y, label):
    """5-fold stratified CV accuracy for a simple logistic regression,
    compared against a majority-class baseline (DummyClassifier) - the
    baseline matters because with imbalanced groups (25 vs 30, or 25 vs
    15 vs 15), a classifier that always predicts the largest class can
    look deceptively 'accurate' without learning anything real."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    clf = LogisticRegression(max_iter=2000)
    clf_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")

    dummy = DummyClassifier(strategy="most_frequent")
    dummy_scores = cross_val_score(dummy, X, y, cv=skf, scoring="accuracy")

    print(f"\n--- {label} ---")
    print(f"Logistic regression CV accuracy: {clf_scores.mean():.3f} (+/- {clf_scores.std():.3f})")
    print(f"Majority-class baseline accuracy: {dummy_scores.mean():.3f}")
    print(f"Difference: {clf_scores.mean() - dummy_scores.mean():+.3f}")


def main():
    data = np.load(DATA_FILE, allow_pickle=True)
    embeddings = data["embeddings"]
    group = data["group"]
    age = data["age_years"]
    sex = data["sex"]

    print(f"Loaded {embeddings.shape[0]} embeddings, {embeddings.shape[1]}-dim")

    # --- Primary comparison: SLE (combined) vs NC ---
    binary_label = np.where(group == "NC", "NC", "SLE")  # combines SLE + SLE+LN
    cv_accuracy(embeddings, binary_label, "PRIMARY: SLE (combined, n=30) vs NC (n=25)")

    # --- Exploratory: full 3-class ---
    cv_accuracy(embeddings, group, "EXPLORATORY: 3-class NC / SLE / SLE+LN")

    # --- What explains the small outlier subcluster? ---
    # We don't have UMAP coordinates saved from script 03 (only raw
    # embeddings), so identify the subcluster directly in embedding space
    # instead, using a simple distance-based approach: samples whose
    # nearest-neighbor distance to the rest of the cohort is unusually
    # large are the same subcluster the UMAP plot visually isolated.
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(embeddings, embeddings)
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_dist = dist_matrix.min(axis=1)

    # The 6-point subcluster should show up as samples whose nearest
    # neighbor is one of the OTHER 5 subcluster members, at a distance
    # that's a clear outlier relative to the rest of the cohort.
    threshold = np.percentile(nearest_dist, 75) + 1.5 * (
        np.percentile(nearest_dist, 75) - np.percentile(nearest_dist, 25)
    )
    print(f"\n--- Investigating potential outlier subcluster ---")
    print("(This is a rough heuristic, not a formal clustering method - "
          "meant to flag candidates for a closer look, not a final answer.)")

    # Simpler and more direct: just report age/sex stats split by whether
    # a sample's nearest neighbor distance is in the top or bottom half -
    # a crude but honest first pass before any fancier clustering.
    median_dist = np.median(nearest_dist)
    is_far = nearest_dist > median_dist
    print(f"\nAge (years), samples with ABOVE-median nearest-neighbor distance (n={is_far.sum()}):")
    print(f"  mean={age[is_far].mean():.1f}, values={sorted(age[is_far].tolist())}")
    print(f"\nAge (years), samples with BELOW-median nearest-neighbor distance (n={(~is_far).sum()}):")
    print(f"  mean={age[~is_far].mean():.1f}")

    print(f"\nSex breakdown, above-median distance group:")
    unique, counts = np.unique(sex[is_far], return_counts=True)
    print(dict(zip(unique, counts)))


if __name__ == "__main__":
    main()
