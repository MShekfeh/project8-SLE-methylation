"""
Project 8, Stage 1 — Zero-shot embedding of GSE82218 through the
Project 6 methylation transformer (never trained on lupus/autoimmune data).

Scientific question: does a model trained exclusively on cancer epigenetic
dysregulation (5 TCGA classes: BRCA/PRAD/LUAD/COAD/Normal) already separate
lupus from healthy PBMC methylation, with zero lupus-specific training?

Depends on:
  ../data/gse82218_zeroshot_input.parquet  (55 samples x 5000 probes + metadata)
  methylation_model.py                      (MethylationTransformer class)
  ../data/best_transformer.pt               (trained Project 6 weights)

Produces:
  ../data/gse82218_zeroshot_embeddings.npz  (raw 128-dim CLS embeddings,
                                               for any downstream analysis)
  ../figures/gse82218_zeroshot_umap.png     (UMAP colored by NC/SLE/SLE+LN)
"""

import numpy as np
import pandas as pd
import torch
import umap
import matplotlib.pyplot as plt

from methylation_model import MethylationTransformer

INPUT_FILE = "../data/gse82218_zeroshot_input_imputed.parquet"
CHECKPOINT_FILE = "../data/best_transformer.pt"


def main():
    df = pd.read_parquet(INPUT_FILE)

    # The parquet has 5000 CpG columns plus group/age_years/sex. Split
    # those apart explicitly rather than assuming column order/count.
    metadata_cols = ["group", "age_years", "sex"]
    cpg_cols = [c for c in df.columns if c not in metadata_cols]
    print(f"Loaded {df.shape[0]} samples, {len(cpg_cols)} CpG columns")
    assert len(cpg_cols) == 5000, f"Expected 5000 CpG columns, got {len(cpg_cols)}"

    beta_values = torch.tensor(df[cpg_cols].values, dtype=torch.float32)

    # --- Load model, verified-matching checkpoint (confirmed strict=True
    # load with zero missing/unexpected keys before this script was written) ---
    model = MethylationTransformer(n_probes=5000, d_model=128, n_heads=4,
                                     n_layers=4, n_classes=5, dropout=0.1)
    state_dict = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    beta_values = beta_values.to(device)
    print(f"Running on: {device}")

    # IMPORTANT: even though we only have 55 samples, we must batch this.
    # Self-attention memory cost is O(batch x heads x seq_len^2), and our
    # sequence length is 5001 (5000 CpG tokens + 1 CLS) - the SEQUENCE
    # LENGTH is what's expensive here, not the sample count. A single
    # forward pass of all 55 samples tries to allocate ~22GB on CPU,
    # which is exactly the constraint methylation_model.py's own
    # docstring warns about ("full self-attention over 5,000 tokens
    # needs more memory than compbio's ~6.2GB budget"). Batch size 4
    # keeps peak memory to roughly 4 x 400MB = ~1.6GB, safely under budget.
    BATCH_SIZE = 4
    n_samples = beta_values.shape[0]
    embeddings_list, logits_list = [], []

    with torch.no_grad():
        for start in range(0, n_samples, BATCH_SIZE):
            batch = beta_values[start:start + BATCH_SIZE]
            emb_batch = model(batch, return_embedding=True).cpu().numpy()
            logit_batch = model(batch, return_embedding=False).cpu().numpy()
            embeddings_list.append(emb_batch)
            logits_list.append(logit_batch)
            print(f"  processed samples {start}-{min(start + BATCH_SIZE, n_samples)}/{n_samples}")

    embeddings = np.concatenate(embeddings_list, axis=0)
    logits = np.concatenate(logits_list, axis=0)

    print(f"Embeddings shape: {embeddings.shape}")

    # Save raw embeddings + labels for any downstream analysis (e.g.
    # a formal separation test, distance-to-centroid metrics, etc.)
    np.savez(
        "../data/gse82218_zeroshot_embeddings.npz",
        embeddings=embeddings,
        group=df["group"].values,
        age_years=df["age_years"].values,
        sex=df["sex"].values,
        tcga_logits=logits,
    )
    print("Saved ../data/gse82218_zeroshot_embeddings.npz")

    # --- UMAP visualization ---
    # n_neighbors default (15) is too large relative to our smallest group
    # (15 samples) - reduce it so local structure isn't washed out by
    # forcing each point to consider more neighbors than its own group has.
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.1)
    umap_coords = reducer.fit_transform(embeddings)

    plt.figure(figsize=(7, 6))
    colors = {"NC": "#4C72B0", "SLE": "#DD8452", "SLE+LN": "#C44E52"}
    for group_name, color in colors.items():
        mask = df["group"].values == group_name
        plt.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                    s=60, alpha=0.8, label=f"{group_name} (n={mask.sum()})",
                    color=color, edgecolors="white", linewidths=0.5)
    plt.legend()
    plt.title("Zero-shot UMAP: GSE82218 through Project 6's cancer-trained\nmethylation transformer (never seen lupus)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig("../figures/gse82218_zeroshot_umap.png", dpi=150)
    print("Saved ../figures/gse82218_zeroshot_umap.png")


if __name__ == "__main__":
    main()
