"""
Project 8, Stage 2 -- Full end-to-end fine-tuning of the Project 6
methylation transformer on GSE82218 (lupus vs. healthy). COLAB (GPU) VERSION.

Why Colab: training needs every layer's attention activations kept in memory
for backprop (5,001 tokens x 4 heads x batch x 4 layers). That exceeded
compbio's ~6.2GB budget and the process was killed. Same constraint as
Project 6 Stage 2.

Design:
- Full fine-tune: all parameters trainable, but backbone and the new
  classification head use different learning rates.
- Nested CV: outer StratifiedKFold gives the reported score; an inner
  validation split carved ONLY from each outer-train fold drives early
  stopping, so the outer test fold is never used for model selection.
- Primary: SLE-combined (n=30) vs NC (n=25), binary AUROC.
  Exploratory: 3-class NC/SLE/SLE+LN, accuracy.
- Mixed precision (fp16 autocast + GradScaler) to cut memory and time.

Setup in Colab (run these in cells BEFORE this script):
    from google.colab import drive
    drive.mount('/content/drive')
    # Runtime > Change runtime type > GPU
"""

import copy
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

DRIVE_DIR = "/content/drive/MyDrive/genomics-ml/project8-SLE-methylation/"
sys.path.insert(0, DRIVE_DIR)  # so `import methylation_model` finds the copy in Drive
from methylation_model import MethylationTransformer

DATA_PATH = DRIVE_DIR + "gse82218_zeroshot_input_imputed.parquet"
CHECKPOINT_PATH = DRIVE_DIR + "best_transformer.pt"
PROBE_LIST_PATH = DRIVE_DIR + "project6_cpg_probe_list.txt"
RESULTS_PATH = DRIVE_DIR + "stage2_finetune_cv_results.json"

DEVICE = torch.device("cuda")
BATCH_SIZE = 4
MAX_EPOCHS = 40
PATIENCE = 5
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
INNER_VAL_FRACTION = 0.20
RANDOM_STATE = 42


class BetaValueDataset(Dataset):
    """Wraps a beta-value matrix and integer labels for DataLoader use."""

    def __init__(self, beta_values: np.ndarray, labels: np.ndarray):
        self.beta_values = torch.tensor(beta_values, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.beta_values[idx], self.labels[idx]


def build_model_for_finetuning(n_classes: int, n_probes: int) -> nn.Module:
    """Load the Project 6 checkpoint, then swap in a fresh classifier head
    sized for this task. Backbone keeps its Project 6 weights."""
    model = MethylationTransformer(n_classes=5, n_probes=n_probes)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)  # fails loudly on any mismatch
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    return model.to(DEVICE)


def get_param_groups(model: nn.Module):
    """Backbone moves slowly; the new, randomly initialised head moves fast."""
    backbone = [p for n, p in model.named_parameters() if "classifier" not in n]
    head = list(model.classifier.parameters())
    return [
        {"params": backbone, "lr": BACKBONE_LR},
        {"params": head, "lr": HEAD_LR},
    ]


def predict_probs(model, loader):
    """Return (probabilities, labels) as numpy arrays for a DataLoader."""
    model.eval()
    probs, labels_out = [], []
    with torch.no_grad():
        for beta, labels in loader:
            with autocast("cuda"):
                logits = model(beta.to(DEVICE))
            probs.append(torch.softmax(logits.float(), dim=1).cpu())
            labels_out.append(labels)
    return torch.cat(probs).numpy(), torch.cat(labels_out).numpy()


def score(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """AUROC for binary, accuracy for multi-class."""
    if n_classes == 2:
        return roc_auc_score(labels, probs[:, 1])
    return float((probs.argmax(axis=1) == labels).mean())


def train_one_fold(model, train_ds, val_ds, n_classes):
    """Train with early stopping on inner-validation score; restore best epoch."""
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(get_param_groups(model), weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda")  # rescales loss so fp16 gradients don't underflow
    criterion = nn.CrossEntropyLoss()

    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    stale = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for beta, labels in train_loader:
            beta, labels = beta.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast("cuda"):
                loss = criterion(model(beta), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        probs, labels_np = predict_probs(model, val_loader)
        val_score = score(probs, labels_np, n_classes)

        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, best_score, epoch + 1


def run_nested_cv(beta_matrix, labels, n_classes, label_name):
    metric = "AUROC" if n_classes == 2 else "accuracy"
    print(f"\n=== {label_name} ({n_classes}-class, metric={metric}) ===", flush=True)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_scores = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(beta_matrix, labels)):
        t0 = time.time()
        # Inner split comes ONLY from the outer-train indices.
        inner_train_idx, inner_val_idx = train_test_split(
            train_idx, test_size=INNER_VAL_FRACTION,
            stratify=labels[train_idx], random_state=RANDOM_STATE,
        )
        train_ds = BetaValueDataset(beta_matrix[inner_train_idx], labels[inner_train_idx])
        val_ds = BetaValueDataset(beta_matrix[inner_val_idx], labels[inner_val_idx])
        test_ds = BetaValueDataset(beta_matrix[test_idx], labels[test_idx])

        torch.manual_seed(RANDOM_STATE + fold_idx)  # reproducible head init per fold
        model = build_model_for_finetuning(n_classes, beta_matrix.shape[1])
        model, inner_score, n_epochs = train_one_fold(model, train_ds, val_ds, n_classes)

        probs, labels_np = predict_probs(
            model, DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False))
        test_score = score(probs, labels_np, n_classes)
        fold_scores.append(test_score)

        print(f"Fold {fold_idx}: {n_epochs} epochs, inner val {metric}={inner_score:.3f} "
              f"-> OUTER TEST {metric}={test_score:.3f}  ({time.time() - t0:.0f}s)", flush=True)

        del model
        torch.cuda.empty_cache()

    arr = np.array(fold_scores)
    print(f"{label_name}: outer test {metric} = {arr.mean():.3f} (+/- {arr.std():.3f})", flush=True)
    return {"metric": metric, "fold_scores": fold_scores,
            "mean": float(arr.mean()), "std": float(arr.std())}


def main():
    df = pd.read_parquet(DATA_PATH)
    with open(PROBE_LIST_PATH) as f:
        probe_cols = [line.strip() for line in f]
    beta_matrix = df[probe_cols].values.astype(np.float32)

    # Sanity checks before spending GPU time (NaN check = standing rule).
    assert beta_matrix.shape == (55, 5000), f"unexpected shape {beta_matrix.shape}"
    assert not np.isnan(beta_matrix).any(), "NaNs present in input matrix"
    print("Group counts:", df["group"].value_counts().to_dict(), flush=True)

    results = {}

    # Primary: SLE-combined (SLE + SLE+LN) vs NC. All 55 samples are used.
    primary_labels = df["group"].map({"NC": 0, "SLE": 1, "SLE+LN": 1}).values
    results["primary_sle_vs_nc"] = run_nested_cv(
        beta_matrix, primary_labels, 2, "Primary: SLE-combined vs NC")

    # Exploratory: 3-class.
    exploratory_labels = df["group"].map({"NC": 0, "SLE": 1, "SLE+LN": 2}).values
    results["exploratory_3class"] = run_nested_cv(
        beta_matrix, exploratory_labels, 3, "Exploratory: NC/SLE/SLE+LN")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
