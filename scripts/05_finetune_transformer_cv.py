# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:32:56 2026

@author: marwa
"""

"""
Project 8, Stage 2 -- Full end-to-end fine-tuning of the Project 6
methylation transformer on GSE82218 (lupus vs. healthy).

Design (see session discussion):
- Full fine-tune: all parameters trainable, but backbone and new
  classification head use different learning rates (backbone moves
  gently, head moves fast since it starts randomly initialized).
- Nested cross-validation: outer StratifiedKFold gives the honest,
  reported accuracy; an inner validation split (carved from each
  outer-train fold only) is used purely for early stopping, so the
  outer test fold is never touched during model selection.
- Primary comparison: SLE-combined (n=30) vs NC (n=25), binary.
  Exploratory: 3-class NC/SLE/SLE+LN, run separately, reported as
  exploratory only (per kickoff Decision 3).
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from methylation_model import MethylationTransformer

DATA_PATH = "../data/gse82218_zeroshot_input_imputed.parquet"
CHECKPOINT_PATH = "../data/best_transformer.pt"
PROBE_LIST_PATH = "../data/project6_cpg_probe_list.txt"

DEVICE = torch.device("cpu")
BATCH_SIZE = 2          # small: full self-attention over 5001 tokens is
                         # memory-heavy even at batch_size=4 for inference;
                         # training (grads + activations) needs headroom
MAX_EPOCHS = 40
PATIENCE = 5             # small on purpose -- little data to justify more
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
INNER_VAL_FRACTION = 0.20
RANDOM_STATE = 42


class BetaValueDataset(Dataset):
    """Wraps a beta-value matrix + integer labels for DataLoader use."""
    def __init__(self, beta_values: np.ndarray, labels: np.ndarray):
        self.beta_values = torch.tensor(beta_values, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.beta_values[idx], self.labels[idx]


def build_model_for_finetuning(n_classes: int, n_probes: int) -> nn.Module:
    """
    Loads the Project 6 checkpoint, then REPLACES the final classifier
    layer with a freshly-initialized one sized for this task's n_classes
    (Project 6's head was sized for 5 cancer classes; ours is 2 for the
    primary comparison, or 3 for the exploratory one).

    Everything else (the backbone: embedding + attention layers) keeps
    its Project 6-trained weights as the starting point -- that's the
    actual "transfer" in transfer learning.
    """
    model = MethylationTransformer(n_classes=5, n_probes=n_probes)  # original head shape
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)  # confirms checkpoint matches architecture exactly

    # Find the input size of the old classifier head so the new one lines up
    old_head_in_features = model.classifier.in_features
    model.classifier = nn.Linear(old_head_in_features, n_classes)  # fresh, random init

    return model.to(DEVICE)


def get_param_groups(model: nn.Module):
    """
    Two learning rates: backbone parameters move slowly (they already
    encode something useful from Project 6), the new classifier head
    moves fast (it starts from random weights and has to catch up).
    """
    backbone_params = [p for name, p in model.named_parameters() if "classifier" not in name]
    head_params = list(model.classifier.parameters())
    return [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ]


def train_one_fold(model, train_ds, val_ds, n_classes):
    """
    Trains with early stopping on inner validation AUROC (binary) or
    accuracy (3-class, since multi-class AUROC needs one-vs-rest handling
    we skip for simplicity at this sample size). Returns the model state
    from the best epoch, not the last epoch.
    """
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(get_param_groups(model), weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for beta, labels in train_loader:
            beta, labels = beta.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(beta)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Inner validation
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for beta, labels in val_loader:
                beta = beta.to(DEVICE)
                logits = model(beta)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(labels)
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        if n_classes == 2:
            val_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            val_score = (all_probs.argmax(axis=1) == all_labels).mean()

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, best_val_score, epoch + 1


def run_nested_cv(beta_matrix: np.ndarray, labels: np.ndarray, n_classes: int, label_name: str):
    print(f"\n=== {label_name} ({n_classes}-class) ===")
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(beta_matrix, labels)):
        # Inner split: carve validation ONLY from the outer-train portion.
        # The outer test_idx is never touched until final scoring below.
        inner_train_idx, inner_val_idx = train_test_split(
            train_idx,
            test_size=INNER_VAL_FRACTION,
            stratify=labels[train_idx],
            random_state=RANDOM_STATE,
        )

        train_ds = BetaValueDataset(beta_matrix[inner_train_idx], labels[inner_train_idx])
        val_ds = BetaValueDataset(beta_matrix[inner_val_idx], labels[inner_val_idx])
        test_ds = BetaValueDataset(beta_matrix[test_idx], labels[test_idx])

        model = build_model_for_finetuning(n_classes=n_classes, n_probes=beta_matrix.shape[1])
        model, inner_val_score, n_epochs = train_one_fold(model, train_ds, val_ds, n_classes)

        # Final, honest evaluation on the untouched outer test fold
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for beta, lbl in test_loader:
                logits = model(beta.to(DEVICE))
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(lbl)
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        if n_classes == 2:
            test_score = roc_auc_score(all_labels, all_probs[:, 1])
            metric_name = "AUROC"
        else:
            test_score = (all_probs.argmax(axis=1) == all_labels).mean()
            metric_name = "accuracy"

        print(f"Fold {fold_idx}: stopped after {n_epochs} epochs "
              f"(inner val {metric_name}={inner_val_score:.3f}) "
              f"-> outer test {metric_name}={test_score:.3f}")
        fold_results.append(test_score)

    fold_results = np.array(fold_results)
    print(f"\n{label_name} outer test {metric_name}: "
          f"{fold_results.mean():.3f} (+/- {fold_results.std():.3f})")
    return fold_results


def main():
    df = pd.read_parquet(DATA_PATH)
    with open(PROBE_LIST_PATH) as f:
        probe_cols = [line.strip() for line in f]

    beta_matrix = df[probe_cols].values.astype(np.float32)

    # Primary comparison: SLE-combined vs NC
    primary_mask = df["group"].isin(["NC", "SLE", "SLE+LN"])
    primary_labels = df["group"].map({"NC": 0, "SLE": 1, "SLE+LN": 1}).values
    run_nested_cv(beta_matrix[primary_mask.values], primary_labels[primary_mask.values],
                  n_classes=2, label_name="Primary: SLE-combined vs NC")

    # Exploratory: 3-class
    exploratory_labels = df["group"].map({"NC": 0, "SLE": 1, "SLE+LN": 2}).values
    run_nested_cv(beta_matrix, exploratory_labels,
                  n_classes=3, label_name="Exploratory: NC/SLE/SLE+LN")


if __name__ == "__main__":
    main()

