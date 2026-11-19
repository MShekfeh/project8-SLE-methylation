"""
Project 8, Stage 2c -- Transformer diagnostic retry (COLAB GPU).

Question isolated: can the Project 6 transformer, fully fine-tuned, fit and
generalize the SLE-vs-NC signal once optimization is not the bottleneck?
(Stage 2b showed ridge regression reaches ~0.97 AUROC on this exact input,
so the signal IS in the 5,000 probes.)

What changes vs. Stage 2 (05_finetune_transformer_cv_colab.py):
  1. NO early stopping and NO inner validation split. Stage 2's log showed
     the "best" epoch was often epoch 1 (selected on ~9 samples), i.e. the
     restored models were nearly untrained. With no model selection there is
     nothing to leak, so all outer-train samples are used for training.
  2. Fixed epoch count (30). We report ONLY the final-epoch outer-test AUROC.
     We never pick epochs or configs by looking at test scores.
  3. Training loss and training AUROC are logged, so we can see whether the
     model can even fit its own training data.
  4. Two configurations, declared in advance, both reported (so the number
     of things tried is visible):
        A: backbone LR 1e-5 (Stage 2's LR, but now with enough steps)
        B: backbone LR 1e-4
Same outer folds as Stage 2 and the ridge baseline (StratifiedKFold, 5
splits, seed 42), so scores are directly comparable.
Binary primary comparison only (SLE-combined vs NC).
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DRIVE_DIR = "/content/drive/MyDrive/genomics-ml/project8-SLE-methylation/"
sys.path.insert(0, DRIVE_DIR)
from methylation_model import MethylationTransformer

DATA_PATH = DRIVE_DIR + "gse82218_zeroshot_input_imputed.parquet"
CHECKPOINT_PATH = DRIVE_DIR + "best_transformer.pt"
PROBE_LIST_PATH = DRIVE_DIR + "project6_cpg_probe_list.txt"
RESULTS_PATH = DRIVE_DIR + "stage2c_transformer_diagnostic_results.json"

DEVICE = torch.device("cuda")
BATCH_SIZE = 4
EPOCHS = 30
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_STATE = 42
CONFIGS = [
    {"name": "A_backbone_lr_1e-5", "backbone_lr": 1e-5},
    {"name": "B_backbone_lr_1e-4", "backbone_lr": 1e-4},
]


class BetaValueDataset(Dataset):
    """Wraps a beta-value matrix and integer labels for DataLoader use."""

    def __init__(self, beta_values: np.ndarray, labels: np.ndarray):
        self.beta_values = torch.tensor(beta_values, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.beta_values[idx], self.labels[idx]


def build_model(n_probes: int) -> nn.Module:
    """Project 6 checkpoint (strict load) with a fresh 2-class head."""
    model = MethylationTransformer(n_classes=5, n_probes=n_probes)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"), strict=True)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model.to(DEVICE)


def predict_probs(model, ds):
    """Return (P(class 1), labels) as numpy arrays."""
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for beta, labels in loader:
            with autocast("cuda"):
                logits = model(beta.to(DEVICE))
            probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
            labs.append(labels)
    return torch.cat(probs).numpy(), torch.cat(labs).numpy()


def train_fixed_epochs(model, train_ds, backbone_lr):
    """Train for EPOCHS epochs; no model selection. Returns loss/AUROC history."""
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    backbone = [p for n, p in model.named_parameters() if "classifier" not in n]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": backbone_lr},
         {"params": model.classifier.parameters(), "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for beta, labels in loader:
            beta, labels = beta.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast("cuda"):
                loss = criterion(model(beta), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if epoch % 5 == 0 or epoch == 1:
            p, l = predict_probs(model, train_ds)
            entry["train_auroc"] = float(roc_auc_score(l, p))
        history.append(entry)
    return history


def main():
    df = pd.read_parquet(DATA_PATH)
    probes = [l.strip() for l in open(PROBE_LIST_PATH)]
    X = df[probes].values.astype(np.float32)
    assert X.shape == (55, 5000) and not np.isnan(X).any()
    y = (df["group"].values != "NC").astype(int)
    print("SLE-combined:", y.sum(), "| NC:", (1 - y).sum(), flush=True)

    results = {}
    for cfg in CONFIGS:
        print(f"\n=== Config {cfg['name']} ===", flush=True)
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        fold_scores, fold_hist = [], []
        for fold, (tr, te) in enumerate(outer.split(X, y)):
            t0 = time.time()
            torch.manual_seed(RANDOM_STATE + fold)
            model = build_model(X.shape[1])
            train_ds = BetaValueDataset(X[tr], y[tr])
            hist = train_fixed_epochs(model, train_ds, cfg["backbone_lr"])
            p, l = predict_probs(model, BetaValueDataset(X[te], y[te]))
            score = float(roc_auc_score(l, p))
            fold_scores.append(score)
            fold_hist.append(hist)
            first, last = hist[0], hist[-1]
            print(f"Fold {fold}: loss {first['train_loss']:.3f} -> {last['train_loss']:.3f} | "
                  f"train AUROC {first['train_auroc']:.3f} -> {last['train_auroc']:.3f} | "
                  f"FINAL-EPOCH OUTER TEST AUROC={score:.3f}  ({time.time() - t0:.0f}s)",
                  flush=True)
            del model
            torch.cuda.empty_cache()
        arr = np.array(fold_scores)
        print(f"{cfg['name']}: outer test AUROC = {arr.mean():.3f} (+/- {arr.std():.3f})",
              flush=True)
        results[cfg["name"]] = {"fold_scores": fold_scores, "mean": float(arr.mean()),
                                "std": float(arr.std()), "history": fold_hist}

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
