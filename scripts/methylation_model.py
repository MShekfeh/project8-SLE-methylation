"""
Project 6, Stage 2 — Methylation transformer architecture.

Runs on: anywhere (pure PyTorch module definition, no data/GPU needed to
import this file). Actual training happens on Colab (see 08_train_transformer.py)
because full self-attention over 5,000 tokens needs more memory than
compbio's ~6.2GB budget comfortably allows — see README "Compute Notes"
for the full reasoning.

Each of the 5,000 CpG probes becomes one input "token." Unlike DNABERT-2,
there is no discrete vocabulary to look up — a token here is built from two
learned pieces added together:
    1. probe_embedding[i]      -> "which CpG site is this" (also serves as
                                   positional info, since probe order is fixed
                                   and identical across all samples)
    2. value_projection(beta)  -> "what's the methylation level here"
A learned [CLS] token is prepended, exactly like BERT/DNABERT-2, and its
final-layer representation is used both for classification and, for Stage 3,
the UMAP/attractor analysis (via return_embedding=True).
"""

import torch
import torch.nn as nn


class MethylationTransformer(nn.Module):
    def __init__(
        self,
        n_probes: int = 5000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        n_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Piece 1: probe identity embedding — one learned vector per CpG site
        self.probe_embedding = nn.Embedding(n_probes, d_model)

        # Piece 2: continuous beta-value projection — maps each scalar beta
        # value into a d_model-dimensional vector (the continuous-value
        # equivalent of a discrete token embedding lookup)
        self.value_projection = nn.Linear(1, d_model)

        # Learned [CLS] token, prepended to every sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

        self.register_buffer("probe_indices", torch.arange(n_probes))

    def forward(self, beta_values, return_embedding=False):
        """
        beta_values: (batch, n_probes) float tensor of methylation values
        return_embedding: if True, return the CLS representation instead of
            classifier logits (used for UMAP/attractor analysis in Stage 3)
        """
        batch_size = beta_values.shape[0]

        probe_emb = self.probe_embedding(self.probe_indices)
        probe_emb = probe_emb.unsqueeze(0).expand(batch_size, -1, -1)

        value_emb = self.value_projection(beta_values.unsqueeze(-1))

        tokens = probe_emb + value_emb

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        encoded = self.encoder(tokens)

        cls_output = encoded[:, 0, :]
        cls_output = self.dropout(cls_output)

        if return_embedding:
            return cls_output

        return self.classifier(cls_output)
