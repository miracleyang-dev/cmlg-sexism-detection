"""CMLG — CKBERT-based Multi-feature Attention-fused Bi-GRU for binary classification."""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import utils as ut

# ========== Configuration ==========
CMLG_CONFIG_PATH = "./configs/cmlg_config.json"
CMLG_CONFIG = ut.load_json_config(CMLG_CONFIG_PATH, tag="cmlg-config")

CMLG_HIDDEN_DIM = int(ut.nested_get(CMLG_CONFIG, ["model", "hidden_dim"], 64))
CMLG_NUM_CLASSES = int(ut.nested_get(CMLG_CONFIG, ["model", "num_classes"], 2))
CMLG_DROPOUT = float(ut.nested_get(CMLG_CONFIG, ["model", "dropout"], 0.3))
CMLG_FC_DIM = int(ut.nested_get(CMLG_CONFIG, ["model", "fc_dim"], 64))
CMLG_PROJECT_DIM = int(ut.nested_get(CMLG_CONFIG, ["model", "project_dim"], 128))
CMLG_ATTN_HEADS = int(ut.nested_get(CMLG_CONFIG, ["model", "attn_heads"], 4))

CMLG_USE_FOCAL_LOSS = bool(ut.nested_get(CMLG_CONFIG, ["loss", "use_focal_loss"], True))
CMLG_FOCAL_GAMMA = float(ut.nested_get(CMLG_CONFIG, ["loss", "focal_gamma"], 2.0))
CMLG_LABEL_SMOOTHING = float(ut.nested_get(CMLG_CONFIG, ["loss", "label_smoothing"], 0.05))
CMLG_WEIGHT_POWER = float(ut.nested_get(CMLG_CONFIG, ["loss", "weight_power"], 1.0))


# ========== Focal Loss ==========
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        num_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target = target.long()
        target_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        if self.label_smoothing > 0:
            true_dist = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(-1), 1.0 - self.label_smoothing)
            target_log_probs = (true_dist * log_probs).sum(dim=-1)
        else:
            target_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        focal = (1.0 - target_probs).clamp(min=0).pow(self.gamma)
        loss = -target_log_probs * focal
        if self.weight is not None:
            loss = loss * self.weight.gather(0, target)
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss


# ========== Feature Attention Fusion ==========
class FeatureAttention(nn.Module):
    """Additive attention across feature streams at each character position.

    Input:  (B, T, K, D) — K feature streams, each projected to D
    Output: (B, T, D)    — attention-weighted fusion
            (B, T, K)    — attention weights (for interpretability)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, features: torch.Tensor):
        # features: (B, T, K, D)
        scores = self.v(torch.tanh(self.W(features)))  # (B, T, K, 1)
        weights = F.softmax(scores, dim=2)              # (B, T, K, 1)
        fused = (weights * features).sum(dim=2)         # (B, T, D)
        return fused, weights.squeeze(-1)               # (B, T, D), (B, T, K)


# ========== Model ==========
class CMLG(nn.Module):
    """Bi-GRU classifier with feature-level attention fusion."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = CMLG_HIDDEN_DIM,
        num_classes: int = CMLG_NUM_CLASSES,
        dropout: float = CMLG_DROPOUT,
        fc_dim: int = CMLG_FC_DIM,
        feature_dims: Optional[list[int]] = None,
        project_dim: int = CMLG_PROJECT_DIM,
    ):
        super().__init__()
        self.feature_dims = feature_dims

        if feature_dims is not None and len(feature_dims) > 1:
            # Per-stream projection to uniform dimension
            self.projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d, project_dim),
                    nn.LayerNorm(project_dim),
                    nn.GELU(),
                ) for d in feature_dims
            ])
            # Attention fusion across streams
            self.attention = FeatureAttention(project_dim)
            gru_input_dim = project_dim
        else:
            # Single feature — just project, no attention needed
            self.projections = None
            self.attention = None
            actual_dim = feature_dims[0] if feature_dims else input_dim
            self.single_proj = nn.Sequential(
                nn.Linear(actual_dim, project_dim),
                nn.LayerNorm(project_dim),
                nn.GELU(),
            )
            gru_input_dim = project_dim

        self.bigru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fc_dim, num_classes)

    # ========== Padding ==========
    @staticmethod
    def prepare_v(V: Sequence[torch.Tensor], device=None):
        seqs = [item.detach().cpu().float() for item in V]
        lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
        padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        if device is not None:
            padded, lengths = padded.to(device), lengths.to(device)
        return padded, lengths

    # ========== Forward ==========
    def forward(self, v: list[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        v_padded, lengths = self.prepare_v(v, device=device)  # (B, T, input_dim)

        if self.projections is not None and self.feature_dims is not None:
            # Split into feature streams and project each
            parts = torch.split(v_padded, self.feature_dims, dim=-1)
            projected = [proj(p) for proj, p in zip(self.projections, parts)]
            stacked = torch.stack(projected, dim=2)  # (B, T, K, project_dim)
            v_fused, _ = self.attention(stacked)      # (B, T, project_dim)
        else:
            v_fused = self.single_proj(v_padded)      # (B, T, project_dim)

        packed = nn.utils.rnn.pack_padded_sequence(
            v_fused, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.bigru(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, hidden*2)

        x = self.fc(h)
        x = self.bn(x)
        x = self.dropout(x)
        return self.classifier(x)

    def get_attention_weights(self, v: list[torch.Tensor]) -> torch.Tensor:
        """Return per-position attention weights (B, T, K) for interpretability."""
        if self.attention is None:
            return None
        device = next(self.parameters()).device
        v_padded, _ = self.prepare_v(v, device=device)
        parts = torch.split(v_padded, self.feature_dims, dim=-1)
        projected = [proj(p) for proj, p in zip(self.projections, parts)]
        stacked = torch.stack(projected, dim=2)
        _, weights = self.attention(stacked)
        return weights.detach().cpu()

    # ========== Loss ==========
    @staticmethod
    def build_criterion(weight=None):
        if CMLG_USE_FOCAL_LOSS:
            return FocalLoss(weight=weight, gamma=CMLG_FOCAL_GAMMA, label_smoothing=CMLG_LABEL_SMOOTHING)
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=CMLG_LABEL_SMOOTHING)

    @torch.no_grad()
    def predict(self, v: list[torch.Tensor]) -> torch.Tensor:
        self.eval()
        return torch.argmax(self.forward(v), dim=-1)


# ========== Utilities ==========
def build_class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=num_classes).float().clamp(min=1.0)
    weights = y.numel() / (num_classes * counts)
    weights = torch.pow(weights, CMLG_WEIGHT_POWER)
    return weights / weights.mean().clamp(min=1e-12)


def iter_batches(n: int, batch_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


@torch.no_grad()
def predict_model(model: CMLG, X: list[torch.Tensor], batch_size: int = 128) -> torch.Tensor:
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        preds.append(torch.argmax(logits, dim=-1).cpu())
    return torch.cat(preds, dim=0)


def _precision_recall_f1(y_true, y_pred, num_classes):
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum().item())
        fp = int(((y_true != c) & (y_pred == c)).sum().item())
        fn = int(((y_true == c) & (y_pred != c)).sum().item())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f1)
    return {"macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1s))}


def _confusion_matrix(y_true, y_pred, num_classes):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for i in range(y_true.shape[0]):
        t, p = int(y_true[i].item()), int(y_pred[i].item())
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1.0
    return cm


def _per_class_from_confusion(cm):
    per_class = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c].item())
        fp = float(cm[:, c].sum().item() - tp)
        fn = float(cm[c, :].sum().item() - tp)
        support = float(cm[c, :].sum().item())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class.append({"precision": p, "recall": r, "f1": f1, "support": support})
    return per_class


@torch.no_grad()
def evaluate_model(model, X, y, num_classes):
    d = evaluate_model_detailed(model, X, y, num_classes)
    return {"acc": d["acc"], "macro_precision": d["macro_precision"],
            "macro_recall": d["macro_recall"], "macro_f1": d["macro_f1"]}


@torch.no_grad()
def evaluate_model_detailed(model, X, y, num_classes):
    y_pred = predict_model(model, X)
    y_true = y.cpu()
    acc = float((y_pred == y_true).float().mean().item())
    prf = _precision_recall_f1(y_true, y_pred, num_classes)
    cm = _confusion_matrix(y_true, y_pred, num_classes)
    return {"acc": acc, **prf, "confusion_matrix": cm,
            "per_class": _per_class_from_confusion(cm)}
