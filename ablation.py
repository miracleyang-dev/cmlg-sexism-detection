"""Ablation study — 5-fold stratified CV for binary classification."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import cmlg as cmlg
import utils as ut

# ========== Configuration ==========
ABLATION_CONFIG_PATH = "./configs/ablation_config.json"

@dataclass(frozen=True)
class AblationSetting:
    name: str
    use_c: bool = True
    use_p: bool = True
    use_w: bool = True
    use_b: bool = False

    @classmethod
    def from_mapping(cls, raw):
        return cls(name=str(raw.get("name", "unnamed")),
                   use_c=bool(raw.get("use_c", True)), use_p=bool(raw.get("use_p", True)),
                   use_w=bool(raw.get("use_w", True)), use_b=bool(raw.get("use_b", False)))

ABLATION_CONFIG = ut.load_json_config(ABLATION_CONFIG_PATH, tag="ablation-config")
TRAIN_CFG = ut.nested_get(ABLATION_CONFIG, ["train"], {})
RAW_SETTINGS = ut.nested_get(ABLATION_CONFIG, ["settings"], [])

def _parse_settings(raw):
    out = []
    for item in raw:
        out.append(item if isinstance(item, AblationSetting) else AblationSetting.from_mapping(item))
    return out if out else [AblationSetting(name="c+p+w+b")]

ABLATION_SETTINGS = _parse_settings(RAW_SETTINGS)

NUM_CLASSES = 2  # Binary classification


@dataclass
class TrainConfig:
    epochs: int = int(ut.nested_get(TRAIN_CFG, ["epochs"], 100))
    batch_size: int = int(ut.nested_get(TRAIN_CFG, ["batch_size"], 32))
    lr: float = float(ut.nested_get(TRAIN_CFG, ["lr"], 5e-4))
    weight_decay: float = float(ut.nested_get(TRAIN_CFG, ["weight_decay"], 5e-4))
    random_state: int = int(ut.nested_get(TRAIN_CFG, ["random_state"], 42))
    val_ratio: float = float(ut.nested_get(TRAIN_CFG, ["val_ratio"], 0.2))
    early_stopping_patience: int = int(ut.nested_get(TRAIN_CFG, ["early_stopping_patience"], 15))
    min_delta: float = float(ut.nested_get(TRAIN_CFG, ["min_delta"], 1e-4))
    cv_folds: int = int(ut.nested_get(TRAIN_CFG, ["cv_folds"], 5))
    verbose: bool = bool(ut.nested_get(TRAIN_CFG, ["verbose"], True))
    grad_clip_norm: float = float(ut.nested_get(TRAIN_CFG, ["grad_clip_norm"], 1.0))
    scheduler_patience: int = int(ut.nested_get(TRAIN_CFG, ["scheduler_patience"], 5))
    scheduler_factor: float = float(ut.nested_get(TRAIN_CFG, ["scheduler_factor"], 0.5))


# ========== Helpers ==========
def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _subset(X, y, idx):
    return [X[i] for i in idx], y[torch.as_tensor(idx, dtype=torch.long)]


# ========== Feature Fusion ==========
def build_fused_features(V_c, V_p, V_w, V_b, labels, use_c=True, use_p=True, use_w=True, use_b=True):
    groups = [g for g, e in ((V_c, use_c), (V_p, use_p), (V_w, use_w), (V_b, use_b)) if e]
    if not groups:
        raise ValueError("At least one feature must be enabled.")
    n = min(len(g) for g in groups + [labels])
    dims = [g[0].shape[1] for g in groups]
    fused, fused_labels = [], []
    for i in range(n):
        seqs = [g[i] for g in groups]
        if any(s.ndim != 2 for s in seqs):
            continue
        base_len = seqs[0].shape[0]
        if any(s.shape[0] != base_len for s in seqs[1:]):
            continue
        fused.append(torch.cat(seqs, dim=1))
        fused_labels.append(int(labels[i]))
    skipped = n - len(fused)
    if skipped > 0:
        print(f"[fusion] WARNING: {skipped}/{n} skipped (length mismatch)")
    print(f"[fusion] features: {dims}, total_dim: {sum(dims)}, samples: {len(fused)}")
    return fused, torch.as_tensor(fused_labels, dtype=torch.long), dims


# ========== Stratified K-Fold ==========
def stratified_kfold_indices(labels, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    labels_np = np.asarray(labels)
    buckets = {}
    for cls in np.unique(labels_np):
        idx = np.where(labels_np == cls)[0]
        rng.shuffle(idx)
        buckets[int(cls)] = np.array_split(idx, n_splits)
    folds = []
    for fi in range(n_splits):
        val_idx, train_idx = [], []
        for cls in np.unique(labels_np):
            parts = buckets[int(cls)]
            val_idx.extend(parts[fi].tolist())
            for pi in range(n_splits):
                if pi != fi:
                    train_idx.extend(parts[pi].tolist())
        rng.shuffle(train_idx); rng.shuffle(val_idx)
        folds.append((train_idx, val_idx))
    return folds


# ========== Training ==========
def _train_epoch(model, X, y, optimizer, criterion, batch_size, seed, clip_norm):
    losses = []
    for batch_idx in cmlg.iter_batches(len(X), batch_size, seed):
        batch_x = [X[i] for i in batch_idx.tolist()]
        batch_y = y[torch.as_tensor(batch_idx, dtype=torch.long)]
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def _train_eval_fold(X, y, train_idx, val_idx, config, feature_dims, setting_name, fold_name):
    device = _device()
    X_train, y_train = _subset(X, y, train_idx)
    X_val, y_val = _subset(X, y, val_idx)

    if config.verbose:
        print(f"[{setting_name}][{fold_name}] start: train={len(X_train)}, val={len(X_val)}, "
              f"input_dim={X_train[0].shape[1]}, classes={NUM_CLASSES}")

    model = cmlg.CMLG(
        input_dim=X_train[0].shape[1], num_classes=NUM_CLASSES, feature_dims=feature_dims,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.scheduler_factor, patience=config.scheduler_patience)

    weights = cmlg.build_class_weights(y_train, NUM_CLASSES).to(device)
    criterion = model.build_criterion(weight=weights)

    best_f1, best_epoch, best_state, bad = -1.0, 0, None, 0

    for ep in range(config.epochs):
        model.train()
        loss = _train_epoch(model, X_train, y_train.to(device), optimizer, criterion,
                            config.batch_size, config.random_state + ep, config.grad_clip_norm)
        val_m = cmlg.evaluate_model(model, X_val, y_val.to(device), NUM_CLASSES)

        if config.verbose:
            print(f"[{setting_name}][{fold_name}] epoch {ep+1}/{config.epochs} "
                  f"loss={loss:.4f} acc={val_m['acc']:.4f} p={val_m['macro_precision']:.4f} "
                  f"r={val_m['macro_recall']:.4f} f1={val_m['macro_f1']:.4f}")

        scheduler.step(val_m["macro_f1"])

        if val_m["macro_f1"] > best_f1 + config.min_delta:
            best_f1 = val_m["macro_f1"]; best_epoch = ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            if config.verbose:
                print(f"[{setting_name}][{fold_name}] best updated: epoch={best_epoch}, f1={best_f1:.4f}")
        else:
            bad += 1
            if config.early_stopping_patience > 0 and bad >= config.early_stopping_patience:
                if config.verbose:
                    print(f"[{setting_name}][{fold_name}] early stopping at epoch {ep+1}")
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    train_m = cmlg.evaluate_model(model, X_train, y_train.to(device), NUM_CLASSES)
    val_d = cmlg.evaluate_model_detailed(model, X_val, y_val.to(device), NUM_CLASSES)

    if config.verbose:
        print(f"[{setting_name}][{fold_name}] done: best_epoch={best_epoch}, "
              f"train_f1={train_m['macro_f1']:.4f}, val_f1={val_d['macro_f1']:.4f}")

    results = {
        "input_dim": float(X_train[0].shape[1]), "best_epoch": float(best_epoch),
        "train_acc": train_m["acc"], "train_macro_f1": train_m["macro_f1"],
        "acc": val_d["acc"], "precision": val_d["macro_precision"],
        "recall": val_d["macro_recall"], "macro_f1": val_d["macro_f1"],
    }
    for ci, cm in enumerate(val_d["per_class"]):
        results[f"p_c{ci}"] = cm["precision"]
        results[f"r_c{ci}"] = cm["recall"]
        results[f"f1_c{ci}"] = cm["f1"]
    cm = val_d["confusion_matrix"]
    for ti in range(NUM_CLASSES):
        for pi in range(NUM_CLASSES):
            results[f"cm_{ti}_{pi}"] = float(cm[ti, pi].item())
    return results


def _avg_folds(fold_metrics):
    agg = {}
    for key in fold_metrics[0]:
        vals = np.array([m[key] for m in fold_metrics], dtype=np.float64)
        agg[f"{key}_mean"] = float(vals.mean())
        agg[f"{key}_std"] = float(vals.std(ddof=0))
    return agg


# ========== Public API ==========
def run_ablation_setting(V_c, V_p, V_w, V_b, labels, setting, config=None):
    config = config or TrainConfig()
    set_seed(config.random_state)
    s = setting if isinstance(setting, AblationSetting) else AblationSetting.from_mapping(setting)

    X, y, feature_dims = build_fused_features(
        V_c, V_p, V_w, V_b, labels,
        use_c=s.use_c, use_p=s.use_p, use_w=s.use_w, use_b=s.use_b)

    folds = stratified_kfold_indices(y.tolist(), n_splits=config.cv_folds, seed=config.random_state)
    fold_results = []
    for fi, (train_idx, val_idx) in enumerate(folds, 1):
        m = _train_eval_fold(X, y, train_idx, val_idx, config, feature_dims, s.name, f"fold{fi}")
        fold_results.append(m)

    agg = _avg_folds(fold_results)
    row = {"setting": s.name, "samples": float(len(y)), "cv_folds": float(config.cv_folds)}
    row.update(agg)
    if config.verbose:
        print(f"[{s.name}] CV: acc={agg['acc_mean']:.4f}±{agg['acc_std']:.4f}, "
              f"f1={agg['macro_f1_mean']:.4f}±{agg['macro_f1_std']:.4f}")
    return row
