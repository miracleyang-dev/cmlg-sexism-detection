import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from pypinyin import pinyin, Style
from pywubi import wubi


def c2PandW(texts: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    """Convert Chinese text list to aligned pinyin and wubi token lists."""
    unk_py, unk_wb, space_tok = "<unk_py>", "<unk_wb>", "<sp>"
    texts_py, texts_wb = [], []
    for t in texts:
        sent_py, sent_wb = [], []
        for ch in str(t):
            if ch.isspace():
                sent_py.append(space_tok); sent_wb.append(space_tok); continue
            py_item = pinyin(ch, style=Style.NORMAL)
            py_tok = py_item[0][0] if py_item and py_item[0] and py_item[0][0] else unk_py
            wb_item = wubi(ch)
            if isinstance(wb_item, (list, tuple)):
                wb_tok = wb_item[0] if len(wb_item) > 0 and wb_item[0] else unk_wb
            else:
                wb_tok = wb_item if wb_item else unk_wb
            sent_py.append(py_tok); sent_wb.append(wb_tok)
        texts_py.append(sent_py); texts_wb.append(sent_wb)
    return texts_py, texts_wb


def tokens2matrix(tokens: list[str], w2v_model: Word2Vec) -> torch.Tensor:
    """Convert token list to a matrix of word embeddings."""
    dim = w2v_model.vector_size
    zero_vec = np.zeros(dim, dtype=np.float32)
    vecs = [w2v_model.wv[t] if t in w2v_model.wv else zero_vec for t in tokens]
    return torch.from_numpy(np.asarray(vecs, dtype=np.float32))


def load_json_config(path: str, tag: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        print(f"[{tag}] failed to load {path}: {e}. Use fallback defaults.")
        return {}


def nested_get(cfg: dict, keys: list[str], default):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_json_list(path: str) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_json_list(items: list[dict], path: str) -> None:
    file_path = Path(path)
    if file_path.parent:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def upsert_json_item(items: list[dict], item: dict, key: str = "setting") -> list[dict]:
    filtered = [old for old in items if old.get(key) != item.get(key)]
    filtered.append(item)
    return filtered


def load_torch_list(path: str) -> list[torch.Tensor]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return torch.load(file_path)


def save_torch_list(items: list[torch.Tensor], path: str) -> None:
    file_path = Path(path)
    if file_path.parent:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, file_path)


def load_results_frame(path: str) -> pd.DataFrame:
    return pd.DataFrame(load_json_list(path))


def format_results_frame(frame: pd.DataFrame, sort_key: str = "macro_f1_mean") -> pd.DataFrame:
    if frame.empty:
        return frame
    metric_cols = [
        "samples", "cv_folds", "input_dim_mean", "input_dim_std",
        "train_acc_mean", "train_acc_std", "train_macro_f1_mean", "train_macro_f1_std",
        "acc_mean", "acc_std", "precision_mean", "precision_std",
        "recall_mean", "recall_std", "macro_f1_mean", "macro_f1_std",
    ]
    out = frame.copy()
    for col in metric_cols:
        if col in out.columns:
            out[col] = out[col].astype(float).round(4)
    if sort_key in out.columns:
        out = out.sort_values(sort_key, ascending=False).reset_index(drop=True)
    return out


def run_and_store_step(step_idx, settings, rows, save_path, runner):
    setting = settings[step_idx]
    print(f"\n===== Step {step_idx + 1}/{len(settings)}: {getattr(setting, 'name', step_idx)} =====")
    row = runner(setting)
    updated_rows = upsert_json_item(rows, row, key="setting")
    save_json_list(updated_rows, save_path)
    print(f"saved {len(updated_rows)} rows to {save_path}")
    return row, updated_rows
