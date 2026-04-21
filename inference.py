"""Inference module — frozen HuggingFace encoders for character-level embeddings.
No random projections. CKBERT outputs 1024d, BGE outputs 768d natively.
"""
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from saver import C2VC_MODEL_PATH, C2VB_MODEL_PATH
import utils as ut

# ========== Configuration ==========
INFERENCE_CONFIG_PATH = "./configs/inference_config.json"
INFERENCE_CONFIG = ut.load_json_config(INFERENCE_CONFIG_PATH, tag="inference-config")

CKBERT_OUTPUT_DIM = int(ut.nested_get(INFERENCE_CONFIG, ["ckbert", "native_output_dim"], 1024))
BGE_OUTPUT_DIM = int(ut.nested_get(INFERENCE_CONFIG, ["bge", "native_output_dim"], 768))
BGE_QUERY_INSTRUCTION = str(ut.nested_get(INFERENCE_CONFIG, ["bge", "query_instruction"], "为这个句子生成表示以用于检索相关文章："))


# ========== Cache ==========
@dataclass
class _Cache:
    tokenizer: Optional[object] = None
    model: Optional[object] = None
    device: Optional[torch.device] = None

_CKBERT_CACHE = _Cache()
_BGE_CACHE = _Cache()


def _get_cache(encoder: str) -> _Cache:
    return _CKBERT_CACHE if encoder == "ckbert" else _BGE_CACHE


def _resolve_model_dir(encoder: str) -> str:
    return C2VC_MODEL_PATH if encoder == "ckbert" else C2VB_MODEL_PATH


# ========== Character-level pooling ==========
def _char_pool(token_emb, attn_mask, offsets, text):
    char_len = len(text)
    if char_len == 0:
        return torch.zeros((0, token_emb.shape[1]), dtype=torch.float32)
    char_sum = torch.zeros((char_len, token_emb.shape[1]), dtype=torch.float32)
    char_cnt = torch.zeros((char_len, 1), dtype=torch.float32)
    for i in range(token_emb.shape[0]):
        if attn_mask[i].item() == 0:
            continue
        s, e = int(offsets[i, 0].item()), int(offsets[i, 1].item())
        if e <= s:
            continue
        s, e = max(0, min(s, char_len)), max(0, min(e, char_len))
        if e <= s:
            continue
        char_sum[s:e] += token_emb[i]
        char_cnt[s:e] += 1.0
    mask = char_cnt.squeeze(-1) > 0
    out = torch.zeros_like(char_sum)
    out[mask] = char_sum[mask] / char_cnt[mask]
    return out


def _mean_pool(token_emb, attn_mask):
    mask = attn_mask.unsqueeze(-1)
    return (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


# ========== Model loading ==========
def _load_model(encoder: str, verbose: bool):
    cache = _get_cache(encoder)
    if cache.model is not None:
        return cache.tokenizer, cache.model, cache.device
    model_dir = _resolve_model_dir(encoder)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    cache.tokenizer, cache.model, cache.device = tokenizer, model, device
    if verbose:
        print(f"[{encoder}] loaded from: {model_dir}, device: {device}")
    return tokenizer, model, device


# ========== Core encode ==========
def _encode_char_level(
    sentences: List[str],
    encoder: str,
    batch_size: int = 32,
    max_length: int = 128,
    verbose: bool = True,
    instruction: Optional[str] = None,
) -> List[torch.Tensor]:
    if not sentences:
        return []

    tokenizer, model, device = _load_model(encoder, verbose)

    # Normalize
    texts = [str(x) if x else " " for x in sentences]

    # Instruction prefix for BGE
    trim_len = 0
    if instruction:
        inst = instruction.strip()
        texts = [f"{inst}{x}" for x in texts]
        trim_len = len(inst)

    results = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        try:
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                                return_tensors="pt", return_offsets_mapping=True)
        except TypeError:
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                                return_tensors="pt")

        offsets = encoded.pop("offset_mapping", None)
        inputs = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        token_emb = outputs.last_hidden_state.detach().cpu().float()
        attn_mask = inputs["attention_mask"].detach().cpu()
        if offsets is not None:
            offsets = offsets.detach().cpu()

        for b_idx, text in enumerate(batch):
            if offsets is not None:
                seq = _char_pool(token_emb[b_idx], attn_mask[b_idx], offsets[b_idx], text)
            else:
                raise RuntimeError("Tokenizer does not support offset mapping.")
            # Trim instruction prefix
            if trim_len > 0 and seq.shape[0] > trim_len:
                seq = seq[trim_len:]
            results.append(seq)

        if verbose and (i == 0 or i + batch_size >= total):
            print(f"[{encoder}] progress: {min(i + batch_size, total)} / {total}")

    if verbose and results:
        print(f"[{encoder}] output: {len(results)} samples, first shape: {tuple(results[0].shape)}")
    return results


# ========== Public API ==========
def infer_vc(texts: List[str], batch_size=32, max_length=128, verbose=True,
             path=None) -> List[torch.Tensor]:
    from saver import VC_EMBEDDINGS_PATH
    p = path or VC_EMBEDDINGS_PATH
    cached = ut.load_torch_list(p)
    if cached and cached[0].ndim == 2 and cached[0].shape[1] == CKBERT_OUTPUT_DIM:
        return cached
    return _encode_char_level(texts, "ckbert", batch_size, max_length, verbose)


def infer_vb(texts: List[str], batch_size=32, max_length=128, verbose=True,
             use_query_instruction=False, path=None) -> List[torch.Tensor]:
    from saver import VB_EMBEDDINGS_PATH
    p = path or VB_EMBEDDINGS_PATH
    cached = ut.load_torch_list(p)
    if cached and cached[0].ndim == 2 and cached[0].shape[1] == BGE_OUTPUT_DIM:
        return cached
    inst = BGE_QUERY_INSTRUCTION if use_query_instruction else None
    return _encode_char_level(texts, "bge", batch_size, max_length, verbose, instruction=inst)
