import os
import torch
from gensim.models import Word2Vec
import utils as ut

# ========== Configuration ==========
SAVER_CONFIG_PATH = "./configs/saver_config.json"
SAVER_CONFIG = ut.load_json_config(SAVER_CONFIG_PATH, tag="saver-config")
os.environ["HF_ENDPOINT"] = str(ut.nested_get(SAVER_CONFIG, ["hf_endpoint"], "https://hf-mirror.com"))

CKBERT_MODEL_ID = str(ut.nested_get(SAVER_CONFIG, ["model_ids", "ckbert"], "alibaba-pai/pai-ckbert-large-zh"))
BGE_MODEL_ID = str(ut.nested_get(SAVER_CONFIG, ["model_ids", "bge"], "BAAI/bge-base-zh-v1.5"))

C2VC_MODEL_PATH = str(ut.nested_get(SAVER_CONFIG, ["paths", "c2vc_model"], "./embedding_models/c2vc_model"))
C2VP_MODEL_PATH = str(ut.nested_get(SAVER_CONFIG, ["paths", "c2vp_model"], "./embedding_models/c2vp_model"))
C2VW_MODEL_PATH = str(ut.nested_get(SAVER_CONFIG, ["paths", "c2vw_model"], "./embedding_models/c2vw_model"))
C2VB_MODEL_PATH = str(ut.nested_get(SAVER_CONFIG, ["paths", "c2vb_model"], "./embedding_models/c2vb_model"))

VC_EMBEDDINGS_PATH = str(ut.nested_get(SAVER_CONFIG, ["paths", "v_ckbert"], "./embeddings/v_ckbert"))
VP_EMBEDDINGS_PATH = str(ut.nested_get(SAVER_CONFIG, ["paths", "v_pinyin"], "./embeddings/v_pinyin"))
VW_EMBEDDINGS_PATH = str(ut.nested_get(SAVER_CONFIG, ["paths", "v_wubi"], "./embeddings/v_wubi"))
VB_EMBEDDINGS_PATH = str(ut.nested_get(SAVER_CONFIG, ["paths", "v_bge"], "./embeddings/v_bge"))

WORD2VEC_OUTPUT_DIM = int(ut.nested_get(SAVER_CONFIG, ["word2vec", "vector_size"], 256))


# ========== Word2Vec Models ==========
def _load_w2v_model(texts: list[list[str]], save_path: str):
    if os.path.exists(save_path):
        return Word2Vec.load(save_path)
    model = Word2Vec(
        sentences=texts,
        vector_size=WORD2VEC_OUTPUT_DIM,
        window=int(ut.nested_get(SAVER_CONFIG, ["word2vec", "window"], 5)),
        min_count=int(ut.nested_get(SAVER_CONFIG, ["word2vec", "min_count"], 1)),
        workers=int(ut.nested_get(SAVER_CONFIG, ["word2vec", "workers"], 4)),
        sg=int(ut.nested_get(SAVER_CONFIG, ["word2vec", "sg"], 1)),
        epochs=int(ut.nested_get(SAVER_CONFIG, ["word2vec", "epochs"], 20)),
        seed=int(ut.nested_get(SAVER_CONFIG, ["word2vec", "seed"], 42)),
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return model

def load_c2vp_model(texts_py): return _load_w2v_model(texts_py, C2VP_MODEL_PATH)
def load_c2vw_model(texts_wb): return _load_w2v_model(texts_wb, C2VW_MODEL_PATH)


# ========== Hugging Face Models ==========
def _load_hf_model(local_dir: str, model_id: str):
    if os.path.exists(local_dir):
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model_id, local_dir=local_dir)

def load_ckbert_model(): _load_hf_model(C2VC_MODEL_PATH, CKBERT_MODEL_ID)
def load_bge_model(): _load_hf_model(C2VB_MODEL_PATH, BGE_MODEL_ID)


# ========== Save / Load Embeddings ==========
def save_vc_embeddings(e): ut.save_torch_list(e, VC_EMBEDDINGS_PATH)
def save_vp_embeddings(e): ut.save_torch_list(e, VP_EMBEDDINGS_PATH)
def save_vw_embeddings(e): ut.save_torch_list(e, VW_EMBEDDINGS_PATH)
def save_vb_embeddings(e): ut.save_torch_list(e, VB_EMBEDDINGS_PATH)

def load_vc_embeddings(): return ut.load_torch_list(VC_EMBEDDINGS_PATH)
def load_vp_embeddings(): return ut.load_torch_list(VP_EMBEDDINGS_PATH)
def load_vw_embeddings(): return ut.load_torch_list(VW_EMBEDDINGS_PATH)
def load_vb_embeddings(): return ut.load_torch_list(VB_EMBEDDINGS_PATH)
