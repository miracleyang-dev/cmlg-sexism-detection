# %%
# =============================================================
# Main experiment runner — 3 datasets × 5 ablation settings
# Runs: original → cleaned_full → cleaned_highconf
# Each dataset gets its own embeddings and results
# =============================================================

import importlib as lib
import shutil
import pandas as pd
from pathlib import Path

# Project modules
import utils
import cmlg
import saver as sv
import inference as inf
import ablation as abl

lib.reload(utils)
lib.reload(cmlg)
lib.reload(abl)
lib.reload(inf)
lib.reload(sv)

# %% [markdown]
# # Dataset Configuration

# %%
DATASETS = {
    "original": {
        "path": "./dataset/SexCommentNew.xlsx",
        "desc": "Original labels (no cleaning)",
    },
    "cleaned_full": {
        "path": "./dataset/SexCommentCleaned_full.xlsx",
        "desc": "LLM 3-model majority vote relabeled (8962 samples)",
    },
    "cleaned_highconf": {
        "path": "./dataset/SexCommentCleaned_highconf.xlsx",
        "desc": "LLM 3-model unanimous agreement only (6793 samples)",
    },
}


def clean_artifacts():
    """Remove per-dataset embeddings, Word2Vec models, and ablation results.
    HuggingFace models (c2vc_model, c2vb_model) are kept — they don't change between datasets.
    """
    # 1. Delete all cached embeddings (per-dataset, must regenerate)
    emb_dir = Path("./embeddings")
    if emb_dir.exists():
        shutil.rmtree(emb_dir)
        print("  Removed dir:  ./embeddings")

    # 2. Delete Word2Vec models only (trained per-dataset, must retrain)
    #    Keep c2vc_model (CKBERT) and c2vb_model (BGE) untouched
    em_dir = Path("./embedding_models")
    if em_dir.exists():
        for pattern in ["c2vp_model*", "c2vw_model*"]:
            for f in sorted(em_dir.glob(pattern)):
                if f.is_file():
                    f.unlink()
                    print(f"  Removed W2V:  {f}")
                elif f.is_dir():
                    shutil.rmtree(f)
                    print(f"  Removed W2V:  {f}/")

    # 3. Delete previous ablation results
    abl_file = Path("./ablation_rows.json")
    if abl_file.exists():
        abl_file.unlink()
        print("  Removed file: ./ablation_rows.json")

    print("  (HuggingFace models kept intact)")


# %% [markdown]
# # Run All Experiments

# %%
all_results = {}

for ds_name, ds_cfg in DATASETS.items():
    print(f"\n{'='*70}")
    print(f"  DATASET: {ds_name}")
    print(f"  {ds_cfg['desc']}")
    print(f"  Path: {ds_cfg['path']}")
    print(f"{'='*70}")

    # --- Step 0: Clean old artifacts ---
    print("\n[Step 0] Cleaning old artifacts...")
    clean_artifacts()

    # --- Step 1: Load data ---
    print("\n[Step 1] Loading data...")
    df = pd.read_excel(ds_cfg["path"])
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    print(f"  label distribution:")
    print(f"  {df['label'].value_counts().sort_index().to_dict()}")

    texts = df["comment_text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    print(f"  texts: {len(texts)}, labels: {len(labels)}")

    # --- Step 2: Feature engineering ---
    print("\n[Step 2] Feature engineering — pinyin & wubi...")
    texts_py, texts_wb = utils.c2PandW(texts)

    print("\n[Step 2b] Training Word2Vec models...")
    Vp_model = sv.load_c2vp_model(texts_py)
    Vw_model = sv.load_c2vw_model(texts_wb)
    print(f"  pinyin: dim={Vp_model.vector_size}, vocab={len(Vp_model.wv)}")
    print(f"  wubi:   dim={Vw_model.vector_size}, vocab={len(Vw_model.wv)}")

    print("\n[Step 2c] Building pinyin & wubi embeddings...")
    V_p = [utils.tokens2matrix(toks, Vp_model) for toks in texts_py]
    V_w = [utils.tokens2matrix(toks, Vw_model) for toks in texts_wb]
    sv.save_vp_embeddings(V_p)
    sv.save_vw_embeddings(V_w)
    print(f"  V_p: {len(V_p)} samples, dim={V_p[0].shape[1]}")
    print(f"  V_w: {len(V_w)} samples, dim={V_w[0].shape[1]}")

    # --- Step 3: HuggingFace models ---
    print("\n[Step 3] Loading HuggingFace models (if not cached)...")
    sv.load_ckbert_model()
    sv.load_bge_model()

    # --- Step 4: CKBERT & BGE embeddings ---
    print("\n[Step 4] CKBERT character-level embeddings...")
    V_c = inf.infer_vc(texts)
    print(f"  V_c: {len(V_c)} samples, dim={V_c[0].shape[1]}")
    sv.save_vc_embeddings(V_c)

    print("\n[Step 4b] BGE character-level embeddings...")
    V_b = inf.infer_vb(texts)
    print(f"  V_b: {len(V_b)} samples, dim={V_b[0].shape[1]}")
    sv.save_vb_embeddings(V_b)

    # --- Step 5: Reload embeddings & verify ---
    print("\n[Step 5] Reloading saved embeddings...")
    V_c = sv.load_vc_embeddings()
    V_p = sv.load_vp_embeddings()
    V_w = sv.load_vw_embeddings()
    V_b = sv.load_vb_embeddings()
    assert len(V_c) == len(V_p) == len(V_w) == len(V_b) == len(labels)
    print(f"  V_c={V_c[0].shape[1]}d, V_p={V_p[0].shape[1]}d, "
          f"V_w={V_w[0].shape[1]}d, V_b={V_b[0].shape[1]}d")

    # --- Step 6: Ablation study ---
    print("\n[Step 6] Running ablation study (5-fold CV)...")
    cfg = abl.TrainConfig()
    results_path = Path("./ablation_rows.json")
    ablation_rows = utils.load_json_list(str(results_path))
    common_kwargs = dict(V_c=V_c, V_p=V_p, V_w=V_w, V_b=V_b, labels=labels, config=cfg)

    run_step = lambda i: utils.run_and_store_step(
        i, abl.ABLATION_SETTINGS, ablation_rows, str(results_path),
        lambda s: abl.run_ablation_setting(setting=s, **common_kwargs))

    print(f"  Settings: {[s.name for s in abl.ABLATION_SETTINGS]}")

    for step_i in range(len(abl.ABLATION_SETTINGS)):
        row, ablation_rows = run_step(step_i)

    # --- Step 7: Collect results ---
    print(f"\n[Step 7] Results for dataset: {ds_name}")
    ablation_df = utils.format_results_frame(utils.load_results_frame(str(results_path)))
    if not ablation_df.empty:
        show_cols = ["setting", "samples", "acc_mean", "acc_std",
                     "precision_mean", "recall_mean", "macro_f1_mean", "macro_f1_std"]
        show_cols = [c for c in show_cols if c in ablation_df.columns]
        print(ablation_df[show_cols].to_string())

        # Per-class diagnostics
        diag_cols = ["setting", "macro_f1_mean",
                     "p_c0_mean", "r_c0_mean", "f1_c0_mean",
                     "p_c1_mean", "r_c1_mean", "f1_c1_mean"]
        diag_cols = [c for c in diag_cols if c in ablation_df.columns]
        print(f"\n  Per-class (c0=neutral, c1=opposing):")
        print(ablation_df[diag_cols].round(4).to_string())

    # Save dataset-specific results
    ds_result_path = f"./results_{ds_name}.json"
    utils.save_json_list(ablation_rows, ds_result_path)
    print(f"  Saved to {ds_result_path}")
    all_results[ds_name] = ablation_df.copy()

# %% [markdown]
# # Cross-Dataset Comparison

# %%
print(f"\n{'='*70}")
print("  CROSS-DATASET COMPARISON")
print(f"{'='*70}")

comparison_rows = []
for ds_name, df_res in all_results.items():
    if df_res.empty:
        continue
    for _, row in df_res.iterrows():
        comparison_rows.append({
            "dataset": ds_name,
            "setting": row.get("setting", ""),
            "samples": row.get("samples", 0),
            "acc_mean": row.get("acc_mean", 0),
            "acc_std": row.get("acc_std", 0),
            "macro_f1_mean": row.get("macro_f1_mean", 0),
            "macro_f1_std": row.get("macro_f1_std", 0),
            "f1_c0_mean": row.get("f1_c0_mean", 0),
            "f1_c1_mean": row.get("f1_c1_mean", 0),
        })

if comparison_rows:
    comp_df = pd.DataFrame(comparison_rows)
    comp_df = comp_df.sort_values(["setting", "dataset"]).reset_index(drop=True)
    print(comp_df.round(4).to_string())

    # Save comparison
    comp_df.to_csv("./results_comparison.csv", index=False)
    comp_df.to_excel("./results_comparison.xlsx", index=False)
    print("\nSaved: results_comparison.csv / results_comparison.xlsx")

    # Best result per dataset
    print("\n--- Best F1 per dataset ---")
    for ds_name in all_results:
        ds_rows = comp_df[comp_df["dataset"] == ds_name]
        if not ds_rows.empty:
            best = ds_rows.loc[ds_rows["macro_f1_mean"].idxmax()]
            print(f"  {ds_name}: {best['setting']} -> F1={best['macro_f1_mean']:.4f}+/-{best['macro_f1_std']:.4f}")

print("\nAll experiments complete.")
