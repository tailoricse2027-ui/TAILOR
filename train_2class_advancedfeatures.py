#!/usr/bin/env python3
"""
train_2class_advancedfeatures.py  —  Phase 2a: Classifier Training
====================================================================
Trains XGBoost / CatBoost / RandomForest classifiers to predict
SHORT vs LONG output class from prompt features (v13 feature set).

Output
------
  deploy/v13_xgb.json          XGBoost model
  deploy/v13_cat.cbm           CatBoost model
  deploy/v13_rf.pkl            RandomForest model
  deploy/v13_vectorizer.pkl    Shared TF-IDF vectorizer

Dataset keys (--datasets argument)
-----------------------------------
  llama2-7b-awq | llama2-13b-dolly | mistral-7b-dolly | mistral-7b-mixed |
  mistral_nemo_12b | deepseek-r1-llama-8b | deepseek-r1-qwen-14b | all

Usage (via TUI)
---------------
  python tailor_ui.py   →  Phase 2a

Usage (direct)
--------------
  python train_2class_advancedfeatures.py \
      --datasets mistral-7b-dolly,mistral-7b-mixed \
      --threshold 300 \
      --classifiers xgb,cat,rf
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

AVAILABLE_DATASETS = {
    "llama2-7b-awq":           "data/vllm_llama2_7b_awqmarlin_dolly15k.csv",
    "llama2-13b-dolly":        "data/vllm_llama2_13b_dolly15k.csv",
    "mistral-7b-dolly":        "data/vllm_mistral_7b_v0_2_dolly15k.csv",
    "mistral-7b-mixed":        "data/vllm_mistral_7b_v0_2_mixed_prompts_v2.csv",
    "mistral-7b-v01-dolly":    "data/vllm_mistral_7b_v0_1_dolly15k.csv",
    "mistral_nemo_12b":        "data/vllm_mistral_nemo_12b.csv",
    "deepseek-r1-llama-8b":    "data/vllm_deepseek_r1_distill_llama_8b_dolly15k.csv",
    "deepseek-r1-qwen-14b":    "data/vllm_deepseek_r1_qwen_14b_mixed_prompts_v2.csv",
    "deepseek_r1_qwen_14b_dolly15k": "data/vllm_deepseek_r1_distill_qwen_14b_dolly15k.csv",
}

MIN_OUTPUT_TOKENS = 9
RANDOM_STATE      = 42


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering  (v13.8 — phrase-aware bigrams)
# ─────────────────────────────────────────────────────────────────────────────

def extract_v13_features(df: pd.DataFrame, vectorizer=None):
    prompt_col = next(
        (c for c in ["prompt_text", "prompt", "input_text"] if c in df.columns),
        "prompt",
    )
    prompts = df[prompt_col].astype(str).fillna("")

    feat_df = pd.DataFrame(index=df.index)
    feat_df["char_count"]     = prompts.str.len()
    feat_df["word_count"]     = prompts.str.split().str.len()
    feat_df["line_count"]     = prompts.str.count(r'\n')
    feat_df["clause_density"] = (
        prompts.str.count(r'[,;:]') / (feat_df["word_count"] + 1)
    )
    lower = prompts.str.lower()
    feat_df["has_code_block"] = lower.str.contains(r'```|\bdef\b|\bclass\b').astype(int)
    feat_df["is_question"]    = lower.str.contains(r'\?').astype(int)

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=150, ngram_range=(1, 2),
            stop_words="english", binary=True,
        )
        tfidf_matrix = vectorizer.fit_transform(prompts)
    else:
        tfidf_matrix = vectorizer.transform(prompts)

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=df.index,
    )
    return pd.concat([feat_df, tfidf_df], axis=1), vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="TAILOR Phase 2a: Classifier Training")
    ap.add_argument(
        "--datasets", default="mistral-7b-dolly,mistral-7b-mixed",
        help="Comma-separated dataset keys (or 'all'). "
             f"Available: {', '.join(AVAILABLE_DATASETS)}",
    )
    ap.add_argument(
        "--threshold", type=int, default=300,
        help="Token count boundary for SHORT/LONG labels",
    )
    ap.add_argument(
        "--classifiers", default="xgb,cat,rf",
        help="Comma-separated list of classifiers to train: xgb, cat, rf",
    )
    ap.add_argument(
        "--deploy_dir", default="deploy",
        help="Directory to save trained models",
    )
    args = ap.parse_args()

    # Resolve dataset keys
    keys = (
        list(AVAILABLE_DATASETS)
        if args.datasets.strip().lower() == "all"
        else [k.strip() for k in args.datasets.split(",")]
    )

    clf_keys = {c.strip().lower() for c in args.classifiers.split(",")}

    # Load data
    paths = [AVAILABLE_DATASETS[k] for k in keys if k in AVAILABLE_DATASETS and
             os.path.exists(AVAILABLE_DATASETS[k])]
    missing = [k for k in keys if k not in AVAILABLE_DATASETS]
    if missing:
        print(f"Warning: unknown dataset keys ignored: {missing}")

    if not paths:
        print("No valid data files found. Check data/ directory."); return

    print(f"Loading {len(paths)} dataset(s)...")
    raw_df = pd.concat(
        [pd.read_csv(p, low_memory=False) for p in paths],
        ignore_index=True,
    )

    before = len(raw_df)
    df = raw_df[raw_df["output_tokens"] >= MIN_OUTPUT_TOKENS].copy()
    print(f"Filtered {before - len(df)} noise samples. Using {len(df)} rows.")
    print(f"Token threshold: {args.threshold}  →  "
          f"SHORT={int((df['output_tokens'] <= args.threshold).sum())}  "
          f"LONG={int((df['output_tokens'] > args.threshold).sum())}")

    # Features + labels
    X, vec = extract_v13_features(df)
    y = (df["output_tokens"] > args.threshold).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )

    os.makedirs(args.deploy_dir, exist_ok=True)
    joblib.dump(vec, f"{args.deploy_dir}/v13_vectorizer.pkl")
    print(f"Vectorizer saved → {args.deploy_dir}/v13_vectorizer.pkl\n")

    # Train
    clfs = {}
    if "xgb" in clf_keys:
        clfs["XGBoost"] = XGBClassifier(
            n_estimators=1000, learning_rate=0.02, max_depth=7,
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    if "cat" in clf_keys:
        clfs["CatBoost"] = CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=6,
            thread_count=-1, verbose=0, random_seed=RANDOM_STATE,
        )
    if "rf" in clf_keys:
        clfs["RandomForest"] = RandomForestClassifier(
            n_estimators=500, max_depth=15, n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    print(f"{'Model':<15} | {'Accuracy':<10} | {'F1 (macro)':<12}")
    print("─" * 42)

    feat_names = X.columns.tolist()
    for name, clf in clfs.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")
        print(f"{name:<15} | {acc:<10.4f} | {f1:<12.4f}")

        # Top-10 features
        importances = clf.feature_importances_
        top10 = np.argsort(importances)[::-1][:10]
        print(f"  Top-10 features for {name}:")
        for rank, idx in enumerate(top10, 1):
            print(f"    {rank:2}. {feat_names[idx]:<22} {importances[idx]:.4f}")
        print("─" * 42)

        # Save
        out_path = {
            "XGBoost":      f"{args.deploy_dir}/v13_xgb.json",
            "CatBoost":     f"{args.deploy_dir}/v13_cat.cbm",
            "RandomForest": f"{args.deploy_dir}/v13_rf.pkl",
        }[name]

        if name == "XGBoost":
            clf.save_model(out_path)
        elif name == "CatBoost":
            clf.save_model(out_path)
        else:
            joblib.dump(clf, out_path)
        print(f"  Saved → {out_path}\n")

    print("All models saved successfully.")


if __name__ == "__main__":
    main()
