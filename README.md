# TAILOR: Tail-Aware Inference with Length-Oriented Routing

TAILOR is an adaptive LLM serving system that reduces tail latency (TTFT P95/P99 and E2E P95) via XGBoost-based request classification and MAPE-K-controlled KV-aware routing across two specialised vLLM workers.

---

## Quick Start

```bash
# 1. Install dependencies
pip install vllm xgboost catboost scikit-learn transformers datasets torch rich

# 2. Launch the interactive pipeline UI
python tailor_ui.py
```

The TUI guides you through all phases interactively with model selection menus.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1 (Optional)   Corpus Creation                          │
│  run_inference_vllm.py                                         │
│  → Runs LLM inference on Dolly-15k                            │
│  → Output: data/vllm_<model>_dolly15k.csv                     │
│  [Skip if pre-built datasets are present in data/]            │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────────┐   ┌────────────────────────────────────────┐
│  Phase 2a           │   │  Phase 2b                              │
│  Classifier         │   │  Concurrency Profiling                 │
│  train_2class_      │   │  knee_new.py                          │
│  advancedfeatures.py│   │  → Sweeps M_Seq values per GPU        │
│  → XGBoost/Cat/RF   │   │  → Finds throughput knee point        │
│  → deploy/v13_*.json│   │  → Output: instance_configs.json      │
└─────────┬───────────┘   └──────────────┬─────────────────────────┘
          └──────────────┬───────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3              Evaluation                               │
│  rq3_eval_final.py                                             │
│  → Runs Static-RR, TAILOR (CFG1/CFG2/CFG3), PreServe          │
│  → Sweeps arrival rates 8–72 req/s                             │
│  → Output: rq3_results_sweep.csv                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
tailor/
├── tailor_ui.py                   ← Interactive pipeline launcher (start here)
├── run_inference_vllm.py          ← Phase 1: corpus creation
├── train_2class_advancedfeatures.py ← Phase 2a: classifier training
├── knee_new.py                    ← Phase 2b: concurrency profiling
├── rq3_eval_final.py              ← Phase 3: full evaluation
├── sensitivity_analysis.py        ← Supplementary: parameter sensitivity
├── feature_extractor.py           ← Shared feature extraction module
├── load_predictor/                ← PreServe mLSTM baseline (adapted)
│   └── predictor.py
├── deploy/                        ← Pre-trained artifacts
│   ├── v13_xgb.json               ← XGBoost classifier
│   ├── v13_vectorizer.pkl         ← TF-IDF vectorizer
│   └── 1_claases-1_prompt-1_resample.pth  ← PreServe mLSTM weights
├── data/                          ← Corpus CSV files (one per model)
│   ├── vllm_mistral_7b_v0_2_dolly15k.csv
│   └── ...
├── instance_configs.json          ← Profiled M_Seq configs (Phase 2b output)
└── README.md
```

---

## Supported Models

| # | Name | HuggingFace ID | Token Threshold |
|---|------|---------------|-----------------|
| 1 | Llama-2-7B-AWQ | `TheBloke/Llama-2-7B-Chat-AWQ` | 200 |
| 2 | Llama-2-13B | `meta-llama/Llama-2-13b-chat-hf` | 400 |
| 3 | Mistral-7B-v0.1 | `mistralai/Mistral-7B-Instruct-v0.1` | 300 |
| 4 | Mistral-7B-v0.2 | `mistralai/Mistral-7B-Instruct-v0.2` | 300 |
| 5 | Mistral-Nemo-12B | `mistralai/Mistral-Nemo-Instruct-2407` | 400 |
| 6 | DeepSeek-R1-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 1000 |
| 7 | DeepSeek-R1-Qwen-14B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 600 |

---

## Hardware Requirements

- **GPUs**: 2× NVIDIA V100/A100 32GB (evaluation requires two GPUs)
- **CUDA**: 11.8 or higher
- **RAM**: 64GB recommended
- **Disk**: ~200GB for all model weights + datasets

---

## Dependencies

```bash
pip install vllm==0.6.x
pip install xgboost catboost scikit-learn
pip install transformers datasets torch
pip install rich                          # TUI
pip install autoawq                       # AWQ models only
pip install nvidia-ml-py                  # GPU SM monitoring
```

> **Note:** For AWQ models (Llama-2-7B-AWQ), `autoawq` must be installed.
> For DeepSeek models, set `trust_remote_code=True` in engine args.

---

## Step-by-Step Instructions

### Option A — Interactive (recommended)

```bash
python tailor_ui.py
```

Select phases from the menu. The TUI handles all argument construction.

---

### Option B — Direct CLI

#### Phase 1: Corpus Creation (skip if using provided data)

```bash
python run_inference_vllm.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.1 \
    --dataset_name databricks/databricks-dolly-15k \
    --num_prompts 5000 \
    --out_csv data/vllm_mistral_7b_dolly15k.csv \
    --batch_size 16
```

#### Phase 2a: Classifier Training

```bash
# Train on one model's data
python train_2class_advancedfeatures.py \
    --datasets mistral-7b-dolly,mistral-7b-mixed \
    --threshold 300 \
    --classifiers xgb,cat,rf

# Train on all available datasets
python train_2class_advancedfeatures.py \
    --datasets all \
    --threshold 300
```

#### Phase 2b: Concurrency Profiling

```bash
python knee_new.py \
    --model mistralai/Mistral-7B-Instruct-v0.1 \
    --threshold 300 \
    --gpu_gib 32 \
    --weight_gib 13.5 \
    --short_configs 128,180,220,256,300 \
    --long_configs 96,128,140,180,220
```

Output: `instance_configs.json`

#### Phase 3: Evaluation

```bash
# Full rate sweep (recommended)
python rq3_eval_final.py \
    --rate-start 8 --rate-step 16 --rate-end 72

# Single rate (for testing)
python rq3_eval_final.py --rate 56.0
```

Output: `rq3_results_sweep.csv`

#### Sensitivity Analysis (supplementary)

```bash
# All parameters
python sensitivity_analysis.py --reps 3

# Single parameter
python sensitivity_analysis.py --param delta_kv --reps 1
```

---

## Expected Outputs

| File | Phase | Description |
|------|-------|-------------|
| `data/vllm_*.csv` | 1 | Per-prompt features + output tokens |
| `deploy/v13_xgb.json` | 2a | XGBoost classifier |
| `deploy/v13_vectorizer.pkl` | 2a | TF-IDF vectorizer |
| `instance_configs.json` | 2b | Optimal M_Seq per class |
| `rq3_results_sweep.csv` | 3 | Latency + throughput per method per rate |
| `sensitivity_results.csv` | — | Per-parameter sensitivity data |
| `sensitivity_summary.csv` | — | Mean ± std summary |

---

## Expected Runtime

| Phase | Duration |
|-------|----------|
| Phase 1 (5000 prompts, Mistral-7B) | ~2–3 hours |
| Phase 2a (XGB + Cat + RF) | ~5–10 minutes |
| Phase 2b (M_Seq sweep, 5 configs each) | ~1–2 hours |
| Phase 3 (5 rates × 5 methods) | ~6–8 hours |
| Sensitivity (1 param, 1 rep) | ~1.5 hours |

---

## Pre-built Artifacts

The `deploy/` directory contains pre-trained artifacts so that **Phase 1 and 2 can be skipped** to reproduce Phase 3 results directly:

- `deploy/v13_xgb.json` — XGBoost classifier trained on Mistral-7B corpus
- `deploy/v13_vectorizer.pkl` — matching TF-IDF vectorizer
- `instance_configs.json` — profiled M_Seq configs for Mistral-7B on V100 32GB

To reproduce with a different model, run Phases 1–2 for that model and update `MODEL_ID` in `rq3_eval_final.py`.

---

## Reproducing Paper Results

```bash
# 1. Ensure deploy/ artifacts are present
ls deploy/v13_xgb.json deploy/v13_vectorizer.pkl
ls instance_configs.json

# 2. Run evaluation
python rq3_eval_final.py --rate-start 8 --rate-step 16 --rate-end 72

# 3. Results are in rq3_results_sweep.csv
```

The CSV contains one row per (rate, method) combination with columns:
`Rate, Method, TTFT_P95_ALL, TTFT_P99_ALL, E2E_P95_ALL, TPS, Total_S, KV_Avg, KV_Peak, Preemptions`.

---

## Notes

- The PreServe baseline requires the mLSTM model weights at
  `deploy/1_claases-1_prompt-1_resample.pth`. If absent, the PRESERVE
  condition is automatically skipped and a warning is printed.
- GPU SM monitoring requires `nvidia-ml-py`. If unavailable, work stealing
  is disabled and a warning is printed (other methods still run normally).
- All IPC files are written to `/tmp/rq3_*` and cleaned between runs.
- Results are saved incrementally after each rate point to prevent data
  loss on interruption.
