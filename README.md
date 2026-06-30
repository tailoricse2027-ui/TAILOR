# TAILOR: Tail-Aware Inference with Length-Oriented Routing

TAILOR is an adaptive LLM serving system that reduces tail latency (TTFT P95/P99 and E2E P95) via XGBoost-based request classification and MAPE-K-controlled KV-aware routing across two specialised vLLM workers.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the interactive pipeline UI
python tailor_ui.py
```

---

## Repository Structure

```
tailor/
│
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── tailor_ui.py                       ← Interactive pipeline launcher (start here)
│
├── phase1_corpus/
│   ├── run_inference_vllm.py          ← Corpus creation via LLM inference
│   └── feature_extractor.py           ← Shared prompt feature extraction (v13)
│
├── phase2_training/
│   ├── train_2class_advancedfeatures.py  ← Classifier training (XGB/Cat/RF)
│   └── knee_new.py                    ← Concurrency profiling (M_Seq sweep)
│
├── phase3_evaluation/
│   ├── rq4_evaluation.py              ← Main evaluation (rate sweep)
│   └── sensitivity_analysis.py        ← Routing parameter sensitivity study
│
├── load_predictor/                    ← PreServe mLSTM baseline (adapted)
│   └── predictor.py
│
├── deploy/                            ← Pre-trained artifacts (provided)
│   ├── v13_xgb.json                   ← XGBoost classifier
│   ├── v13_cat.cbm                    ← CatBoost classifier
│   ├── v13_rf.pkl                     ← RandomForest classifier
│   ├── v13_vectorizer.pkl             ← TF-IDF vectorizer
│   └── 1_claases-1_prompt-1_resample.pth  ← PreServe mLSTM weights
│
├── data/                              ← Corpus CSV files (one per model)
│   ├── vllm_mistral_7b_v0_2_dolly15k.csv
│   ├── vllm_mistral_7b_v0_2_mixed_prompts_v2.csv
│   └── ...                            ← See "Datasets" section below
│
└── instance_configs.json              ← Profiled M_Seq configs (Phase 2b output)
```

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1 (Optional)   Corpus Creation                          │
│  phase1_corpus/run_inference_vllm.py                           │
│  Runs LLM inference on Dolly-15k and writes per-prompt         │
│  features + output token counts to CSV.                        │
│  → Output: data/vllm_<model>_dolly15k.csv                     │
│  Skip if pre-built datasets are present in data/               │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌──────────────────────┐  ┌─────────────────────────────────────┐
│  Phase 2a            │  │  Phase 2b                           │
│  Classifier Training │  │  Concurrency Profiling              │
│  phase2_training/    │  │  phase2_training/knee_new.py        │
│  train_2class_...py  │  │  Sweeps M_Seq on a single GPU to   │
│  XGBoost / CatBoost  │  │  find the throughput knee point     │
│  / RandomForest      │  │  per request class.                 │
│  → deploy/v13_*.json │  │  → instance_configs.json            │
└──────────┬───────────┘  └──────────────┬──────────────────────┘
           └──────────────┬──────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3              Evaluation                               │
│  phase3_evaluation/rq4_evaluation.py                           │
│  Runs Static-RR, TAILOR (CFG1/CFG2/CFG3), and PreServe        │
│  across arrival rates 8–72 req/s on two GPUs.                 │
│  → Output: rq3_results_sweep.csv                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Supported Models

| # | Name | HuggingFace ID | Token Threshold | Notes |
|---|------|---------------|-----------------|-------|
| 1 | Llama-2-7B-AWQ | `TheBloke/Llama-2-7B-Chat-AWQ` | 200 | |
| 2 | Llama-2-13B | `meta-llama/Llama-2-13b-chat-hf` | 200 | 🔒 Gated — requires HF token + license acceptance |
| 3 | Mistral-7B-v0.1 | `mistralai/Mistral-7B-Instruct-v0.1` | 300 | |
| 4 | Mistral-7B-v0.2 | `mistralai/Mistral-7B-Instruct-v0.2` | 300 | |
| 5 | Mistral-Nemo-12B | `mistralai/Mistral-Nemo-Instruct-2407` | 300 | |
| 6 | DeepSeek-R1-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 1000 | |
| 7 | DeepSeek-R1-Qwen-14B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 600 | |

> **🔒 Gated models** require a HuggingFace account token and acceptance of the model's license on its HF page. The TUI will prompt for the token automatically when a gated model is selected. Tokens can be created at https://huggingface.co/settings/tokens.

> **Note on GPU compatibility:** AWQ quantization (Llama-2-7B-AWQ) requires GPU compute capability ≥ 7.5 (Turing or newer). It will not run on V100 (compute 7.0). All other models run on V100 with `dtype=float16`.

---

## Hardware Requirements

- **GPUs**: 2× NVIDIA V100/A100 32GB (Phase 3 requires two GPUs)
- **CUDA**: 11.8 or higher
- **RAM**: 64GB recommended
- **Disk**: ~200GB for all model weights + datasets

---

## Installation

```bash
pip install -r requirements.txt
```

For AWQ models (Llama-2-7B-AWQ):
```bash
pip install autoawq
```

> **Important:** Do **not** install the old `pynvml` package. Install `nvidia-ml-py` instead.
> If both are installed, `pynvml` takes precedence and causes errors.

---

## Usage

### Option A — Interactive TUI (recommended)

```bash
python tailor_ui.py
```

The TUI provides a numbered menu, handles model selection, dataset picking, HF token prompting for gated models, and constructs all commands automatically. Phases 1 and 2a are marked **Optional** and can be skipped if pre-built artifacts in `deploy/` and `data/` are present.

### Option B — Direct CLI

#### Phase 1: Corpus Creation *(skip if using provided data)*

```bash
python phase1_corpus/run_inference_vllm.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.1 \
    --dataset_name databricks/databricks-dolly-15k \
    --num_prompts 5000 \
    --out_csv data/vllm_mistral_7b_dolly15k.csv
```

#### Phase 2a: Classifier Training

```bash
python phase2_training/train_2class_advancedfeatures.py \
    --datasets mistral-7b-dolly,mistral-7b-mixed \
    --threshold 300 \
    --classifiers xgb,cat,rf
```

#### Phase 2b: Concurrency Profiling

```bash
python phase2_training/knee_new.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --threshold 300 \
    --gpu_gib 32 \
    --weight_gib 13.5 \
    --short_configs 128,180,220,256,300 \
    --long_configs 96,128,140,180,220
```

#### Phase 3: Evaluation

```bash
# Full sweep (all rates) — produces rq3_results_sweep.csv
python phase3_evaluation/rq4_evaluation.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --rate-start 8 --rate-step 16 --rate-end 72

# Single rate (quick check)
python phase3_evaluation/rq4_evaluation.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --rate 56.0
```

#### Sensitivity Analysis

```bash
python phase3_evaluation/sensitivity_analysis.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --reps 3

# Single parameter
python phase3_evaluation/sensitivity_analysis.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --param delta_kv --reps 1
```

---

## Reproducing Paper Results

Pre-trained artifacts in `deploy/` allow Phase 3 to be run directly:

```bash
# Verify artifacts exist
ls deploy/v13_xgb.json deploy/v13_vectorizer.pkl instance_configs.json

# Run evaluation
python phase3_evaluation/rq4_evaluation.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --rate-start 8 --rate-step 16 --rate-end 72
```

Results are written to `rq3_results_sweep.csv` with columns:
`Rate, Method, TTFT_P95_ALL, TTFT_P99_ALL, E2E_P95_ALL, TPS, Total_S, KV_Avg, KV_Peak, Preemptions`.

---

## Datasets

Pre-built corpus CSVs are provided in `data/`. Each file corresponds to one model and contains per-prompt features and output token counts.

| File | Model |
|------|-------|
| `vllm_mistral_7b_v0_2_dolly15k.csv` | Mistral-7B-v0.2 |
| `vllm_mistral_7b_v0_2_mixed_prompts_v2.csv` | Mistral-7B-v0.2 (mixed) |
| `vllm_llama2_7b_awqmarlin_dolly15k.csv` | Llama-2-7B-AWQ |
| `vllm_llama2_13b_dolly15k.csv` | Llama-2-13B |
| `vllm_mistral_nemo_12b.csv` | Mistral-Nemo-12B |
| `vllm_deepseek_r1_distill_llama_8b_dolly15k.csv` | DeepSeek-R1-Llama-8B |
| `vllm_deepseek_r1_distill_qwen_14b_dolly15k.csv` | DeepSeek-R1-Qwen-14B |

---

## Expected Outputs

| File | Phase | Description |
|------|-------|-------------|
| `data/vllm_*.csv` | 1 | Corpus: per-prompt features + output tokens |
| `deploy/v13_xgb.json` | 2a | XGBoost classifier |
| `deploy/v13_vectorizer.pkl` | 2a | TF-IDF vectorizer |
| `instance_configs.json` | 2b | Optimal M_Seq per class |
| `rq3_results_sweep.csv` | 3 | Latency + throughput per method per rate |
| `sensitivity_results.csv` | — | Per-parameter sweep data |
| `sensitivity_summary.csv` | — | Mean ± std per parameter value |

---

## Expected Runtime

| Phase | Estimated Time |
|-------|---------------|
| Phase 1 — 5000 prompts, Mistral-7B | ~2–3 hours |
| Phase 2a — XGB + Cat + RF | ~5–10 minutes |
| Phase 2b — M_Seq sweep (5 configs each class) | ~1–2 hours |
| Phase 3 — 5 rates × 5 methods | ~6–8 hours |
| Sensitivity — 1 parameter, 1 rep | ~1.5 hours |

---

## Notes

- The PreServe baseline requires `deploy/1_claases-1_prompt-1_resample.pth`.
  If absent, the PRESERVE condition is automatically skipped.
- GPU SM monitoring requires `nvidia-ml-py`. If unavailable, work stealing
  is disabled and other methods run normally.
- All IPC temp files are written to `/tmp/rq3_*` and cleaned between runs.
- Results are saved incrementally after each rate point to prevent data loss on interruption.
- `PROJECT_ROOT` is derived automatically from the script location — no hardcoded paths. The repo can be cloned anywhere.
- The `--model` flag is required for Phases 2b, 3, and Sensitivity Analysis when running directly via CLI. The TUI handles this automatically.
