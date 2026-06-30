#!/usr/bin/env python3
"""
run_inference_vllm.py  —  Phase 1: Corpus Creation
===================================================
Runs inference with a selected LLM and writes per-prompt features + metrics
to a CSV file used for classifier training (Phase 2a).

Supported models
----------------
  1  TheBloke/Llama-2-7B-Chat-AWQ          (AWQ quantized)
  2  meta-llama/Llama-2-13b-chat-hf
  3  mistralai/Mistral-7B-Instruct-v0.1
  4  mistralai/Mistral-7B-Instruct-v0.2
  5  mistralai/Mistral-Nemo-Instruct-2407
  6  deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  7  deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

Usage (via TUI)
---------------
  python tailor_ui.py   →  Phase 1

Usage (direct)
--------------
  python run_inference_vllm.py \
      --model_name mistralai/Mistral-7B-Instruct-v0.1 \
      --dataset_name databricks/databricks-dolly-15k \
      --num_prompts 5000 \
      --out_csv data/vllm_mistral_7b_dolly15k.csv
"""

import os, sys, csv, json, time, uuid, argparse, random
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
from feature_extractor import extract_prompt_features  # noqa


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "TheBloke/Llama-2-7B-Chat-AWQ":              {"quant": "awq",  "ctx": 4096},
    "meta-llama/Llama-2-13b-chat-hf":            {"quant": None,   "ctx": 4096},
    "mistralai/Mistral-7B-Instruct-v0.1":        {"quant": None,   "ctx": 4096},
    "mistralai/Mistral-7B-Instruct-v0.2":        {"quant": None,   "ctx": 4096},
    "mistralai/Mistral-Nemo-Instruct-2407":      {"quant": None,   "ctx": 4096},
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {"quant": None,   "ctx": 4096},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {"quant": None,   "ctx": 4096},
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_and_sample_dolly(dataset_name: str, num_prompts: int) -> List[Dict]:
    print(f"Loading {dataset_name}...")
    ds: Dataset = load_dataset(dataset_name, split="train")
    domains = ds.unique("category")
    samples_per_domain = num_prompts // len(domains)
    remainder = num_prompts % len(domains)
    rows = []
    for idx, domain in enumerate(domains):
        k = samples_per_domain + (1 if idx < remainder else 0)
        domain_ds = ds.filter(lambda x, d=domain: x["category"] == d)
        k = min(k, len(domain_ds))
        if k > 0:
            for item in domain_ds.select(
                random.sample(range(len(domain_ds)), k)
            ).to_list():
                text = item["instruction"]
                if item.get("context"):
                    text = f"{item['context']}\n\n{item['instruction']}"
                rows.append({
                    "prompt_id":   f"dolly_{domain}_{uuid.uuid4().hex[:8]}",
                    "prompt_text": text,
                    "domain":      item["category"],
                    "source":      dataset_name,
                })
    print(f"Sampled {len(rows)} prompts across {len(domains)} domains.")
    return rows


def read_prompts_csv(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def write_batch(path, rows, header):
    if not rows:
        return
    mode = "w" if header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           escapechar="\\", quoting=csv.QUOTE_MINIMAL)
        if header:
            w.writeheader()
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Row builder
# ─────────────────────────────────────────────────────────────────────────────

def build_row(prompt_id, prompt, domain, source, gen, cfg):
    feats = extract_prompt_features(prompt)
    return {
        "sample_id":     str(uuid.uuid4()),
        "prompt_id":     prompt_id,
        "prompt_text":   prompt,
        "domain":        domain or "unknown",
        "source":        source or "synthetic",
        "temperature":   cfg.get("temperature", 0.7),
        "top_p":         cfg.get("top_p", 0.95),
        "max_new_tokens": 0,
        "natural_stop":  bool(cfg.get("natural_stop", False)),
        "ctx_cap":       cfg.get("ctx_cap", 4096),
        "engine":        "vllm",
        "model_name":    cfg.get("model_name", "unknown"),
        "features_json": json.dumps(feats, separators=(",", ":")),
        **gen,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chunked iterator
# ─────────────────────────────────────────────────────────────────────────────

def chunked(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf; buf = []
    if buf:
        yield buf


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="TAILOR Phase 1: Corpus Creation")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompts_csv",   help="Path to input CSV with prompts")
    src.add_argument("--dataset_name",  help="HuggingFace dataset name")

    ap.add_argument("--model_name",   required=True,
                    choices=list(MODEL_REGISTRY),
                    help="HuggingFace model ID")
    ap.add_argument("--out_csv",      required=True,  help="Output CSV path")
    ap.add_argument("--num_prompts",  type=int,       help="Prompts to sample (dataset only)")
    ap.add_argument("--batch_size",   type=int, default=16)
    ap.add_argument("--natural_stop", action="store_true")
    ap.add_argument("--temperature",  type=float, default=0.7)
    ap.add_argument("--top_p",        type=float, default=0.95)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    # Load prompts
    if args.prompts_csv:
        rows_in = read_prompts_csv(args.prompts_csv)
    else:
        if args.num_prompts is None:
            ap.error("--num_prompts required with --dataset_name")
        name = args.dataset_name
        if "dolly" in name.lower():
            name = "databricks/databricks-dolly-15k"
        rows_in = load_and_sample_dolly(name, args.num_prompts)

    if not rows_in:
        print("No prompts loaded."); sys.exit(1)

    # Model config
    minfo = MODEL_REGISTRY[args.model_name]
    ctx   = minfo["ctx"]
    quant = minfo["quant"]

    print(f"Loading model: {args.model_name}")
    llm_kwargs = dict(
        model=args.model_name,
        dtype="float16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=ctx,
        trust_remote_code=True,
    )
    if quant:
        llm_kwargs["quantization"] = quant
    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()

    def input_len(p): return len(tok.encode(p))

    # Filter oversized prompts
    print(f"Filtering prompts > {ctx} tokens...")
    before = len(rows_in)
    rows_in = [r for r in rows_in if input_len(str(r["prompt_text"])) <= ctx]
    print(f"Dropped {before - len(rows_in)} long prompts. Using {len(rows_in)}.")

    cfg = dict(
        temperature=args.temperature, top_p=args.top_p,
        natural_stop=args.natural_stop, ctx_cap=ctx,
        model_name=args.model_name,
    )

    def batch_max_tokens(prompts):
        lens = [input_len(p) for p in prompts]
        return max(1, min(ctx - l - 1 for l in lens))

    if os.path.exists(args.out_csv):
        os.remove(args.out_csv)
    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    total = 0
    for batch_idx, batch in enumerate(chunked(rows_in, args.batch_size)):
        prompts   = [str(r["prompt_text"]) for r in batch]
        max_tokens = batch_max_tokens(prompts)

        sp = SamplingParams(
            temperature=0.0 if args.natural_stop else args.temperature,
            top_p=1.0 if args.natural_stop else args.top_p,
            max_tokens=max_tokens,
            stop_token_ids=[tok.eos_token_id] if tok.eos_token_id else None,
        )

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        e2e_ms = (time.perf_counter() - t0) * 1000.0

        batch_out = []
        for r_in, out in zip(batch, outputs):
            text = out.outputs[0].text if out.outputs else ""
            in_t  = getattr(out.metrics, "prompt_tokens",    None) or input_len(r_in["prompt_text"])
            out_t = getattr(out.metrics, "generated_tokens", None) or input_len(text)
            gen = {
                "response_text":   text,
                "input_tokens":    int(in_t),
                "output_tokens":   int(out_t),
                "ttft_ms":         None,
                "tpot_ms":         e2e_ms / max(1, int(out_t)),
                "e2e_latency_ms":  float(e2e_ms),
            }
            batch_out.append(build_row(
                prompt_id=str(r_in.get("prompt_id", uuid.uuid4())),
                prompt=str(r_in["prompt_text"]),
                domain=str(r_in.get("domain", "unknown")),
                source=str(r_in.get("source", "synthetic")),
                gen=gen, cfg=cfg,
            ))

        write_batch(args.out_csv, batch_out, header=(batch_idx == 0))
        total += len(batch_out)
        print(f"Batch {batch_idx+1}: {len(batch_out)} rows (total {total})")

    print(f"\nDone. Wrote {total} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
