"""
vllm_kv_benchmark.py
─────────────────────────────────────────────────────────────────────────────
Profiles SHORT and LONG request groups across M_Seq sweep.
Selects optimal config per group based on MAX TPS.
Writes instance_configs.json consumed by the main serving program.

Output file: instance_configs.json
    {
      "SHORT": {
        "optimal_m_seq":  512,
        "gpu_mem_util":   0.40,
        "tps":            2744.10,
        "tpot_ms":        0.36,
        "avg_latency_s":  24.1,
        "p95_latency_s":  24.1,
        "kv_peak_pct":    68.2,
        "kv_avg_pct":     47.3
      },
      "LONG": { ... }
    }
"""

import time
import gc
import re
import json
import logging
import torch
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer


# =============================================================================
# CONFIGURATION  ← edit these before running
# =============================================================================

MODEL_ID        = "TheBloke/Llama-2-7B-Chat-AWQ"
TOKEN_THRESHOLD = 200       # output-token boundary between SHORT and LONG

SHORT_CONFIGS   = [128, 200, 256, 300, 350]
LONG_CONFIGS    = [96, 128, 192, 256, 300]

# Single-GPU memory partitioning
GPU_GIB         = 40.0     # ← your GPU VRAM in GiB (40 / 80 for A100, 80 for H100)
WEIGHT_GIB      =  3.67    # model weights per instance  (from vLLM startup log)
GRAPH_GIB       =  0.54    # CUDA graphs per instance    (from vLLM startup log)
SAFETY_MARGIN   =  0.05    # fraction reserved for driver / OS
SHORT_KV_SHARE  =  0.40    # SHORT gets 40% of usable KV budget
LONG_KV_SHARE   =  0.60    # LONG  gets 60% of usable KV budget

OUTPUT_JSON     = "instance_configs.json"
OUTPUT_CSV      = "vllm_kv_saturation_analysis.csv"


# =============================================================================
# 1. DATASET
# =============================================================================

def get_large_test_pool(model_id: str, num_samples: int = 800) -> list[str]:
    print(f"Loading {num_samples} samples...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    gsm_ds       = load_dataset("gsm8k", "main", split="test")
    gsm8k        = list(gsm_ds.select(range(min(300, len(gsm_ds))))['question'])

    he_ds        = load_dataset("openai_humaneval", split="test")
    humaneval    = list(he_ds.select(range(min(160, len(he_ds))))['prompt'])

    chat_ds      = load_dataset("allenai/WildChat-1M", split="train", streaming=True).take(340)
    chat_prompts = [item['conversation'][0]['content'] for item in chat_ds]

    raw = gsm8k + humaneval + chat_prompts
    processed = [
        tokenizer.decode(
            tokenizer.encode(p, truncation=True, max_length=3500),
            skip_special_tokens=True
        )
        for p in raw
    ]
    return processed[:num_samples]


# =============================================================================
# 2. KV CAPTURE  (logger interception — works with vLLM V1 subprocess engine)
# =============================================================================

class KVCaptureHandler(logging.Handler):
    _PATTERN = re.compile(r'GPU KV cache usage:\s*([\d.]+)%')

    def __init__(self):
        super().__init__()
        self.samples: list[float] = []

    def emit(self, record: logging.LogRecord) -> None:
        m = self._PATTERN.search(record.getMessage())
        if m:
            self.samples.append(float(m.group(1)) / 100.0)


# =============================================================================
# 3. BENCHMARK ENGINE
# =============================================================================

def run_benchmark(prompts: list[str], m_seq: int, model_id: str, name: str = "Batch") -> dict:
    if not prompts:
        return {"tokens": 0, "time": 0.001, "lengths": [],
                "latencies": [], "kv_cache_util": 0.0, "kv_cache_avg": 0.0}

    print(f"\n>>> {name} | M_Seq={m_seq} | Prompts={len(prompts)}")

    llm = LLM(
        model=model_id,
        max_num_seqs=m_seq,
        gpu_memory_utilization=0.95,
        disable_log_stats=False,       # must be False — drives KV log lines
        enable_prefix_caching=False,   # disabled for clean benchmarking
    )
    sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)

    handler = KVCaptureHandler()
    for lg in ("vllm", "vllm.engine.async_llm_engine", "vllm.core.scheduler"):
        logging.getLogger(lg).addHandler(handler)

    t0      = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    duration = time.perf_counter() - t0

    for lg in ("vllm", "vllm.engine.async_llm_engine", "vllm.core.scheduler"):
        logging.getLogger(lg).removeHandler(handler)

    kv      = handler.samples
    peak_kv = float(max(kv))     if kv else 0.0
    avg_kv  = float(np.mean(kv)) if kv else 0.0

    print(f"   KV samples={len(kv)}  raw={[f'{v*100:.1f}%' for v in kv]}")
    print(f"   Avg KV={avg_kv*100:.2f}%  Peak KV={peak_kv*100:.2f}%")
    if not kv:
        print("   ⚠  No KV samples — run < 10 s or disable_log_stats=True")

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "tokens":        total_tokens,
        "time":          duration,
        "lengths":       [len(o.outputs[0].token_ids) for o in outputs],
        "latencies":     [duration for _ in outputs],
        "kv_cache_util": peak_kv,
        "kv_cache_avg":  avg_kv,
    }


# =============================================================================
# 4. GPU MEMORY PARTITIONING  (single-GPU two-instance)
# =============================================================================

def compute_gpu_mem_splits(
    total_gpu_gib:  float = GPU_GIB,
    weight_gib:     float = WEIGHT_GIB,
    graph_gib:      float = GRAPH_GIB,
    safety_margin:  float = SAFETY_MARGIN,
    short_kv_share: float = SHORT_KV_SHARE,
    long_kv_share:  float = LONG_KV_SHARE,
) -> tuple[float, float]:
    """
    Partitions GPU VRAM between two vLLM instances on a single GPU.
    Returns (short_gpu_mem_util, long_gpu_mem_util).
    Their sum leaves safety_margin free for driver / OS.
    """
    overhead  = (weight_gib + graph_gib) / total_gpu_gib   # per instance
    usable    = 1.0 - 2 * overhead - safety_margin

    if usable <= 0:
        raise RuntimeError(
            f"Not enough VRAM: weights+graphs consume "
            f"{2*(weight_gib+graph_gib):.2f} GiB on a {total_gpu_gib} GiB GPU."
        )

    short_util = round(overhead + usable * short_kv_share, 2)
    long_util  = round(overhead + usable * long_kv_share,  2)

    print(f"\n📐 Single-GPU partition ({total_gpu_gib:.0f} GiB)")
    print(f"   Per-instance overhead : {overhead*100:.1f}%  ({weight_gib+graph_gib:.2f} GiB)")
    print(f"   Usable KV budget      : {usable*100:.1f}%  ({usable*total_gpu_gib:.2f} GiB)")
    print(f"   SHORT gpu_mem_util    : {short_util}  ({short_util*total_gpu_gib:.2f} GiB)")
    print(f"   LONG  gpu_mem_util    : {long_util}  ({long_util*total_gpu_gib:.2f} GiB)")
    print(f"   Combined              : {short_util+long_util:.2f}  ({(short_util+long_util)*total_gpu_gib:.2f} GiB)")

    return short_util, long_util


# =============================================================================
# 5. BEST CONFIG SELECTION  (pure max TPS)
# =============================================================================

def select_best_config(gdf: pd.DataFrame, group_name: str) -> dict:
    """
    Picks the M_Seq with the highest TPS — no KV filtering.
    Returns a clean dict written to instance_configs.json.
    """
    best = gdf.loc[gdf['TPS'].idxmax()]

    print(f"\n📌 {group_name} → best config (max TPS)")
    print(f"   M_Seq         : {int(best['M_Seq'])}")
    print(f"   TPS           : {best['TPS']:.2f}")
    print(f"   TPOT_ms       : {best['TPOT_ms']:.2f}")
    print(f"   Avg Latency   : {best['Avg_Lat']:.2f} s")
    print(f"   P95 Latency   : {best['P95_Lat']:.2f} s")
    print(f"   KV_Peak       : {best['KV_Peak']:.2f}%")
    print(f"   KV_Avg        : {best['KV_Avg']:.2f}%")

    return {
        "group":         group_name,
        "optimal_m_seq": int(best['M_Seq']),
        "gpu_mem_util":  None,          # filled after partitioning
        "tps":           round(float(best['TPS']),     2),
        "tpot_ms":       round(float(best['TPOT_ms']), 2),
        "avg_latency_s": round(float(best['Avg_Lat']), 2),
        "p95_latency_s": round(float(best['P95_Lat']), 2),
        "kv_peak_pct":   round(float(best['KV_Peak']), 2),
        "kv_avg_pct":    round(float(best['KV_Avg']),  2),
    }


# =============================================================================
# 6. MAIN ANALYSIS
# =============================================================================

def analyze_800(single_gpu: bool = True, gpu_gib: float = GPU_GIB) -> dict:
    """
    Full sweep → selects best M_Seq per group → writes instance_configs.json.

    Args:
        single_gpu : True  → two instances share one GPU (memory partitioned)
                     False → instances on separate GPUs  (each gets 0.95)
        gpu_gib    : total VRAM of your GPU in GiB

    Returns:
        instance_configs dict (same content as instance_configs.json)
    """
    all_prompts = get_large_test_pool(MODEL_ID, 800)

    # ── Phase A: profiling pass ───────────────────────────────────────────────
    print("\n[PHASE A] Profiling (M_Seq=512)...")
    prof    = run_benchmark(all_prompts, 512, MODEL_ID, name="Profiling")
    short_p = [all_prompts[i] for i, l in enumerate(prof['lengths']) if l <= TOKEN_THRESHOLD]
    long_p  = [all_prompts[i] for i, l in enumerate(prof['lengths']) if l >  TOKEN_THRESHOLD]
    print(f"\nSplit → SHORT: {len(short_p)}  LONG: {len(long_p)}\n")

    summary_data = []

    # ── Phase B: SHORT sweep ──────────────────────────────────────────────────
    print(f"\n{'='*55}\n  SHORT sweep ({len(short_p)} samples)\n{'='*55}")
    for m in SHORT_CONFIGS:
        if m > len(short_p):
            print(f"  [SKIP] M_Seq={m} > {len(short_p)}")
            continue
        res  = run_benchmark(short_p, m, MODEL_ID, name="SHORT")
        tps  = res['tokens'] / res['time']          if res['time']   > 0 else 0.0
        tpot = (res['time'] / res['tokens']) * 1000 if res['tokens'] > 0 else 0.0
        summary_data.append({
            "Group":   "SHORT", "M_Seq": m, "Tokens": res['tokens'],
            "TPS":     tps,     "TPOT_ms": tpot,
            "Avg_Lat": float(np.mean(res['latencies'])),
            "P95_Lat": float(np.percentile(res['latencies'], 95)),
            "KV_Peak": res['kv_cache_util'] * 100.0,
            "KV_Avg":  res['kv_cache_avg']  * 100.0,
        })

    # ── Phase C: LONG sweep ───────────────────────────────────────────────────
    print(f"\n{'='*55}\n  LONG sweep ({len(long_p)} samples)\n{'='*55}")
    for m in LONG_CONFIGS:
        if m > len(long_p):
            print(f"  [SKIP] M_Seq={m} > {len(long_p)}")
            continue
        res  = run_benchmark(long_p, m, MODEL_ID, name="LONG")
        tps  = res['tokens'] / res['time']          if res['time']   > 0 else 0.0
        tpot = (res['time'] / res['tokens']) * 1000 if res['tokens'] > 0 else 0.0
        summary_data.append({
            "Group":   "LONG",  "M_Seq": m, "Tokens": res['tokens'],
            "TPS":     tps,     "TPOT_ms": tpot,
            "Avg_Lat": float(np.mean(res['latencies'])),
            "P95_Lat": float(np.percentile(res['latencies'], 95)),
            "KV_Peak": res['kv_cache_util'] * 100.0,
            "KV_Avg":  res['kv_cache_avg']  * 100.0,
        })

    df = pd.DataFrame(summary_data)
    df['TPS_Gain_%'] = df.groupby('Group')['TPS'].pct_change() * 100

    # ── Report table ──────────────────────────────────────────────────────────
    W = 165
    print("\n" + "=" * W)
    print(f"{'GROUP':<7} | {'M_SEQ':<7} | {'TPS':<10} | {'GAIN %':<10} | "
          f"{'TPOT(ms)':<10} | {'AVG_LAT':<10} | {'P95_LAT':<10} | "
          f"{'KV_Peak%':<10} | {'KV_Avg%':<10}")
    print("-" * W)
    for _, row in df.iterrows():
        gain = f"{row['TPS_Gain_%']:>8.1f}%" if not pd.isna(row['TPS_Gain_%']) else "   ---  "
        print(f"{row['Group']:<7} | {int(row['M_Seq']):<7} | {row['TPS']:<10.2f} | {gain} | "
              f"{row['TPOT_ms']:<10.2f} | {row['Avg_Lat']:<10.2f} | {row['P95_Lat']:<10.2f} | "
              f"{row['KV_Peak']:<10.2f} | {row['KV_Avg']:<10.2f}")
    print("=" * W)

    # ── Best config selection (max TPS) ───────────────────────────────────────
    print("\n" + "=" * 55)
    print("  BEST CONFIG SELECTION  (criterion: max TPS)")
    print("=" * 55)

    instance_configs = {}
    for group_name in ["SHORT", "LONG"]:
        gdf = df[df['Group'] == group_name].copy()
        if gdf.empty:
            continue
        instance_configs[group_name] = select_best_config(gdf, group_name)

    # ── Assign gpu_mem_util ───────────────────────────────────────────────────
    if single_gpu:
        short_mem, long_mem = compute_gpu_mem_splits(total_gpu_gib=gpu_gib)
    else:
        short_mem, long_mem = 0.95, 0.95
        print("\n📐 Multi-GPU mode: each instance gets gpu_mem_util=0.95")

    instance_configs["SHORT"]["gpu_mem_util"] = short_mem
    instance_configs["LONG"]["gpu_mem_util"]  = long_mem

    # ── Save outputs ──────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Raw sweep data     → {OUTPUT_CSV}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(instance_configs, f, indent=2)
    print(f"✅ Instance configs   → {OUTPUT_JSON}")
    print("\n--- instance_configs.json content ---")
    print(json.dumps(instance_configs, indent=2))

    return instance_configs


# =============================================================================
# 7. INSTANCE LAUNCHER  — import this in your main serving program
# =============================================================================

def launch_instances(
    config_path: str   = OUTPUT_JSON,
    model_id:    str   = MODEL_ID,
    single_gpu:  bool  = True,
    gpu_gib:     float = GPU_GIB,
) -> dict:
    """
    Reads instance_configs.json and returns two ready LLM handles.

    In your main serving program:
        from vllm_kv_benchmark import launch_instances
        instances = launch_instances()
        group     = catboost_model.predict(features)[0]   # "SHORT" or "LONG"
        result    = instances[group].generate([prompt], SamplingParams(...))
    """
    with open(config_path) as f:
        configs = json.load(f)

    if single_gpu:
        short_mem, long_mem = compute_gpu_mem_splits(total_gpu_gib=gpu_gib)
        configs["SHORT"]["gpu_mem_util"] = short_mem
        configs["LONG"]["gpu_mem_util"]  = long_mem
        print(f"\n⚡ Single-GPU | SHORT={short_mem} | LONG={long_mem} | "
              f"Total={short_mem+long_mem:.2f}")
    else:
        print(f"\n⚡ Multi-GPU  | SHORT={configs['SHORT']['gpu_mem_util']} | "
              f"LONG={configs['LONG']['gpu_mem_util']}")

    instances = {}

    # SHORT first — smaller allocation, reduces fragmentation risk
    sc = configs["SHORT"]
    print(f"\n🚀 SHORT | M_Seq={sc['optimal_m_seq']} | gpu_mem_util={sc['gpu_mem_util']}")
    instances["SHORT"] = LLM(
        model=model_id,
        max_num_seqs=sc['optimal_m_seq'],
        gpu_memory_utilization=sc['gpu_mem_util'],
        disable_log_stats=False,
        enable_prefix_caching=False,
    )
    print("   ✓ SHORT ready")

    # LONG second — uses remaining memory
    lc = configs["LONG"]
    print(f"\n🚀 LONG  | M_Seq={lc['optimal_m_seq']} | gpu_mem_util={lc['gpu_mem_util']}")
    instances["LONG"] = LLM(
        model=model_id,
        max_num_seqs=lc['optimal_m_seq'],
        gpu_memory_utilization=lc['gpu_mem_util'],
        disable_log_stats=False,
        enable_prefix_caching=False,
    )
    print("   ✓ LONG ready")

    return instances   # {"SHORT": LLM(...), "LONG": LLM(...)}


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    # Set single_gpu=True  → both instances share one GPU
    # Set single_gpu=False → each instance on a dedicated GPU
    # Set gpu_gib to your actual VRAM (40 for A100-40GB, 80 for A100-80GB / H100)
    analyze_800(single_gpu=True, gpu_gib=GPU_GIB)