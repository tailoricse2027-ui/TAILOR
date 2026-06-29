"""
rq3_evaluation_v4.py — Dual V100 + Rate Sweep + Real-Time KV Routing
─────────────────────────────────────────────────────────────────────────────
Fixes vs v3:
  1. max_model_len = 4096   (was 8192 → only 12 concurrent sequences on V100)
  2. prompt max_length = 2000  (2000 prompt + 2048 output = 4048 < 4096)
  3. max_tokens = 2048          (realistic output length)
 
     

V100 32GB KV budget at util=0.90:
  available KV = 32×0.90 − 16 (weights) = 12.8 GB
  KV per token (Llama3.1 8B GQA) = 128 KB
  KV token pool ≈ 102,400 tokens
  max concurrent (max_model_len=4096) ≈ 25 sequences   ← correct
  max concurrent (max_model_len=8192) ≈ 12 sequences   ← was broken

Usage:
    python3 rq3_evaluation_v4.py                     # sweep
    python3 rq3_evaluation_v4.py --rate 24.0         # single rate
    python3 rq3_evaluation_v4.py --rate-start 8 \\
        --rate-step 8 --rate-end 48
    python3 rq3_evaluation_v4.py --rate 24 --debug-kv
"""

import argparse
import asyncio
import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
import joblib

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# ── PreServe preserve baseline ────────────────────────────────────────────────
import sys as _sys
_PRESERVE_ROOT = "/workspace"
if _PRESERVE_ROOT not in _sys.path:
    _sys.path.insert(0, _PRESERVE_ROOT)

PRESERVE_MODEL_PATH = "/workspace/deploy/1_claases-1_prompt-1_resample.pth"  # ← update if filename differs

# ── GPU SM utilisation monitoring via pynvml ─────────────────────────────────
try:
    import pynvml as _nvml
    _GPU_HANDLES = {
        0: _nvml.nvmlDeviceGetHandleByIndex(0),
        1: _nvml.nvmlDeviceGetHandleByIndex(1),
    }
    def read_gpu_sm(gpu_id: int) -> int:
        """Return GPU SM utilisation (0-100). Falls back to 50 on error."""
        try:
            return _nvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLES[gpu_id]).gpu
        except Exception:
            return 50
    HAS_NVML = True
    print("✓ pynvml initialised — real-time SM monitoring enabled")
except Exception as _e:
    HAS_NVML = False
    def read_gpu_sm(gpu_id: int) -> int:
        return 50   # assume busy; work-stealing disabled
    print(f"⚠  pynvml not available ({_e}) — SM monitoring disabled")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = "/workspace"

MODEL_ID     = "mistralai/Mistral-7B-Instruct-v0.1"
VEC_PATH     = f"{PROJECT_ROOT}/deploy/v13_vectorizer.pkl"
XGB_PATH     = f"{PROJECT_ROOT}/deploy/v13_xgb.json"
INSTANCE_CFG = f"{PROJECT_ROOT}/instance_configs.json"

MODEL_THRESHOLDS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 750,
    "mistralai/Mistral-7B-Instruct-v0.1":     300,
    "mistralai/Mistral-7B-Instruct-v0.2":     300,
    "meta-llama/Llama-2-13b-chat-hf":         400,
    "mistralai/Mistral-NeMo-12B-Instruct":    400,
    "deepseek-ai/DeepSeek-V2-Lite-Chat":      250,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 1000,

}
TOKEN_THRESHOLD = MODEL_THRESHOLDS.get(MODEL_ID, 500)

GPU_A = 0
GPU_B = 1

UTIL_PER_GPU = 0.90

# ── Engine configuration ──────────────────────────────────────────────────────
MAX_MODEL_LEN    = 4096   # prompt(2000) + output(2048) = 4048 < 4096 ✓
PROMPT_MAX_LEN   = 2000   # truncate prompts to leave room for 2048 output tokens
MAX_OUTPUT_TOKENS = 2048  # realistic output length

# TCP ports for adaptive server mode
WORKER_PORTS = {GPU_A: 9200, GPU_B: 9201}
KV_STAT_DIR  = "/tmp/rq3_kv"

# ── Routing parameters ────────────────────────────────────────────────────────
# score = (active / m_seq) + kv
# Send to preferred (class-matched) server unless preferred_score > alt_score + CLASS_BONUS
CLASS_BONUS = 0.0    # pure load balance; increase toward 0.2 to add class preference

STATIC_M_SEQS   = [ 180,240]   # tuned for max_model_len=4096, max_tokens=2048
SINGLE_M_SEQS   = []

# ── Adaptive M_Seq configurations ────────────────────────────────────────────
# CFG1 = loaded from instance_configs.json (profiled optimal)
# CFG2 and CFG3 = user-defined overrides  ← edit these before running
ADAPTIVE_CFG2 = {"short_m_seq": 300, "long_m_seq": 180}
#ADAPTIVE_CFG3 = {"short_m_seq": 300, "long_m_seq": 200}
# Global params used by static/single baselines (max_tokens=2048, no class info)
SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,
    max_tokens=MAX_OUTPUT_TOKENS,
    min_tokens=1,
    stop=["<|eot_id|>", "<|end_of_text|>"],
)

# Per-class token budgets for ADAPTIVE mode
# Classifier predicts SHORT → cap at 512 tokens (fast decode, slots freed quickly)
# Classifier predicts LONG  → allow up to 2048 tokens (full output)
MAX_TOKENS_SHORT = 512
MAX_TOKENS_LONG  = 2048

def make_sampling_params(cls: str) -> SamplingParams:
    """Create per-request SamplingParams based on classifier prediction."""
    max_out = MAX_TOKENS_SHORT if cls == "SHORT" else MAX_TOKENS_LONG
    return SamplingParams(
        temperature=0.7,
        max_tokens=max_out,
        min_tokens=1,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )
TARGET_PER_TYPE = 400
RANDOM_SEED     = 42

BARRIER_DIR = "/tmp/rq3_barrier"
RESULT_DIR  = "/tmp/rq3_results"
PROMPTS_DIR = "/tmp/rq3_prompts"

VLLM_LOGGERS = (
    "vllm", "vllm.engine", "vllm.engine.llm_engine",
    "vllm.engine.async_llm_engine", "vllm.core.scheduler",
)


# =============================================================================
# 1. DATASET
# =============================================================================

def get_test_pool(model_id: str, target_per_type: int = TARGET_PER_TYPE) -> list[str]:
    print("Loading test pool...")
    tokenizer    = AutoTokenizer.from_pretrained(model_id)
    gsm_ds       = load_dataset("gsm8k", "main", split="test")
    gsm8k        = list(gsm_ds.select(range(min(target_per_type, len(gsm_ds))))['question'])
    he_ds        = load_dataset("openai_humaneval", split="test")
    humaneval    = list(he_ds.select(range(min(target_per_type, len(he_ds))))['prompt'])
    chat_ds      = load_dataset("allenai/WildChat-1M", split="train",
                                streaming=True).take(target_per_type)
    chat_prompts = [item['conversation'][0]['content'] for item in chat_ds]
    raw = gsm8k + humaneval + chat_prompts
    return [
        tokenizer.decode(
            tokenizer.encode(p, truncation=True, max_length=PROMPT_MAX_LEN),
            skip_special_tokens=True
        )
        for p in raw
    ]


# =============================================================================
# 2. FEATURE EXTRACTION  (matches training script v13.8)
# =============================================================================

def extract_v13_features(prompts_input, vectorizer=None):
    if isinstance(prompts_input, list):
        df_inner = pd.DataFrame({"prompt": prompts_input})
    else:
        df_inner = prompts_input
    prompt_col = next(
        (c for c in ["prompt_text", "prompt", "input_text"] if c in df_inner.columns),
        "prompt"
    )
    prompts = df_inner[prompt_col].astype(str).fillna("")

    feat_df = pd.DataFrame(index=df_inner.index)
    feat_df["char_count"]     = prompts.str.len()
    feat_df["word_count"]     = prompts.str.split().str.len()
    feat_df["line_count"]     = prompts.str.count(r'\n')
    feat_df["clause_density"] = prompts.str.count(r'[,;:]') / (feat_df["word_count"] + 1)
    lower = prompts.str.lower()
    feat_df["has_code_block"] = lower.str.contains(r'```|\bdef\b|\bclass\b').astype(int)
    feat_df["is_question"]    = lower.str.contains(r'\?').astype(int)

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=150, ngram_range=(1, 2),
            stop_words='english', binary=True
        )
        tfidf_matrix = vectorizer.fit_transform(prompts)
    else:
        tfidf_matrix = vectorizer.transform(prompts)

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=df_inner.index
    )
    return pd.concat([feat_df, tfidf_df], axis=1), vectorizer


# =============================================================================
# 3. ARRIVAL TIME GENERATION
# =============================================================================

def generate_poisson_arrivals(n: int, rate: float, seed: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.exponential(1.0 / rate, size=n))


# =============================================================================
# 4. KV STATS HANDLER
# =============================================================================

class StatsHandler(logging.Handler):
    _KV_PATTERNS = [
        re.compile(r'GPU KV cache usage:\s*([\d.]+)%',    re.IGNORECASE),
        re.compile(r'gpu_cache_usage_perc[^\d]*([\d.]+)', re.IGNORECASE),
        re.compile(r'KV cache usage[:\s]+([\d.]+)%',      re.IGNORECASE),
        re.compile(r'cache_usage.*?([\d.]+)%',            re.IGNORECASE),
        re.compile(r'Avg KV Cache.*?([\d.]+)%',           re.IGNORECASE),
    ]
    _PRE_PATTERN = re.compile(r'Preempted:\s*(\d+)\s*reqs', re.IGNORECASE)

    def __init__(self, debug: bool = False):
        super().__init__()
        self.kv_samples:     list[float] = []
        self.preempt_counts: list[int]   = []
        self.current_kv:     float       = 0.0
        self.debug = debug

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        if self.debug and any(k in msg.lower() for k in ('kv', 'cache', 'preempt', 'usage')):
            print(f"[KV-DEBUG] {msg}", flush=True)
        for pat in self._KV_PATTERNS:
            m = pat.search(msg)
            if m:
                val = float(m.group(1))
                v   = val / 100.0 if val > 1.0 else val
                self.kv_samples.append(v)
                self.current_kv = v
                break
        p = self._PRE_PATTERN.search(msg)
        if p:
            self.preempt_counts.append(int(p.group(1)))


# =============================================================================
# 5. FILE HELPERS
# =============================================================================

def write_prompts(run_id: str, role: str, prompts: list[str]) -> str:
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    path = os.path.join(PROMPTS_DIR, f"{run_id}_{role}.json")
    with open(path, 'w') as f:
        json.dump(prompts, f)
    return path


def write_arrivals(run_id: str, role: str, arrivals: np.ndarray) -> str:
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    path = os.path.join(PROMPTS_DIR, f"{run_id}_{role}_arrivals.json")
    with open(path, 'w') as f:
        json.dump(arrivals.tolist(), f)
    return path


def write_token_budgets(run_id: str, role: str, budgets: list[int]) -> str:
    """
    Write per-prompt max_tokens list so static/single workers apply the same
    token budget as the adaptive run. Each value is either MAX_TOKENS_SHORT or
    MAX_TOKENS_LONG based on the classifier prediction for that prompt.
    """
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    path = os.path.join(PROMPTS_DIR, f"{run_id}_{role}_budgets.json")
    with open(path, 'w') as f:
        json.dump(budgets, f)
    return path


def _save_worker_result(result_file, role, wall, results, handler):
    kv      = handler.kv_samples
    kv_avg  = float(np.mean(kv)) if kv else 0.0
    kv_peak = float(max(kv))     if kv else 0.0
    preempt = sum(handler.preempt_counts)
    tokens  = sum(r['tokens'] for r in results)
    ttfts   = [r['ttft'] for r in results]
    e2es    = [r['e2e']  for r in results]
    p95     = float(np.percentile(ttfts, 95)) if ttfts else 0.0
    print(f"[{role}] Done | n={len(results)} tokens={tokens} wall={wall:.0f}s "
          f"TTFT_P95={p95:.3f}s "
          f"KV_Avg={kv_avg*100:.1f}% KV_Peak={kv_peak*100:.1f}% "
          f"KV_samples={len(kv)} Preempt={preempt}")
    with open(result_file, 'w') as f:
        json.dump({
            "role": role, "wall": wall, "tokens": tokens,
            "kv_avg": kv_avg, "kv_peak": kv_peak, "preemptions": preempt,
            "ttfts": ttfts, "e2es": e2es,
        }, f)


# =============================================================================
# 6. WORKER
# =============================================================================

def _make_engine_args(m_seq: int, gpu_mem: float, role: str) -> AsyncEngineArgs:
    """
    Build AsyncEngineArgs.
    Note: max_num_batched_tokens omitted — vLLM 0.6.x requires it >= max_model_len.
    vLLM's default chunked prefill handles interleaving automatically.
    """
    return AsyncEngineArgs(
        model=MODEL_ID,
        #enforce_eager=True,   # ← disables CUDA graph capture entirely  deepseek
        max_num_seqs=m_seq,
        gpu_memory_utilization=gpu_mem,
        disable_log_stats=False,
        enable_prefix_caching=False,
        dtype="float16",
        max_model_len=MAX_MODEL_LEN,
        #trust_remote_code=True, #for deepseek
    )


async def worker_main(args):
    os.makedirs(BARRIER_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR,  exist_ok=True)

    run_id  = args.run_id
    role    = args.role
    m_seq   = args.m_seq
    gpu_mem = args.gpu_mem

    loaded_file = os.path.join(BARRIER_DIR, f"{run_id}_{role}_loaded")
    go_file     = os.path.join(BARRIER_DIR, f"{run_id}_go")
    result_file = os.path.join(RESULT_DIR,  f"{run_id}_{role}.json")
    visible     = os.environ.get("CUDA_VISIBLE_DEVICES", "all")

    os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    handler = StatsHandler(debug=args.debug_kv)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    for lg in VLLM_LOGGERS:
        l = logging.getLogger(lg)
        l.setLevel(logging.INFO)
        l.propagate = True

    engine_args = _make_engine_args(m_seq, gpu_mem, role)

    if args.server_mode:
        await _worker_server_mode(args, engine_args, handler,
                                  loaded_file, go_file, result_file, visible)
        root_logger.removeHandler(handler)
        return

    # ── Pre-assignment mode (static / single) ─────────────────────────────────
    prompts  = json.load(open(args.prompts_file))
    arrivals = np.array(json.load(open(args.arrivals_file)))

    # Per-prompt token budgets — set by caller when using classifier-matched budgets.
    # Falls back to global SAMPLING_PARAMS (max_tokens=2048) if file absent.
    budgets: list[int] | None = None
    if args.budgets_file and os.path.exists(args.budgets_file):
        budgets = json.load(open(args.budgets_file))

    budget_label = f"per-prompt ({MAX_TOKENS_SHORT}/{MAX_TOKENS_LONG})" if budgets else str(MAX_OUTPUT_TOKENS)
    print(f"[{role}] Loading  M_Seq={m_seq}  GPU={visible}  n={len(prompts)}"
          f"  max_model_len={MAX_MODEL_LEN}  max_tokens={budget_label}")

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    open(loaded_file, 'w').close()
    print(f"[{role}] Loaded ✓ — waiting for GO...")
    while not os.path.exists(go_file):
        await asyncio.sleep(0.05)

    print(f"[{role}] GO — serving {len(prompts)} requests")
    wall_start = time.perf_counter()

    async def serve_request(prompt, arrival_offset, req_id, max_tok: int | None = None):
        elapsed = time.perf_counter() - wall_start
        if (s := arrival_offset - elapsed) > 0:
            await asyncio.sleep(s)
        sp = (SamplingParams(temperature=0.7, max_tokens=max_tok, min_tokens=1,
                             stop=["<|eot_id|>", "<|end_of_text|>"])
              if max_tok is not None else SAMPLING_PARAMS)
        submit_t = time.perf_counter()
        ttft, last_out = None, None
        try:
            async for out in engine.generate(prompt, sp, req_id):
                if ttft is None and out.outputs[0].token_ids:
                    ttft = time.perf_counter() - submit_t
                last_out = out
        except Exception as e:
            print(f"[{role}] {req_id}: {e}")
        e2e    = time.perf_counter() - submit_t
        tokens = len(last_out.outputs[0].token_ids) if last_out and last_out.outputs else 0
        return {"ttft": ttft or e2e, "e2e": e2e, "tokens": tokens}

    results = await asyncio.gather(
        *[serve_request(prompts[i], float(arrivals[i]), f"{role}-{i}",
                        budgets[i] if budgets else None)
          for i in range(len(prompts))],
        return_exceptions=True
    )
    results = [r for r in results if not isinstance(r, Exception)]
    wall    = time.perf_counter() - wall_start

    _save_worker_result(result_file, role, wall, results, handler)
    root_logger.removeHandler(handler)
    del engine
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# =============================================================================
# 7. ADAPTIVE SERVER MODE
# =============================================================================

async def _worker_server_mode(args, engine_args, handler,
                               loaded_file, go_file, result_file, visible):
    role   = args.role
    m_seq  = args.m_seq
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    port   = WORKER_PORTS[gpu_id]
    done_f = os.path.join(BARRIER_DIR, f"{args.run_id}_{role}_done")

    os.makedirs(KV_STAT_DIR, exist_ok=True)
    kv_stat_file = os.path.join(KV_STAT_DIR, f"{role}.json")

    print(f"[{role}] Server mode  M_Seq={m_seq}  GPU={visible}  port={port}"
          f"  max_model_len={MAX_MODEL_LEN}")

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    active_requests = 0
    results_store:  list[dict] = []
    stop_broadcast  = asyncio.Event()

    async def kv_broadcast():
        while not stop_broadcast.is_set():
            try:
                with open(kv_stat_file, 'w') as f:
                    json.dump({
                        "kv":     handler.current_kv,
                        "active": active_requests,
                        "m_seq":  m_seq,
                        "ts":     time.time(),
                        "role":   role,
                    }, f)
            except Exception:
                pass
            await asyncio.sleep(0.3)

    asyncio.create_task(kv_broadcast())

    open(loaded_file, 'w').close()
    print(f"[{role}] Loaded ✓ — waiting for GO...")
    while not os.path.exists(go_file):
        await asyncio.sleep(0.05)
    print(f"[{role}] GO — TCP on 127.0.0.1:{port}")

    wall_start = time.perf_counter()

    async def handle_client(reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter):
        nonlocal active_requests
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                req = json.loads(line.decode())
                if req.get("cmd") == "shutdown":
                    writer.write(b'{"ok":true}\n')
                    await writer.drain()
                    break
                prompt, req_id = req["prompt"], req["req_id"]
                if "max_tokens" in req:  # preserve dispatcher sends exact token budget
                    sp = SamplingParams(temperature=0.7, max_tokens=int(req["max_tokens"]),
                                        min_tokens=1, stop=["<|eot_id|>", "<|end_of_text|>"])
                else:
                    cls = req.get("cls", "LONG")
                    sp  = make_sampling_params(cls) if cls in ("SHORT","LONG") else SAMPLING_PARAMS
                active_requests += 1
                submit_t = time.perf_counter()
                ttft, last_out = None, None
                try:
                    async for out in engine.generate(prompt, sp, req_id):
                        if ttft is None and out.outputs[0].token_ids:
                            ttft = time.perf_counter() - submit_t
                        last_out = out
                except Exception as e:
                    print(f"[{role}] {req_id}: {e}")
                finally:
                    active_requests = max(0, active_requests - 1)
                e2e    = time.perf_counter() - submit_t
                tokens = (len(last_out.outputs[0].token_ids)
                          if last_out and last_out.outputs else 0)
                result = {"ttft": ttft or e2e, "e2e": e2e,
                          "tokens": tokens, "role": role}
                results_store.append(result)
                writer.write((json.dumps(result) + "\n").encode())
                await writer.drain()
        except Exception as e:
            print(f"[{role}] client error: {e}")
        finally:
            writer.close()

    server = await asyncio.start_server(handle_client, "127.0.0.1", port)
    async with server:
        while not os.path.exists(done_f):
            await asyncio.sleep(0.2)

    stop_broadcast.set()
    wall = time.perf_counter() - wall_start
    _save_worker_result(result_file, role, wall, results_store, handler)
    del engine
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# =============================================================================
# 8. SUBPROCESS HELPERS
# =============================================================================

def _worker_env(gpu_id: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
    return env


def _make_worker_cmd(role, run_id, m_seq, gpu_mem, pfile, afile,
                     debug_kv, server_mode=False, bfile=None):
    cmd = [
        sys.executable, __file__,
        "--worker",
        "--role",          role,
        "--run-id",        run_id,
        "--m-seq",         str(m_seq),
        "--gpu-mem",       str(gpu_mem),
        "--prompts-file",  pfile or "/dev/null",
        "--arrivals-file", afile or "/dev/null",
        "--budgets-file",  bfile or "",
    ]
    if debug_kv:    cmd.append("--debug-kv")
    if server_mode: cmd.append("--server-mode")
    return cmd


# =============================================================================
# 9. STATIC / SINGLE LAUNCHERS
# =============================================================================

def launch_parallel(
    run_id, role_a, prompts_a, arrivals_a, m_seq_a, gpu_a,
    role_b, prompts_b, arrivals_b, m_seq_b, gpu_b, util, debug_kv=False,
    budgets_a=None, budgets_b=None,
):
    os.makedirs(BARRIER_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR,  exist_ok=True)

    loaded_a = os.path.join(BARRIER_DIR, f"{run_id}_{role_a}_loaded")
    loaded_b = os.path.join(BARRIER_DIR, f"{run_id}_{role_b}_loaded")
    go_file  = os.path.join(BARRIER_DIR, f"{run_id}_go")
    res_a    = os.path.join(RESULT_DIR,  f"{run_id}_{role_a}.json")
    res_b    = os.path.join(RESULT_DIR,  f"{run_id}_{role_b}.json")

    for f in [loaded_a, loaded_b, go_file, res_a, res_b]:
        if os.path.exists(f): os.remove(f)

    pfile_a = write_prompts(run_id, role_a, prompts_a)
    pfile_b = write_prompts(run_id, role_b, prompts_b)
    afile_a = write_arrivals(run_id, role_a, arrivals_a)
    afile_b = write_arrivals(run_id, role_b, arrivals_b)
    bfile_a = write_token_budgets(run_id, role_a, budgets_a) if budgets_a else None
    bfile_b = write_token_budgets(run_id, role_b, budgets_b) if budgets_b else None

    print(f"\n  ▶ {role_a} (GPU={gpu_a}, M_Seq={m_seq_a}, util={util})")
    proc_a = subprocess.Popen(
        _make_worker_cmd(role_a, run_id, m_seq_a, util, pfile_a, afile_a, debug_kv, bfile=bfile_a),
        env=_worker_env(gpu_a))
    t0 = time.time()
    while not os.path.exists(loaded_a):
        if time.time()-t0 > 300: proc_a.kill(); raise TimeoutError(role_a)
        if proc_a.poll() is not None: raise RuntimeError(f"{role_a} died")
        time.sleep(0.2)
    print(f"  ✓ {role_a} loaded")

    print(f"\n  ▶ {role_b} (GPU={gpu_b}, M_Seq={m_seq_b}, util={util})")
    proc_b = subprocess.Popen(
        _make_worker_cmd(role_b, run_id, m_seq_b, util, pfile_b, afile_b, debug_kv, bfile=bfile_b),
        env=_worker_env(gpu_b))
    t0 = time.time()
    while not os.path.exists(loaded_b):
        if time.time()-t0 > 300: proc_a.kill(); proc_b.kill(); raise TimeoutError(role_b)
        if proc_b.poll() is not None: proc_a.kill(); raise RuntimeError(f"{role_b} died")
        time.sleep(0.2)
    print(f"  ✓ {role_b} loaded — both ready")

    dt = time.perf_counter()
    open(go_file, 'w').close()
    print(f"  ✓ GO at {time.strftime('%H:%M:%S')}")
    proc_a.wait(); proc_b.wait()
    wc = time.perf_counter() - dt

    if proc_a.returncode or proc_b.returncode:
        raise RuntimeError(f"Worker failed: {role_a}={proc_a.returncode}"
                           f" {role_b}={proc_b.returncode}")

    with open(res_a) as f: ra = json.load(f)
    with open(res_b) as f: rb = json.load(f)
    print(f"\n   ⏱  {wc:.1f}s  ({role_a}={ra['wall']:.1f}s | {role_b}={rb['wall']:.1f}s)")
    return ra, rb


def launch_single(run_id, role, prompts, arrivals, m_seq, gpu, util, debug_kv=False, budgets=None):
    os.makedirs(BARRIER_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR,  exist_ok=True)

    loaded_f = os.path.join(BARRIER_DIR, f"{run_id}_{role}_loaded")
    go_file  = os.path.join(BARRIER_DIR, f"{run_id}_go")
    res_f    = os.path.join(RESULT_DIR,  f"{run_id}_{role}.json")

    for f in [loaded_f, go_file, res_f]:
        if os.path.exists(f): os.remove(f)

    pfile = write_prompts(run_id, role, prompts)
    afile = write_arrivals(run_id, role, arrivals)
    bfile = write_token_budgets(run_id, role, budgets) if budgets else None

    print(f"\n  ▶ {role} (GPU={gpu}, M_Seq={m_seq}, n={len(prompts)})")
    proc = subprocess.Popen(
        _make_worker_cmd(role, run_id, m_seq, util, pfile, afile, debug_kv, bfile=bfile),
        env=_worker_env(gpu))
    t0 = time.time()
    while not os.path.exists(loaded_f):
        if time.time()-t0 > 300: proc.kill(); raise TimeoutError(role)
        if proc.poll() is not None: raise RuntimeError(f"{role} died")
        time.sleep(0.2)
    print(f"  ✓ {role} loaded")

    dt = time.perf_counter()
    open(go_file, 'w').close()
    print(f"  ✓ GO at {time.strftime('%H:%M:%S')}")
    proc.wait()
    wc = time.perf_counter() - dt

    if proc.returncode: raise RuntimeError(f"{role} failed (exit={proc.returncode})")
    with open(res_f) as f: result = json.load(f)
    print(f"\n   ⏱  {wc:.1f}s")
    return result


# =============================================================================
# 10. ADAPTIVE LAUNCHER  (real-time KV dispatcher)
# =============================================================================

def launch_adaptive_realtime(
    run_id, short_m_seq, long_m_seq, all_prompts, preds, rate, util, debug_kv=False,
):
    os.makedirs(BARRIER_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR,  exist_ok=True)
    os.makedirs(KV_STAT_DIR, exist_ok=True)

    loaded_a  = os.path.join(BARRIER_DIR, f"{run_id}_SHORT_loaded")
    loaded_b  = os.path.join(BARRIER_DIR, f"{run_id}_LONG_loaded")
    go_file   = os.path.join(BARRIER_DIR, f"{run_id}_go")
    done_a    = os.path.join(BARRIER_DIR, f"{run_id}_SHORT_done")
    done_b    = os.path.join(BARRIER_DIR, f"{run_id}_LONG_done")
    res_a     = os.path.join(RESULT_DIR,  f"{run_id}_SHORT.json")
    res_b     = os.path.join(RESULT_DIR,  f"{run_id}_LONG.json")
    kv_a_file = os.path.join(KV_STAT_DIR, "SHORT.json")
    kv_b_file = os.path.join(KV_STAT_DIR, "LONG.json")

    for f in [loaded_a, loaded_b, go_file, done_a, done_b, res_a, res_b,
              kv_a_file, kv_b_file]:
        if os.path.exists(f): os.remove(f)

    print(f"\n  ▶ SHORT server (GPU={GPU_A}, M_Seq={short_m_seq})")
    proc_a = subprocess.Popen(
        _make_worker_cmd("SHORT", run_id, short_m_seq, util,
                         None, None, debug_kv, server_mode=True),
        env=_worker_env(GPU_A))
    t0 = time.time()
    while not os.path.exists(loaded_a):
        if time.time()-t0 > 300: proc_a.kill(); raise TimeoutError("SHORT")
        if proc_a.poll() is not None: raise RuntimeError("SHORT died")
        time.sleep(0.2)
    print(f"  ✓ SHORT loaded  (port {WORKER_PORTS[GPU_A]})")

    print(f"\n  ▶ LONG  server (GPU={GPU_B}, M_Seq={long_m_seq})")
    proc_b = subprocess.Popen(
        _make_worker_cmd("LONG",  run_id, long_m_seq, util,
                         None, None, debug_kv, server_mode=True),
        env=_worker_env(GPU_B))
    t0 = time.time()
    while not os.path.exists(loaded_b):
        if time.time()-t0 > 300: proc_a.kill(); proc_b.kill(); raise TimeoutError("LONG")
        if proc_b.poll() is not None: proc_a.kill(); raise RuntimeError("LONG died")
        time.sleep(0.2)
    print(f"  ✓ LONG  loaded  (port {WORKER_PORTS[GPU_B]}) — both ready")

    open(go_file, 'w').close()
    print(f"  ✓ GO at {time.strftime('%H:%M:%S')}")
    time.sleep(0.5)

    arrivals = generate_poisson_arrivals(len(all_prompts), rate, RANDOM_SEED)
    dispatch_stats = asyncio.run(_dispatcher(
        all_prompts, arrivals, preds,
        WORKER_PORTS[GPU_A], WORKER_PORTS[GPU_B],
        kv_a_file, kv_b_file,
        short_m_seq, long_m_seq,
    ))

    open(done_a, 'w').close()
    open(done_b, 'w').close()
    proc_a.wait(); proc_b.wait()

    with open(res_a) as f: ra = json.load(f)
    with open(res_b) as f: rb = json.load(f)

    print(f"\n  Routing: SHORT={len(ra.get('ttfts',[]))}  "
          f"LONG={len(rb.get('ttfts',[]))}  "
          f"spillovers={dispatch_stats.get('spillovers',0)}  "
          f"steals={dispatch_stats.get('steals',0)}  "
          f"overflows={dispatch_stats.get('overflows',0)}")

    # Save permanent validation snapshot — /tmp/rq3_results is wiped each rate
    val_path = os.path.join(PROJECT_ROOT, "rq3_adaptive_validation.json")
    with open(val_path, "w") as vf:
        json.dump({
            "run_id":      run_id,
            "rate":        rate,
            "short_m_seq": short_m_seq,
            "long_m_seq":  long_m_seq,
            "spillovers":  dispatch_stats.get("spillovers", 0),
            "SHORT":       ra,
            "LONG":        rb,
        }, vf)
    print(f"  Validation snapshot -> {val_path}")

    return ra, rb


async def _dispatcher(
    prompts, arrivals, preds,
    port_short, port_long,
    kv_a_file, kv_b_file,
    m_seq_short, m_seq_long,
):
    """
    Bidirectional SM-aware work-stealing dispatcher.

    Primary routing  : load score = (active/m_seq) + kv_util
    Work-stealing    : triggered by real GPU SM utilisation (via pynvml)
                       when EITHER GPU drops below SM_IDLE_THRESHOLD
    Token budget     : always from ORIGINAL classifier prediction
                       pred=0 → cls="SHORT" → max_tokens=512
                       pred=1 → cls="LONG"  → max_tokens=2048
                       Routing destination ≠ token budget

    Overflow queue   : requests that can't be dispatched immediately go here.
                       The SM monitor drains to whichever GPU is more idle.

    SM_IDLE_THRESHOLD: GPU SM% below which the GPU is considered idle.
    STEAL_POLL_S     : how often the SM monitor checks both GPUs (seconds).
    MAX_STEAL_ACTIVE : cap on concurrent stolen LONG requests on SHORT server
                       (prevents KV exhaustion: 102,400 tokens / 2048 = 50 safe).
    """
    SM_IDLE_THRESHOLD = 15    # GPU SM% below which GPU is idle → steal work
    STEAL_POLL_S      = 0.5   # SM poll interval
    MAX_STEAL_ACTIVE  = 45    # max concurrent 2048-token requests on SHORT server
    OVERFLOW_THRESHOLD = 0.60 # score above which requests go to overflow

    # ── helpers ───────────────────────────────────────────────────────────────
    def read_stat(path):
        try:
            with open(path) as f:
                d = json.load(f)
            return float(d.get("kv", 0.0)), int(d.get("active", 0))
        except Exception:
            return 0.0, 0

    def load_score(kv, active, m_seq):
        return (active / max(m_seq, 1)) + kv

    # ── state ─────────────────────────────────────────────────────────────────
    wall_start      = time.perf_counter()
    pending         = []
    spillovers      = 0
    steals          = 0
    overflows       = 0
    route_counts    = {"SHORT": 0, "LONG": 0}
    overflow_queue  = asyncio.Queue()  # (prompt, req_id, orig_cls)
    dispatch_done   = asyncio.Event()
    steal_active    = 0   # live count of 2048-token requests on SHORT server

    # ── send helper ───────────────────────────────────────────────────────────
    async def send(prompt, req_id, port, orig_cls):
        """orig_cls sets token budget, NOT server destination."""
        r, w = await asyncio.open_connection("127.0.0.1", port)
        w.write((json.dumps({"prompt": prompt, "req_id": req_id,
                              "cls": orig_cls}) + "\n").encode())
        await w.drain()
        resp = await r.readline()
        w.close()
        await w.wait_closed()
        return json.loads(resp.decode())

    async def send_and_track(prompt, req_id, port, orig_cls):
        """Like send(), but tracks steal_active for LONG budget on SHORT server."""
        nonlocal steal_active
        steal_active += 1
        try:
            return await send(prompt, req_id, port, orig_cls)
        finally:
            steal_active -= 1

    # ── SM-based bidirectional work-stealer ───────────────────────────────────
    async def sm_work_stealer():
        """
        Monitors BOTH GPUs every STEAL_POLL_S seconds.

        When GPU_A (SHORT) SM < SM_IDLE_THRESHOLD:
          → send overflow to SHORT server
          → if orig_cls="LONG": use send_and_track (respects MAX_STEAL_ACTIVE)
          → if orig_cls="SHORT": use send normally

        When GPU_B (LONG) SM < SM_IDLE_THRESHOLD:
          → send overflow to LONG server
          → any cls: LONG server can handle both budgets

        Both GPUs idle: send to SHORT first (higher M_Seq, more capacity)
        After dispatch_done: keep draining until overflow empty
        """
        nonlocal steals, steal_active

        while not (dispatch_done.is_set() and overflow_queue.empty()):
            if overflow_queue.empty():
                await asyncio.sleep(STEAL_POLL_S)
                continue

            sm_a = read_gpu_sm(GPU_A)   # SHORT server SM%
            sm_b = read_gpu_sm(GPU_B)   # LONG  server SM%

            # Determine where to steal
            a_idle = sm_a < SM_IDLE_THRESHOLD or dispatch_done.is_set()
            b_idle = sm_b < SM_IDLE_THRESHOLD or dispatch_done.is_set()

            if not a_idle and not b_idle:
                await asyncio.sleep(STEAL_POLL_S)
                continue

            # Pick destination: prefer the more idle GPU
            if a_idle and (not b_idle or sm_a <= sm_b):
                target_port = port_short
                target_name = "SHORT"
            else:
                target_port = port_long
                target_name = "LONG"

            # Peek at next item class to enforce MAX_STEAL_ACTIVE
            # (can't peek asyncio.Queue directly — just check the cap)
            if target_name == "SHORT":
                # Temporarily skip if SHORT would exceed safe LONG capacity
                # We check steal_active as a proxy (not perfect but effective)
                if steal_active >= MAX_STEAL_ACTIVE:
                    # Try LONG server instead
                    if b_idle:
                        target_port = port_long
                        target_name = "LONG"
                    else:
                        await asyncio.sleep(STEAL_POLL_S)
                        continue

            try:
                prompt_ov, req_id_ov, orig_cls_ov = overflow_queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(STEAL_POLL_S)
                continue

            route_counts[target_name] += 1
            steals += 1

            if target_name == "SHORT" and orig_cls_ov == "LONG":
                # LONG budget on SHORT server: track to enforce cap
                task = asyncio.create_task(
                    send_and_track(prompt_ov, req_id_ov, target_port, orig_cls_ov)
                )
            else:
                task = asyncio.create_task(
                    send(prompt_ov, req_id_ov, target_port, orig_cls_ov)
                )
            pending.append(task)

            print(f"  [steal] → {target_name}  SM_A={sm_a}%  SM_B={sm_b}%"
                  f"  cls={orig_cls_ov}  steal_active={steal_active}"
                  f"  overflow_remaining={overflow_queue.qsize()}",
                  flush=True)

            await asyncio.sleep(0.15)   # throttle: avoid burst flooding

    stealer_task = asyncio.create_task(sm_work_stealer())

    # ── main dispatch loop ────────────────────────────────────────────────────
    for i, (prompt, arrival, pred) in enumerate(zip(prompts, arrivals, preds)):
        elapsed = time.perf_counter() - wall_start
        if (s := float(arrival) - elapsed) > 0:
            await asyncio.sleep(s)

        orig_cls = "SHORT" if pred == 0 else "LONG"

        kv_s, act_s = read_stat(kv_a_file)
        kv_l, act_l = read_stat(kv_b_file)
        sc_s = load_score(kv_s, act_s, m_seq_short)
        sc_l = load_score(kv_l, act_l, m_seq_long)

        if pred == 0:   # SHORT class
            if sc_s <= sc_l + CLASS_BONUS:
                # Normal: SHORT → SHORT server
                route_counts["SHORT"] += 1
                pending.append(asyncio.create_task(
                    send(prompt, f"req-{i}", port_short, orig_cls)
                ))
            elif sc_l < OVERFLOW_THRESHOLD:
                # Immediate spillover to LONG server
                route_counts["LONG"] += 1
                pending.append(asyncio.create_task(
                    send(prompt, f"req-{i}", port_long, orig_cls)
                ))
                spillovers += 1
            else:
                # Both overloaded → overflow
                await overflow_queue.put((prompt, f"req-{i}", orig_cls))
                overflows += 1

        else:           # LONG class
            if sc_l <= sc_s + CLASS_BONUS:
                # Normal: LONG → LONG server
                route_counts["LONG"] += 1
                pending.append(asyncio.create_task(
                    send(prompt, f"req-{i}", port_long, orig_cls)
                ))
            elif sc_s < OVERFLOW_THRESHOLD and steal_active < MAX_STEAL_ACTIVE:
                # Immediate spillover to SHORT server (with LONG budget)
                route_counts["SHORT"] += 1
                pending.append(asyncio.create_task(
                    send_and_track(prompt, f"req-{i}", port_short, orig_cls)
                ))
                spillovers += 1
            else:
                # Overloaded or cap reached → overflow for SM-stealer
                await overflow_queue.put((prompt, f"req-{i}", orig_cls))
                overflows += 1

    dispatch_done.set()

    # Work-stealer drains overflow; wait for it to finish
    await stealer_task
    await asyncio.gather(*pending, return_exceptions=True)

    print(f"    spillovers={spillovers}  overflows={overflows}  steals={steals}")
    return {**route_counts, "spillovers": spillovers,
            "overflows": overflows, "steals": steals}


# =============================================================================
# 10b. PRESERVE BASELINE  (PreServe load_5 routing + mLSTM predictor)
# =============================================================================

class _InstState:
    """Per-instance load state for PreServe load_5 routing."""
    __slots__ = ("prefill", "expected")
    def __init__(self):           self.prefill = 0;  self.expected = 0
    def load_5(self, p=2):        return self.prefill * p + self.expected
    def on_dispatch(self, pl, po): self.prefill += pl; self.expected += pl + po
    def on_done(self, pl, po):    self.prefill -= pl; self.expected -= (pl + po); self.expected = max(0, self.expected)


def _build_load_predictor(model_path: str):
    from load_predictor.predictor import LoadPredictor
    pred = LoadPredictor({"req_predictor_model_path": model_path})
    pred.predict("warm up")   # eliminate cold-start latency
    print(f"[preserve] LoadPredictor ready — {model_path}", flush=True)
    return pred


async def _preserve_dispatcher(prompts, arrivals, load_predictor, port_a, port_b):
    """
    PreServe preserve routing:  route to min(load_5) instance.
    Uses one TCP connection per request (matches existing worker handle_client design).
    Returns (results_gpu_a, results_gpu_b).
    """
    n      = len(prompts)
    ports  = [port_a, port_b]
    states = [_InstState(), _InstState()]

    # Pre-predict all output lengths (CPU, ~3ms each; done upfront to avoid dispatch lag)
    print(f"[preserve] Predicting output lengths for {n} prompts...", flush=True)
    pred_outs = []
    for p in prompts:
        raw = load_predictor.predict(p)
        pred_outs.append(min(max(int(raw), 1), MAX_MODEL_LEN - 1))
    print(f"[preserve] pred_out  mean={np.mean(pred_outs):.0f}"
          f"  p50={np.percentile(pred_outs,50):.0f}"
          f"  p95={np.percentile(pred_outs,95):.0f}", flush=True)

    wall_start = time.perf_counter()
    pending    = []   # list of (task, gpu_idx)

    async def send_one(prompt, req_id, port, max_tokens):
        r, w = await asyncio.open_connection("127.0.0.1", port)
        w.write((json.dumps({"prompt": prompt, "req_id": req_id,
                              "max_tokens": max_tokens}) + "\n").encode())
        await w.drain()
        resp = await r.readline()
        w.close()
        await w.wait_closed()
        return json.loads(resp.decode())

    for i in range(n):
        elapsed = time.perf_counter() - wall_start
        if (s := float(arrivals[i]) - elapsed) > 0:
            await asyncio.sleep(s)

        pred_out = pred_outs[i]
        p_len    = len(prompts[i].split())   # word-count proxy for routing state
        gpu_idx  = min(range(2), key=lambda j: states[j].load_5(2))
        states[gpu_idx].on_dispatch(p_len, pred_out)

        task = asyncio.create_task(send_one(prompts[i], f"pre-{i}", ports[gpu_idx], pred_out))
        pending.append((task, gpu_idx, p_len, pred_out))

    all_results = await asyncio.gather(*[t for t, *_ in pending], return_exceptions=True)

    res_a, res_b = [], []
    for (task, gpu_idx, p_len, pred_out), result in zip(pending, all_results):
        states[gpu_idx].on_done(p_len, pred_out)
        if isinstance(result, Exception):
            continue
        bucket = res_a if gpu_idx == 0 else res_b
        bucket.append(result)

    print(f"[preserve] GPU_A={len(res_a)}  GPU_B={len(res_b)}", flush=True)
    return res_a, res_b


def launch_preserve_realtime(
    run_id, short_m_seq, long_m_seq, prompts, preds, rate, util, load_predictor, debug_kv=False,
):
    """Launch preserve routing experiment; returns (ra, rb) in same format as launch_adaptive_realtime."""
    os.makedirs(BARRIER_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR,  exist_ok=True)
    os.makedirs(KV_STAT_DIR, exist_ok=True)

    loaded_a = os.path.join(BARRIER_DIR, f"{run_id}_A_loaded")
    loaded_b = os.path.join(BARRIER_DIR, f"{run_id}_B_loaded")
    go_file  = os.path.join(BARRIER_DIR, f"{run_id}_go")
    done_a   = os.path.join(BARRIER_DIR, f"{run_id}_A_done")
    done_b   = os.path.join(BARRIER_DIR, f"{run_id}_B_done")
    res_a    = os.path.join(RESULT_DIR,  f"{run_id}_A.json")
    res_b    = os.path.join(RESULT_DIR,  f"{run_id}_B.json")

    for f in [loaded_a, loaded_b, go_file, done_a, done_b, res_a, res_b]:
        if os.path.exists(f): os.remove(f)

    print(f"\n  ▶ PRESERVE-A (GPU={GPU_A}, M_Seq={short_m_seq})")
    proc_a = subprocess.Popen(
        _make_worker_cmd("A", run_id, short_m_seq, util, None, None, debug_kv, server_mode=True),
        env=_worker_env(GPU_A))
    t0 = time.time()
    while not os.path.exists(loaded_a):
        if time.time()-t0 > 300: proc_a.kill(); raise TimeoutError("A")
        if proc_a.poll() is not None: raise RuntimeError("preserve-A died")
        time.sleep(0.2)
    print(f"  ✓ PRESERVE-A loaded  (port {WORKER_PORTS[GPU_A]})")

    print(f"\n  ▶ PRESERVE-B (GPU={GPU_B}, M_Seq={long_m_seq})")
    proc_b = subprocess.Popen(
        _make_worker_cmd("B", run_id, long_m_seq, util, None, None, debug_kv, server_mode=True),
        env=_worker_env(GPU_B))
    t0 = time.time()
    while not os.path.exists(loaded_b):
        if time.time()-t0 > 300: proc_a.kill(); proc_b.kill(); raise TimeoutError("B")
        if proc_b.poll() is not None: proc_a.kill(); raise RuntimeError("preserve-B died")
        time.sleep(0.2)
    print(f"  ✓ PRESERVE-B loaded  (port {WORKER_PORTS[GPU_B]}) — both ready")

    open(go_file, 'w').close()
    print(f"  ✓ GO at {time.strftime('%H:%M:%S')}")
    time.sleep(0.5)

    arrivals  = generate_poisson_arrivals(len(prompts), rate, RANDOM_SEED)
    raw_a, raw_b = asyncio.run(_preserve_dispatcher(
        prompts, arrivals, load_predictor,
        WORKER_PORTS[GPU_A], WORKER_PORTS[GPU_B],
    ))

    open(done_a, 'w').close()
    open(done_b, 'w').close()
    proc_a.wait(); proc_b.wait()

    def _to_result(raw_list, role):
        ttfts = [r.get("ttft", 0.0) for r in raw_list]
        e2es  = [r.get("e2e",  0.0) for r in raw_list]
        tok   = sum(r.get("tokens", 0) for r in raw_list)
        wall  = max(e2es) if e2es else 0.0
        return {"role": role, "wall": wall, "tokens": tok,
                "kv_avg": 0.0, "kv_peak": 0.0, "preemptions": 0,
                "ttfts": ttfts, "e2es": e2es}

    ra = _to_result(raw_a, "A")
    rb = _to_result(raw_b, "B")

    with open(res_a, 'w') as f: json.dump(ra, f)
    with open(res_b, 'w') as f: json.dump(rb, f)

    print(f"\n  Preserve routing: A={len(raw_a)}  B={len(raw_b)}")
    return ra, rb

def _pct(arr, p):
    return float(np.percentile(arr, p)) if arr else 0.0


def compute_stats_parallel(res_a, res_b, label_a, label_b):
    ttft_a, ttft_b = res_a['ttfts'], res_b['ttfts']
    e2e_a,  e2e_b  = res_a['e2es'],  res_b['e2es']
    all_ttft = ttft_a + ttft_b
    all_e2e  = e2e_a  + e2e_b
    wc  = max(res_a['wall'], res_b['wall'])
    tok = res_a['tokens'] + res_b['tokens']
    avg_kv = np.mean([res_a['kv_avg'],  res_b['kv_avg']])
    pk_kv  = max(res_a['kv_peak'], res_b['kv_peak'])
    pre    = res_a['preemptions'] + res_b['preemptions']
    n      = len(ttft_a) + len(ttft_b)

    return {
        "wall_clock":          wc,
        "total_tokens":        tok,
        "tps":                 tok / wc if wc > 0 else 0.0,
        "ttft_p50_all":        _pct(all_ttft, 50),
        "ttft_p95_all":        _pct(all_ttft, 95),
        "ttft_p99_all":        _pct(all_ttft, 99),
        "e2e_p95_all":         _pct(all_e2e,  95),
        f"n_{label_a}":        len(ttft_a),
        f"ttft_p95_{label_a}": _pct(ttft_a, 95),
        f"e2e_p95_{label_a}":  _pct(e2e_a,  95),
        f"n_{label_b}":        len(ttft_b),
        f"ttft_p95_{label_b}": _pct(ttft_b, 95),
        f"e2e_p95_{label_b}":  _pct(e2e_b,  95),
        "kv_avg_pct":          round(avg_kv * 100, 2),
        "kv_peak_pct":         round(pk_kv  * 100, 2),
        "preemptions":         pre,
        "preemption_rate":     round(pre / n * 100, 2) if n > 0 else 0.0,
    }


def compute_stats_single(res):
    ttfts, e2es = res['ttfts'], res['e2es']
    n, pre = len(ttfts), res['preemptions']
    return {
        "wall_clock":      res['wall'],
        "total_tokens":    res['tokens'],
        "tps":             res['tokens'] / res['wall'] if res['wall'] > 0 else 0.0,
        "ttft_p50_all":    _pct(ttfts, 50),
        "ttft_p95_all":    _pct(ttfts, 95),
        "ttft_p99_all":    _pct(ttfts, 99),
        "e2e_p95_all":     _pct(e2es,  95),
        "kv_avg_pct":      round(res['kv_avg']  * 100, 2),
        "kv_peak_pct":     round(res['kv_peak'] * 100, 2),
        "preemptions":     pre,
        "preemption_rate": round(pre / n * 100, 2) if n > 0 else 0.0,
    }


# =============================================================================
# 12. CONFIGS + CLASSIFIER
# =============================================================================

def load_instance_configs():
    if not os.path.exists(INSTANCE_CFG):
        raise FileNotFoundError(f"'{INSTANCE_CFG}' not found. Run vllm_kv_benchmark.py first.")
    with open(INSTANCE_CFG) as f:
        cfg = json.load(f)
    print(f"\n📋 {INSTANCE_CFG}")
    for g, c in cfg.items():
        print(f"   {g}: M_Seq={c['optimal_m_seq']}  TPS={c['tps']}  KV_Avg={c['kv_avg_pct']}%")
    return cfg


def load_classifier():
    vectorizer = joblib.load(VEC_PATH)
    xgb_model  = XGBClassifier()
    xgb_model.load_model(XGB_PATH)
    print(f"   XGBoost: {XGB_PATH}  threshold={TOKEN_THRESHOLD}")
    return xgb_model, vectorizer


# =============================================================================
# 13. SINGLE-RATE EVALUATION
# =============================================================================

def evaluate(rate, cfg, prompts, preds, debug_kv=False, load_predictor=None):
    short_m_seq = cfg["SHORT"]["optimal_m_seq"]
    long_m_seq  = cfg["LONG"]["optimal_m_seq"]
    n           = len(prompts)
    n_short     = int((preds == 0).sum())
    n_long      = int((preds == 1).sum())

    print(f"\n{'#'*70}")
    print(f"  RATE={rate}  SHORT_M={short_m_seq}  LONG_M={long_m_seq}")
    print(f"  max_model_len={MAX_MODEL_LEN}  max_tokens={MAX_OUTPUT_TOKENS}"
          f"  prompt_max={PROMPT_MAX_LEN}")
    print(f"  Classifier: SHORT={n_short}  LONG={n_long}  CLASS_BONUS={CLASS_BONUS}")
    print(f"  Token budgets: SHORT=max_tokens={MAX_TOKENS_SHORT}  LONG=max_tokens={MAX_TOKENS_LONG}")
    print(f"{'#'*70}")

    half_rate = rate / 2.0
    rr_a      = [prompts[i] for i in range(0, n, 2)]
    rr_b      = [prompts[i] for i in range(1, n, 2)]
    rr_a_arr  = generate_poisson_arrivals(len(rr_a), half_rate, RANDOM_SEED)
    rr_b_arr  = generate_poisson_arrivals(len(rr_b), half_rate, RANDOM_SEED+1)
    all_arr   = generate_poisson_arrivals(n, rate, RANDOM_SEED)

    # ── Per-prompt token budgets (classifier-matched) ─────────────────────────
    # Applied to ALL methods so the only variable is routing strategy.
    # SHORT predicted → max_tokens=512 | LONG predicted → max_tokens=2048
    all_budgets = [MAX_TOKENS_SHORT if p == 0 else MAX_TOKENS_LONG for p in preds]
    rr_a_budgets = [all_budgets[i] for i in range(0, n, 2)]
    rr_b_budgets = [all_budgets[i] for i in range(1, n, 2)]

    n_short_budget = sum(1 for b in all_budgets if b == MAX_TOKENS_SHORT)
    n_long_budget  = n - n_short_budget
    print(f"  Token budgets (applied to ALL methods):")
    print(f"    SHORT ({n_short_budget} prompts) → max_tokens={MAX_TOKENS_SHORT}")
    print(f"    LONG  ({n_long_budget}  prompts) → max_tokens={MAX_TOKENS_LONG}")

    results_all = []

    def row(base): return {"Rate": rate, **base}

    # ── Single ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  SINGLE  (rate={rate})")
    print("="*60)
    for m in SINGLE_M_SEQS:
        print(f"\n>>> Single-{m}")
        res = launch_single(f"single_{m}_r{int(rate)}", f"Single{m}",
                            prompts, all_arr, m, GPU_A, UTIL_PER_GPU, debug_kv,
                            budgets=all_budgets)
        s = compute_stats_single(res)
        results_all.append(row({
            "Method": f"Single-{m}", "Routing": "none", "Instances": 1,
            "GPU": f"{GPU_A}", "M_Seq": str(m), "N_SHORT": n, "N_LONG": 0,
            "TPS": s['tps'],
            "TTFT_P50_ALL": s['ttft_p50_all'], "TTFT_P95_ALL": s['ttft_p95_all'],
            "TTFT_P99_ALL": s['ttft_p99_all'],
            "TTFT_P95_SHORT": s['ttft_p95_all'], "TTFT_P95_LONG": 0.0,
            "E2E_P95_ALL": s['e2e_p95_all'],
            "KV_Avg": s['kv_avg_pct'], "KV_Peak": s['kv_peak_pct'],
            "Preemptions": s['preemptions'], "Preempt_Rate%": s['preemption_rate'],
            "Total_S": s['wall_clock'],
        }))

    # ── Static ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  STATIC  (rate={rate})")
    print("="*60)
    for m in STATIC_M_SEQS:
        print(f"\n>>> Static-{m}")
        ra, rb = launch_parallel(
            f"static_{m}_r{int(rate)}",
            f"Sta{m}A", rr_a, rr_a_arr, m, GPU_A,
            f"Sta{m}B", rr_b, rr_b_arr, m, GPU_B,
            UTIL_PER_GPU, debug_kv,
            budgets_a=rr_a_budgets, budgets_b=rr_b_budgets,
        )
        s = compute_stats_parallel(ra, rb, "A", "B")
        results_all.append(row({
            "Method": f"Static-{m}", "Routing": "round-robin", "Instances": 2,
            "GPU": f"{GPU_A}+{GPU_B}", "M_Seq": f"{m}/{m}",
            "N_SHORT": s['n_A'], "N_LONG": s['n_B'],
            "TPS": s['tps'],
            "TTFT_P50_ALL": s['ttft_p50_all'], "TTFT_P95_ALL": s['ttft_p95_all'],
            "TTFT_P99_ALL": s['ttft_p99_all'],
            "TTFT_P95_SHORT": s['ttft_p95_A'], "TTFT_P95_LONG": s['ttft_p95_B'],
            "E2E_P95_ALL": s['e2e_p95_all'],
            "KV_Avg": s['kv_avg_pct'], "KV_Peak": s['kv_peak_pct'],
            "Preemptions": s['preemptions'], "Preempt_Rate%": s['preemption_rate'],
            "Total_S": s['wall_clock'],
        }))

    # ── Adaptive (3 configs) ─────────────────────────────────────────────────
    # CFG1 = from instance_configs.json, CFG2/CFG3 = user-defined at top of file
    adaptive_cfgs = [
        ("ADAPTIVE_CFG1", short_m_seq,                        long_m_seq),
        ("ADAPTIVE_CFG2", ADAPTIVE_CFG2["short_m_seq"],       ADAPTIVE_CFG2["long_m_seq"]),
        #("ADAPTIVE_CFG3", ADAPTIVE_CFG3["short_m_seq"],       ADAPTIVE_CFG3["long_m_seq"]),
    ]

    for cfg_label, s_mseq, l_mseq in adaptive_cfgs:
        print("\n" + "="*60)
        print(f"  {cfg_label}  SHORT_M={s_mseq}  LONG_M={l_mseq}  (rate={rate})")
        print("="*60)

        # Clean IPC files between adaptive runs
        for d in [BARRIER_DIR, RESULT_DIR, PROMPTS_DIR, KV_STAT_DIR]:
            if os.path.exists(d):
                for fname in os.listdir(d):
                    try: os.remove(os.path.join(d, fname))
                    except Exception: pass

        rs, rl = launch_adaptive_realtime(
            f"{cfg_label.lower()}_r{int(rate)}", s_mseq, l_mseq,
            prompts, preds, rate, UTIL_PER_GPU, debug_kv,
        )
        s = compute_stats_parallel(rs, rl, "SHORT", "LONG")
        results_all.append(row({
            "Method":        cfg_label,
            "Routing":       "XGBoost+RealTimeKV",
            "Instances":     2,
            "GPU":           f"{GPU_A}+{GPU_B}",
            "M_Seq":         f"{s_mseq}/{l_mseq}",
            "N_SHORT":       len(rs.get("ttfts", [])),
            "N_LONG":        len(rl.get("ttfts", [])),
            "TPS":           s["tps"],
            "TTFT_P50_ALL":  s["ttft_p50_all"],
            "TTFT_P95_ALL":  s["ttft_p95_all"],
            "TTFT_P99_ALL":  s["ttft_p99_all"],
            "TTFT_P95_SHORT": s["ttft_p95_SHORT"],
            "TTFT_P95_LONG":  s["ttft_p95_LONG"],
            "E2E_P95_ALL":   s["e2e_p95_all"],
            "KV_Avg":        s["kv_avg_pct"],
            "KV_Peak":       s["kv_peak_pct"],
            "Preemptions":   s["preemptions"],
            "Preempt_Rate%": s["preemption_rate"],
            "Total_S":       s["wall_clock"],
        }))

    # ── Preserve (PreServe mLSTM routing baseline) ───────────────────────────
    if load_predictor is not None:
        print("\n" + "="*60)
        print(f"  PRESERVE  short_m={short_m_seq}  long_m={long_m_seq}  (rate={rate})")
        print("="*60)

        for d in [BARRIER_DIR, RESULT_DIR, PROMPTS_DIR, KV_STAT_DIR]:
            if os.path.exists(d):
                for fname in os.listdir(d):
                    try: os.remove(os.path.join(d, fname))
                    except Exception: pass

        rs, rl = launch_preserve_realtime(
            f"preserve_r{int(rate)}", short_m_seq, long_m_seq,
            prompts, preds, rate, UTIL_PER_GPU, load_predictor, debug_kv,
        )
        s = compute_stats_parallel(rs, rl, "A", "B")
        results_all.append(row({
            "Method":         "PRESERVE",
            "Routing":        "PreServe-mLSTM",
            "Instances":      2,
            "GPU":            f"{GPU_A}+{GPU_B}",
            "M_Seq":          f"{short_m_seq}/{long_m_seq}",
            "N_SHORT":        len(rs.get("ttfts", [])),
            "N_LONG":         len(rl.get("ttfts", [])),
            "TPS":            s["tps"],
            "TTFT_P50_ALL":   s["ttft_p50_all"],
            "TTFT_P95_ALL":   s["ttft_p95_all"],
            "TTFT_P99_ALL":   s["ttft_p99_all"],
            "TTFT_P95_SHORT": s["ttft_p95_A"],
            "TTFT_P95_LONG":  s["ttft_p95_B"],
            "E2E_P95_ALL":    s["e2e_p95_all"],
            "KV_Avg":         s["kv_avg_pct"],
            "KV_Peak":        s["kv_peak_pct"],
            "Preemptions":    s["preemptions"],
            "Preempt_Rate%":  s["preemption_rate"],
            "Total_S":        s["wall_clock"],
        }))

    # ── Per-rate summary ──────────────────────────────────────────────────────
    df        = pd.DataFrame(results_all)
    static_df = df[df["Method"].str.startswith("Static")]
    adp_df    = df[df["Method"].str.startswith("ADAPTIVE")]
    best_ttft = static_df["TTFT_P95_ALL"].min() if not static_df.empty else 0
    best_e2e  = static_df["E2E_P95_ALL"].min()  if not static_df.empty else 0

    print(f"\n  ── Rate={rate} summary ──────────────────────────────────")
    print(f"  {'Method':<16} {'TTFT_P95':>10} {'Δ_TTFT':>8}  {'E2E_P95':>9} {'Δ_E2E':>7}  {'N_S':>5} {'N_L':>5}")
    print(f"  {'─'*16} {'─'*10} {'─'*8}  {'─'*9} {'─'*7}  {'─'*5} {'─'*5}")
    for _, r in adp_df.iterrows():
        dt = (r["TTFT_P95_ALL"] - best_ttft) / best_ttft * 100 if best_ttft else 0
        de = (r["E2E_P95_ALL"]  - best_e2e)  / best_e2e  * 100 if best_e2e  else 0
        print(f"  {r['Method']:<16} {r['TTFT_P95_ALL']:>10.2f} {dt:>+8.1f}%"
              f"  {r['E2E_P95_ALL']:>9.2f} {de:>+7.1f}%"
              f"  {int(r['N_SHORT']):>5} {int(r['N_LONG']):>5}")
    print(f"  BestStatic       {best_ttft:>10.2f} {'(base)':>9}  {best_e2e:>9.2f} {'(base)':>8}")

    return results_all


# =============================================================================
# 14. RATE SWEEP
# =============================================================================

def evaluate_rate_sweep(rate_start=24.0, rate_step=16.0, rate_end=76.0, debug_kv=False):
    rates = list(np.arange(rate_start, rate_end + 1e-9, rate_step))
    print(f"\n🔁 Rates: {[round(r,1) for r in rates]}")
    print(f"   max_model_len={MAX_MODEL_LEN}  prompt_max={PROMPT_MAX_LEN}"
          f"  max_tokens={MAX_OUTPUT_TOKENS}")

    cfg     = load_instance_configs()
    prompts = get_test_pool(MODEL_ID)
    n       = len(prompts)
    print(f"\nPrompts={n}  GPU_A={GPU_A}(SHORT)  GPU_B={GPU_B}(LONG)"
          f"  util={UTIL_PER_GPU}  threshold={TOKEN_THRESHOLD}")

    print("\n[Classifier] Running once...")
    xgb_model, vectorizer = load_classifier()
    features, _           = extract_v13_features(prompts, vectorizer)
    preds                 = xgb_model.predict(features.values).flatten()
    print(f"   SHORT={int((preds==0).sum())}  LONG={int((preds==1).sum())}")

    # ── Load mLSTM predictor once for preserve baseline ───────────────────────
    load_predictor = None
    if os.path.exists(PRESERVE_MODEL_PATH):
        try:
            load_predictor = _build_load_predictor(PRESERVE_MODEL_PATH)
        except Exception as e:
            print(f"⚠  PreServe predictor load failed ({e}) — PRESERVE skipped")
    else:
        print(f"⚠  {PRESERVE_MODEL_PATH} not found — PRESERVE skipped (train first)")

    all_rows, failed = [], []

    for i, rate in enumerate(rates):
        print(f"\n{'═'*70}")
        print(f"  RATE {i+1}/{len(rates)} : {rate}")
        print(f"{'═'*70}")

        for d in [BARRIER_DIR, RESULT_DIR, PROMPTS_DIR, KV_STAT_DIR]:
            if os.path.exists(d):
                for fname in os.listdir(d):
                    try: os.remove(os.path.join(d, fname))
                    except Exception: pass

        try:
            rows = evaluate(rate, cfg, prompts, preds, debug_kv, load_predictor)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv("rq3_results_sweep.csv", index=False)
            print(f"\n  💾 {len(all_rows)} rows → rq3_results_sweep.csv")
        except Exception as e:
            print(f"\n  ❌ Rate={rate} FAILED: {e}")
            failed.append(rate)

    df = pd.DataFrame(all_rows)
    print(f"\n{'═'*100}")
    print(f"  COMPLETE — {len(rates)-len(failed)}/{len(rates)} rates")
    if failed: print(f"  Failed: {failed}")
    print(f"{'═'*100}")

    adp_labels = ["ADAPTIVE_CFG1", "ADAPTIVE_CFG2"]#, "ADAPTIVE_CFG3"]
    print(f"\n  {'Rate':>6}  {'Method':<16} {'TTFT_P95':>10} {'Δ_TTFT':>8}  {'E2E_P95':>9} {'Δ_E2E':>7}  {'N_S':>5} {'N_L':>5}")
    print(f"  {'─'*6}  {'─'*16} {'─'*10} {'─'*8}  {'─'*9} {'─'*7}  {'─'*5} {'─'*5}")
    for rate in rates:
        rdf = df[df['Rate'] == rate]
        sta = rdf[rdf['Method'].str.startswith('Static')]
        if sta.empty: continue
        bt = sta['TTFT_P95_ALL'].min()
        be = sta['E2E_P95_ALL'].min()
        for label in adp_labels:
            adp = rdf[rdf['Method'] == label]
            if adp.empty: continue
            a  = adp.iloc[0]
            dt = (a['TTFT_P95_ALL'] - bt) / bt * 100
            de = (a['E2E_P95_ALL']  - be) / be * 100
            print(f"  {rate:>6.1f}  {label:<16} {a['TTFT_P95_ALL']:>10.2f} {dt:>+8.1f}%"
                  f"  {a['E2E_P95_ALL']:>9.2f} {de:>+7.1f}%"
                  f"  {int(a['N_SHORT']):>5} {int(a['N_LONG']):>5}")
        print(f"         {'BestStatic':<16} {bt:>10.2f} {'(base)':>9}  {be:>9.2f} {'(base)':>8}")
        print()

    df.to_csv("rq3_results_sweep.csv", index=False)
    print(f"\n✅ rq3_results_sweep.csv  ({len(df)} rows)")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker",        action="store_true")
    parser.add_argument("--server-mode",   action="store_true")
    parser.add_argument("--role",          type=str,   default=None)
    parser.add_argument("--run-id",        type=str,   default=None)
    parser.add_argument("--m-seq",         type=int,   default=20)
    parser.add_argument("--gpu-mem",       type=float, default=UTIL_PER_GPU)
    parser.add_argument("--prompts-file",  type=str,   default=None)
    parser.add_argument("--arrivals-file", type=str,   default=None)
    parser.add_argument("--budgets-file",  type=str,   default="")
    parser.add_argument("--debug-kv",      action="store_true")
    parser.add_argument("--rate",          type=float, default=None)
    parser.add_argument("--rate-start",    type=float, default=24.0)
    parser.add_argument("--rate-step",     type=float, default=16.0)
    parser.add_argument("--rate-end",      type=float, default=76.0)
    args = parser.parse_args()

    if args.worker:
        asyncio.run(worker_main(args))
    elif args.rate is not None:
        cfg           = load_instance_configs()
        prompts       = get_test_pool(MODEL_ID)
        xgb_model, v  = load_classifier()
        features, _   = extract_v13_features(prompts, v)
        preds         = xgb_model.predict(features.values).flatten()
        print(f"   SHORT={int((preds==0).sum())}  LONG={int((preds==1).sum())}")
        rows = evaluate(args.rate, cfg, prompts, preds, args.debug_kv)
        out  = f"rq3_results_r{int(args.rate)}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"✅ {out}")
    else:
        evaluate_rate_sweep(args.rate_start, args.rate_step, args.rate_end, args.debug_kv)
