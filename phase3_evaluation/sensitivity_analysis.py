"""
sensitivity_analysis.py — TAILOR routing parameter sensitivity analysis
=======================================================================
Supplementary material: systematically varies each TAILOR routing parameter
independently (one-at-a-time) while holding all others at their default values.

Parameters analysed
-------------------
1. delta_kv        (OVERFLOW_THRESHOLD) — KV utilisation threshold above which
                   a server is considered overloaded and requests spill over.
2. class_bonus     (CLASS_BONUS)        — load-score margin given to the
                   class-matched server before routing to the alternate.
3. sm_idle_thresh  (SM_IDLE_THRESHOLD)  — GPU SM% below which a worker is
                   considered idle and eligible for work stealing.
4. max_steal       (MAX_STEAL_ACTIVE)   — cap on concurrent LONG-budget
                   requests on the SHORT server to prevent KV exhaustion.
5. token_threshold (TOKEN_THRESHOLD)   — prompt-token boundary used by the
                   XGBoost classifier to define SHORT vs LONG.

For each parameter sweep we fix rate=56 req/s (peak load),
run 3 repetitions to quantify variance, and report:
  TTFT P95, TTFT P99, E2E P95, TPS, SHORT/LONG routing ratio.

Usage
-----
    python sensitivity_analysis.py                   # all sweeps
    python sensitivity_analysis.py --param delta_kv  # single param
    python sensitivity_analysis.py --rate 40         # different rate
    python sensitivity_analysis.py --reps 2          # fewer reps (faster)

Output
------
    sensitivity_results.csv   — all runs
    sensitivity_summary.csv   — mean ± std per parameter value
"""

import argparse
import os
import sys
import time
import json
import copy

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace")
import rq3_eval_final as m

# ─────────────────────────────────────────────────────────────────────────────
# Default parameter values (from rq3_eval_final.py)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "delta_kv":       0.80,   # OVERFLOW_THRESHOLD
    "class_bonus":    0.00,   # CLASS_BONUS
    "sm_idle_thresh": 15,     # SM_IDLE_THRESHOLD
    "max_steal":      45,     # MAX_STEAL_ACTIVE
    "token_threshold":300,    # TOKEN_THRESHOLD (Mistral-7B default)
}

# Parameter sweep grids
SWEEP_GRIDS = {
    "delta_kv":        [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    "class_bonus":     [0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
    "sm_idle_thresh":  [5,    10,   15,   20,   25,   30],
    "max_steal":       [15,   25,   35,   45,   55],
    "token_threshold": [150,  200,  250,  300,  400,  500],
}

PARAM_LABELS = {
    "delta_kv":        r"δ_KV (overflow threshold)",
    "class_bonus":     "CLASS_BONUS",
    "sm_idle_thresh":  "SM_IDLE_THRESHOLD (%)",
    "max_steal":       "MAX_STEAL_ACTIVE",
    "token_threshold": "Token threshold (tokens)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Patched dispatcher factory
# ─────────────────────────────────────────────────────────────────────────────

def make_patched_dispatcher(delta_kv, class_bonus, sm_idle_thresh, max_steal):
    """
    Returns an async dispatcher with the given parameter values injected.
    Wraps _dispatcher from rq3_eval_final but overrides the four constants.
    """
    import asyncio

    async def patched_dispatcher(
        prompts, arrivals, preds,
        port_short, port_long,
        kv_a_file, kv_b_file,
        m_seq_short, m_seq_long,
    ):
        # ── same logic as rq3_eval_final._dispatcher ──────────────────────
        OVERFLOW_THRESHOLD = delta_kv
        CLASS_BONUS_       = class_bonus
        SM_IDLE_THRESHOLD  = sm_idle_thresh
        STEAL_POLL_S       = 0.5
        MAX_STEAL_ACTIVE   = max_steal

        def read_stat(path):
            try:
                with open(path) as f:
                    d = json.load(f)
                return float(d.get("kv", 0.0)), int(d.get("active", 0))
            except Exception:
                return 0.0, 0

        def load_score(kv, active, mseq):
            return (active / max(mseq, 1)) + kv

        wall_start     = time.perf_counter()
        pending        = []
        spillovers     = 0
        steals         = 0
        overflows      = 0
        route_counts   = {"SHORT": 0, "LONG": 0}
        overflow_queue = asyncio.Queue()
        dispatch_done  = asyncio.Event()
        steal_active   = 0

        async def send(prompt, req_id, port, orig_cls):
            r, w = await asyncio.open_connection("127.0.0.1", port)
            w.write((json.dumps({"prompt": prompt, "req_id": req_id,
                                  "cls": orig_cls}) + "\n").encode())
            await w.drain()
            resp = await r.readline()
            w.close()
            await w.wait_closed()
            return json.loads(resp.decode())

        async def send_and_track(prompt, req_id, port, orig_cls):
            nonlocal steal_active
            steal_active += 1
            try:
                return await send(prompt, req_id, port, orig_cls)
            finally:
                steal_active -= 1

        async def sm_work_stealer():
            nonlocal steals, steal_active
            while not (dispatch_done.is_set() and overflow_queue.empty()):
                if overflow_queue.empty():
                    await asyncio.sleep(STEAL_POLL_S)
                    continue
                sm_a = m.read_gpu_sm(m.GPU_A)
                sm_b = m.read_gpu_sm(m.GPU_B)
                a_idle = sm_a < SM_IDLE_THRESHOLD or dispatch_done.is_set()
                b_idle = sm_b < SM_IDLE_THRESHOLD or dispatch_done.is_set()
                if not a_idle and not b_idle:
                    await asyncio.sleep(STEAL_POLL_S)
                    continue
                if a_idle and (not b_idle or sm_a <= sm_b):
                    target_port, target_name = port_short, "SHORT"
                else:
                    target_port, target_name = port_long, "LONG"
                if target_name == "SHORT" and steal_active >= MAX_STEAL_ACTIVE:
                    if b_idle:
                        target_port, target_name = port_long, "LONG"
                    else:
                        await asyncio.sleep(STEAL_POLL_S)
                        continue
                try:
                    prompt_ov, req_id_ov, cls_ov = overflow_queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(STEAL_POLL_S)
                    continue
                route_counts[target_name] += 1
                steals += 1
                if target_name == "SHORT" and cls_ov == "LONG":
                    task = asyncio.create_task(
                        send_and_track(prompt_ov, req_id_ov, target_port, cls_ov))
                else:
                    task = asyncio.create_task(
                        send(prompt_ov, req_id_ov, target_port, cls_ov))
                pending.append(task)
                await asyncio.sleep(0.15)

        stealer_task = asyncio.create_task(sm_work_stealer())

        for i, (prompt, arrival, pred) in enumerate(zip(prompts, arrivals, preds)):
            elapsed = time.perf_counter() - wall_start
            if (s := float(arrival) - elapsed) > 0:
                await asyncio.sleep(s)
            orig_cls = "SHORT" if pred == 0 else "LONG"
            kv_s, act_s = read_stat(kv_a_file)
            kv_l, act_l = read_stat(kv_b_file)
            sc_s = load_score(kv_s, act_s, m_seq_short)
            sc_l = load_score(kv_l, act_l, m_seq_long)

            if pred == 0:
                if sc_s <= sc_l + CLASS_BONUS_:
                    route_counts["SHORT"] += 1
                    pending.append(asyncio.create_task(
                        send(prompt, f"req-{i}", port_short, orig_cls)))
                elif sc_l < OVERFLOW_THRESHOLD:
                    route_counts["LONG"] += 1
                    pending.append(asyncio.create_task(
                        send(prompt, f"req-{i}", port_long, orig_cls)))
                    spillovers += 1
                else:
                    await overflow_queue.put((prompt, f"req-{i}", orig_cls))
                    overflows += 1
            else:
                if sc_l <= sc_s + CLASS_BONUS_:
                    route_counts["LONG"] += 1
                    pending.append(asyncio.create_task(
                        send(prompt, f"req-{i}", port_long, orig_cls)))
                elif sc_s < OVERFLOW_THRESHOLD and steal_active < MAX_STEAL_ACTIVE:
                    route_counts["SHORT"] += 1
                    pending.append(asyncio.create_task(
                        send_and_track(prompt, f"req-{i}", port_short, orig_cls)))
                    spillovers += 1
                else:
                    await overflow_queue.put((prompt, f"req-{i}", orig_cls))
                    overflows += 1

        dispatch_done.set()
        await stealer_task
        await asyncio.gather(*pending, return_exceptions=True)
        return {**route_counts, "spillovers": spillovers,
                "overflows": overflows, "steals": steals}

    return patched_dispatcher


# ─────────────────────────────────────────────────────────────────────────────
# Single run with injected parameters
# ─────────────────────────────────────────────────────────────────────────────

def run_one(cfg, prompts, preds, rate, params, rep_id):
    """
    Run TAILOR with overridden parameters. Returns a metrics dict.
    params: dict with keys from DEFAULTS.
    """
    import asyncio

    delta_kv       = params.get("delta_kv",       DEFAULTS["delta_kv"])
    class_bonus    = params.get("class_bonus",     DEFAULTS["class_bonus"])
    sm_idle_thresh = params.get("sm_idle_thresh",  DEFAULTS["sm_idle_thresh"])
    max_steal      = params.get("max_steal",       DEFAULTS["max_steal"])
    token_threshold= params.get("token_threshold", DEFAULTS["token_threshold"])

    # Recompute preds if token_threshold changed
    if token_threshold != DEFAULTS["token_threshold"]:
        # Re-threshold: count tokens in prompt and re-label
        # Approximate via word count * 1.3 as token proxy
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(m.MODEL_ID)
        new_preds = []
        for p in prompts:
            n_tok = len(_tok.encode(p))
            new_preds.append(0 if n_tok <= token_threshold else 1)
        preds_run = np.array(new_preds)
    else:
        preds_run = preds

    short_m_seq = cfg["SHORT"]["optimal_m_seq"]
    long_m_seq  = cfg["LONG"]["optimal_m_seq"]

    run_id = f"sens_{rep_id}"

    # Clean IPC dirs
    for d in [m.BARRIER_DIR, m.RESULT_DIR, m.PROMPTS_DIR, m.KV_STAT_DIR]:
        if os.path.exists(d):
            for fname in os.listdir(d):
                try: os.remove(os.path.join(d, fname))
                except Exception: pass
    for d in [m.BARRIER_DIR, m.RESULT_DIR, m.PROMPTS_DIR, m.KV_STAT_DIR]:
        os.makedirs(d, exist_ok=True)

    # Patch dispatcher
    patched = make_patched_dispatcher(
        delta_kv, class_bonus, sm_idle_thresh, max_steal)
    original_dispatcher = m._dispatcher
    m._dispatcher = patched

    try:
        ra, rb = m.launch_adaptive_realtime(
            run_id, short_m_seq, long_m_seq,
            prompts, preds_run, rate, m.UTIL_PER_GPU,
        )
    finally:
        m._dispatcher = original_dispatcher

    s = m.compute_stats_parallel(ra, rb, "SHORT", "LONG")

    n_short = len(ra.get("ttfts", []))
    n_long  = len(rb.get("ttfts", []))
    ratio   = n_short / max(n_long, 1)

    return {
        "rep":            rep_id,
        "rate":           rate,
        "delta_kv":       delta_kv,
        "class_bonus":    class_bonus,
        "sm_idle_thresh": sm_idle_thresh,
        "max_steal":      max_steal,
        "token_threshold":token_threshold,
        "ttft_p95":       s["ttft_p95_all"],
        "ttft_p99":       s["ttft_p99_all"],
        "e2e_p95":        s["e2e_p95_all"],
        "tps":            s["tps"],
        "wall":           s["wall_clock"],
        "n_short":        n_short,
        "n_long":         n_long,
        "short_long_ratio": ratio,
        "kv_avg":         s["kv_avg_pct"],
        "kv_peak":        s["kv_peak_pct"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sweep runner
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(param_name, rate, reps, cfg, prompts, preds):
    grid   = SWEEP_GRIDS[param_name]
    rows   = []
    total  = len(grid) * reps
    done   = 0

    print(f"\n{'='*70}")
    print(f"  Sweeping: {PARAM_LABELS[param_name]}")
    print(f"  Grid: {grid}")
    print(f"  Rate={rate}  Reps={reps}  Total runs={total}")
    print(f"{'='*70}")

    for val in grid:
        for rep in range(reps):
            done += 1
            params = dict(DEFAULTS)
            params[param_name] = val
            print(f"\n  [{done}/{total}] {param_name}={val}  rep={rep+1}", flush=True)
            try:
                row = run_one(cfg, prompts, preds, rate, params, f"{param_name}_{val}_r{rep}")
                row["param_name"]  = param_name
                row["param_value"] = val
                rows.append(row)
                print(f"    TTFT_P95={row['ttft_p95']:.3f}s  "
                      f"E2E_P95={row['e2e_p95']:.1f}s  "
                      f"TPS={row['tps']:.0f}", flush=True)
            except Exception as e:
                print(f"    FAILED: {e}", flush=True)
                rows.append({
                    "param_name": param_name, "param_value": val,
                    "rep": rep, "rate": rate, "error": str(e),
                    **{k: DEFAULTS[k] for k in DEFAULTS},
                })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Summary stats
# ─────────────────────────────────────────────────────────────────────────────

def summarise(df):
    metrics = ["ttft_p95", "ttft_p99", "e2e_p95", "tps", "short_long_ratio"]
    rows = []
    for (pname, pval), g in df.groupby(["param_name", "param_value"]):
        row = {"param_name": pname, "param_value": pval,
               "is_default": pval == DEFAULTS.get(pname)}
        for met in metrics:
            if met in g.columns:
                row[f"{met}_mean"] = g[met].mean()
                row[f"{met}_std"]  = g[met].std()
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param",  type=str,   default=None,
        help=f"Parameter to sweep. One of: {list(SWEEP_GRIDS)}. Default: all.")
    parser.add_argument("--rate",   type=float, default=56.0,
        help="Arrival rate in req/s (default: 56)")
    parser.add_argument("--reps",   type=int,   default=3,
        help="Repetitions per grid point (default: 3)")
    args = parser.parse_args()

    params_to_run = [args.param] if args.param else list(SWEEP_GRIDS.keys())

    # Load once
    print("Loading dataset and classifier...")
    cfg     = m.load_instance_configs()
    prompts = m.get_test_pool(m.MODEL_ID)
    xgb, v  = m.load_classifier()
    feats,_ = m.extract_v13_features(prompts, v)
    preds   = xgb.predict(feats.values).flatten()
    print(f"Prompts={len(prompts)}  SHORT={int((preds==0).sum())}  LONG={int((preds==1).sum())}")

    all_rows = []
    for pname in params_to_run:
        rows = run_sweep(pname, args.rate, args.reps, cfg, prompts, preds)
        all_rows.extend(rows)
        # Save after each parameter in case of interruption
        pd.DataFrame(all_rows).to_csv("sensitivity_results.csv", index=False)
        print(f"\n  💾 Saved {len(all_rows)} rows → sensitivity_results.csv")

    df = pd.DataFrame(all_rows)
    df.to_csv("sensitivity_results.csv", index=False)

    summary = summarise(df)
    summary.to_csv("sensitivity_summary.csv", index=False)

    print(f"\n{'='*70}")
    print(f"  COMPLETE")
    print(f"  sensitivity_results.csv  ({len(df)} rows)")
    print(f"  sensitivity_summary.csv  ({len(summary)} rows)")
    print(f"{'='*70}")

    # Print summary table per parameter
    for pname in params_to_run:
        sub = summary[summary["param_name"] == pname]
        if sub.empty: continue
        print(f"\n  {PARAM_LABELS[pname]}")
        print(f"  {'Value':>10}  {'TTFT_P95':>10}  {'±':>6}  {'E2E_P95':>9}  {'TPS':>7}  {'Default':>8}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*9}  {'─'*7}  {'─'*8}")
        for _, r in sub.iterrows():
            marker = " ◄" if r["is_default"] else ""
            print(f"  {r['param_value']:>10}  "
                  f"{r['ttft_p95_mean']:>10.3f}  "
                  f"{r.get('ttft_p95_std', 0):>6.3f}  "
                  f"{r['e2e_p95_mean']:>9.1f}  "
                  f"{r['tps_mean']:>7.0f}"
                  f"{marker}")