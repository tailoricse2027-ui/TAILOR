#!/usr/bin/env python3
"""
tailor_ui.py — TAILOR Pipeline Terminal Interface
==================================================
Interactive terminal menu for running all three phases of the TAILOR
evaluation pipeline. Requires Python 3.10+ and the `rich` library.

Usage:
    python tailor_ui.py
"""

import os
import sys
import subprocess
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    from rich.rule import Rule
except ImportError:
    print("Installing rich...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    from rich.rule import Rule

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "1": {
        "name":    "Llama-2-7B-AWQ",
        "hf_id":   "TheBloke/Llama-2-7B-Chat-AWQ",
        "quant":   "awq",
        "thresh":  200,
        "dataset": "llama2-7b-awq",
        "gated":   False,
    },
    "2": {
        "name":    "Llama-2-13B",
        "hf_id":   "meta-llama/Llama-2-13b-chat-hf",
        "quant":   None,
        "thresh":  200,
        "dataset": "llama2-13b-dolly",
        "gated":   True,
    },
    "3": {
        "name":    "Mistral-7B-v0.1",
        "hf_id":   "mistralai/Mistral-7B-Instruct-v0.1",
        "quant":   None,
        "thresh":  300,
        "dataset": "mistral-7b-dolly",
        "gated":   False,
    },
    "4": {
        "name":    "Mistral-7B-v0.2",
        "hf_id":   "mistralai/Mistral-7B-Instruct-v0.2",
        "quant":   None,
        "thresh":  300,
        "dataset": "mistral-7b-mixed",
        "gated":   False,
    },
    "5": {
        "name":    "Mistral-Nemo-12B",
        "hf_id":   "mistralai/Mistral-Nemo-Instruct-2407",
        "quant":   None,
        "thresh":  300,
        "dataset": "mistral_nemo_12b",
        "gated":   False,
    },
    "6": {
        "name":    "DeepSeek-R1-Distill-Llama-8B",
        "hf_id":   "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "quant":   None,
        "thresh":  1000,
        "dataset": "deepseek-r1-llama-8b",
        "gated":   False,
    },
    "7": {
        "name":    "DeepSeek-R1-Distill-Qwen-14B",
        "hf_id":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quant":   None,
        "thresh":  600,
        "dataset": "deepseek-r1-qwen-14b",
        "gated":   False,
    },
}

SCRIPTS = {
    "phase1":  "phase1_corpus/run_inference_vllm.py",
    "phase2a": "phase2_training/train_2class_advancedfeatures.py",
    "phase2b": "phase2_training/knee_new.py",
    "phase3":  "phase3_evaluation/rq4_evaluation.py",
}


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def header():
    console.print()
    console.print(Panel(
        Text("TAILOR  ·  Tail-Aware Inference with Length-Oriented Routing\n"
             "LLM Serving Evaluation Pipeline", justify="center", style="bold white"),
        style="bold blue",
        padding=(1, 4),
    ))
    console.print()


def section(title: str):
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))
    console.print()


def model_table(title="Select a model"):
    section(title)
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    t.add_column("#",      style="bold yellow", width=4)
    t.add_column("Name",   style="white",       width=30)
    t.add_column("HuggingFace ID",              width=46)
    t.add_column("Token threshold", style="dim", width=16)
    t.add_column("",       style="dim",         width=4)
    for k, m in MODELS.items():
        lock = "[yellow]🔒[/yellow]" if m["gated"] else ""
        t.add_row(k, m["name"], m["hf_id"], str(m["thresh"]), lock)
    console.print(t)
    if any(m["gated"] for m in MODELS.values()):
        console.print("  [dim]🔒 = requires HuggingFace login (gated model)[/dim]\n")


def pick_model(allow_all=False) -> list[dict]:
    model_table()
    hint = "[0=all] " if allow_all else ""
    choices = list(MODELS.keys()) + (["0"] if allow_all else [])
    choice = Prompt.ask(
        f"  Select a model {hint}",
        choices=choices,
        show_choices=False,
    )
    if choice == "0":
        return list(MODELS.values())
    return [MODELS[choice]]


def maybe_ask_hf_token(models: list[dict]) -> dict:
    """If any selected model is gated and HF_TOKEN isn't already set, prompt for it."""
    if not any(m.get("gated") for m in models):
        return {}
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return {}
    console.print()
    console.print("  [yellow]This model requires a HuggingFace access token.[/yellow]")
    console.print("  [dim]Get one at https://huggingface.co/settings/tokens — "
                  "you must also accept the model's license on its HF page.[/dim]")
    token = Prompt.ask("  HuggingFace token (hf_...)", password=True)
    return {"HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token}


def run_cmd(cmd: list[str], label: str, extra_env: dict | None = None):
    console.print(f"\n  [dim]Running:[/dim] {' '.join(cmd)}\n")
    env = {**os.environ, **(extra_env or {})}
    result = subprocess.run(cmd, env=env)
    if result.returncode == 0:
        console.print(f"\n  [bold green]✓ {label} completed successfully.[/bold green]")
    else:
        console.print(f"\n  [bold red]✗ {label} exited with code {result.returncode}.[/bold red]")
    Prompt.ask("\n  Press Enter to return to menu", default="")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Corpus creation
# ─────────────────────────────────────────────────────────────────────────────

def phase1():
    header()
    section("Phase 1 · Corpus Creation  (run_inference_vllm.py)")

    console.print("  This phase generates inference data for classifier training.\n"
                  "  [dim]Skip this phase if pre-built datasets are present in [bold]data/[/bold].[/dim]\n")

    if not Confirm.ask("  Proceed with corpus creation?", default=False):
        return

    model = pick_model()[0]
    hf_env = maybe_ask_hf_token([model])

    cmd = [sys.executable, SCRIPTS["phase1"], "--model_name", model["hf_id"]]

    # Build dataset choices: always include Dolly, then scan data/ for local CSVs
    dataset_options = {
        "1": {
            "label": "Dolly-15k  (HuggingFace: databricks/databricks-dolly-15k)",
            "kind":  "hf",
            "hf_id": "databricks/databricks-dolly-15k",
        },
    }
    local_csvs = sorted(Path("data").glob("*.csv")) if Path("data").is_dir() else []
    for i, csv_file in enumerate(local_csvs, start=2):
        dataset_options[str(i)] = {
            "label": f"Local CSV  →  {csv_file}",
            "kind":  "csv",
            "path":  str(csv_file),
        }
    dataset_options[str(len(dataset_options) + 1)] = {
        "label": "Custom CSV path  (enter manually)",
        "kind":  "custom_csv",
    }

    console.print()
    console.print("  [bold cyan]Available input datasets:[/bold cyan]")
    ds_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    ds_table.add_column("#",     style="bold yellow", width=4)
    ds_table.add_column("Dataset", style="white")
    for k, v in dataset_options.items():
        ds_table.add_row(k, v["label"])
    console.print(ds_table)

    ds_choice = Prompt.ask(
        "  Select dataset",
        choices=list(dataset_options.keys()),
        show_choices=False,
    )
    chosen_ds = dataset_options[ds_choice]

    if chosen_ds["kind"] == "hf":
        num = Prompt.ask("  Number of prompts to sample", default="5000")
        cmd += ["--dataset_name", chosen_ds["hf_id"], "--num_prompts", num]
    elif chosen_ds["kind"] == "csv":
        cmd += ["--prompts_csv", chosen_ds["path"]]
    else:
        csv_path = Prompt.ask("  Path to prompts CSV (e.g. data/my_prompts.csv)")
        cmd += ["--prompts_csv", csv_path]

    out = Prompt.ask(
        "  Output CSV path",
        default=f"data/vllm_{model['name'].lower().replace('-','_')}_dolly15k.csv",
    )
    cmd += ["--out_csv", out]

    batch = Prompt.ask("  Batch size", default="16")
    cmd += ["--batch_size", batch]

    if Confirm.ask("  Use natural stop (greedy decoding)?", default=False):
        cmd.append("--natural_stop")

    run_cmd(cmd, "Phase 1 corpus creation", hf_env)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2a — Classifier training
# ─────────────────────────────────────────────────────────────────────────────

def phase2a():
    header()
    section("Phase 2a · Classifier Training  (train_2class_advancedfeatures.py)")

    console.print("  Trains XGBoost / CatBoost / RandomForest classifiers on corpus data.\n"
                  "  Output: [bold]deploy/v13_xgb.json[/bold], "
                  "[bold]deploy/v13_cat.cbm[/bold], "
                  "[bold]deploy/v13_rf.pkl[/bold], "
                  "[bold]deploy/v13_vectorizer.pkl[/bold]\n")

    selected = pick_model(allow_all=True)
    dataset_keys = [m["dataset"] for m in selected]
    thresh = selected[0]["thresh"] if len(selected) == 1 else 300

    if len(selected) > 1:
        thresh = int(Prompt.ask(
            "  Token threshold for SHORT/LONG boundary",
            default="300",
        ))

    console.print(f"\n  [dim]Datasets:[/dim] {dataset_keys}")
    console.print(f"  [dim]Threshold:[/dim] {thresh}\n")

    cmd = [
        sys.executable, SCRIPTS["phase2a"],
        "--datasets", ",".join(dataset_keys),
        "--threshold", str(thresh),
    ]

    classifiers = []
    if Confirm.ask("  Train XGBoost?",     default=True):  classifiers.append("xgb")
    if Confirm.ask("  Train CatBoost?",    default=True):  classifiers.append("cat")
    if Confirm.ask("  Train RandomForest?",default=True):  classifiers.append("rf")
    cmd += ["--classifiers", ",".join(classifiers)]

    run_cmd(cmd, "Phase 2a classifier training")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2b — Concurrency profiling (knee point)
# ─────────────────────────────────────────────────────────────────────────────

def phase2b():
    header()
    section("Phase 2b · Concurrency Profiling  (knee_new.py)")

    console.print("  Sweeps M_Seq values to find the optimal concurrency knee point.\n"
                  "  Output: [bold]instance_configs.json[/bold]\n")

    model = pick_model()[0]
    hf_env = maybe_ask_hf_token([model])

    gpu_vram = Prompt.ask("  GPU VRAM in GiB (e.g. 32 for V100, 40/80 for A100)", default="32")
    weight   = Prompt.ask("  Model weight size in GiB (from vLLM startup log)",    default="13.5")

    console.print()
    console.print("  [dim]M_Seq sweep ranges:[/dim]")
    short_range = Prompt.ask("  SHORT M_Seq values (comma-separated)", default="128,180,220,256,300")
    long_range  = Prompt.ask("  LONG  M_Seq values (comma-separated)", default="96,128,140,180,220")

    cmd = [
        sys.executable, SCRIPTS["phase2b"],
        "--model",        model["hf_id"],
        "--threshold",    str(model["thresh"]),
        "--gpu_gib",      gpu_vram,
        "--weight_gib",   weight,
        "--short_configs", short_range,
        "--long_configs",  long_range,
    ]

    run_cmd(cmd, "Phase 2b concurrency profiling", hf_env)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def phase3():
    header()
    section("Phase 3 · Evaluation  (rq3_eval_final.py)")

    console.print("  Runs the full rate sweep evaluation across all methods.\n"
                  "  Output: [bold]rq3_results_sweep.csv[/bold]\n")

    model = pick_model()[0]
    hf_env = maybe_ask_hf_token([model])
    console.print(f"  [dim]Selected:[/dim] {model['hf_id']}\n")

    console.print("  [bold cyan]Run mode:[/bold cyan]")
    console.print("  [bold yellow]sweep[/bold yellow]  Runs the model at multiple arrival rates (e.g. 8 → 72 req/s in steps of 16).")
    console.print("         Each rate is a separate ~90s experiment. All routing methods")
    console.print("         (TAILOR, round-robin, single-queue) are compared at every rate.")
    console.print("         Output: [bold]rq3_results_sweep.csv[/bold] — use this for the final paper results.\n")
    console.print("  [bold yellow]single[/bold yellow] Runs just one arrival rate. Much faster (~90s total).")
    console.print("         Useful for a quick sanity check or debugging a specific load point.\n")

    mode = Prompt.ask(
        "  Run mode",
        choices=["sweep", "single"],
        default="sweep",
    )

    cmd = [sys.executable, SCRIPTS["phase3"], "--model", model["hf_id"]]

    if mode == "single":
        rate = Prompt.ask("  Arrival rate (req/s)", default="56.0")
        cmd += ["--rate", rate]
    else:
        r_start = Prompt.ask("  Rate start (req/s)", default="8")
        r_step  = Prompt.ask("  Rate step  (req/s)", default="16")
        r_end   = Prompt.ask("  Rate end   (req/s)", default="72")
        cmd += ["--rate-start", r_start, "--rate-step", r_step, "--rate-end", r_end]

    if Confirm.ask("  Enable KV debug logging?", default=False):
        cmd.append("--debug-kv")

    est = "~90s per rate point × number of steps" if mode == "sweep" else "~90s"
    console.print(f"\n  [yellow]⚠  Estimated runtime: {est}[/yellow]")
    console.print(f"  [dim]Instance configs read from instance_configs.json[/dim]\n")

    if Confirm.ask("  Start evaluation?", default=True):
        run_cmd(cmd, "Phase 3 evaluation", hf_env)


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity():
    header()
    section("Sensitivity Analysis  (sensitivity_analysis.py)")

    console.print("  Sweeps routing parameters and produces sensitivity tables.\n"
                  "  Output: [bold]sensitivity_results.csv[/bold], "
                  "[bold]sensitivity_summary.csv[/bold]\n")

    model = pick_model()[0]
    hf_env = maybe_ask_hf_token([model])

    param = Prompt.ask(
        "  Parameter to sweep",
        choices=["delta_kv", "sm_idle_thresh", "max_steal", "class_bonus",
                 "token_threshold", "all"],
        default="all",
    )
    rate = Prompt.ask("  Rate (req/s)", default="56.0")
    reps = Prompt.ask("  Repetitions per grid point", default="1")

    cmd = [sys.executable, "phase3_evaluation/sensitivity_analysis.py",
           "--model", model["hf_id"], "--rate", rate, "--reps", reps]
    if param != "all":
        cmd += ["--param", param]

    run_cmd(cmd, "Sensitivity analysis", hf_env)


# ─────────────────────────────────────────────────────────────────────────────
# Environment check
# ─────────────────────────────────────────────────────────────────────────────

def check_env():
    header()
    section("Environment Check")

    checks = [
        ("Python ≥ 3.10",     sys.version_info >= (3, 10)),
        ("vLLM",              _importable("vllm")),
        ("xgboost",           _importable("xgboost")),
        ("rich",              _importable("rich")),
        ("transformers",      _importable("transformers")),
        ("datasets",          _importable("datasets")),
        ("torch",             _importable("torch")),
        ("pynvml / nvidia-ml-py", _importable("pynvml") or _importable("nvidia_ml_py")),
        ("deploy/v13_xgb.json",   Path("deploy/v13_xgb.json").exists()),
        ("deploy/v13_vectorizer.pkl", Path("deploy/v13_vectorizer.pkl").exists()),
        ("instance_configs.json",  Path("instance_configs.json").exists()),
        ("data/ directory",        Path("data").is_dir()),
    ]

    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column("Check", style="white", width=35)
    t.add_column("Status", width=10)
    for name, ok in checks:
        status = "[bold green]✓ OK[/bold green]" if ok else "[bold red]✗ Missing[/bold red]"
        t.add_row(name, status)
    console.print(t)
    Prompt.ask("\n  Press Enter to return", default="")


def _importable(pkg: str) -> bool:
    import importlib
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main menu
# ─────────────────────────────────────────────────────────────────────────────

MENU = [
    ("0", "Environment check",           check_env),
    ("1", "Phase 1  — Corpus creation  [dim](Optional)[/dim]",  phase1),
    ("2", "Phase 2a — Train classifier [dim](Optional)[/dim]", phase2a),
    ("3", "Phase 2b — Concurrency profiling (knee point)", phase2b),
    ("4", "Phase 3  — Evaluation (rate sweep / single rate)", phase3),
    ("5", "Sensitivity analysis",        sensitivity),
    ("q", "Quit",                        None),
]


def main_menu():
    while True:
        header()
        t = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
        t.add_column("Key",   style="bold yellow", width=5)
        t.add_column("Action", style="white")
        for key, label, _ in MENU:
            t.add_row(key, label)
        console.print(t)
        console.print()

        choice = Prompt.ask(
            "  Select",
            choices=[k for k, _, _ in MENU],
            show_choices=False,
        )

        if choice == "q":
            console.print("\n  Goodbye.\n")
            sys.exit(0)

        for key, _, fn in MENU:
            if choice == key and fn:
                fn()
                break


if __name__ == "__main__":
    main_menu()
