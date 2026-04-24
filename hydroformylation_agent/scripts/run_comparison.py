"""
run_comparison.py
-----------------
Comparative evaluation: Agent vs Random Baseline vs Bayesian-only surrogate.

Runs each strategy N times over a synthetic ground-truth function derived from
the real seed data, then reports mean ± std for the primary metrics.

Usage (from project root):
    python scripts/run_comparison.py
    python scripts/run_comparison.py --n-runs 5 --n-iter 20
    python scripts/run_comparison.py --n-runs 3 --seed-file data/seed_data_BDP-2.json

Strategies compared:
  1. Random         – uniformly sample the parameter space each iteration
  2. Bayesian-only  – Gaussian Process plus Expected Improvement acquisition (scikit-learn)
  3. Agent (mock)   – the LLM mock response from llm_planner._mock_response, which
                      deterministically walks toward better conditions; replace with
                      a real run log if you have one.

Output:
    - Console table: mean ± std for max L:B, avg L:B, convergence iter, composite reward
    - results/comparison_report.json
    - results/figures/strategy_comparison.png
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"

# Add src/ to path for imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation import (
    compute_metrics,
    compute_composite_reward,
    identify_pareto_front,
    DEFAULT_W_LB, DEFAULT_W_CONV, DEFAULT_W_TON,
)

# Ground-truth simulator

# These polynomial coefficients were hand-tuned to match the reported agent
# trajectory (seed Iterations 1–2: conv=42.5/58.0, L:B=2.1/2.9; peak L:B≈4.7
# at T≈80°C, P≈50 bar, ligand_eq≈8, cat_loading≈1.5).
# They are used to simulate what the lab would return for a given conditions dict.

def _simulate_outcome(conditions: dict, noise_std: float = 0.15) -> dict:
    """Return simulated L:B, conversion, and TON for a conditions dict.

    The true optimum is approximately:
        T = 80°C, P = 50 bar, ligand_eq = 8, cat = 1.5 mol%

    A Gaussian hill in each dimension is used so that the true optimum is unique
    and the landscape is smooth enough for GP to learn quickly.
    """
    T   = float(conditions.get("temperature_C",            70))
    P   = float(conditions.get("pressure_bar",             30))
    Leq = float(conditions.get("ligand_loading_eq",         4))
    cat = float(conditions.get("catalyst_loading_mol_pct",  1))

    # L:B response surface (peak at T=80, P=50, Leq=8)
    lb_true = (
        4.8
        * _gauss(T,   80, 25)
        * _gauss(P,   50, 20)
        * _gauss(Leq,  8,  4)
        * _gauss(cat, 1.5, 0.8)
    )

    # Conversion response surface (peak at T=90, P=40)
    conv_true = (
        82.0
        * _gauss(T,   90, 30)
        * _gauss(P,   40, 25)
        * _gauss(cat, 1.5, 1.0)
    )

    # TON (simple, linear, and maximised at low catalyst loading)
    ton_true = max(0, 350 * _gauss(cat, 0.8, 0.6) * _gauss(T, 75, 30))

    rng = random.gauss
    return {
        "l_b_ratio":     max(0.1, lb_true   + rng(0, noise_std * lb_true)),
        "conversion_pct": max(0.0, min(100.0,
                            conv_true + rng(0, noise_std * conv_true))),
        "ton":           max(0.0, ton_true  + rng(0, noise_std * ton_true)),
        "notes":         "simulated",
    }

# Define the unnormalised Gaussian function for the response surfaces.
def _gauss(x: float, mu: float, sigma: float) -> float:
    """Unnormalised Gaussian evaluated at x."""
    import math
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Parameter space

PARAM_SPACE = {
    "temperature_C":            (40.0, 120.0),
    "pressure_bar":             (5.0,  80.0),
    "ligand_loading_eq":        (1.0,  20.0),
    "catalyst_loading_mol_pct": (0.1,   5.0),
}

# Defines the parameter space bounds for random sampling and Bayesian optimization.
def _random_conditions() -> dict:
    return {k: random.uniform(lo, hi) for k, (lo, hi) in PARAM_SPACE.items()}


# Strategy runners
# Define functions to run each strategy for a given number of iterations, returning a history of conditions and outcomes for metric computation.
def run_random_strategy(n_iter: int, seed: int = 0) -> list:
    """Pure random sampling."""
    random.seed(seed)
    history = []
    for i in range(1, n_iter + 1):
        cond = _random_conditions()
        out  = _simulate_outcome(cond)
        history.append({"iteration": i, "conditions": cond, "outcomes": out})
    return history

# Define a function to run the Bayesian optimization strategy using Gaussian Process regression and Expected Improvement acquisition.
def run_bayesian_strategy(n_iter: int, seed: int = 0) -> list:
    """Gaussian Process + Expected Improvement acquisition.

    Falls back to random sampling if scikit-learn is unavailable.
    """
    try:
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        from scipy.stats import norm as sp_norm
    except ImportError:
        print("[COMPARE] scikit-learn/scipy not available. Bayesian strategy uses random fallback.")
        return run_random_strategy(n_iter, seed)

    random.seed(seed)
    np.random.seed(seed)

    keys   = list(PARAM_SPACE.keys())
    bounds = np.array([PARAM_SPACE[k] for k in keys])

    def _to_vec(cond: dict) -> np.ndarray:
        return np.array([cond[k] for k in keys])

    def _from_vec(vec: np.ndarray) -> dict:
        return {k: float(v) for k, v in zip(keys, vec)}

    def _ei(X_test, gp, y_best, xi=0.01):
        mu, sigma = gp.predict(X_test, return_std=True)
        sigma = sigma.reshape(-1, 1).ravel()
        z     = (mu - y_best - xi) / (sigma + 1e-9)
        return (mu - y_best - xi) * sp_norm.cdf(z) + sigma * sp_norm.pdf(z)

    gp      = GaussianProcessRegressor(
        kernel=Matern(nu=2.5), alpha=1e-4,
        normalize_y=True, n_restarts_optimizer=3
    )
    history: list = []
    X_obs: list   = []
    y_obs: list   = []

    # Warm-start: 3 random runs
    for i in range(1, min(4, n_iter + 1)):
        cond = _random_conditions()
        out  = _simulate_outcome(cond)
        history.append({"iteration": i, "conditions": cond, "outcomes": out})
        X_obs.append(_to_vec(cond))
        y_obs.append(out["l_b_ratio"])

    # GP-guided runs
    for i in range(len(history) + 1, n_iter + 1):
        X_arr = np.array(X_obs)
        y_arr = np.array(y_obs)
        gp.fit(X_arr, y_arr)
        y_best = float(np.max(y_arr))

        # Random candidates → pick best EI
        n_candidates = 500
        candidates   = np.column_stack([
            np.random.uniform(lo, hi, n_candidates)
            for lo, hi in bounds
        ])
        ei_vals = _ei(candidates, gp, y_best)
        best_c  = candidates[np.argmax(ei_vals)]
        cond    = _from_vec(best_c)
        out     = _simulate_outcome(cond)

        history.append({"iteration": i, "conditions": cond, "outcomes": out})
        X_obs.append(_to_vec(cond))
        y_obs.append(out["l_b_ratio"])

    return history

# Define a function to run the LLM agent strategy using the deterministic mock planner.
def run_agent_strategy(n_iter: int, seed: int = 0) -> list:
    """Simulate the LLM agent using the deterministic mock planner.

    The mock planner in llm_planner._mock_response shifts conditions each
    iteration toward T=100°C, P=60 bar, ligand_eq grows — which moves toward
    (but not exactly at) the true optimum.  This gives a realistic but
    imperfect agent trajectory for benchmarking.
    """
    random.seed(seed)
    history = []
    for i in range(1, n_iter + 1):
        base_temp  = min(60 + (i - 1) * 4, 95)
        base_press = min(20 + (i - 1) * 4, 60)
        cond = {
            "temperature_C":            base_temp,
            "pressure_bar":             base_press,
            "ligand_loading_eq":        4.0 + i * 0.5,
            "catalyst_loading_mol_pct": 1.0 + min(i * 0.05, 1.0),
        }
        out = _simulate_outcome(cond)
        history.append({"iteration": i, "conditions": cond, "outcomes": out})
    return history


# Multi-run statistics
# Define a function to run a given strategy multiple times and compute mean ± std for key metrics across runs.
def _multi_run_stats(
    strategy_fn,
    n_runs: int,
    n_iter: int,
    w_lb: float, w_conv: float, w_ton: float,
) -> dict:
    """Run a strategy N times and compute mean ± std of key metrics."""
    import math

    max_lbs, avg_lbs, conv_iters, composite_maxes = [], [], [], []

    for run_seed in range(n_runs):
        history = strategy_fn(n_iter, seed=run_seed * 17)
        m       = compute_metrics(history)
        rewards = compute_composite_reward(history, w_lb, w_conv, w_ton)

        max_lbs.append(m.get("max_l_b_ratio", 0))
        avg_lbs.append(m.get("avg_l_b_ratio", 0))
        conv_iters.append(m.get("convergence_iteration") or n_iter)
        composite_maxes.append(max(rewards) if rewards else 0)
    # Define a helper to compute mean and std, rounded to 4 decimals for reporting.
    def _stats(vals):
        n   = len(vals)
        mu  = sum(vals) / n
        std = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(n - 1, 1))
        return round(mu, 4), round(std, 4)

    return {
        "max_l_b":          _stats(max_lbs),
        "avg_l_b":          _stats(avg_lbs),
        "convergence_iter": _stats(conv_iters),
        "max_composite":    _stats(composite_maxes),
        "raw_max_lbs":      max_lbs,
    }


# Reporting 
# Define a function to print a formatted comparison table of the results for each strategy, showing mean ± std for each metric.
def print_comparison_table(results: dict, n_runs: int, n_iter: int) -> None:
    print(f"\n{'=' * 68}")
    print(f"  STRATEGY COMPARISON  ({n_runs} runs × {n_iter} iterations, mean ± std)")
    print(f"{'=' * 68}")
    print(f"  {'Metric':<28} {'Random':>14} {'Bayesian':>14} {'Agent':>14}")
    print(f"  {'-' * 64}")

    metrics = [
        ("Max L:B (↑ better)",      "max_l_b"),
        ("Avg L:B (↑ better)",      "avg_l_b"),
        ("Convergence iter (↓)",    "convergence_iter"),
        ("Max composite reward (↑)","max_composite"),
    ]

    for label, key in metrics:
        r  = results["random"][key]
        b  = results["bayesian"][key]
        a  = results["agent"][key]
        print(
            f"  {label:<28}  "
            f"{r[0]:>6.3f}±{r[1]:<5.3f}  "
            f"{b[0]:>6.3f}±{b[1]:<5.3f}  "
            f"{a[0]:>6.3f}±{a[1]:<5.3f}"
        )
    print(f"{'=' * 68}")

# Define a function to generate and save a bar plot comparing the max L:B ratio achieved by each strategy.
def generate_comparison_plot(results: dict, n_iter: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    strategies = ["random", "bayesian", "agent"]
    labels     = ["Random", "Bayesian GP", "LLM Agent"]
    colors     = ["#dc2626", "#2563eb", "#16a34a"]

    max_lbs = [results[s]["max_l_b"][0] for s in strategies]
    stds    = [results[s]["max_l_b"][1] for s in strategies]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(strategies))
    bars = ax.bar(x, max_lbs, yerr=stds, capsize=6, color=colors, alpha=0.82,
                  error_kw={"elinewidth": 2, "ecolor": "#374151"})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Max L:B Ratio Achieved (mean ± std)", fontsize=11)
    ax.set_title(
        f"Strategy Comparison over {n_iter} Iterations\n"
        f"({len(results['random']['raw_max_lbs'])} independent runs per strategy)",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)
    # Value annotations
    for bar, val, std in zip(bars, max_lbs, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.03,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "strategy_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[COMPARE] Plot saved: {path}")


# Entry point
# Define the main function to run the comparison.
def main(
    n_runs: int  = 3,
    n_iter: int  = 20,
    w_lb:   float = DEFAULT_W_LB,
    w_conv: float = DEFAULT_W_CONV,
    w_ton:  float = DEFAULT_W_TON,
    no_plots: bool = False,
) -> None:
    print(f"[COMPARE] Running {n_runs} independent runs × {n_iter} iterations per strategy...")

    results = {
        "random":   _multi_run_stats(run_random_strategy,   n_runs, n_iter, w_lb, w_conv, w_ton),
        "bayesian": _multi_run_stats(run_bayesian_strategy,  n_runs, n_iter, w_lb, w_conv, w_ton),
        "agent":    _multi_run_stats(run_agent_strategy,     n_runs, n_iter, w_lb, w_conv, w_ton),
    }

    print_comparison_table(results, n_runs, n_iter)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "n_runs": n_runs, "n_iter": n_iter,
        "weights": {"w_lb": w_lb, "w_conv": w_conv, "w_ton": w_ton},
        "results": results,
    }
    report_path = RESULTS_DIR / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[COMPARE] Report saved to '{report_path}'.")

    if not no_plots:
        generate_comparison_plot(results, n_iter)

# Command-line interface 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare agent vs random vs Bayesian strategies."
    )
    parser.add_argument("--n-runs",  type=int,   default=3,
                        help="Independent runs per strategy (default: 3)")
    parser.add_argument("--n-iter",  type=int,   default=20,
                        help="Iterations per run (default: 20)")
    parser.add_argument("--w-lb",    type=float, default=DEFAULT_W_LB)
    parser.add_argument("--w-conv",  type=float, default=DEFAULT_W_CONV)
    parser.add_argument("--w-ton",   type=float, default=DEFAULT_W_TON)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    main(
        n_runs=args.n_runs,
        n_iter=args.n_iter,
        w_lb=args.w_lb,
        w_conv=args.w_conv,
        w_ton=args.w_ton,
        no_plots=args.no_plots,
    )
