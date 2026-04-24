"""
evaluation.py
--------------
Evaluate the agent's optimization performance.

Metrics computed:
  1. L:B ratio progression across iterations (primary metric)
  2. Conversion % progression
  3. TON progression
  4. Convergence speed (iterations to reach 80% of max observed L:B)
  5. Comparison vs a random-sampling baseline
  6. Multi-objective composite reward (weighted L:B + conversion + TON)
  7. Pareto front identification across [L:B, conversion, TON]

Usage (from project root):
    python scripts/evaluation.py
    python scripts/evaluation.py --log data/experiment_log.json
    python scripts/evaluation.py --compare data/random_baseline.json
    python scripts/evaluation.py --log data/experiment_log.json --no-plots
    python scripts/evaluation.py --w-lb 0.5 --w-conv 0.3 --w-ton 0.2

Output:
    - Console summary table with ASCII bar chart
    - results/evaluation_report.json
    - results/figures/lb_convergence.png
    - results/figures/yield_convergence.png
    - results/figures/parameter_lb_scatter.png
    - results/figures/composite_reward.png
    - results/figures/pareto_front.png
"""

# Import standard libraries
import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional

# Define paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"

# Default composite-reward weights (must sum to 1.0)
DEFAULT_W_LB   = 0.50
DEFAULT_W_CONV = 0.30
DEFAULT_W_TON  = 0.20

# Define functions for loading history
def load_history(filepath: str) -> list:
    """Load experiment log from a JSON file."""
    if not os.path.exists(filepath):
        print(f"[EVAL] File not found: {filepath}")
        return []
    with open(filepath, "r") as f:
        return json.load(f)

# Define function to compute metrics
def compute_metrics(history: list) -> dict:
    """Compute optimization performance metrics from a run history."""
    if not history:
        return {}
    # Extract relevant metrics from history
    lb_values   = [r["outcomes"].get("l_b_ratio",      0) for r in history]
    conv_values = [r["outcomes"].get("conversion_pct",  0) for r in history]
    ton_values  = [r["outcomes"].get("ton",             0) for r in history]

    max_lb    = max(lb_values)
    target_lb = 0.80 * max_lb

    # Determine convergence iteration (first iteration to reach 80% of max L:B)
    convergence_iter = None
    for i, lb in enumerate(lb_values, 1):
        if lb >= target_lb:
            convergence_iter = i
            break

    best_run = history[lb_values.index(max_lb)]

    # Compile metrics into a dictionary
    return {
        "total_runs":              len(history),
        "final_l_b_ratio":         lb_values[-1],
        "max_l_b_ratio":           max_lb,
        "max_l_b_iteration":       lb_values.index(max_lb) + 1,
        "final_conversion_pct":    conv_values[-1],
        "max_conversion_pct":      max(conv_values),
        "avg_l_b_ratio":           round(sum(lb_values)   / len(lb_values),   3),
        "avg_conversion_pct":      round(sum(conv_values) / len(conv_values),  2),
        "avg_ton":                 round(sum(ton_values)  / len(ton_values),   1), 
        "convergence_iteration":   convergence_iter,
        "best_conditions":         best_run.get("conditions", {}),
        "l_b_progression":         lb_values,
        "conversion_progression":  conv_values,
        "ton_progression":         ton_values,
    }


# Multi-objective composite reward
# Define a composite reward that combines L:B ratio, conversion %, and TON into a single score.
def compute_composite_reward(
    history: list,
    w_lb:   float = DEFAULT_W_LB,
    w_conv: float = DEFAULT_W_CONV,
    w_ton:  float = DEFAULT_W_TON,
) -> list:
    """Compute a per-iteration composite reward score.

    Each metric is min-max normalised to [0, 1] across the run history before
    applying the weights.  This makes the composite score comparable across
    different experimental campaigns regardless of absolute magnitude.

    Composite reward = w_lb * lb_norm + w_conv * conv_norm + w_ton * ton_norm

    Parameters
    ----------
    history  : list of run records (each with an 'outcomes' dict)
    w_lb     : weight for L:B ratio          (default 0.50)
    w_conv   : weight for conversion %        (default 0.30)
    w_ton    : weight for TON                 (default 0.20)

    Returns
    -------
    list of float  – one composite score per run (same length as history)
    """
    if not history:
        return []

    lb_values   = [r["outcomes"].get("l_b_ratio",     0.0) for r in history]
    conv_values = [r["outcomes"].get("conversion_pct", 0.0) for r in history]
    ton_values  = [r["outcomes"].get("ton",            0.0) for r in history]

    def _minmax(vals: list) -> list:
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return [0.5] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    lb_n   = _minmax(lb_values)
    conv_n = _minmax(conv_values)
    ton_n  = _minmax(ton_values)

    return [
        round(w_lb * lb + w_conv * cv + w_ton * tn, 4)
        for lb, cv, tn in zip(lb_n, conv_n, ton_n)
    ]

# Define function to identify Pareto front
def identify_pareto_front(history: list) -> list:
    """Identify the Pareto-optimal runs across three objectives: L:B, conversion, TON.

    A run is Pareto-optimal if no other run is at least as good on *all three*
    objectives and strictly better on at least one.

    Parameters
    ----------
    history : list of run records

    Returns
    -------
    list of int  – indices (0-based) of Pareto-optimal runs in ``history``
    """
    if not history:
        return []

    points = [
        (
            r["outcomes"].get("l_b_ratio",     0.0),
            r["outcomes"].get("conversion_pct", 0.0),
            r["outcomes"].get("ton",            0.0),
        )
        for r in history
    ]

    pareto_indices = []
    n = len(points)
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is >= i on all objectives and > on at least one
            if (
                points[j][0] >= points[i][0] and
                points[j][1] >= points[i][1] and
                points[j][2] >= points[i][2] and
                (points[j][0] > points[i][0] or
                 points[j][1] > points[i][1] or
                 points[j][2] > points[i][2])
            ):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)

    return pareto_indices

# Define function to print Pareto summary
def print_pareto_summary(history: list, pareto_indices: list) -> None:
    """Print a human-readable summary of Pareto-optimal runs."""
    print(f"\n{'=' * 58}")
    print(f"  PARETO-OPTIMAL RUNS  ({len(pareto_indices)} of {len(history)} total)")
    print(f"{'=' * 58}")
    print(f"  {'Iter':>5}  {'L:B':>7}  {'Conv%':>7}  {'TON':>7}")
    print(f"  {'-' * 46}")
    for idx in sorted(pareto_indices):
        out = history[idx]["outcomes"]
        print(
            f"  {history[idx].get('iteration', idx+1):>5}  "
            f"{out.get('l_b_ratio', 0):>7.3f}  "
            f"{out.get('conversion_pct', 0):>7.1f}  "
            f"{out.get('ton', 0):>7.1f}"
        )
    print(f"{'=' * 58}")

# Define function to generate random baseline
def generate_random_baseline(n_runs: int, seed: int = 42) -> list:
    """
    Generate a random-sampling baseline history for comparison.
    Samples from the same parameter space as the agent.
    """
    random.seed(seed)
    history = []
    for i in range(1, n_runs + 1):
        conditions = {
            "temperature_C":            random.uniform(40, 120),
            "pressure_bar":             random.uniform(5, 80),
            "co_h2_ratio":              random.choice(["1:1", "1:2", "2:1"]),
            "catalyst_loading_mol_pct": random.uniform(0.1, 5.0),
            "ligand_loading_eq":        random.uniform(1, 20),
            "reaction_time_h":          random.uniform(1, 24),
        }
        outcomes = {
            "conversion_pct": random.uniform(10, 75),
            "l_b_ratio":      random.uniform(0.5, 3.5),
            "ton":            random.uniform(20, 200),
            "notes":          "random baseline simulation",
        }
        history.append({"iteration": i, "conditions": conditions, "outcomes": outcomes})
    return history

# Define function to print summary table
def print_summary_table(metrics: dict, label: str = "Agent") -> None:
    """Print a human-readable summary table with ASCII bar chart."""
    print(f"\n{'=' * 58}")
    print(f"  EVALUATION REPORT — {label}")
    print(f"{'=' * 58}")
    print(f"  Total runs              : {metrics.get('total_runs', 'N/A')}")
    print(f"  Max L:B ratio achieved  : {metrics.get('max_l_b_ratio', 'N/A')} "
          f"(iteration {metrics.get('max_l_b_iteration', '?')})")
    print(f"  Final L:B ratio         : {metrics.get('final_l_b_ratio', 'N/A')}")
    print(f"  Average L:B ratio       : {metrics.get('avg_l_b_ratio', 'N/A')}")
    print(f"  Max conversion (%)      : {metrics.get('max_conversion_pct', 'N/A')}")
    print(f"  Average conversion (%)  : {metrics.get('avg_conversion_pct', 'N/A')}")
    print(f"  Average TON             : {metrics.get('avg_ton', 'N/A')}")  
    print(f"  Convergence iteration   : {metrics.get('convergence_iteration', 'Did not converge')}")
    print(f"{'=' * 58}")

    # Add ASCII bar chart for L:B progression
    prog = metrics.get("l_b_progression", [])
    if prog:
        print(f"\n  L:B Ratio Progression:")
        for i, lb in enumerate(prog, 1):
            bar = "\u2588" * int(lb * 3)
            print(f"    Run {i:2d}: {lb:.3f}  {bar}")

# Define function to compare agent vs baseline
def compare_agent_vs_baseline(agent_metrics: dict, baseline_metrics: dict) -> None:
    """Print a side-by-side comparison."""
    print(f"\n{'=' * 58}")
    print(f"  AGENT vs RANDOM BASELINE COMPARISON")
    print(f"{'=' * 58}")
    print(f"  {'Metric':<32} {'Agent':>10} {'Baseline':>10}")
    print(f"  {'-' * 54}")

    # Define rows to compare
    rows = [
        ("Max L:B Ratio",       "max_l_b_ratio"),
        ("Avg L:B Ratio",       "avg_l_b_ratio"),
        ("Max Conversion (%)",  "max_conversion_pct"),
        ("Avg Conversion (%)",  "avg_conversion_pct"),
        ("Average TON",         "avg_ton"),              
        ("Convergence Iter.",   "convergence_iteration"),
    ]

    # Print each metric with a checkmark if agent outperforms baseline
    for label, key in rows:
        a_val = agent_metrics.get(key, "N/A")
        b_val = baseline_metrics.get(key, "N/A")
        try:
            flag = " ✓" if float(a_val) > float(b_val) else (" ✗" if float(a_val) < float(b_val) else "")
        except (TypeError, ValueError):
            flag = ""
        print(f"  {label:<32} {str(a_val):>10} {str(b_val):>10}{flag}")

    print(f"{'=' * 58}")

# Define function to generate plots
def generate_plots(agent_metrics: dict, baseline_metrics: dict) -> None:
    """Generate and save convergence and scatter plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for headless environments
        import matplotlib.pyplot as plt
    except ImportError:
        print("[EVAL] matplotlib not available. Skipping plot generation.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: L:B convergence progression
    agent_lb    = agent_metrics.get("l_b_progression", [])
    baseline_lb = baseline_metrics.get("l_b_progression", [])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(agent_lb) + 1), agent_lb,
            "o-", color="#2563eb", linewidth=2, markersize=7, label="Agent")
    if baseline_lb:
        ax.plot(range(1, len(baseline_lb) + 1), baseline_lb,
                "s--", color="#dc2626", linewidth=1.5, markersize=5,
                alpha=0.7, label="Random Baseline")
    if agent_lb:
        ax.axhline(max(agent_lb), color="#2563eb", linestyle=":", alpha=0.4,
                   label=f"Agent best: {max(agent_lb):.2f}")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("L:B Selectivity Ratio", fontsize=12)
    ax.set_title("Linear-to-Branch Selectivity vs Iteration", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    lb_path = FIGURES_DIR / "lb_convergence.png"
    plt.savefig(lb_path, dpi=150)
    plt.close()
    print(f"[EVAL] Plot saved: {lb_path}")

    # Plot 2: Conversion progression
    agent_conv    = agent_metrics.get("conversion_progression", [])
    baseline_conv = baseline_metrics.get("conversion_progression", [])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(range(1, len(agent_conv) + 1), agent_conv, alpha=0.15, color="#16a34a")
    ax.plot(range(1, len(agent_conv) + 1), agent_conv,
            "o-", color="#16a34a", linewidth=2, markersize=7, label="Agent")
    if baseline_conv:
        ax.plot(range(1, len(baseline_conv) + 1), baseline_conv,
                "s--", color="#dc2626", linewidth=1.5, markersize=5,
                alpha=0.7, label="Random Baseline")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Conversion (%)", fontsize=12)
    ax.set_title("Substrate Conversion vs Iteration", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    conv_path = FIGURES_DIR / "yield_convergence.png"
    plt.savefig(conv_path, dpi=150)
    plt.close()
    print(f"[EVAL] Plot saved: {conv_path}")


def generate_multiobjective_plots(
    agent_history: list,
    baseline_history: list,
    w_lb: float   = DEFAULT_W_LB,
    w_conv: float = DEFAULT_W_CONV,
    w_ton: float  = DEFAULT_W_TON,
) -> None:
    """Generate composite-reward and Pareto-front plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[EVAL] matplotlib not available. Skipping multi-objective plots.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Plot 3: Composite reward progression ─────────────────────────────────
    agent_reward    = compute_composite_reward(agent_history, w_lb, w_conv, w_ton)
    baseline_reward = compute_composite_reward(baseline_history, w_lb, w_conv, w_ton)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(agent_reward) + 1), agent_reward,
            "o-", color="#7c3aed", linewidth=2, markersize=7, label="Agent")
    if baseline_reward:
        ax.plot(range(1, len(baseline_reward) + 1), baseline_reward,
                "s--", color="#dc2626", linewidth=1.5, markersize=5,
                alpha=0.7, label="Random Baseline")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(
        f"Composite Reward\n(w_LB={w_lb}, w_conv={w_conv}, w_TON={w_ton})",
        fontsize=10
    )
    ax.set_title("Multi-Objective Composite Reward vs Iteration",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    reward_path = FIGURES_DIR / "composite_reward.png"
    plt.savefig(reward_path, dpi=150)
    plt.close()
    print(f"[EVAL] Plot saved: {reward_path}")

    # ── Plot 4: Pareto front (L:B vs Conversion, TON as bubble size) ─────────
    if not agent_history:
        return

    pareto_indices = identify_pareto_front(agent_history)
    pareto_set     = set(pareto_indices)

    lb_vals   = [r["outcomes"].get("l_b_ratio",     0.0) for r in agent_history]
    conv_vals = [r["outcomes"].get("conversion_pct", 0.0) for r in agent_history]
    ton_vals  = [r["outcomes"].get("ton",            1.0) for r in agent_history]
    max_ton   = max(ton_vals) if max(ton_vals) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8, 6))

    # Non-Pareto points
    non_p_idx = [i for i in range(len(agent_history)) if i not in pareto_set]
    if non_p_idx:
        ax.scatter(
            [lb_vals[i]   for i in non_p_idx],
            [conv_vals[i] for i in non_p_idx],
            s=[200 * ton_vals[i] / max_ton for i in non_p_idx],
            c="#94a3b8", alpha=0.6, label="Dominated runs", zorder=2
        )

    # Pareto-optimal points
    if pareto_indices:
        ax.scatter(
            [lb_vals[i]   for i in pareto_indices],
            [conv_vals[i] for i in pareto_indices],
            s=[200 * ton_vals[i] / max_ton for i in pareto_indices],
            c="#f59e0b", edgecolors="#b45309", linewidths=1.5,
            alpha=0.9, label="Pareto-optimal", zorder=3
        )
        # Annotate iteration numbers on Pareto runs
        for i in pareto_indices:
            ax.annotate(
                str(agent_history[i].get("iteration", i + 1)),
                (lb_vals[i], conv_vals[i]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, color="#92400e"
            )

    ax.set_xlabel("L:B Selectivity Ratio", fontsize=12)
    ax.set_ylabel("Conversion (%)", fontsize=12)
    ax.set_title(
        "Pareto Front: L:B vs Conversion\n(bubble size ∝ TON)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pareto_path = FIGURES_DIR / "pareto_front.png"
    plt.savefig(pareto_path, dpi=150)
    plt.close()
    print(f"[EVAL] Plot saved: {pareto_path}")

# Define main evaluation function
def run_evaluation(
    log_file: str = str(PROJECT_ROOT / "data" / "experiment_log.json"),
    compare_file: Optional[str] = None,
    no_plots: bool = False,
    w_lb:   float = DEFAULT_W_LB,
    w_conv: float = DEFAULT_W_CONV,
    w_ton:  float = DEFAULT_W_TON,
) -> None:
    """Main evaluation entry point."""
    history = load_history(log_file)
    if not history:
        print("[EVAL] No history found. Run the agent first.")
        return
    
    # Compute metrics for the agent
    agent_metrics = compute_metrics(history)
    print_summary_table(agent_metrics, label="Agent")
    
    baseline_history = (
        load_history(compare_file) if compare_file
        else generate_random_baseline(len(history))
    )
    baseline_metrics = compute_metrics(baseline_history)
    print_summary_table(baseline_metrics, label="Random Baseline")

    compare_agent_vs_baseline(agent_metrics, baseline_metrics)

    # ── Multi-objective analysis ──────────────────────────────────────────────
    agent_reward = compute_composite_reward(history, w_lb, w_conv, w_ton)
    agent_metrics["composite_reward_progression"] = agent_reward
    agent_metrics["composite_reward_weights"] = {
        "w_lb": w_lb, "w_conv": w_conv, "w_ton": w_ton
    }

    pareto_indices = identify_pareto_front(history)
    agent_metrics["pareto_optimal_iterations"] = [
        history[i].get("iteration", i + 1) for i in pareto_indices
    ]
    print_pareto_summary(history, pareto_indices)

    # Save report to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "evaluation_report.json"
    report = {"agent": agent_metrics, "baseline": baseline_metrics}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[EVAL] Report saved to '{report_path}'.")

    if not no_plots:
        generate_plots(agent_metrics, baseline_metrics)
        generate_multiobjective_plots(history, baseline_history, w_lb, w_conv, w_ton)

# Entry point for command-line execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent optimization performance.")
    parser.add_argument(
        "--log", default=str(PROJECT_ROOT / "data" / "experiment_log.json"),
        help="Path to the agent experiment log JSON file."
    )
    parser.add_argument(
        "--compare", default=None,
        help="Path to a baseline log JSON file (optional)."
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib plot generation."
    )
    parser.add_argument(
        "--w-lb", type=float, default=DEFAULT_W_LB,
        help=f"Composite reward weight for L:B ratio (default: {DEFAULT_W_LB})"
    )
    parser.add_argument(
        "--w-conv", type=float, default=DEFAULT_W_CONV,
        help=f"Composite reward weight for conversion %% (default: {DEFAULT_W_CONV})"
    )
    parser.add_argument(
        "--w-ton", type=float, default=DEFAULT_W_TON,
        help=f"Composite reward weight for TON (default: {DEFAULT_W_TON})"
    )
    args = parser.parse_args()
    run_evaluation(
        log_file=args.log,
        compare_file=args.compare,
        no_plots=args.no_plots,
        w_lb=args.w_lb,
        w_conv=args.w_conv,
        w_ton=args.w_ton,
    )
