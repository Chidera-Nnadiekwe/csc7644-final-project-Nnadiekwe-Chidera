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

Usage:
    python evaluation.py
    python evaluation.py --log my_experiment_log.json
    python evaluation.py --compare random_baseline.json

Output:
    - Console summary table
    - evaluation_report.json  (detailed metrics)
"""

import json
import os
import argparse
import random
from typing import Optional


def load_history(filepath: str) -> list:
    if not os.path.exists(filepath):
        print(f"[EVAL] File not found: {filepath}")
        return []
    with open(filepath, "r") as f:
        return json.load(f)


def compute_metrics(history: list) -> dict:
    """Compute optimization performance metrics from a run history."""
    if not history:
        return {}

    lb_values    = [r["outcomes"].get("l_b_ratio", 0)      for r in history]
    conv_values  = [r["outcomes"].get("conversion_pct", 0) for r in history]
    ton_values   = [r["outcomes"].get("ton", 0)            for r in history]

    max_lb   = max(lb_values)
    target_lb = 0.80 * max_lb  # 80% of max as convergence criterion

    # Convergence speed: first iteration where L:B >= 80% of eventual max
    convergence_iter = None
    for i, lb in enumerate(lb_values, 1):
        if lb >= target_lb:
            convergence_iter = i
            break

    best_run = history[lb_values.index(max_lb)]

    return {
        "total_runs":           len(history),
        "final_l_b_ratio":      lb_values[-1],
        "max_l_b_ratio":        max_lb,
        "max_l_b_iteration":    lb_values.index(max_lb) + 1,
        "final_conversion_pct": conv_values[-1],
        "max_conversion_pct":   max(conv_values),
        "avg_l_b_ratio":        round(sum(lb_values) / len(lb_values), 3),
        "avg_conversion_pct":   round(sum(conv_values) / len(conv_values), 2),
        "convergence_iteration": convergence_iter,
        "best_conditions":      best_run.get("conditions", {}),
        "l_b_progression":      lb_values,
        "conversion_progression": conv_values,
    }


def generate_random_baseline(n_runs: int, seed: int = 42) -> list:
    """
    Generate a fake random-sampling baseline for comparison.
    Samples from the same parameter space the agent explores.
    """
    random.seed(seed)
    history = []
    for i in range(1, n_runs + 1):
        conditions = {
            "temperature_C":           random.uniform(40, 120),
            "pressure_bar":            random.uniform(5, 80),
            "co_h2_ratio":             random.choice(["1:1", "1:2", "2:1"]),
            "catalyst_loading_mol_pct": random.uniform(0.1, 5.0),
            "ligand_loading_eq":       random.uniform(1, 20),
            "reaction_time_h":         random.uniform(1, 24),
        }
        # Simulate random outcomes (poor on average compared to agent)
        outcomes = {
            "conversion_pct": random.uniform(10, 75),
            "l_b_ratio":      random.uniform(0.5, 3.5),
            "ton":            random.uniform(20, 200),
            "notes":          "random baseline simulation"
        }
        history.append({"iteration": i, "conditions": conditions, "outcomes": outcomes})
    return history


def print_summary_table(metrics: dict, label: str = "Agent"):
    """Print a human-readable summary table."""
    print(f"\n{'='*55}")
    print(f"  EVALUATION REPORT — {label}")
    print(f"{'='*55}")
    print(f"  Total runs              : {metrics.get('total_runs', 'N/A')}")
    print(f"  Max L:B ratio achieved  : {metrics.get('max_l_b_ratio', 'N/A')} "
          f"(iteration {metrics.get('max_l_b_iteration', '?')})")
    print(f"  Final L:B ratio         : {metrics.get('final_l_b_ratio', 'N/A')}")
    print(f"  Average L:B ratio       : {metrics.get('avg_l_b_ratio', 'N/A')}")
    print(f"  Max conversion (%)      : {metrics.get('max_conversion_pct', 'N/A')}")
    print(f"  Average conversion (%)  : {metrics.get('avg_conversion_pct', 'N/A')}")
    print(f"  Convergence iteration   : {metrics.get('convergence_iteration', 'Did not converge')}")
    print(f"{'='*55}")

    prog = metrics.get("l_b_progression", [])
    if prog:
        print(f"\n  L:B Ratio Progression:")
        for i, lb in enumerate(prog, 1):
            bar = "█" * int(lb * 3)
            print(f"    Run {i:2d}: {lb:.3f}  {bar}")


def compare_agent_vs_baseline(agent_metrics: dict, baseline_metrics: dict):
    """Print a side-by-side comparison."""
    print(f"\n{'='*55}")
    print(f"  AGENT vs RANDOM BASELINE COMPARISON")
    print(f"{'='*55}")
    print(f"  {'Metric':<30} {'Agent':>10} {'Baseline':>10}")
    print(f"  {'-'*50}")

    metrics_to_compare = [
        ("Max L:B Ratio",        "max_l_b_ratio"),
        ("Avg L:B Ratio",        "avg_l_b_ratio"),
        ("Max Conversion (%)",   "max_conversion_pct"),
        ("Avg Conversion (%)",   "avg_conversion_pct"),
        ("Convergence Iter.",    "convergence_iteration"),
    ]

    for label, key in metrics_to_compare:
        agent_val    = agent_metrics.get(key, "N/A")
        baseline_val = baseline_metrics.get(key, "N/A")

        # Highlight improvements
        try:
            if float(agent_val) > float(baseline_val):
                flag = " ✓"
            elif float(agent_val) < float(baseline_val):
                flag = " ✗"
            else:
                flag = ""
        except (TypeError, ValueError):
            flag = ""

        print(f"  {label:<30} {str(agent_val):>10} {str(baseline_val):>10}{flag}")

    print(f"{'='*55}")


def run_evaluation(log_file: str = "experiment_log.json", compare_file: Optional[str] = None):
    """Main evaluation entry point."""
    history = load_history(log_file)
    if not history:
        print("[EVAL] No history found. Run the agent first.")
        return

    agent_metrics = compute_metrics(history)
    print_summary_table(agent_metrics, label="Agent")

    # Generate or load baseline
    baseline_history = (
        load_history(compare_file) if compare_file
        else generate_random_baseline(len(history))
    )
    baseline_metrics = compute_metrics(baseline_history)
    print_summary_table(baseline_metrics, label="Random Baseline")

    compare_agent_vs_baseline(agent_metrics, baseline_metrics)

    # Save report
    report = {
        "agent":    agent_metrics,
        "baseline": baseline_metrics,
    }
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n[EVAL] Report saved to 'evaluation_report.json'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent optimization performance.")
    parser.add_argument("--log",     default="experiment_log.json",
                        help="Path to the agent experiment log JSON file.")
    parser.add_argument("--compare", default=None,
                        help="Path to a baseline log JSON file (optional).")
    args = parser.parse_args()

    run_evaluation(log_file=args.log, compare_file=args.compare)
