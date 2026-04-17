"""
memory_store.py
----------------
Handles saving and loading the experimental history to/from a JSON file.

The history is a list of run records, each containing:
  - iteration      : int
  - conditions     : dict (proposed by the LLM)
  - outcomes       : dict (measured in the lab)
  - reasoning      : str (LLM chain-of-thought)
  - retrieved_passages : list of str (snippets from literature)

For datasets < 20 runs, the full history is always returned.
For larger datasets, a summarized version is returned to save tokens.
"""

import json
import os
from typing import Optional


SUMMARY_THRESHOLD = 20   # After this many runs, summarize older ones


class MemoryStore:
    def __init__(self, filepath: str = "experiment_log.json"):
        self.filepath = filepath

    def load_history(self) -> list:
        """Load and return the full experiment history list."""
        if not os.path.exists(self.filepath):
            return []
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError) as e:
            print(f"[MEMORY] Warning: Could not load history — {e}")
            return []

    def save_history(self, history: list) -> None:
        """Save the full history list to the JSON file."""
        try:
            with open(self.filepath, "w") as f:
                json.dump(history, f, indent=2)
        except IOError as e:
            print(f"[MEMORY] Warning: Could not save history — {e}")

    def get_history_for_prompt(self, history: list) -> list:
        """
        Return history in a form suitable for injection into the LLM prompt.
        - If <= SUMMARY_THRESHOLD runs: return full records
        - If > SUMMARY_THRESHOLD: summarize older runs, keep the last 5 in full
        """
        if len(history) <= SUMMARY_THRESHOLD:
            return history

        # Summarize all but the last 5 runs
        recent = history[-5:]
        older = history[:-5]

        summary = self._summarize_older_runs(older)
        return [{"summary_of_older_runs": summary}] + recent

    def _summarize_older_runs(self, runs: list) -> str:
        """Create a brief textual summary of older experimental runs."""
        if not runs:
            return "No older runs."

        best = max(runs, key=lambda r: r.get("outcomes", {}).get("l_b_ratio", 0))
        worst = min(runs, key=lambda r: r.get("outcomes", {}).get("l_b_ratio", 0))
        avg_conv = sum(r.get("outcomes", {}).get("conversion_pct", 0) for r in runs) / len(runs)
        avg_lb = sum(r.get("outcomes", {}).get("l_b_ratio", 0) for r in runs) / len(runs)

        return (
            f"Summary of {len(runs)} older runs: "
            f"Average conversion = {avg_conv:.1f}%, average L:B = {avg_lb:.2f}. "
            f"Best L:B = {best['outcomes'].get('l_b_ratio', '?')} "
            f"at conditions {best['conditions']}. "
            f"Worst L:B = {worst['outcomes'].get('l_b_ratio', '?')} "
            f"at conditions {worst['conditions']}."
        )

    def add_run(self, run_record: dict) -> None:
        """Convenience method: load, append, and save in one call."""
        history = self.load_history()
        history.append(run_record)
        self.save_history(history)

    def get_best_run(self, metric: str = "l_b_ratio") -> Optional[dict]:
        """Return the run with the highest value for a given outcome metric."""
        history = self.load_history()
        if not history:
            return None
        valid = [r for r in history if metric in r.get("outcomes", {})]
        if not valid:
            return None
        return max(valid, key=lambda r: r["outcomes"][metric])
