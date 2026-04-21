"""
agent_controller.py
--------------------
Main controller for the Agentic LLM Experimental Optimizer.
This script runs the full agent loop:
  Retrieve -> Build Prompt -> Call LLM -> Validate -> Display -> Await Results -> Parse -> Log

Run from the project root:
    python src/agent_controller.py

CLI options:
    --max-iter       Maximum number of optimization iterations (default: 10)
    --target-lb      L:B ratio for early stopping (default: 5.0)
    --target-conv    % conversion for early stopping (default: 80.0)
    --substrate      Substrate name or SMILES (default: 1-hexene)
    --memory-file    Path to experiment log JSON (default: data/experiment_log.json)
    --ingest-mode    Result entry mode: manual | json | gc (default: manual)
    --seed-file      Path to seed data JSON to pre-load on first run
    --target-ton     Minimum TON for stopping (default: 0.0, not enforced)
    --consecutive    Consecutive runs all targets must be met before stopping (default: 2)

Before running:
    pip install -r requirements.txt
    Set OPENAI_API_KEY and OPENROUTER_API_KEY in a .env file
"""

# Import standard libraries
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src/ to path for imports
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

# Import project modules
from memory_store import MemoryStore
from rag_retriever import RAGRetriever
from llm_planner import LLMPlanner
from result_parser import parse_experimental_result, load_seed_data
from tool_layer import validate_smiles

load_dotenv(PROJECT_ROOT / ".env")


# DEFAULTS
DEFAULT_MAX_ITERATIONS    = 10
DEFAULT_TARGET_L_B_RATIO  = 5.0
DEFAULT_TARGET_CONVERSION = 80.0
DEFAULT_CORPUS_PATH       = str(PROJECT_ROOT / "data" / "corpus")
DEFAULT_MEMORY_FILE       = str(PROJECT_ROOT / "data" / "experiment_log.json")
DEFAULT_SEED_FILE         = str(PROJECT_ROOT / "data" / "seed_data.json")

# Defined a function to check stopping criteria based on iteration count and target metrics
def check_stopping_criteria(history: list, max_iter: int,
                             target_lb: float, target_conv: float,
                             target_ton: float = 0.0,
                             consecutive_required: int = 2) -> bool:
    """Return True if the agent should stop.
    
    Stops when ALL three metrics (L:B, conversion, TON) meet their targets
    simultaneously for `consecutive_required` runs in a row, or when
    max_iter is reached.
    """
    if len(history) >= max_iter:
        print(f"\n[STOP] Reached maximum iterations ({max_iter}).")
        return True

    if len(history) < consecutive_required:
        return False

    # Check the last N runs all satisfy every threshold simultaneously
    recent = history[-consecutive_required:]
    all_met = all(
        r.get("outcomes", {}).get("l_b_ratio", 0)       >= target_lb   and
        r.get("outcomes", {}).get("conversion_pct", 0)  >= target_conv and
        r.get("outcomes", {}).get("ton", 0)             >= target_ton
        for r in recent
    )

    if all_met:
        print(f"\n[STOP] All targets met for {consecutive_required} consecutive run(s).")
        print(f"       L:B ≥ {target_lb}, Conversion ≥ {target_conv}%, TON ≥ {target_ton}")
        return True

    return False

# Defined a function to display proposed conditions in a formatted way
def display_proposed_conditions(conditions: dict) -> None:
    """Pretty-print the conditions the agent is proposing."""
    print("\n" + "=" * 55)
    print("  AGENT PROPOSED CONDITIONS")
    print("=" * 55)
    for key, value in conditions.items():
        print(f"  {key:<30}: {value}")
    print("=" * 55)

# Defined functions to get experimental results from different input modes (manual, JSON, GC CSV)
def get_experimental_result_manual() -> dict:
    """Interactive CLI prompt for the lab chemist to enter observed results."""
    print("\n[INPUT REQUIRED] Run the experiment with the conditions above.")
    print("Enter the observed results (press Enter to use the default value):\n")

    def prompt_float(label: str, default: float) -> float:
        raw = input(f"  {label} [{default}]: ").strip()
        try:
            return float(raw) if raw else default
        except ValueError:
            print(f"  [WARN] Invalid number; using default {default}.")
            return default

    conversion = prompt_float("Conversion (%)",  45.0)
    l_b_ratio  = prompt_float("L:B Ratio",        2.5)
    ton        = prompt_float("TON",              120.0)
    notes      = input("  Notes (optional): ").strip()
    return parse_experimental_result(conversion, l_b_ratio, ton, notes)

# Defined a function to get experimental results from a JSON string input
def get_experimental_result_json() -> dict:
    """Accept a JSON string from stdin for scripted/automated use."""
    print("\n[INPUT REQUIRED] Paste a JSON result dict and press Enter:")
    raw = input("  JSON> ").strip()
    try:
        data = json.loads(raw)
        return parse_experimental_result(
            conversion_pct=data.get("conversion_pct", 0.0),
            l_b_ratio=data.get("l_b_ratio", 0.0),
            ton=data.get("ton", 0.0),
            notes=data.get("notes", "")
        )
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse error: {e}. Using zeros.")
        return parse_experimental_result(0.0, 0.0, 0.0, "parse_error")

# Defined a function to get experimental results from a GC export CSV file
def get_experimental_result_gc() -> dict:
    """Parse results from a GC export CSV file (path prompted at runtime)."""
    from result_parser import parse_from_gc_csv
    gc_csv_path = input("\n[INPUT REQUIRED] Path to GC CSV file: ").strip()
    result = parse_from_gc_csv(gc_csv_path)
    if result is None:
        print("  [WARN] Could not parse GC CSV. Using zeros.")
        return parse_experimental_result(0.0, 0.0, 0.0, "gc_parse_failed")
    return result

# Defined the main function to run the agent loop
def run_agent(args: argparse.Namespace) -> None:
    """Main agent loop."""
    print("\n" + "=" * 65)
    print("  AGENTIC LLM OPTIMIZER: Hydroformylation / Isomerization")
    print("=" * 65)

    memory    = MemoryStore(filepath=args.memory_file)
    retriever = RAGRetriever(
        corpus_dir=args.corpus_path,
        index_dir=str(PROJECT_ROOT / "data" / "faiss_index")
    )
    planner   = LLMPlanner()

    # Load history; seed from seed_data.json on first run
    history = memory.load_history()
    if not history and os.path.exists(args.seed_file):
        print(f"\n[MEMORY] No prior history. Loading seed data from '{args.seed_file}'.")
        history = load_seed_data(args.seed_file)
        if history:
            memory.save_history(history)

    print(f"\n[MEMORY] Loaded {len(history)} prior experimental run(s).")

    iteration = len(history) + 1

    # All arguments passed through
    while not check_stopping_criteria(
        history, args.max_iter, args.target_lb, args.target_conv,
        args.target_ton, args.consecutive_required
    ):
        print(f"\n{'─' * 65}")
        print(f"  ITERATION {iteration}  |  Substrate: {args.substrate}")
        print(f"{'─' * 65}")

        # Step 1: Build retrieval query
        query = (
            f"hydroformylation isomerization L:B selectivity {args.substrate} "
            "linear aldehyde optimization catalyst ligand"
        )
        if history:
            last_cond = history[-1].get("conditions", {})
            query += (
                f" temperature {last_cond.get('temperature_C', '?')}C "
                f"pressure {last_cond.get('pressure_bar', '?')}bar"
            )

        print(f"\n[RAG] Query: '{query[:70]}...'")
        retrieved_passages = retriever.retrieve(query, top_k=3)
        print(f"[RAG] Retrieved {len(retrieved_passages)} passage(s).")

        # Step 2: Build prompt history (summarized if > 20 runs)
        history_for_prompt = memory.get_history_for_prompt(history)

        # Step 3: Call LLM planner
        print("\n[LLM] Calling planner (this may take a moment)...")
        response = planner.propose_conditions(
            history=history_for_prompt,
            retrieved_passages=retrieved_passages,
            iteration=iteration
        )

        if response is None:
            print("[ERROR] LLM did not return a valid response. Stopping.")
            break

        proposed_conditions = response.get("proposed_conditions", {})
        reasoning           = response.get("reasoning", "No reasoning provided.")

        # Step 4: Validate SMILES
        smiles = proposed_conditions.get("substrate_smiles", "")
        if smiles:
            is_valid = validate_smiles(smiles)
            if not is_valid:
                print(
                    f"[TOOL] Warning: SMILES '{smiles}' failed RDKit validation. "
                    "Logging warning but continuing."
                )

        # Step 5: Display conditions and reasoning
        display_proposed_conditions(proposed_conditions)
        print(f"\n[REASONING]\n{reasoning}\n")

        # Step 6: Collect results
        if args.ingest_mode == "manual":
            outcomes = get_experimental_result_manual()
        elif args.ingest_mode == "json":
            outcomes = get_experimental_result_json()
        elif args.ingest_mode == "gc":
            outcomes = get_experimental_result_gc()
        else:
            print(f"[WARN] Unknown ingest-mode '{args.ingest_mode}'. Defaulting to manual.")
            outcomes = get_experimental_result_manual()

        # Step 7: Log run
        run_record = {
            "iteration":          iteration,
            "conditions":         proposed_conditions,
            "outcomes":           outcomes,
            "reasoning":          reasoning,
            "retrieved_passages": [p["text"][:200] for p in retrieved_passages],
        }
        history.append(run_record)
        memory.save_history(history)
        print(f"\n[MEMORY] Run {iteration} saved to '{args.memory_file}'.")
        print(
            f"[RESULT] Conversion={outcomes['conversion_pct']:.1f}%  "
            f"L:B={outcomes['l_b_ratio']:.3f}  TON={outcomes['ton']:.1f}"
        )

        iteration += 1

    # Completion summary
    print("\n[DONE] Agent loop complete.")
    print(f"Total runs logged: {len(history)}")
    if history:
        best = max(history, key=lambda r: r["outcomes"].get("l_b_ratio", 0))
        print(f"\nBest run (by L:B ratio):")
        print(f"  Iteration : {best['iteration']}")
        print(f"  Conditions: {json.dumps(best['conditions'], indent=4)}")
        print(f"  Outcomes  : {best['outcomes']}")

# Defined a function to parse command-line arguments for configuring the agent's behavior
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic LLM optimizer for olefin hydroformylation/isomerization."
    )
    parser.add_argument("--max-iter",    type=int,   default=DEFAULT_MAX_ITERATIONS,
                        dest="max_iter",
                        help="Maximum optimization iterations (default: 10)")
    parser.add_argument("--target-lb",   type=float, default=DEFAULT_TARGET_L_B_RATIO,
                        dest="target_lb",
                        help="L:B ratio for early stopping (default: 5.0)")
    parser.add_argument("--target-conv", type=float, default=DEFAULT_TARGET_CONVERSION,
                        dest="target_conv",
                        help="Conversion %% for early stopping (default: 80.0)")
    parser.add_argument("--substrate",   type=str,   default="1-hexene",
                        help="Substrate name or SMILES (default: 1-hexene)")
    parser.add_argument("--corpus-path", type=str,   default=DEFAULT_CORPUS_PATH,
                        dest="corpus_path",
                        help="Path to corpus .txt directory")
    parser.add_argument("--memory-file", type=str,   default=DEFAULT_MEMORY_FILE,
                        dest="memory_file",
                        help="Path to experiment log JSON")
    parser.add_argument("--ingest-mode", type=str,   default="manual",
                        dest="ingest_mode", choices=["manual", "json", "gc"],
                        help="Result entry mode: manual | json | gc")
    parser.add_argument("--seed-file",   type=str,   default=DEFAULT_SEED_FILE,
                        dest="seed_file",
                        help="Path to seed data JSON")
    parser.add_argument("--target-ton", type=float, default=0.0,
                        dest="target_ton",
                        help="Minimum TON for stopping (default: 0.0, i.e. not enforced)")
    parser.add_argument("--consecutive", type=int, default=2,
                        dest="consecutive_required",
                        help="Consecutive runs all targets must be met before stopping (default: 2)")
    return parser.parse_args()


if __name__ == "__main__":
    run_agent(parse_args())
