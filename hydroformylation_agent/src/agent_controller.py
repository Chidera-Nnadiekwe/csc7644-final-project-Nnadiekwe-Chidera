"""
agent_controller.py
--------------------
Main controller for the Agentic LLM Experimental Optimizer.
This script runs the full agent loop:
  Retrieve → Build Prompt → Call LLM → Parse Output → Log → Await Next Results

Run:
    python agent_controller.py

Before running:
    pip install openai faiss-cpu numpy python-dotenv rdkit
    Set OPENAI_API_KEY and OPENROUTER_API_KEY in a .env file
"""

import json
import os
from dotenv import load_dotenv

from memory_store import MemoryStore
from rag_retriever import RAGRetriever
from llm_planner import LLMPlanner
from result_parser import parse_experimental_result
from tool_layer import validate_smiles

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION — edit these to match your run
# ─────────────────────────────────────────────
MAX_ITERATIONS = 10          # Stop after this many agent cycles
TARGET_L_B_RATIO = 5.0       # Stop early if L:B ratio exceeds this
TARGET_CONVERSION = 80.0     # Stop early if % conversion exceeds this
CORPUS_PATH = "corpus/"      # Folder containing your .txt literature files
MEMORY_FILE = "experiment_log.json"  # Where run history is saved


def check_stopping_criteria(history: list) -> bool:
    """Return True if the agent should stop (target reached or max iterations)."""
    if len(history) >= MAX_ITERATIONS:
        print(f"\n[STOP] Reached maximum iterations ({MAX_ITERATIONS}).")
        return True

    if history:
        last = history[-1]
        outcomes = last.get("outcomes", {})
        lb = outcomes.get("l_b_ratio", 0)
        conv = outcomes.get("conversion_pct", 0)
        if lb >= TARGET_L_B_RATIO and conv >= TARGET_CONVERSION:
            print(f"\n[STOP] Target achieved! L:B = {lb}, Conversion = {conv}%")
            return True

    return False


def display_proposed_conditions(conditions: dict):
    """Pretty-print the conditions the agent is proposing."""
    print("\n" + "="*50)
    print("  AGENT PROPOSED CONDITIONS")
    print("="*50)
    for key, value in conditions.items():
        print(f"  {key:<25}: {value}")
    print("="*50)


def get_experimental_result_from_user(conditions: dict) -> dict:
    """
    In a real lab setting, you would run the experiment and then
    enter the results here. This function simulates that input step.
    
    Returns a dict with keys: conversion_pct, l_b_ratio, ton, notes
    """
    print("\n[INPUT REQUIRED] Run the experiment with the conditions above.")
    print("Enter the observed results (press Enter to use a simulated value):\n")

    def prompt_float(label, default):
        raw = input(f"  {label} [{default}]: ").strip()
        return float(raw) if raw else default

    conversion = prompt_float("Conversion (%)", 45.0)
    l_b_ratio  = prompt_float("L:B Ratio",     2.5)
    ton        = prompt_float("TON",            120.0)
    notes      = input("  Notes (optional): ").strip()

    return {
        "conversion_pct": conversion,
        "l_b_ratio": l_b_ratio,
        "ton": ton,
        "notes": notes
    }


def run_agent():
    """Main agent loop."""
    print("\n" + "="*60)
    print("  AGENTIC LLM OPTIMIZER: Hydroformylation / Isomerization")
    print("="*60)

    # 1. Initialize components
    memory = MemoryStore(filepath=MEMORY_FILE)
    retriever = RAGRetriever(corpus_dir=CORPUS_PATH)
    planner = LLMPlanner()

    # 2. Load existing history (if any)
    history = memory.load_history()
    print(f"\n[MEMORY] Loaded {len(history)} prior experimental runs.")

    # 3. Main loop
    iteration = len(history) + 1

    while not check_stopping_criteria(history):
        print(f"\n{'─'*60}")
        print(f"  ITERATION {iteration}")
        print(f"{'─'*60}")

        # Step A: Retrieve relevant literature
        query = f"hydroformylation isomerization optimization L:B selectivity"
        if history:
            last = history[-1]
            cond = last.get("conditions", {})
            query += f" temperature {cond.get('temperature_C','?')}C pressure {cond.get('pressure_bar','?')}bar"

        print(f"\n[RAG] Retrieving literature for query: '{query[:60]}...'")
        retrieved_passages = retriever.retrieve(query, top_k=3)
        print(f"[RAG] Retrieved {len(retrieved_passages)} passages.")

        # Step B: Build prompt and call LLM planner
        print("\n[LLM] Calling planner (this may take a moment)...")
        response = planner.propose_conditions(
            history=history,
            retrieved_passages=retrieved_passages,
            iteration=iteration
        )

        if response is None:
            print("[ERROR] LLM did not return a valid response. Stopping.")
            break

        proposed_conditions = response.get("proposed_conditions", {})
        reasoning = response.get("reasoning", "No reasoning provided.")

        # Step C: Validate the substrate SMILES using RDKit
        smiles = proposed_conditions.get("substrate_smiles", "")
        if smiles:
            is_valid = validate_smiles(smiles)
            if not is_valid:
                print(f"[TOOL] Warning: SMILES '{smiles}' failed RDKit validation. Proceeding anyway.")

        # Step D: Display what the agent proposes
        display_proposed_conditions(proposed_conditions)
        print(f"\n[REASONING]\n{reasoning}\n")

        # Step E: Get experimental result (real or simulated)
        outcomes = get_experimental_result_from_user(proposed_conditions)

        # Step F: Log everything to memory
        run_record = {
            "iteration": iteration,
            "conditions": proposed_conditions,
            "outcomes": outcomes,
            "reasoning": reasoning,
            "retrieved_passages": [p["text"][:200] for p in retrieved_passages]  # Save snippet only
        }
        history.append(run_record)
        memory.save_history(history)
        print(f"\n[MEMORY] Run {iteration} saved to '{MEMORY_FILE}'.")

        iteration += 1

    print("\n[DONE] Agent loop complete.")
    print(f"Total runs: {len(history)}")
    if history:
        best = max(history, key=lambda r: r["outcomes"].get("l_b_ratio", 0))
        print(f"\nBest run (by L:B ratio):")
        print(f"  Iteration : {best['iteration']}")
        print(f"  Conditions: {best['conditions']}")
        print(f"  Outcomes  : {best['outcomes']}")


if __name__ == "__main__":
    run_agent()
