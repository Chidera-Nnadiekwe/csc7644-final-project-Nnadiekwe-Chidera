# Agentic LLM for Adaptive Experimental Optimization of Olefin Hydroformylation and Isomerization

**Final Project for CSC 7644: Applied LLM Development**  
**Author:** Chidera C. Nnadiekwe

---

## Project Overview

This project implements an **agentic LLM system** that autonomously closes the experimental optimization loop for rhodium- and cobalt-catalyzed **olefin hydroformylation** and **isomerization** reactions. Rather than requiring a Researcher to manually interpret results and propose new reaction conditions between every trial, this agentic system:

1. Reads accumulated experimental history from a persistent JSON log,
2. Retrieves relevant literature passages from a FAISS-indexed corpus using RAG,
3. Calls **Llama 3.1 70B** (OpenRouter) with a structured CoT prompt to propose the next set of assay conditions,
4. Validates the proposal against physical/chemical constraints through RDKit, and
5. Takes in the experimental results and logs everything. This process then repeats.

The primary target metrics are **aldehyde yield (%)** and **linear-to-branch (L:B) product selectivity ratio** for 1-hexene hydroformylation. However, the architecture generalises to any homogeneous catalysis workflow where experimental output is the limiting factor.

---

## Key Features / Capabilities

The following features distinguish this agentic LLM system from a more traditional human-in-the-loop workflow:
- **Agentic optimization loop** — fully autonomous multi-turn cycle (Retrieve → Prompt → Validate → Log → Repeat) up to a configurable iteration budget or early-stopping threshold.
- **Chain-of-thought (CoT) chemical reasoning** — Claude reasons step-by-step from mechanistic first principles before proposing conditions, producing interpretable rationale traces alongside each JSON proposal.
- **RAG-grounded proposals** — a FAISS index over 50–80 hydroformylation/isomerization literature documents grounds every proposal in published knowledge and reduces chemically unrealistic suggestions.
- **Persistent memory store** — all prior runs, conditions, outcomes, and reasoning traces are persisted in a structured JSON log; the agent injects full history (≤ 20 runs) or a compressed summary (> 20 runs) into each prompt.
- **Chemical validation layer** — RDKit SMILES validation plus rule-based physical constraint checks (temperature, pressure, CO/H₂ ratio, ligand loading) reject or warn on out-of-bounds proposals before they reach the lab.
- **Flexible result ingestion** — supports manual CLI entry, structured JSON, and GC-MS area-% text parsing.
- **Evaluation & visualisation:** convergence plots, parameter scatter analyses, and quantitative metrics (best L:B, best yield, iterations-to-threshold) through `scripts/evaluate.py` and `notebooks/results_analysis.ipynb`.

---