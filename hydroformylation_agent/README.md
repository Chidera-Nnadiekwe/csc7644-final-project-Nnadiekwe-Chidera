# Agentic LLM for Adaptive Experimental Optimization of Olefin Hydroformylation and Isomerization

**Final Project — CSC 7644: Applied LLM Development**  
**Author:** Chidera C. Nnadiekwe

---

## Project Overview

This project implements an **agentic LLM system** that autonomously closes the experimental optimization loop for rhodium- and cobalt-catalyzed **olefin hydroformylation** and **isomerization** reactions. Rather than requiring a chemist to manually interpret results and propose new conditions between every trial, this agent:

1. Reads accumulated experimental history from a persistent JSON log.
2. Retrieves relevant literature passages from a FAISS-indexed corpus using RAG.
3. Calls **Llama 3.1 70B** (OpenRouter) with a structured chain-of-thought prompt to propose the next set of assay conditions.
4. Validates the proposal against physical/chemical constraints (via RDKit and rule-based guardrails).
5. Ingests the experimental results and logs everything — then repeats.

The primary target metrics are **linear aldehyde yield (%)** and **linear-to-branch (L:B) selectivity ratio** for 1-hexene hydroformylation, but the architecture generalises to any homogeneous catalysis workflow where experimental throughput is the limiting factor.

---

## Key Features / Capabilities