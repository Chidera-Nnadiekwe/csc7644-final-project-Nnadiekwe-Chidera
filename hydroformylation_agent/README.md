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
- **Agentic optimization loop:** A fully autonomous, iterative decision-making framework that performs a closed-loop optimization process until a stopping criteria is met (e.g., max iterations, % conversion and/or L:B selectivity ratio).
- **CoT chemical reasoning:** The agent reasons step-by-step from mechanistic standpoints before proposing conditions and producing interpretable rationale traces alongside each JSON proposal.
- **RAG Grounding:** A FAISS index over 50–80 hydroformylation/isomerization literature documents grounds every proposal in published knowledge and reduces chemically unrealistic suggestions.
- **Persistent memory store:** All previous runs, conditions, outcomes, and reasoning traces are persisted in a structured JSON log. The agent injects full history (≤ 20 runs) or a compressed summary (> 20 runs) into each prompt.
- **Chemical validation layer:** RDKit SMILES validation and rule-based physical constraint checks (including temperature, pressure, syngas ratio, ligand) reject or warn on out-of-domain proposals before they reach the lab.
- **Flexible result ingestion:** Supports manual CLI entry, structured JSON, and GC-MS area text parsing.
- **Evaluation & visualisation:** Convergence plots, parameter scatter analyses, and quantitative metrics (best L:B, best yield, iterations-to-threshold) through `src/evaluation.py` and `notebooks/results_analysis.ipynb`.

---

## Tech Stack and Architecture

| Layer | Technology |
|---|---|
| LLM planner | Llama 3.1 70B (`llama3.1-70b`) through `openrouter` Python SDK |
| Embeddings | OpenAI `text-embedding-3-small` through `openai` Python SDK |
| Vector store | FAISS (`faiss-cpu`) — flat inner-product index with L2-normalised vectors |
| Chemistry validation | RDKit |
| Data persistence | JSON flat files |
| Evaluation & plotting | `matplotlib`, `pandas`, `numpy` |
| Notebook interface | Jupyter |

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        agent_controller.py                                       │
│                                                                                  │
│  ┌────────────────┐   ┌────────────────┐   ┌──────────────┐   ┌────────────────┐ │
│  │ memory_store.py│──▶│rag_retriever.py│──▶│llm_planner.py│──▶│result_parser.py│ │
│  │   (JSON log)   │   │     (FAISS)    │   │    (Llama)   │   │   (results)    │ │
│  └────────────────┘   └────────────────┘   └──────────────┘   └────────────────┘ │
│          ▲                    ▲                   │                   │          │
│          │               ┌────┘                   ▼                   │          │
│          │               │                  ┌──────────────┐          │          │
│          │               │                  │ tool_layer.py│          │          │
│          │               │                  │(RDKit+rules) │          │          │
│          │               │                  └──────────────┘          │          │
│          └───────────────┴────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Data flow per iteration:**
1. `memory_store.py` → builds history block from JSON log
2. `rag_retriever.py` → queries FAISS index, returns top-3 literature passages
3. `llm_planner.py` → sends structured prompt to Llama 3.1 70B, receives CoT + JSON conditions
4. `tool_layer.py` → checks SMILES validity + physical bounds; rejects bad proposals
5. User runs experiment; `result_parser.py` → ingests GC/NMR results
6. `memory_store.py` → appends run record to JSON log; repeat

---

## Setup Instructions

### Prerequisites

- Python ≥ 3.10
- `pip` (Python package manager)
- Active API credentials for:
  - Meta Llama 3.1 70B through `openrouter` Python SDK (for LLM planning)
  - OpenAI API (for embeddings)
- RDKit (for chemical validation)

### 1. Clone the repository

```bash
git clone https://github.com/Chidera-Nnadiekwe/csc7644-final-project-Nnadiekwe-Chidera.git
cd csc7644-final-project-Nnadiekwe-Chidera/hydroformylation_agent/
```

### 2. Create and activate a virtual environment

```bash
conda create -n csc7644_env python=3.12 # create env with Python 3.12
conda activate csc7644_env              # activate the environment
conda deactivate                        # deactivate when done
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on RDKit:** If `pip install rdkit` fails on your platform, follow the [official RDKit installation guide](https://www.rdkit.org/docs/Install.html). The agent degrades gracefully and SMILES validation is skipped with a warning if RDKit is absent.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in your API keys:

```
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
```

Then load them into your shell (or use `python-dotenv` — already imported in each module):

```bash
export $(grep -v '^#' .env | xargs)
```

### 5. Add literature files

Place plain-text (`.txt`) versions of your hydroformylation/isomerization literature in `data/corpus/`. The more documents are added, the better the RAG retrieval.

### 6. Build the FAISS index

This only needs to be run once (or again when you add new literature):

```bash
python scripts/build_index.py
```

---

## Running the Application

### Start the optimizat ion agent

```bash
python src/agent.py
```

The agent will:
1. Load the existing experiment log (or create a fresh one).
2. Retrieve literature, call Llama 3.1 70B, validate, and display proposed conditions.
3. Prompt you to enter experimental results after each iteration.
4. Save everything and loop until `--max-iter` is reached or a target is hit.

**Full CLI options:**

```bash
python src/agent_controller.py \
  --MAX_ITERATIONS 10 \
  --TARGET_L_B_RATIO 5.0 \
  --TARGET_CONVERSION 90.0 \
  --SUBSTRATE "1-hexene" \
  --CORPUS_PATH "corpus/" 
  --MEMORY_FILE "experiment_log.json" 
  --ingest-mode manual
```

| Flag | Default | Description |
|---|---|---|
| `--MAX_ITERATIONS` | 10 | Maximum number of optimization iterations |
| `--TARGET_L_B_RATIO` | 5.0 | L:B ratio for early stopping |
| `--TARGET_CONVERSION` | 90.0 | Aldehyde yield % for early stopping |
| `--SUBSTRATE` | `1-hexene` | Substrate name or SMILES |
| `--MEMORY_FILE` | `data/experiment_log.json` | Path to experiment log |
| `--ingest-mode` | `manual` | Result entry: `manual`, `json`, or `gc` |

### Result ingestion modes

- **`manual`** — interactive CLI prompts for each outcome value (default for development).
- **`json`** — supply a JSON string of outcomes (for scripted/automated use).
- **`gc`** — paste a GC-MS area text block; the parser extracts conversion, yield, and L:B automatically.

### Run evaluation and generate plots

After at least two experimental iterations:

```bash
python scripts/evaluation.py
# Figures saved to results/figures/
```

### Open the results notebook

```bash
jupyter notebook notebooks/results_analysis.ipynb
```

---

## Repository Organisation

```
hydroformylation_agent/
├── data/
│   ├── experiment_log.json        # Persistent memory — all run records
│   ├── corpus/                    # Plain-text .txt files for RAG corpus
│   │   ├── ligand_effects_hydroformylation.txt
│   │   └── isomerization_hydroformylation_tandem.txt
│   └── faiss_index/               # Built FAISS index (generated by build_index.py)
│       ├── index.faiss
│       ├── chunks.json
│       └── metadata.json
├── src/
│   ├── agent_controller.py        # Main controller loop — entry point
│   ├── memory_store.py            # Load / save / query experiment log
│   ├── rag_retriever.py           # FAISS RAG pipeline (embed → search → format)
│   ├── llm_planner.py             # Llama 3.1 70B API call, CoT and JSON parsing
│   ├── tool_layer.py              # RDKit SMILES check and physical constraint guardrails
│   ├── result_parser.py           # Result ingestion (GC text, JSON, manual)
│   └── prompts_templ.py           # All prompt templates (system + iteration + history)
├── scripts/
│   ├── build_index.py             # One-time: chunk → embed → FAISS index
│   └── evaluation.py              # Metrics computation + convergence plots
├── notebooks/
│   └── results_analysis.ipynb     # Interactive visualisation of agent performance
├── results/
│   └── figures/                   # Auto-generated plots (created at evaluation time)
├── .env.template                  # Environment variable template (safe to commit)
├── .gitignore                     # Excludes .env, venv/, __pycache__, *.faiss, etc.
├── requirements.txt
└── README.md
```

**Where to look for what:**
- Start reading at `src/agent_controller.py` — it calls all other modules in sequence.
- Prompt engineering lives entirely in `src/prompts_templ.py`.
- To tune the LLM call (model, max_tokens, temperature), edit `src/llm_planner.py`.
- Evaluation plots are generated by `scripts/evaluation.py` and visualised in the notebook.

---

## Evaluation Design

Success is assessed across three dimensions:

**Quantitative:**
- Primary: Aldehyde yield (%) and L:B selectivity ratio per iteration vs a random-sampling baseline within the same parameter search space.
- Secondary: convergence speed (iterations to reach a predefined L:B threshold) and final L:B ratio.

**Qualitative:**
- CoT reasoning traces are assessed for
  1. chemical correctness,
  2. correct use of retrieved literature, and
  3. coherence between reasoning and proposed conditions.

**System diagnostics:**
- Validation pass rate, retrieval relevance (logged similarity scores), prompt token cost per iteration.

---

## Attributions and Citations

This project builds on the following published work and open-source libraries:

- **Llama 3.1 70B API** — LLM planner backbone. https://docs.llama.com
- **OpenAI Embeddings API** — `text-embedding-3-small` for RAG. https://platform.openai.com/docs/guides/embeddings
- **FAISS** (Facebook AI Research) — efficient similarity search. https://github.com/facebookresearch/faiss
- **RDKit** — cheminformatics toolkit for SMILES validation. https://www.rdkit.org
- **Bran et al. (2023)** — ChemCrow: Augmenting large-language models with chemistry tools. arXiv:2304.05376.
- **Shields et al. (2021)** — Bayesian reaction optimization as a tool for chemical synthesis. *Nature*, 590, 89–96.
- **Szymanski et al. (2023)** — An autonomous laboratory for the accelerated synthesis of inorganic materials. *Nature*, 624, 86–91.
- **Hood et al. (2020)** — Highly active cationic cobalt(II) hydroformylation catalysts. *Science*, 367, 542–548.
- **Kearnes et al. (2021)** — The Open Reaction Database. *JACS*, 143, 18820–18826.

No external code was directly adapted. All module implementations are original, informed by the above references and the Meta Llama 3.1 70B (OpenRouter) and OpenAI API documentation.

---

## License

This repository is for academic use as part of CSC 7644 Final Project. All rights reserved by Chidera C. Nnadiekwe.