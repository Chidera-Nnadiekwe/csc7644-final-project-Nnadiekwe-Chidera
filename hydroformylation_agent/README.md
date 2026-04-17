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
│          │               │                  ┌─────────────┐           │          │
│          │               │                  │ validator.py│           │          │
│          │               │                  │(RDKit+rules)│           │          │
│          │               │                  └─────────────┘           │          │
│          └───────────────┴────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Data flow per iteration:**
1. `memory.py` → builds history block from JSON log
2. `retriever.py` → queries FAISS index, returns top-3 literature passages
3. `planner.py` → sends structured prompt to Claude, receives CoT + JSON conditions
4. `validator.py` → checks SMILES validity + physical bounds; rejects bad proposals
5. User runs experiment; `parser.py` → ingests GC/NMR results
6. `memory.py` → appends run record to JSON log; repeat

---

## Setup Instructions

### Prerequisites

- Python 3.10 or later
- `pip` (comes with Python)
- Active Anthropic API key (for Claude)
- Active OpenAI API key (for embeddings)
- RDKit (optional but recommended for chemical validation)

> Tested on macOS 14 and Ubuntu 22.04. Windows should work via WSL or native Python, but is not officially tested.

### 1. Clone the repository

```bash
git clone https://github.com/csc7644-final-project-nnadiekwe/olefin_agent.git
cd olefin_agent
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on RDKit:** If `pip install rdkit` fails on your platform, follow the [official RDKit installation guide](https://www.rdkit.org/docs/Install.html). The agent degrades gracefully — SMILES validation is skipped with a warning if RDKit is absent.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in your API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Then load them into your shell (or use `python-dotenv` — already imported in each module):

```bash
export $(grep -v '^#' .env | xargs)
```

### 5. Add literature files

Place plain-text (`.txt`) versions of your hydroformylation/isomerization literature in `data/literature/`. Two sample files are included. The more documents you add, the better the RAG retrieval.

### 6. Build the FAISS index

This only needs to be run once (or again when you add new literature):

```bash
python scripts/build_index.py
```

Optional arguments:
```
--literature-dir   data/literature/   (default)
--index-dir        data/faiss_index/  (default)
--chunk-size       300                (words per chunk)
--chunk-overlap    50                 (overlap words)
```

---

## Running the Application

### Start the optimization agent

```bash
python src/agent.py
```

The agent will:
1. Load the existing experiment log (or create a fresh one).
2. Retrieve literature, call Claude, validate, and display proposed conditions.
3. Prompt you to enter experimental results after each iteration.
4. Save everything and loop until `--max-iter` is reached or a target is hit.

**Full CLI options:**

```bash
python src/agent.py \
  --max-iter 10 \
  --target-lb 5.0 \
  --target-yield 90.0 \
  --substrate "1-hexene" \
  --log-path data/experiment_log.json \
  --ingest-mode manual
```

| Flag | Default | Description |
|---|---|---|
| `--max-iter` | 10 | Maximum number of optimization iterations |
| `--target-lb` | 5.0 | L:B ratio for early stopping |
| `--target-yield` | 90.0 | Aldehyde yield % for early stopping |
| `--substrate` | `1-hexene` | Substrate name or SMILES |
| `--log-path` | `data/experiment_log.json` | Path to experiment log |
| `--ingest-mode` | `manual` | Result entry: `manual`, `json`, or `gc` |

### Result ingestion modes

- **`manual`** — interactive CLI prompts for each outcome value (default for development).
- **`json`** — supply a JSON string of outcomes (for scripted/automated use).
- **`gc`** — paste a GC-MS area-% text block; the parser extracts conversion, yield, and L:B automatically.

### Run evaluation and generate plots

After at least two experimental iterations:

```bash
python scripts/evaluate.py
# Figures saved to results/figures/

# Optional: compare against a random-sampling baseline log
python scripts/evaluate.py --baseline-log data/baseline_log.json
```

### Open the results notebook

```bash
jupyter notebook notebooks/results_analysis.ipynb
```

---

## Repository Organisation

```
olefin_agent/
├── data/
│   ├── experiment_log.json        # Persistent memory — all run records
│   ├── literature/                # Plain-text .txt files for RAG corpus
│   │   ├── ligand_effects_hydroformylation.txt
│   │   └── isomerization_hydroformylation_tandem.txt
│   └── faiss_index/               # Built FAISS index (generated by build_index.py)
│       ├── index.faiss
│       ├── chunks.json
│       └── metadata.json
├── src/
│   ├── agent.py                   # Main controller loop — entry point
│   ├── memory.py                  # Load / save / query experiment log
│   ├── retriever.py               # FAISS RAG pipeline (embed → search → format)
│   ├── planner.py                 # Claude API call + CoT + JSON parsing
│   ├── validator.py               # RDKit SMILES check + physical constraint guardrails
│   ├── parser.py                  # Result ingestion (GC text, JSON, manual)
│   └── prompts.py                 # All prompt templates (system + iteration + history)
├── scripts/
│   ├── build_index.py             # One-time: chunk → embed → FAISS index
│   └── evaluate.py                # Metrics computation + convergence plots
├── notebooks/
│   └── results_analysis.ipynb    # Interactive visualisation of agent performance
├── results/
│   └── figures/                   # Auto-generated plots (created at evaluation time)
├── .env.example                   # Environment variable template (safe to commit)
├── .gitignore                     # Excludes .env, venv/, __pycache__, *.faiss, etc.
├── requirements.txt
└── README.md
```

**Where to look for what:**
- Start reading at `src/agent.py` — it calls all other modules in sequence.
- Prompt engineering lives entirely in `src/prompts.py`.
- To tune the LLM call (model, max_tokens, temperature), edit `src/planner.py`.
- To adjust physical constraint bounds, edit the `CONSTRAINTS` dict in `src/validator.py`.
- Evaluation plots are generated by `scripts/evaluate.py` and visualised in the notebook.

---

## Evaluation Design

Success is assessed across three dimensions (per the project proposal):

**Quantitative:**
- Primary: L:B selectivity ratio and aldehyde yield (%) per iteration vs a random-sampling baseline within the same parameter search space.
- Secondary: convergence speed (iterations to reach a predefined L:B threshold) and final L:B ratio.

**Qualitative:**
- CoT reasoning traces are assessed for (1) chemical correctness, (2) correct use of retrieved literature, and (3) coherence between reasoning and proposed conditions.

**System diagnostics:**
- Validation pass rate, retrieval relevance (logged similarity scores), prompt token cost per iteration.

---

## Attributions and Citations

This project builds on the following published work and open-source libraries:

- **Anthropic Claude API** — LLM planner backbone. https://docs.anthropic.com
- **OpenAI Embeddings API** — `text-embedding-3-small` for RAG. https://platform.openai.com/docs/guides/embeddings
- **FAISS** (Facebook AI Research) — efficient similarity search. https://github.com/facebookresearch/faiss
- **RDKit** — cheminformatics toolkit for SMILES validation. https://www.rdkit.org
- **Bran et al. (2023)** — ChemCrow: Augmenting large-language models with chemistry tools. arXiv:2304.05376.
- **Shields et al. (2021)** — Bayesian reaction optimization as a tool for chemical synthesis. *Nature*, 590, 89–96.
- **Szymanski et al. (2023)** — An autonomous laboratory for the accelerated synthesis of inorganic materials. *Nature*, 624, 86–91.
- **Hood et al. (2020)** — Highly active cationic cobalt(II) hydroformylation catalysts. *Science*, 367, 542–548.
- **Kearnes et al. (2021)** — The Open Reaction Database. *JACS*, 143, 18820–18826.

No external code was directly adapted. All module implementations are original, informed by the above references and the Anthropic and OpenAI API documentation.

---

## License

This repository is for academic use as part of CSC 7644. All rights reserved by the author.