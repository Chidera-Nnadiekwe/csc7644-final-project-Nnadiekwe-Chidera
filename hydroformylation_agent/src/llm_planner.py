"""
llm_planner.py
---------------
The LLM Planner is the "brain" of the agent.

It receives:
  - The full (or summarized) experimental history
  - Relevant literature passages from RAG
  - The current iteration number

It returns:
  - A dict with 'proposed_conditions' (structured JSON) and 'reasoning' (chain-of-thought text)

Model used: Llama 3.1 70B via OpenRouter API (compatible with OpenAI SDK format).

Requirements:
    pip install openai python-dotenv
    Set OPENROUTER_API_KEY in your .env file
"""

import json
import os
import re
from typing import Optional

from openai import OpenAI


# Fallback: also support direct OpenAI if OpenRouter key is not set
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = "meta-llama/llama-3.1-70b-instruct"

# Conditions the agent is allowed to propose (with safe ranges)
CONDITION_SCHEMA = {
    "substrate_smiles":   "SMILES string of the olefin substrate",
    "catalyst":           "e.g., RhCl(PPh3)3 or Co2(CO)8",
    "ligand":             "e.g., PPh3, BISBI, Xantphos",
    "ligand_loading_eq":  "molar equivalents of ligand vs catalyst (e.g., 4.0)",
    "catalyst_loading_mol_pct": "mol% catalyst vs substrate (e.g., 1.0)",
    "temperature_C":      "reaction temperature in Celsius (40–120)",
    "pressure_bar":       "total syngas pressure in bar (5–80)",
    "co_h2_ratio":        "CO:H2 molar ratio as a string, e.g. '1:1' or '1:2'",
    "solvent":            "e.g., toluene, THF, DCM",
    "reaction_time_h":    "reaction duration in hours (1–24)",
}

SYSTEM_PROMPT = """You are an expert process chemist specializing in homogeneous catalysis,
specifically olefin hydroformylation and isomerization. Your job is to propose the next
experimental conditions to maximize linear-to-branch (L:B) aldehyde selectivity and
substrate conversion.

You will receive:
1. A history of prior experimental runs with their conditions and measured outcomes.
2. Relevant passages from the scientific literature (retrieved by a RAG system).
3. A target: maximize L:B ratio (linear selectivity) and overall conversion %.

Your response MUST follow this exact two-part format:

REASONING:
<Write 3–6 sentences of step-by-step chemical reasoning. Explain WHY you are proposing
these conditions based on the history and literature. Reference specific trends you observe.>

JSON:
{
  "proposed_conditions": {
    "substrate_smiles": "<SMILES>",
    "catalyst": "<catalyst name>",
    "ligand": "<ligand name>",
    "ligand_loading_eq": <number>,
    "catalyst_loading_mol_pct": <number>,
    "temperature_C": <number>,
    "pressure_bar": <number>,
    "co_h2_ratio": "<ratio string>",
    "solvent": "<solvent name>",
    "reaction_time_h": <number>
  }
}

Rules:
- Always include ALL fields in proposed_conditions.
- Keep temperature_C between 40 and 120.
- Keep pressure_bar between 5 and 80.
- ligand_loading_eq should be between 1 and 20.
- catalyst_loading_mol_pct should be between 0.1 and 5.0.
- reaction_time_h should be between 1 and 24.
- Only suggest chemically realistic combinations.
- Do NOT repeat conditions from a prior run that performed poorly.
"""


class LLMPlanner:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[LLM] Warning: No API key found. Set OPENROUTER_API_KEY in your .env file.")
            self.client = None
        else:
            # OpenRouter uses the same interface as OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL if os.getenv("OPENROUTER_API_KEY") else None
            )

    def _build_user_message(
        self,
        history: list,
        retrieved_passages: list,
        iteration: int
    ) -> str:
        """Assemble the user-side of the prompt from history and retrieved literature."""

        # ── Section 1: Retrieved Literature ──
        lit_section = "## RETRIEVED LITERATURE PASSAGES\n"
        if retrieved_passages:
            for i, passage in enumerate(retrieved_passages, 1):
                source = passage.get("source", "unknown")
                text   = passage.get("text", "")[:600]   # Limit length to save tokens
                lit_section += f"\n[Passage {i} — {source}]\n{text}\n"
        else:
            lit_section += "No passages retrieved.\n"

        # ── Section 2: Experimental History ──
        hist_section = "\n## EXPERIMENTAL HISTORY\n"
        if not history:
            hist_section += "No prior runs yet. This is the first iteration.\n"
        else:
            for record in history:
                it = record.get("iteration", "?")
                cond = record.get("conditions", {})
                out  = record.get("outcomes", {})
                hist_section += (
                    f"\n--- Run {it} ---\n"
                    f"Conditions : {json.dumps(cond, indent=4)}\n"
                    f"Outcomes   : conversion={out.get('conversion_pct','?')}%, "
                    f"L:B={out.get('l_b_ratio','?')}, TON={out.get('ton','?')}\n"
                    f"Notes      : {out.get('notes', 'none')}\n"
                )

        # ── Section 3: Task ──
        task_section = (
            f"\n## YOUR TASK\n"
            f"This is iteration {iteration}. "
            f"Based on the history above and the retrieved literature, "
            f"propose the next experimental conditions that will increase the L:B selectivity "
            f"and conversion. Follow the REASONING + JSON format exactly."
        )

        return lit_section + hist_section + task_section

    def propose_conditions(
        self,
        history: list,
        retrieved_passages: list,
        iteration: int
    ) -> Optional[dict]:
        """
        Call the LLM and parse its response.
        Returns dict with keys: 'proposed_conditions', 'reasoning'
        Returns None on failure.
        """
        if self.client is None:
            print("[LLM] No client available. Returning mock conditions for testing.")
            return self._mock_response(iteration)

        user_message = self._build_user_message(history, retrieved_passages, iteration)

        try:
            response = self.client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message}
                ],
                max_tokens=1000,
                temperature=0.3,   # Lower temperature = more consistent/chemical outputs
            )
            raw_text = response.choices[0].message.content
            return self._parse_response(raw_text)

        except Exception as e:
            print(f"[LLM] API call failed: {e}")
            return None

    def _parse_response(self, raw_text: str) -> Optional[dict]:
        """
        Extract reasoning and JSON conditions from the raw LLM output.
        Handles minor formatting variations gracefully.
        """
        reasoning = ""
        conditions = {}

        # Extract REASONING section
        reasoning_match = re.search(r"REASONING:\s*(.*?)(?=JSON:|$)", raw_text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Extract JSON section
        json_match = re.search(r"JSON:\s*(\{.*?\})\s*$", raw_text, re.DOTALL | re.IGNORECASE)
        if not json_match:
            # Try to find any JSON block as fallback
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

        if json_match:
            try:
                parsed = json.loads(json_match.group(1) if "JSON:" in raw_text else json_match.group(0))
                conditions = parsed.get("proposed_conditions", parsed)
            except json.JSONDecodeError as e:
                print(f"[LLM] JSON parse error: {e}")
                print(f"[LLM] Raw JSON string: {json_match.group(0)[:300]}")
                return None
        else:
            print("[LLM] Could not find JSON block in LLM response.")
            print(f"[LLM] Raw response preview: {raw_text[:400]}")
            return None

        return {
            "proposed_conditions": conditions,
            "reasoning": reasoning or "Reasoning not extracted."
        }

    def _mock_response(self, iteration: int) -> dict:
        """
        Returns a deterministic mock response for testing without API access.
        Conditions shift slightly each iteration to simulate optimization.
        """
        base_temp = 60 + (iteration - 1) * 5   # 60, 65, 70, ...
        base_press = 20 + (iteration - 1) * 5  # 20, 25, 30, ...
        return {
            "proposed_conditions": {
                "substrate_smiles": "CCCCCC=C",    # 1-heptene
                "catalyst": "RhCl(PPh3)3",
                "ligand": "PPh3",
                "ligand_loading_eq": 4.0,
                "catalyst_loading_mol_pct": 1.0,
                "temperature_C": min(base_temp, 100),
                "pressure_bar": min(base_press, 60),
                "co_h2_ratio": "1:1",
                "solvent": "toluene",
                "reaction_time_h": 6.0
            },
            "reasoning": (
                f"Mock reasoning for iteration {iteration}: "
                "Incrementally increasing temperature and pressure to explore their effect "
                "on L:B selectivity. Prior runs suggest higher pressure favors linear product."
            )
        }
