"""
llm_planner.py
---------------
The LLM Planner is the "brain" of the agent.

It receives:
  - The full (or summarized) experimental history
  - Relevant literature passages from RAG
  - The current iteration number

It returns:
  - A dict with 'proposed_conditions' (structured JSON) and 'reasoning' (CoT text)

Model used: Llama 3.1 70B via OpenRouter API (OpenAI-compatible endpoint).
Temperature: 0.3 (consistent outputs, as described in the report).

Requirements:
    pip install openai python-dotenv
    Set OPENROUTER_API_KEY in your .env file
"""

# Import necessary libraries
import json
import os
import re
from typing import Optional

from openai import OpenAI

# Import prompts from the canonical template module
from prompts_templ import (
    SYSTEM_PROMPT,
    ITERATION_PROMPT_TEMPLATE,
    HISTORY_ENTRY_TEMPLATE,
    LITERATURE_PASSAGE_TEMPLATE,
    NO_LITERATURE_FOUND,
)

# Constants for OpenRouter API
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = "meta-llama/llama-3.1-70b-instruct"

# LLMPlanner class definition
class LLMPlanner:
    def __init__(self):
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key     = os.getenv("OPENAI_API_KEY")

        if openrouter_key:
            print("[LLM] Using OpenRouter API (Llama 3.1 70B).")
            self.client = OpenAI(api_key=openrouter_key, base_url=OPENROUTER_BASE_URL)
        elif openai_key:
            print("[LLM] OPENROUTER_API_KEY not found; falling back to OpenAI API.")
            self.client = OpenAI(api_key=openai_key)
        else:
            print("[LLM] Warning: No API key found. Will use mock responses for testing.")
            self.client = None

    # Define methods for formatting history and literature blocks
    def _format_history_block(self, history: list) -> str:
        """Format experimental history records into the prompt history block."""
        if not history:
            return "No prior runs yet. This is the first iteration."

        lines = []
        for record in history:
            # Handle summarized older-run block
            if "summary_of_older_runs" in record:
                lines.append(f"[{record['summary_of_older_runs']}]")
                continue

            it   = record.get("iteration", "?")
            cond = record.get("conditions", {})
            out  = record.get("outcomes", {})
            rsn  = record.get("reasoning", "")[:200]

            # Format each history entry using the HISTORY_ENTRY_TEMPLATE
            lines.append(
                HISTORY_ENTRY_TEMPLATE.format(
                    iteration=it,
                    temperature_C=cond.get("temperature_C", "?"),
                    pressure_bar=cond.get("pressure_bar", "?"),
                    co_h2_ratio=cond.get("co_h2_ratio", "?"),
                    catalyst=cond.get("catalyst", "?"),
                    ligand=cond.get("ligand", "?"),
                    ligand_loading_eq=cond.get("ligand_loading_eq", "?"),
                    catalyst_loading_mol_pct=cond.get("catalyst_loading_mol_pct", "?"),
                    solvent=cond.get("solvent", "?"),
                    reaction_time_h=cond.get("reaction_time_h", "?"),
                    conversion_pct=out.get("conversion_pct", "?"),
                    l_b_ratio=out.get("l_b_ratio", "?"),
                    ton=out.get("ton", "?"),
                    notes=out.get("notes", ""),
                    reasoning_trace=rsn,
                )
            )
        return "\n".join(lines)

    # Define method for formatting literature block
    def _format_literature_block(self, passages: list) -> str:
        """Format retrieved passages into the prompt literature block."""
        if not passages:
            return NO_LITERATURE_FOUND

        # Format each retrieved passage using the LITERATURE_PASSAGE_TEMPLATE
        blocks = []
        for p in passages:
            blocks.append(
                LITERATURE_PASSAGE_TEMPLATE.format(
                    source=p.get("source", "unknown"),
                    score=p.get("score", 0.0),
                    passage=p.get("text", "")[:600],
                )
            )
        return "\n".join(blocks)

    # Define method to build the user message for the LLM
    def _build_user_message(
        self, history: list, retrieved_passages: list, iteration: int
    ) -> str:
        """Assemble the user-side of the prompt from history and retrieved literature."""
        history_block    = self._format_history_block(history)
        literature_block = self._format_literature_block(retrieved_passages)

        return ITERATION_PROMPT_TEMPLATE.format(
            substrate="1-hexene (CCCCC=C)",
            history_block=history_block,
            literature_block=literature_block,
            iteration=iteration,
        )

    # Define the main method to propose conditions by calling the LLM and parsing its response
    def propose_conditions(
        self,
        history: list,
        retrieved_passages: list,
        iteration: int,
    ) -> Optional[dict]:
        """
        Call the LLM and parse its response.

        Returns:
            dict with keys 'proposed_conditions' and 'reasoning', or None on failure.
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
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=1000,
                temperature=0.3,
            )
            raw_text = response.choices[0].message.content
            return self._parse_response(raw_text)

        except Exception as e:
            print(f"[LLM] API call failed: {e}")
            return None

    # Define method to parse the LLM response
    def _parse_response(self, raw_text: str) -> Optional[dict]:
        """
        Extract reasoning and JSON conditions from the raw LLM output.
        Handles minor formatting variations gracefully.
        """
        reasoning  = ""
        conditions = {}

        # Extract REASONING section
        reasoning_match = re.search(
            r"REASONING:\s*(.*?)(?=JSON:|$)", raw_text, re.DOTALL | re.IGNORECASE
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Extract JSON section — prefer the block after "JSON:" label
        json_match = re.search(
            r"JSON:\s*(\{.*?\})\s*$", raw_text, re.DOTALL | re.IGNORECASE
        )
        if not json_match:
            # Fallback: find any JSON object in the response
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

        if json_match:
            try:
                parsed = json.loads(json_match.group(1) if "JSON:" in raw_text.upper()
                                    else json_match.group(0))
                # Support both {"proposed_conditions": {...}} and direct {...} formats
                conditions = parsed.get("proposed_conditions", parsed)
            except json.JSONDecodeError as e:
                print(f"[LLM] JSON parse error: {e}")
                print(f"[LLM] Raw text preview: {raw_text[:400]}")
                return None
        else:
            print("[LLM] Could not locate a JSON block in the LLM response.")
            print(f"[LLM] Raw response preview: {raw_text[:400]}")
            return None

        return {
            "proposed_conditions": conditions,
            "reasoning": reasoning or "Reasoning not extracted.",
        }

    # Define a method to return a deterministic mock response for testing without API access
    def _mock_response(self, iteration: int) -> dict:
        """
        Deterministic mock response for testing without API access.
        Conditions shift each iteration to simulate optimization.
        """
        base_temp  = min(60 + (iteration - 1) * 5, 100)
        base_press = min(20 + (iteration - 1) * 5, 60)
        return {
            "proposed_conditions": {
                "substrate_smiles":          "CCCCC=C",          # 1-hexene
                "catalyst":                  "RhCl(PPh3)3",
                "ligand":                    "PPh3",
                "ligand_loading_eq":         4.0 + iteration * 0.5,
                "catalyst_loading_mol_pct":  1.0,
                "temperature_C":             base_temp,
                "pressure_bar":              base_press,
                "co_h2_ratio":               "1:1",
                "solvent":                   "toluene",
                "reaction_time_h":           6.0,
            },
            "reasoning": (
                f"Mock reasoning for iteration {iteration}: "
                "Incrementally increasing temperature, pressure, and ligand loading "
                "to explore their effect on L:B selectivity. Prior literature suggests "
                "higher CO pressure and bulkier phosphine ligands favor linear product."
            ),
        }
