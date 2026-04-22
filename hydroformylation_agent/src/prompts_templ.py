"""
prompts_templ.py
-----------------
All prompt templates used by the olefin hydroformylation/isomerization agent.

Templates are kept here to allow easy iteration on prompt engineering without
touching controller logic. The system prompt here is the canonical version;
llm_planner.py imports SYSTEM_PROMPT directly from this module.
"""

# System prompt: Imported by llm_planner.py
SYSTEM_PROMPT = """\
You are an expert process chemist specializing in cationic Co(II)-Trisphosphine olefin \
hydroformylation and isomerization reactions on 1-hexene as substrate.

You assist in designing optimal experimental conditions to maximize linear aldehyde \
conversion (%) and linear-to-branch (L:B) selectivity ratio. You reason step-by-step \
from mechanistic standpoint, prior experimental history, and retrieved literature before \
proposing new conditions.

You MUST respond with exactly two sections:

REASONING:
<Write 3-6 sentences of step-by-step chemical reasoning. Cite specific trends from the
history and retrieved literature. Explain why the proposed changes are expected to
improve L:B or conversion.>

JSON:
{
  "proposed_conditions": {
    "substrate_smiles":          "<canonical SMILES string of the olefin substrate>",
    "catalyst":                  "<HCoCO(Ph2PC2H4(PPh)C2H4PPh2) or HCoCO(CH3C(CH2PPh2)3)",
    "ligand":                    "<Ph2PC2H4(PPh)C2H4PPh2 or CH3C(CH2PPh2)3>",
    "ligand_loading_eq":         <float, molar equiv vs catalyst, 1.0-20.0>,
    "catalyst_loading_mol_pct":  <float, mol% vs substrate, 0.1-5.0>,
    "temperature_C":             <float, 40-160>,
    "pressure_bar":              <float, 5-80>,
    "co_h2_ratio":               "<string, e.g. '1:1' or '1:2'>",
    "solvent":                   "<e.g., tetraglyme, toluene, THF, DCM>",
    "reaction_time_h":           <float, 1-24>
  }
}

Rules:
- ALL fields in proposed_conditions are REQUIRED.
- Keep temperature_C between 40 and 160.
- Keep pressure_bar between 5 and 80.
- ligand_loading_eq must be between 1.0 and 20.0.
- catalyst_loading_mol_pct must be between 0.1 and 5.0.
- reaction_time_h must be between 1 and 24.
- Only propose chemically realistic catalyst/ligand/solvent combinations.
- Do NOT repeat conditions from a prior run that performed poorly.
- substrate_smiles must be a valid SMILES string for the olefin substrate.
"""

ITERATION_PROMPT_TEMPLATE = """\
## Experimental Target
Substrate: {substrate}
Goal: Maximize linear aldehyde conversion (%) and linear-to-branch (L:B) selectivity ratio.

## Prior Experimental History
{history_block}

## Retrieved Literature Passages
{literature_block}

## Task
This is iteration {iteration}. Based on the experimental history and the retrieved \
literature above, reason step-by-step about what changes to the reaction conditions \
are most likely to improve L:B selectivity and conversion. Consider temperature, \
pressure, CO/H2 ratio, ligand identity and loading, catalyst loading, and reaction time.

After your CoT reasoning, output the next proposed conditions as a JSON \
object conforming to the schema in your system instructions. \
Do not propose conditions already attempted.
"""

# History Entry Template: Used to format each prior run's conditions and outcomes into a text block for the prompt
HISTORY_ENTRY_TEMPLATE = """\
Iteration {iteration}:
  Conditions: T={temperature_C}°C, P={pressure_bar} bar, CO/H2={co_h2_ratio}, \
catalyst={catalyst}, ligand={ligand} ({ligand_loading_eq} equiv), \
cat. loading={catalyst_loading_mol_pct} mol%, solvent={solvent}, t={reaction_time_h} h
  Outcomes: conversion={conversion_pct}%, L:B={l_b_ratio}, TON={ton}
  Notes: {notes}
  Agent reasoning summary: {reasoning_trace}
"""

# History Summary Template: Used to create a concise summary of all prior runs for the prompt
HISTORY_SUMMARY_TEMPLATE = """\
[Summary of {n_runs} prior runs]
Best L:B ratio achieved: {best_lb} (Iteration {best_lb_iter})
Best conversion: {best_conv}% (Iteration {best_conv_iter})
Most recent run: Iteration {last_iter} — \
T={last_T}°C, P={last_P} bar, conversion={last_conv}%, L:B={last_lb}
Trend: {trend_note}
"""

# Literature Passage Template: Used to format each retrieved literature passage for the prompt
LITERATURE_PASSAGE_TEMPLATE = """\
[Source: {source}, similarity score: {score:.3f}]
{passage}
"""

# No Literature Found Template: Used when no relevant literature passages are retrieved
NO_LITERATURE_FOUND = """\
No closely matching literature passages were retrieved for the current conditions. \
Proceed from first principles and prior experimental history alone.
"""
