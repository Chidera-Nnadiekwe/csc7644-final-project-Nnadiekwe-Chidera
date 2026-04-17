"""
prompts.py
----------
All prompt templates used by the olefin hydroformylation/isomerization agent.

Templates are kept here to allow easy iteration on prompt engineering without
touching controller logic.
"""

SYSTEM_PROMPT = """\
You are an expert process chemist specializing in homogeneous catalysis, specifically \
rhodium- and cobalt-catalyzed olefin hydroformylation and isomerization reactions.

You assist in designing optimal experimental conditions to maximize linear aldehyde \
yield and linear-to-branch (L:B) selectivity. You reason step-by-step from mechanistic \
principles, prior experimental history, and retrieved literature before proposing \
new conditions.

You always respond with:
1. A detailed chain-of-thought reasoning section explaining your chemical logic.
2. A structured JSON block specifying the next set of experimental conditions.

Your JSON output MUST follow this schema exactly:
{
  "temperature_C": <float>,
  "pressure_bar": <float>,
  "co_h2_ratio": <float>,
  "catalyst": <string>,
  "ligand": <string>,
  "ligand_equiv": <float>,
  "substrate_conc_M": <float>,
  "reaction_time_h": <float>,
  "rationale_summary": <string, one sentence>
}
"""

ITERATION_PROMPT_TEMPLATE = """\
## Experimental Target
Substrate: {substrate}
Goal: Maximize linear aldehyde yield (%) and linear-to-branch (L:B) selectivity ratio.

## Prior Experimental History
{history_block}

## Retrieved Literature Passages
{literature_block}

## Task
Based on the experimental history and the retrieved literature above, reason step-by-step \
about what changes to the reaction conditions are most likely to improve linear aldehyde \
yield and L:B selectivity. Consider temperature, pressure, CO/H2 ratio, ligand loading, \
and reaction time.

After your chain-of-thought reasoning, output the next proposed conditions as a JSON \
object conforming to the schema described in your system instructions. \
Do not propose conditions already attempted.
"""

HISTORY_ENTRY_TEMPLATE = """\
Iteration {iteration}:
  Conditions: T={temperature_C}°C, P={pressure_bar} bar, CO/H2={co_h2_ratio}, \
catalyst={catalyst}, ligand={ligand} ({ligand_equiv} equiv), \
[substrate]={substrate_conc_M} M, t={reaction_time_h} h
  Outcomes: conversion={conversion_pct}%, aldehyde yield={aldehyde_yield_pct}%, \
L:B={lb_ratio}, TON={ton}
  Agent reasoning summary: {reasoning_trace}
"""

HISTORY_SUMMARY_TEMPLATE = """\
[Summary of {n_runs} prior runs]
Best L:B ratio achieved: {best_lb} (Iteration {best_lb_iter})
Best aldehyde yield: {best_yield}% (Iteration {best_yield_iter})
Most recent run: Iteration {last_iter} — \
T={last_T}°C, P={last_P} bar, yield={last_yield}%, L:B={last_lb}
Trend: {trend_note}
"""

LITERATURE_PASSAGE_TEMPLATE = """\
[Source: {source}, relevance score: {score:.2f}]
{passage}
"""

NO_LITERATURE_FOUND = """\
No closely matching literature passages were retrieved for the current conditions. \
Proceed from first principles and prior experimental history.
"""
