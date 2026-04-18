"""
result_parser.py
-----------------
Converts raw experimental output (GC peak areas, NMR integrations) into a
structured dictionary that the memory store and agent can work with.

In a real workflow:
  - You would load a CSV or text file exported from GC-MS or NMR software
  - This module parses it into { conversion_pct, l_b_ratio, ton, notes }

Currently supports:
  - parse_from_gc_csv()   : reads a simple CSV with compound/area columns
  - parse_experimental_result() : takes raw numbers and validates/normalizes them
  - compute_l_b_ratio()   : calculates L:B from individual area values
"""

# Import necessary libraries
import csv
import json
import os
from typing import Optional

# Define functions for parsing and processing experimental results
def parse_experimental_result(
    conversion_pct: float,
    l_b_ratio: float,
    ton: float,
    notes: str = ""
) -> dict:
    """
    Validate and package experimental outcome numbers into a standard dict.

    Args:
        conversion_pct : substrate conversion as a percentage (0–100)
        l_b_ratio      : linear-to-branch aldehyde ratio (> 0)
        ton            : turnover number (> 0)
        notes          : any free-text observation from the chemist

    Returns:
        dict with validated outcome values
    """
    # Clamp values to realistic ranges
    conversion_pct = max(0.0, min(100.0, float(conversion_pct)))
    l_b_ratio      = max(0.0, float(l_b_ratio))
    ton            = max(0.0, float(ton))

    return {
        "conversion_pct": round(conversion_pct, 2),
        "l_b_ratio":      round(l_b_ratio, 3),
        "ton":            round(ton, 1),
        "notes":          str(notes).strip()
    }

# Define function to compute L:B ratio from GC peak areas
def compute_l_b_ratio(linear_area: float, branch_area: float) -> float:
    """
    Compute the linear-to-branch (L:B) ratio from GC peak areas.
    Uses a correction factor of 1.0 by default (same response factor).

    Args:
        linear_area : GC peak area for linear aldehyde (e.g., heptanal)
        branch_area : GC peak area for branched aldehyde (e.g., 2-methylhexanal)

    Returns:
        L:B ratio as a float; returns 0.0 if branch_area is zero.
    """
    if branch_area == 0:
        print("[PARSER] Warning: branch_area is 0, cannot compute L:B ratio.")
        return 0.0
    return round(linear_area / branch_area, 3)

# Define function to parse GC results from a CSV file
def parse_from_gc_csv(filepath: str) -> Optional[dict]:
    """
    Parse a GC results CSV file into a structured outcome dict.

    Expected CSV format (two columns, no header required):
        compound_name, peak_area
    
    Example CSV contents:
        substrate,1500.0
        linear_aldehyde,4800.0
        branch_aldehyde,1200.0
        internal_standard,2000.0

    Returns:
        dict with conversion_pct, l_b_ratio, ton, notes
        Returns None if the file cannot be read or parsed.
    """
    if not os.path.exists(filepath):
        print(f"[PARSER] File not found: {filepath}")
        return None
    # Read the CSV and extract areas into a dict
    areas = {}
    try:
        with open(filepath, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    name = row[0].strip().lower().replace(" ", "_")
                    try:
                        area = float(row[1].strip())
                        areas[name] = area
                    except ValueError:
                        continue   # Skip header rows or non-numeric entries
    except IOError as e:
        print(f"[PARSER] Could not read file: {e}")
        return None

    # Calculate metrics from areas
    substrate_area  = areas.get("substrate", 0)
    linear_area     = areas.get("linear_aldehyde", 0)
    branch_area     = areas.get("branch_aldehyde", 0)
    total_products  = linear_area + branch_area

    # Conversion: fraction of substrate consumed
    if (substrate_area + total_products) > 0:
        conversion_pct = (total_products / (substrate_area + total_products)) * 100
    else:
        conversion_pct = 0.0

    l_b_ratio = compute_l_b_ratio(linear_area, branch_area)

    # TON cannot be computed from GC alone without knowing catalyst amount;
    # use a placeholder of 0 if not available
    ton = areas.get("ton", 0.0)

    return parse_experimental_result(
        conversion_pct=conversion_pct,
        l_b_ratio=l_b_ratio,
        ton=ton,
        notes=f"Parsed from {os.path.basename(filepath)}"
    )

# Define function to parse results from a dict (e.g., manual input)
def parse_from_dict(raw: dict) -> dict:
    """
    Parse and validate a result dict that may come from manual input or another source.
    Tolerant of missing keys — uses 0.0 as defaults.
    """
    return parse_experimental_result(
        conversion_pct=raw.get("conversion_pct", raw.get("conversion", 0.0)),
        l_b_ratio=raw.get("l_b_ratio", raw.get("lb_ratio", 0.0)),
        ton=raw.get("ton", raw.get("TON", 0.0)),
        notes=raw.get("notes", raw.get("note", ""))
    )

# Define function to load seed data from a JSON file
def load_seed_data(filepath: str) -> list:
    """
    Load initial experimental data from a JSON file to seed the memory store.
    This is useful for loading your existing PhD experimental runs at the start.

    Expected JSON format:
    [
      {
        "iteration": 1,
        "conditions": { ... },
        "outcomes": { "conversion_pct": 45.0, "l_b_ratio": 2.3, "ton": 100 }
      },
      ...
    ]
    """
    if not os.path.exists(filepath):
        print(f"[PARSER] Seed data file not found: {filepath}")
        return []

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        print(f"[PARSER] Loaded {len(data)} seed runs from '{filepath}'.")
        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"[PARSER] Error reading seed data: {e}")
        return []
    