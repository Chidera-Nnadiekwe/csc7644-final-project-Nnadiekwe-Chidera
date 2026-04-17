"""
validator.py
------------
Validates proposed experimental conditions before they are sent to the lab.

Two layers of validation are applied:
1. Chemical validity — checks that substrate SMILES is parseable by RDKit.
2. Physical/experimental constraints — ensures all numeric parameters are within
   safe and practically achievable bounds for a standard autoclave setup.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constraint bounds (inclusive on both ends)
# ---------------------------------------------------------------------------
CONSTRAINTS: dict = {
    "temperature_C": (40.0, 150.0),      # °C — below solvent bp, above cat. activation
    "pressure_bar": (5.0, 60.0),         # bar total syngas
    "co_h2_ratio": (0.5, 3.0),           # molar ratio
    "ligand_equiv": (1.0, 20.0),         # mol equiv relative to Rh
    "substrate_conc_M": (0.05, 2.0),     # M in solvent
    "reaction_time_h": (0.5, 24.0),      # hours
}

ALLOWED_CATALYSTS = {
    "Rh(acac)(CO)2",
    "Rh(CO)2Cl]2",
    "[Rh(CO)2Cl]2",
    "Co2(CO)8",
    "HCo(CO)4",
    "HRh(CO)(PPh3)3",
}

ALLOWED_LIGANDS = {
    "PPh3",
    "BISBI",
    "BIPHEPHOS",
    "dppe",
    "dppp",
    "Xantphos",
    "P(OPh)3",
    "PCy3",
    "none",
}


@dataclass
class ValidationResult:
    """Container for the outcome of a validation check.

    Attributes
    ----------
    valid:
        True if all checks passed.
    errors:
        List of human-readable error messages (empty if valid).
    warnings:
        List of non-fatal advisory messages.
    """

    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Record a validation error and mark the result as invalid."""
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        """Record a non-fatal warning (does not invalidate)."""
        self.warnings.append(msg)


def validate_substrate_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
    """Check that a SMILES string represents a valid, parseable molecule.

    Uses RDKit for parsing.  Falls back gracefully if RDKit is not installed
    (returns a warning rather than raising).

    Parameters
    ----------
    smiles:
        SMILES string to validate.

    Returns
    -------
    (is_valid, error_message)
        ``is_valid`` is True when the molecule parses without errors.
        ``error_message`` is None when valid, otherwise a descriptive string.
    """
    try:
        from rdkit import Chem  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "RDKit not installed — skipping SMILES validation. "
            "Install it with: pip install rdkit"
        )
        return True, None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, f"RDKit could not parse SMILES: '{smiles}'"

    # Check that molecule contains at least one carbon-carbon double bond.
    has_double_bond = any(
        bond.GetBondTypeAsDouble() == 2.0
        for bond in mol.GetBonds()
        if {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()} == {6}
    )
    if not has_double_bond:
        return (
            False,
            f"Substrate '{smiles}' does not appear to contain a C=C double bond "
            "required for hydroformylation.",
        )

    return True, None


def validate_conditions(conditions: dict) -> ValidationResult:
    """Validate a proposed conditions dictionary against physical constraints.

    Parameters
    ----------
    conditions:
        Dictionary of proposed experimental parameters, as returned by the planner.

    Returns
    -------
    ValidationResult
        Object containing validity flag, errors, and warnings.
    """
    result = ValidationResult()

    # --- Numeric bounds ---
    for param, (lo, hi) in CONSTRAINTS.items():
        value = conditions.get(param)
        if value is None:
            result.add_error(f"Missing required parameter: '{param}'")
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            result.add_error(f"Parameter '{param}' must be numeric; got {value!r}")
            continue

        if not (lo <= value <= hi):
            result.add_error(
                f"Parameter '{param}' = {value} is out of bounds [{lo}, {hi}]."
            )

    # --- Catalyst check ---
    catalyst = conditions.get("catalyst", "")
    if catalyst not in ALLOWED_CATALYSTS:
        result.add_warning(
            f"Catalyst '{catalyst}' is not in the known list {ALLOWED_CATALYSTS}. "
            "Proceed only if intentional."
        )

    # --- Ligand check ---
    ligand = conditions.get("ligand", "")
    if ligand not in ALLOWED_LIGANDS:
        result.add_warning(
            f"Ligand '{ligand}' is not in the known list {ALLOWED_LIGANDS}. "
            "Verify compatibility before running."
        )

    # --- Soft advisory: high temperature reduces L:B ---
    temp = conditions.get("temperature_C", 0)
    if isinstance(temp, (int, float)) and temp > 110:
        result.add_warning(
            f"Temperature {temp}°C is high — literature suggests L:B selectivity "
            "decreases above 100°C due to faster isomerisation."
        )

    if result.valid:
        logger.info("Condition validation passed.")
    else:
        logger.warning("Condition validation FAILED: %s", result.errors)

    return result


def validate_all(conditions: dict, substrate_smiles: Optional[str] = None) -> ValidationResult:
    """Run both SMILES and constraint validation in one call.

    Parameters
    ----------
    conditions:
        Proposed experimental conditions dictionary.
    substrate_smiles:
        Optional SMILES string for the substrate. Skipped if None.

    Returns
    -------
    ValidationResult
        Combined result from both validation layers.
    """
    result = validate_conditions(conditions)

    if substrate_smiles:
        smiles_valid, smiles_error = validate_substrate_smiles(substrate_smiles)
        if not smiles_valid:
            result.add_error(smiles_error)

    return result
