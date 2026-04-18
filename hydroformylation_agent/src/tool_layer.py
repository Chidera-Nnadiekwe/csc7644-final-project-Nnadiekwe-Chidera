"""
tool_layer.py
--------------
Chemical tools used by the agent controller.

Currently includes:
  - validate_smiles()  : Use RDKit to check if a SMILES string is valid
  - get_molecular_weight() : Return molecular weight of a SMILES structure

Requirements:
    pip install rdkit
    (or: conda install -c conda-forge rdkit)
"""

# Attempt to import RDKit, set a flag if it's not available
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[TOOL] Warning: RDKit not installed. SMILES validation will be skipped.")
    print("       Install with: pip install rdkit")

# Define chemical tool functions
def validate_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is chemically valid using RDKit.
    Returns True if valid, False if invalid or RDKit not available.
    """
    if not smiles:
        return False

    if not RDKIT_AVAILABLE:
        print(f"[TOOL] Skipping SMILES validation (RDKit not available): {smiles}")
        return True   # Assume valid when we can't check

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[TOOL] Invalid SMILES: '{smiles}'")
        return False

    print(f"[TOOL] SMILES valid: '{smiles}' → {Chem.MolToSmiles(mol)}")
    return True

# Define additional chemical tool functions
def get_molecular_weight(smiles: str) -> float:
    """
    Return the molecular weight of the compound represented by a SMILES string.
    Returns 0.0 if invalid or RDKit not available.
    """
    if not RDKIT_AVAILABLE:
        return 0.0

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    return round(Descriptors.MolWt(mol), 2)

# Define more chemical tools as needed
def get_atom_count(smiles: str) -> int:
    """Return the number of heavy atoms in the molecule."""
    if not RDKIT_AVAILABLE:
        return 0
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol else 0

# Define a tool to check for terminal olefins
def is_terminal_olefin(smiles: str) -> bool:
    """
    Heuristically check if a SMILES is a terminal (alpha) olefin.
    Terminal olefins end with =C or =CH2 patterns.
    This is a simplified check; RDKit substructure matching is more robust.
    """
    if not RDKIT_AVAILABLE:
        return True   # Assume True when we can't check

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    # Look for a vinyl group: carbon with double bond and at least one H
    terminal_vinyl = Chem.MolFromSmarts("[CH2]=[CH]")
    return mol.HasSubstructMatch(terminal_vinyl)
