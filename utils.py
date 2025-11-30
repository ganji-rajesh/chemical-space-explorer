"""Utility functions for loading data and configuration."""

import os
import yaml
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from rdkit import Chem
from rdkit.Chem import PandasTools

warnings.filterwarnings('ignore')


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
    """Validate a SMILES string and return canonical form."""
    if pd.isna(smiles) or not smiles:
        return False, None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return True, canonical
    except:
        return False, None


def load_molecules(file_path: str, config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """Load molecules from various file formats."""
    errors = []
    file_ext = Path(file_path).suffix.lower()

    try:
        # Load based on file format
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext == '.sdf':
            df = PandasTools.LoadSDF(file_path)
            if 'ROMol' in df.columns:
                df['smiles'] = df['ROMol'].apply(
                    lambda x: Chem.MolToSmiles(x) if x else None
                )
        elif file_ext in ['.smi', '.txt']:
            with open(file_path, 'r') as f:
                lines = [line.strip().split() for line in f if line.strip()]
            if len(lines[0]) == 1:
                df = pd.DataFrame({'smiles': [l[0] for l in lines]})
            else:
                df = pd.DataFrame({
                    'smiles': [l[0] for l in lines],
                    'name': [l[1] if len(l) > 1 else f'mol_{i}' for i, l in enumerate(lines)]
                })
        elif file_ext in ['.mol', '.mol2']:
            mol = Chem.MolFromMolFile(file_path)
            if mol:
                df = pd.DataFrame({'smiles': [Chem.MolToSmiles(mol)]})
            else:
                raise ValueError("Could not parse MOL file")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        required_col = config['input']['required_column']
        if required_col not in df.columns:
            raise ValueError(f"Required column '{required_col}' not found")

        # Validate and canonicalize SMILES
        valid_mask = []
        canonical_smiles = []

        for idx, smi in enumerate(df[required_col]):
            is_valid, canonical = validate_smiles(smi)
            valid_mask.append(is_valid)
            canonical_smiles.append(canonical)
            if not is_valid:
                errors.append(f"Row {idx}: Invalid SMILES '{smi}'")

        df = df[valid_mask].copy()
        df['smiles'] = [canonical_smiles[i] for i, v in enumerate(valid_mask) if v]

        max_mols = config['input']['max_molecules']
        if len(df) > max_mols:
            errors.append(f"Dataset truncated from {len(df)} to {max_mols} molecules")
            df = df.head(max_mols)

        if 'name' not in df.columns:
            df['name'] = [f'mol_{i}' for i in range(len(df))]

        return df, errors

    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")
