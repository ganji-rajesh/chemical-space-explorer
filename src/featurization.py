"""Molecular featurization and property computation."""

import numpy as np
import pandas as pd
from typing import Dict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski


def compute_fingerprints(df: pd.DataFrame, config: Dict) -> np.ndarray:
    """Compute Morgan fingerprints for molecules."""
    fp_config = config['fingerprint']
    radius = fp_config['radius']
    n_bits = fp_config['n_bits']

    fingerprints = []
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fingerprints.append(np.zeros(n_bits))
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits
        )
        arr = np.zeros(n_bits, dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)

    return np.array(fingerprints)


def compute_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Compute molecular properties for all molecules."""
    properties = {
        'LogP': [], 'Molecular Weight': [], 'TPSA': [],
        'HBA': [], 'HBD': [], 'Rotatable Bonds': [], 'Aromatic Rings': []
    }

    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            for key in properties:
                properties[key].append(np.nan)
            continue

        properties['LogP'].append(Descriptors.MolLogP(mol))
        properties['Molecular Weight'].append(Descriptors.MolWt(mol))
        properties['TPSA'].append(Descriptors.TPSA(mol))
        properties['HBA'].append(Lipinski.NumHAcceptors(mol))
        properties['HBD'].append(Lipinski.NumHDonors(mol))
        properties['Rotatable Bonds'].append(Lipinski.NumRotatableBonds(mol))
        properties['Aromatic Rings'].append(Lipinski.NumAromaticRings(mol))

    result_df = df.copy()
    for key, values in properties.items():
        result_df[key] = values

    return result_df
