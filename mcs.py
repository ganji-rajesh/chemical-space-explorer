"""Maximum Common Substructure (MCS) computation."""

import signal
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import rdFMCS


class TimeoutException(Exception):
    """Custom exception for MCS timeout."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("MCS computation timed out")


def compute_mcs(smiles_list: List[str], timeout: int = 5) -> Optional[str]:
    """Compute Maximum Common Substructure for a list of molecules."""
    if not smiles_list or len(smiles_list) < 2:
        return None

    try:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        mols = [m for m in mols if m is not None]

        if len(mols) < 2:
            return None

        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        except (AttributeError, ValueError):
            pass

        result = rdFMCS.FindMCS(
            mols,
            timeout=timeout,
            completeRingsOnly=True,
            ringMatchesRingOnly=True,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder
        )

        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass

        if result.numAtoms > 0:
            return result.smartsString
        return None

    except TimeoutException:
        return None
    except Exception:
        return None
    finally:
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass


def smarts_to_mol(smarts: str) -> Optional[Chem.Mol]:
    """Convert SMARTS string to RDKit molecule."""
    try:
        return Chem.MolFromSmarts(smarts)
    except:
        return None
