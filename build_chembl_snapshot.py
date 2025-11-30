"""Script to build ChEMBL approved drugs snapshot."""

import requests
import pandas as pd
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import validate_smiles, load_config
from src.featurization import compute_properties, compute_fingerprints
from src.embedding import compute_umap_embedding
from src.clustering import perform_clustering


def fetch_chembl_approved_drugs(max_results=2000):
    """Fetch approved drugs from ChEMBL API."""
    base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule"
    params = {
        'max_phase': 4,
        'molecule_type': 'Small molecule',
        'format': 'json',
        'limit': 1000
    }

    molecules = []
    offset = 0

    print("Fetching approved drugs from ChEMBL...")

    while len(molecules) < max_results:
        params['offset'] = offset

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'molecules' not in data or not data['molecules']:
                break

            molecules.extend(data['molecules'])
            print(f"Fetched {len(molecules)} molecules...")

            offset += params['limit']
            time.sleep(0.5)

        except Exception as e:
            print(f"Error: {e}")
            break

    return molecules[:max_results]


def process_chembl_data(molecules):
    """Process ChEMBL molecules into DataFrame."""
    records = []

    for mol in molecules:
        record = {
            'chembl_id': mol.get('molecule_chembl_id'),
            'smiles': mol.get('molecule_structures', {}).get('canonical_smiles'),
            'name': mol.get('pref_name', mol.get('molecule_chembl_id')),
            'max_phase': mol.get('max_phase'),
        }

        if 'atc_classifications' in mol and mol['atc_classifications']:
            atc = mol['atc_classifications'][0]
            record['ATC Level 1'] = atc[:1] if atc else None
            record['ATC Level 2'] = atc[:3] if len(atc) >= 3 else None

        if 'molecule_hierarchy' in mol:
            hierarchy = mol['molecule_hierarchy']
            record['Target Family'] = hierarchy.get('protein_class_desc')

        records.append(record)

    df = pd.DataFrame(records)

    print("Validating SMILES...")
    valid_mask = []
    canonical_smiles = []

    for smiles in df['smiles']:
        is_valid, canonical = validate_smiles(smiles)
        valid_mask.append(is_valid)
        canonical_smiles.append(canonical)

    df = df[valid_mask].copy()
    df['smiles'] = [canonical_smiles[i] for i, v in enumerate(valid_mask) if v]
    df['FDA Approved'] = True

    if 'ATC Level 1' in df.columns:
        df['Therapeutic Area'] = df['ATC Level 1'].map({
            'A': 'Alimentary', 'B': 'Blood', 'C': 'Cardiovascular',
            'D': 'Dermatological', 'G': 'Genitourinary', 'H': 'Hormones',
            'J': 'Anti-infectives', 'L': 'Antineoplastic', 'M': 'Musculoskeletal',
            'N': 'Nervous system', 'P': 'Antiparasitic', 'R': 'Respiratory',
            'S': 'Sensory organs', 'V': 'Various'
        }).fillna('Unknown')

    return df


def main():
    """Main execution function."""
    print("="*80)
    print("ChEMBL Approved Drugs Snapshot Builder")
    print("="*80)

    molecules = fetch_chembl_approved_drugs(max_results=2000)
    print(f"\nFetched {len(molecules)} molecules from ChEMBL")

    df = process_chembl_data(molecules)
    print(f"\nProcessed {len(df)} valid molecules")

    config = load_config("config.yaml")

    print("\nComputing molecular properties...")
    df = compute_properties(df)

    print("Computing fingerprints...")
    fingerprints = compute_fingerprints(df, config)

    print("Computing UMAP embedding...")
    embedding = compute_umap_embedding(fingerprints, config)

    print("Performing clustering...")
    labels = perform_clustering(embedding, config)

    df['umap_x'] = embedding[:, 0]
    df['umap_y'] = embedding[:, 1]
    df['cluster'] = labels

    output_path = Path("data/chembl_approved_drugs.parquet")
    output_path.parent.mkdir(exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(f"\nSaved dataset to {output_path}")
    print(f"Total molecules: {len(df)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
