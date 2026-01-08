"""Functions for generating molecular fingerprints using RDKit.

Matches notebook 01_2-rdkit_fingerprints.ipynb logic.
"""

import pickle
from typing import Any

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA

from .config import RAW_DATA_DIR


def save_fingerprints_data(obj: Any, filename: str) -> None:
    """Save object as pickle in raw data directory."""
    filepath = RAW_DATA_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {filepath}")


def load_fingerprints_data(filename: str) -> Any:
    """Load pickle from raw data directory."""
    filepath = RAW_DATA_DIR / filename
    if filepath.exists():
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None


def generate_fingerprints(
    drugs_df: pd.DataFrame,
    use_morgan: bool = True,
    n_components: int = 100,
    use_only_safe_drugs: bool = True,
    force: bool = False
) -> pd.DataFrame:
    """
    Generate molecular fingerprints for drugs and reduce dimensionality with PCA.
    
    Matches notebook 01_2-rdkit_fingerprints.ipynb logic.
    
    Args:
        drugs_df: DataFrame with drug information (must have 'canonical_smiles' column)
        use_morgan: If True, use Morgan fingerprints; otherwise use RDK fingerprints
        n_components: Number of PCA components to keep
        use_only_safe_drugs: If True, only use drugs that passed phase 3 or 4
        force: If True, regenerate even if cached data exists
        
    Returns:
        DataFrame with drug_id and fingerprint PCA components
    """
    filename = "fingerprints_df.pkl"
    
    if not force:
        cached = load_fingerprints_data(filename)
        if cached is not None:
            print(f"Loaded cached fingerprints: {len(cached)} drugs")
            return cached
    
    print("Generating molecular fingerprints...")
    
    # Filter to safe drugs if requested
    if use_only_safe_drugs:
        drugs_df = drugs_df[
            (drugs_df["max_phase"] == "4.0") | (drugs_df["max_phase"] == "3.0")
        ]
    
    # Filter to drugs with valid SMILES
    drugs_df = drugs_df[drugs_df["canonical_smiles"].notna()]
    
    smiles_list = drugs_df["canonical_smiles"]
    
    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    # Generate fingerprints
    if use_morgan:
        fpgen = AllChem.GetMorganGenerator(radius=2)
        fps = np.array([
            np.fromiter(fpgen.GetFingerprint(mol).ToBitString(), dtype=int)
            for mol in mols if mol is not None
        ])
    else:
        fps = np.array([
            np.fromiter(Chem.RDKFingerprint(mol).ToBitString(), dtype=int)
            for mol in mols if mol is not None
        ])
    
    # Reduce dimensionality with PCA
    print(f"Reducing dimensionality to {n_components} components...")
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(fps)
    
    # Create DataFrame with PCA components
    fp_df = pd.DataFrame(
        X_reduced,
        columns=[f'FP_{i+1}' for i in range(X_reduced.shape[1])]
    )
    
    # Combine with drug IDs
    fingerprints_df = pd.concat([
        drugs_df["drug_id"].reset_index(drop=True),
        fp_df
    ], axis=1)
    
    save_fingerprints_data(fingerprints_df, filename)
    print(f"Generated fingerprints for {len(fingerprints_df)} drugs")
    
    return fingerprints_df


def compute_fingerprint_similarity(mol1_smiles: str, mol2_smiles: str) -> float:
    """
    Compute Tanimoto similarity between two molecules based on their fingerprints.
    
    Args:
        mol1_smiles: SMILES string for first molecule
        mol2_smiles: SMILES string for second molecule
        
    Returns:
        Tanimoto similarity score (0-1)
    """
    from rdkit import DataStructs
    
    mol1 = Chem.MolFromSmiles(mol1_smiles)
    mol2 = Chem.MolFromSmiles(mol2_smiles)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fpgen = AllChem.GetMorganGenerator(radius=2)
    fp1 = fpgen.GetFingerprint(mol1)
    fp2 = fpgen.GetFingerprint(mol2)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)
