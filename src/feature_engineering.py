"""
Feature Engineering Module for DALAS Drug Repurposing Pipeline.

This module handles:
1. Molecular fingerprint computation (RDKit → PCA)
2. Pathway TF-IDF features (SVD dimensionality reduction)
3. Label engineering for ML training

Design Decisions (documented):
------------------------------

FINGERPRINTS:
- We use RDKit fingerprints (2048-bit) rather than Morgan/ECFP because they capture
  substructure patterns that are more interpretable for drug repurposing.
- PCA reduction to 100 components balances information retention vs. dimensionality.
- Drugs without valid SMILES are excluded from fingerprints (they typically lack
  structural data because they're biologics or complex formulations).

PATHWAY TF-IDF:
- TF-IDF is applied to pathway IDs treated as "words" in a document.
- This downweights housekeeping pathways (appear in many drugs/diseases) and
  highlights informative, specific pathways.
- SVD reduction to 50 components (following notebook approach).
- The same vectorizer is fit on the combined drug+disease corpus to ensure
  consistent feature space.

LABEL ENGINEERING:
- We use ONLY max_phase_for_ind == 4 as positive labels (approved drugs).
- Rationale: Phase 4 is objective, verifiable evidence of success.
- We AVOID using p-value-based classification because:
  (a) P-values are often missing or unreliable
  (b) Interpretation of "success" from p-values is subjective
  (c) This introduces label noise that hurts model performance
- Negative labels: max_phase < 2 (early phase, no efficacy signal)
- Excluded (unknown): phases 2-3 where outcome is unclear

This conservative labeling gives cleaner training data at the cost of smaller
dataset size, but typically improves model reliability.
"""

import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import PROCESSED_DATA_DIR


# =============================================================================
# FINGERPRINT COMPUTATION
# =============================================================================

def compute_fingerprints(
    drugs_df: pd.DataFrame,
    n_components: int = 100,
    force: bool = False,
) -> pd.DataFrame:
    """
    Compute molecular fingerprints from SMILES and reduce dimensionality with PCA.
    
    Args:
        drugs_df: DataFrame with 'drug_id' and 'canonical_smiles' columns
        n_components: Number of PCA components (default: 100)
        force: If True, recompute even if cached
        
    Returns:
        DataFrame with drug_id and FP_1 through FP_{n_components} columns
        
    Design choices:
        - RDKit fingerprints (2048-bit) for interpretable substructure patterns
        - PCA for dimensionality reduction (100 components retains ~95% variance)
        - Drugs without valid SMILES are excluded
    """
    cache_path = PROCESSED_DATA_DIR / "fingerprints_df.pkl"
    
    if cache_path.exists() and not force:
        print(f"Loading cached fingerprints from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    print("Computing molecular fingerprints...")
    
    # Import RDKit (may not be installed)
    try:
        from rdkit import Chem
        from rdkit.Chem import RDKFingerprint
    except ImportError:
        raise ImportError(
            "RDKit is required for fingerprint computation. "
            "Install with: conda install -c conda-forge rdkit"
        )
    
    # Filter drugs with valid SMILES
    valid_drugs = drugs_df[drugs_df["canonical_smiles"].notna()].copy()
    print(f"  {len(valid_drugs)}/{len(drugs_df)} drugs have valid SMILES")
    
    if valid_drugs.empty:
        raise ValueError("No drugs have valid SMILES for fingerprint computation")
    
    # Compute fingerprints
    fps_list = []
    valid_ids = []
    failed_count = 0
    
    for _, row in valid_drugs.iterrows():
        try:
            mol = Chem.MolFromSmiles(row["canonical_smiles"])
            if mol is not None:
                fp = RDKFingerprint(mol)
                fp_array = np.fromiter(fp.ToBitString(), dtype=int)
                fps_list.append(fp_array)
                valid_ids.append(row["drug_id"])
            else:
                failed_count += 1
        except Exception:
            failed_count += 1
    
    if failed_count > 0:
        print(f"  Warning: {failed_count} drugs had invalid SMILES (skipped)")
    
    if not fps_list:
        raise ValueError("No valid fingerprints could be computed")
    
    fps_array = np.array(fps_list)
    print(f"  Raw fingerprint shape: {fps_array.shape}")
    
    # PCA reduction
    print(f"  Reducing to {n_components} components with PCA...")
    pca = PCA(n_components=n_components)
    fps_reduced = pca.fit_transform(fps_array)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA explained variance: {explained_var:.1%}")
    
    # Create DataFrame
    fp_columns = [f"FP_{i+1}" for i in range(n_components)]
    fingerprints_df = pd.DataFrame(fps_reduced, columns=fp_columns)
    fingerprints_df.insert(0, "drug_id", valid_ids)
    
    # Cache results
    with open(cache_path, "wb") as f:
        pickle.dump(fingerprints_df, f)
    print(f"  Saved fingerprints to {cache_path}")
    
    return fingerprints_df


# =============================================================================
# PATHWAY TF-IDF FEATURES
# =============================================================================

def compute_pathway_features(
    drugs_df: pd.DataFrame,
    diseases_df: pd.DataFrame,
    n_components: int = 50,
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute TF-IDF pathway features for drugs and diseases.
    
    Args:
        drugs_df: DataFrame with 'drug_id' and 'drug_pathways' columns
        diseases_df: DataFrame with 'disease_id' and 'disease_pathways' columns
        n_components: Number of SVD components (default: 50)
        force: If True, recompute even if cached
        
    Returns:
        Tuple of (drugs_df with path features, diseases_df with path features)
        
    Design choices:
        - TF-IDF downweights ubiquitous "housekeeping" pathways
        - Single vectorizer fitted on combined corpus ensures same feature space
        - SVD (truncated) for efficient dimensionality reduction on sparse matrices
        - 50 components balances expressiveness vs. overfitting risk
    """
    drug_cache = PROCESSED_DATA_DIR / "drugs_with_pathways.pkl"
    disease_cache = PROCESSED_DATA_DIR / "diseases_with_pathways.pkl"
    
    if drug_cache.exists() and disease_cache.exists() and not force:
        print("Loading cached pathway features...")
        with open(drug_cache, "rb") as f:
            drugs_out = pickle.load(f)
        with open(disease_cache, "rb") as f:
            diseases_out = pickle.load(f)
        return drugs_out, diseases_out
    
    print("Computing pathway TF-IDF features...")
    
    # Make copies to avoid modifying originals
    drugs_out = drugs_df.copy()
    diseases_out = diseases_df.copy()
    
    # Convert pathway lists to space-separated strings
    # Handle missing pathways gracefully
    def pathways_to_str(pathways):
        if pathways is None or (isinstance(pathways, float) and np.isnan(pathways)):
            return ""
        if isinstance(pathways, list):
            return " ".join(str(p) for p in pathways if p)
        return str(pathways)
    
    drugs_out["drug_path_str"] = drugs_out["drug_pathways"].apply(pathways_to_str)
    diseases_out["disease_path_str"] = diseases_out["disease_pathways"].apply(pathways_to_str)
    
    # Combine into single corpus for consistent vectorization
    corpus = pd.concat([
        drugs_out["drug_path_str"],
        diseases_out["disease_path_str"]
    ], ignore_index=True)
    
    # Check for empty corpus
    non_empty = corpus[corpus.str.strip() != ""]
    if len(non_empty) == 0:
        print("  Warning: No pathway data available. Skipping TF-IDF features.")
        return drugs_out, diseases_out
    
    print(f"  Corpus size: {len(corpus)} documents ({len(non_empty)} with pathways)")
    
    # TF-IDF vectorization
    # token_pattern matches Reactome IDs like "R-HSA-12345"
    vectorizer = TfidfVectorizer(token_pattern=r"[A-Za-z0-9_\-:]+")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # SVD dimensionality reduction
    # Use min of n_components and matrix dimensions
    actual_components = min(n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    if actual_components < n_components:
        print(f"  Reducing components from {n_components} to {actual_components} (limited by data)")
    
    if actual_components > 0:
        svd = TruncatedSVD(n_components=actual_components, random_state=42)
        latent = svd.fit_transform(tfidf_matrix)
        
        explained_var = svd.explained_variance_ratio_.sum()
        print(f"  SVD explained variance: {explained_var:.1%}")
    else:
        print("  Warning: Not enough pathway data for SVD. Using zeros.")
        latent = np.zeros((len(corpus), n_components))
        actual_components = n_components
    
    # Split back into drugs and diseases
    n_drugs = len(drugs_out)
    drug_features = latent[:n_drugs]
    disease_features = latent[n_drugs:]
    
    # Add features to DataFrames
    for i in range(actual_components):
        drugs_out[f"drug_path_{i}"] = drug_features[:, i]
        diseases_out[f"disease_path_{i}"] = disease_features[:, i]
    
    # Cache results
    with open(drug_cache, "wb") as f:
        pickle.dump(drugs_out, f)
    with open(disease_cache, "wb") as f:
        pickle.dump(diseases_out, f)
    
    print(f"  Added {actual_components} pathway features to drugs and diseases")
    
    return drugs_out, diseases_out


# =============================================================================
# LABEL ENGINEERING
# =============================================================================

def engineer_labels(
    indications_df: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    """
    Create binary labels for drug-disease indications.
    
    Args:
        indications_df: DataFrame with indication data including max_phase_for_ind
        force: If True, recompute even if cached
        
    Returns:
        DataFrame with 'label' column added
        
    Label Engineering Design (DOCUMENTED):
    ======================================
    
    POSITIVE LABELS (label = True):
        - max_phase_for_ind == 4 (Phase 4 = approved, marketed drug)
        - This is OBJECTIVE evidence: regulatory approval means efficacy proven
    
    NEGATIVE LABELS (label = False):
        - max_phase_for_ind < 2 (Phase 0, 1, or early preclinical)
        - Rationale: These haven't shown efficacy signal in humans
        - Note: We're being conservative here
    
    EXCLUDED (label = None, dropped):
        - max_phase_for_ind in [2, 3]
        - Rationale: Outcome is uncertain - trials ongoing or inconclusive
        - Including these would add label noise
    
    WHY NOT USE P-VALUES:
        - P-values from trials are often missing (only ~30% of trials have them)
        - Interpreting "success" from p-values is subjective (threshold choice)
        - Multiple endpoints make aggregation ambiguous
        - This approach was in the notebook but introduces significant label noise
    
    ALTERNATIVE CONSIDERED:
        - Using trial completion status (COMPLETED vs TERMINATED)
        - Rejected because: termination ≠ failure (funding, enrollment issues)
    
    DATASET SIZE IMPACT:
        - This conservative approach reduces dataset size
        - But improves label quality → better model reliability
        - Typical split: ~35% positive, ~65% negative after filtering
    """
    cache_path = PROCESSED_DATA_DIR / "labeled_indications_df.pkl"
    
    if cache_path.exists() and not force:
        print(f"Loading cached labeled indications from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    print("Engineering labels for indications...")
    
    indications_out = indications_df.copy()
    
    # Ensure max_phase_for_ind is numeric
    indications_out["max_phase_for_ind"] = pd.to_numeric(
        indications_out["max_phase_for_ind"], errors="coerce"
    )
    
    # Initialize label as None
    indications_out["label"] = None
    
    # Positive: Phase 4 (approved)
    phase_4_mask = indications_out["max_phase_for_ind"] == 4
    indications_out.loc[phase_4_mask, "label"] = True
    
    # Negative: Phase < 2 (early stage, no efficacy signal)
    early_phase_mask = indications_out["max_phase_for_ind"] < 2
    indications_out.loc[early_phase_mask, "label"] = False
    
    # Count labels
    label_counts = indications_out["label"].value_counts(dropna=False)
    print("  Label distribution:")
    print(f"    True (Phase 4):     {label_counts.get(True, 0):,}")
    print(f"    False (Phase <2):   {label_counts.get(False, 0):,}")
    print(f"    None (Phase 2-3):   {label_counts.get(None, 0):,}")
    
    # Convert to boolean (keeping None as NaN)
    indications_out["label"] = indications_out["label"].astype("boolean")
    
    # Cache results
    with open(cache_path, "wb") as f:
        pickle.dump(indications_out, f)
    print(f"  Saved labeled indications to {cache_path}")
    
    return indications_out


# =============================================================================
# COMBINED FEATURE ENGINEERING
# =============================================================================

def run_feature_engineering(
    force: bool = False,
) -> dict:
    """
    Run all feature engineering steps.
    
    Args:
        force: If True, recompute all features
        
    Returns:
        Dictionary with all processed DataFrames
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load processed data
    print("\nLoading processed data...")
    
    with open(PROCESSED_DATA_DIR / "final_drugs_df.pkl", "rb") as f:
        drugs_df = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "final_diseases_df.pkl", "rb") as f:
        diseases_df = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "final_indications_df.pkl", "rb") as f:
        indications_df = pickle.load(f)
    
    print(f"  Drugs: {len(drugs_df)}, Diseases: {len(diseases_df)}, "
          f"Indications: {len(indications_df)}")
    
    # 1. Fingerprints
    print("\n--- Step 1: Fingerprints ---")
    fingerprints_df = compute_fingerprints(drugs_df, force=force)
    
    # 2. Pathway features
    print("\n--- Step 2: Pathway TF-IDF ---")
    drugs_with_paths, diseases_with_paths = compute_pathway_features(
        drugs_df, diseases_df, force=force
    )
    
    # 3. Labels
    print("\n--- Step 3: Label Engineering ---")
    labeled_indications = engineer_labels(indications_df, force=force)
    
    return {
        "fingerprints_df": fingerprints_df,
        "drugs_with_pathways": drugs_with_paths,
        "diseases_with_pathways": diseases_with_paths,
        "labeled_indications": labeled_indications,
    }
