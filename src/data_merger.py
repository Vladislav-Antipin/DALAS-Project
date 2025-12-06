"""
Data Merger Module for DALAS Drug Repurposing Pipeline.

This module creates the final ML-ready dataset by merging:
- Drug features (properties, fingerprints, pathway embeddings)
- Disease features (pathway embeddings)
- Indication labels
- Name similarity embeddings

Design Decisions (documented):
------------------------------

MERGE STRATEGY:
- Inner join on drug_id and disease_id to ensure all features present
- This drops indications where drug or disease data is missing

FILTERING:
- Exclude biotherapeutics (biologics have different mechanisms than small molecules)
- Exclude rows with missing labels (unknown outcome)
- Rationale: Cleaner training data, focused on small molecule drugs

FEATURE SELECTION:
- Drop object/string columns (not directly usable in ML)
- Drop identifier columns (drug_id, disease_id, names)
- Drop columns with >50% missing values
- Keep: numeric drug properties, fingerprints, pathway features, embeddings

COLUMN NAMING:
- Consistent naming: drug_*, disease_*, FP_*, label
- Original ChEMBL property names preserved for interpretability
"""

import pickle

import pandas as pd

from .config import PROCESSED_DATA_DIR, RESULTS_DIR


def merge_all_features(
    force: bool = False,
) -> pd.DataFrame:
    """
    Merge all features into a single ML-ready dataset.
    
    Args:
        force: If True, recompute even if cached
        
    Returns:
        DataFrame ready for model training with 'label' column
        
    Merge order:
        1. indications (with labels) 
        2. + drugs (with fingerprints + pathway features)
        3. + diseases (with pathway features)
        4. + embeddings (name similarity)
    """
    cache_path = PROCESSED_DATA_DIR / "merged_df.pkl"
    
    if cache_path.exists() and not force:
        print(f"Loading cached merged dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    print("\n" + "=" * 60)
    print("DATA MERGING")
    print("=" * 60)
    
    # Load all required datasets
    print("\nLoading datasets...")
    
    # Labeled indications
    indications_path = PROCESSED_DATA_DIR / "labeled_indications_df.pkl"
    if not indications_path.exists():
        raise FileNotFoundError(
            f"Labeled indications not found at {indications_path}. "
            "Run feature engineering first."
        )
    with open(indications_path, "rb") as f:
        indications_df = pickle.load(f)
    
    # Drugs with pathway features
    drugs_path = PROCESSED_DATA_DIR / "drugs_with_pathways.pkl"
    if not drugs_path.exists():
        # Fall back to final_drugs_df
        drugs_path = PROCESSED_DATA_DIR / "final_drugs_df.pkl"
    with open(drugs_path, "rb") as f:
        drugs_df = pickle.load(f)
    
    # Diseases with pathway features  
    diseases_path = PROCESSED_DATA_DIR / "diseases_with_pathways.pkl"
    if not diseases_path.exists():
        # Fall back to final_diseases_df
        diseases_path = PROCESSED_DATA_DIR / "final_diseases_df.pkl"
    with open(diseases_path, "rb") as f:
        diseases_df = pickle.load(f)
    
    # Fingerprints
    fingerprints_path = PROCESSED_DATA_DIR / "fingerprints_df.pkl"
    if fingerprints_path.exists():
        with open(fingerprints_path, "rb") as f:
            fingerprints_df = pickle.load(f)
        has_fingerprints = True
    else:
        print("  Warning: Fingerprints not found. Skipping.")
        has_fingerprints = False
    
    # Embeddings (name similarity)
    embeddings_path = PROCESSED_DATA_DIR / "embeddings_df.pkl"
    if embeddings_path.exists():
        with open(embeddings_path, "rb") as f:
            embeddings_df = pickle.load(f)
        has_embeddings = True
    else:
        print("  Warning: Embeddings not found. Skipping.")
        has_embeddings = False
    
    print(f"  Indications: {len(indications_df)}")
    print(f"  Drugs: {len(drugs_df)}")
    print(f"  Diseases: {len(diseases_df)}")
    if has_fingerprints:
        print(f"  Fingerprints: {len(fingerprints_df)}")
    if has_embeddings:
        print(f"  Embeddings: {len(embeddings_df)}")
    
    # === STEP 1: Ensure consistent ID columns ===
    print("\n--- Step 1: Standardizing ID columns ---")
    
    # Indications: ensure disease_id column exists
    if "disease_id" not in indications_df.columns:
        if "efo_id" in indications_df.columns:
            indications_df["disease_id"] = indications_df["efo_id"]
            print("  Created disease_id from efo_id in indications")
    
    # Diseases: ensure disease_id column exists
    if "disease_id" not in diseases_df.columns:
        if "id" in diseases_df.columns:
            diseases_df["disease_id"] = diseases_df["id"]
            print("  Created disease_id from id in diseases")
    
    # Normalize disease IDs (handle underscore vs colon format)
    # EFO IDs can be "EFO_0000685" or "EFO:0000685"
    def normalize_disease_id(id_val):
        if pd.isna(id_val):
            return id_val
        return str(id_val).replace("_", ":")
    
    indications_df["disease_id"] = indications_df["disease_id"].apply(normalize_disease_id)
    diseases_df["disease_id"] = diseases_df["disease_id"].apply(normalize_disease_id)
    
    if has_embeddings:
        embeddings_df["disease_id"] = embeddings_df["disease_id"].apply(normalize_disease_id)
    
    # === STEP 2: Merge drugs with fingerprints ===
    print("\n--- Step 2: Merging drugs with fingerprints ---")
    
    if has_fingerprints:
        drugs_merged = drugs_df.merge(fingerprints_df, on="drug_id", how="left")
        n_with_fp = drugs_merged[drugs_merged["FP_1"].notna()].shape[0]
        print(f"  {n_with_fp}/{len(drugs_merged)} drugs have fingerprints")
    else:
        drugs_merged = drugs_df
    
    # === STEP 3: Main merge ===
    print("\n--- Step 3: Main merge (indications + drugs + diseases) ---")
    
    # Start with indications
    merged = indications_df.copy()
    print(f"  Starting with {len(merged)} indications")
    
    # Merge drugs
    merged = merged.merge(drugs_merged, on="drug_id", how="inner")
    print(f"  After drug merge: {len(merged)} rows")
    
    # Merge diseases
    merged = merged.merge(diseases_df, on="disease_id", how="inner")
    print(f"  After disease merge: {len(merged)} rows")
    
    # Merge embeddings if available
    if has_embeddings:
        merged = merged.merge(
            embeddings_df[["drug_id", "disease_id", "name_similarity"]], 
            on=["drug_id", "disease_id"], 
            how="left"
        )
        n_with_emb = merged["name_similarity"].notna().sum()
        print(f"  After embeddings merge: {len(merged)} rows ({n_with_emb} with similarity)")
    
    # === STEP 4: Filtering ===
    print("\n--- Step 4: Filtering ---")
    
    initial_count = len(merged)
    
    # Filter out biotherapeutics (if column exists)
    if "biotherapeutic" in merged.columns:
        # Convert to numeric/boolean for filtering
        bio_mask = merged["biotherapeutic"].astype(bool)
        merged = merged[~bio_mask]
        print(f"  Removed {initial_count - len(merged)} biotherapeutics")
    
    # Filter out rows without labels
    before_label_filter = len(merged)
    merged = merged[merged["label"].notna()]
    print(f"  Removed {before_label_filter - len(merged)} rows without labels")
    
    print(f"  Final dataset: {len(merged)} rows")
    
    # === STEP 5: Feature selection ===
    print("\n--- Step 5: Feature selection ---")
    
    # Columns to drop (identifiers, text, redundant)
    drop_cols = [
        # Identifiers (not features)
        "drug_id", "disease_id", "efo_id", "efo_term", "mesh_id",
        "drug_name", "name",
        # Redundant or processed
        "indication_refs", "nct_ids", "drug_pathways", "disease_pathways",
        "drug_path_str", "disease_path_str",
        "associatedTargets", "phenotypes", "targets",
        "canonical_smiles", "standard_inchi", "standard_inchi_key",
        "description", "helm_notation", "full_molformula",
        "usan_stem_definition",
        # Trial-related (raw, not engineered)
        "n_trials", "max_trial_phase", "has_completed_trial",
        # Phase info (used for labels, would leak)
        "max_phase_for_ind", "max_phase",
    ]
    
    # Only drop columns that exist
    cols_to_drop = [c for c in drop_cols if c in merged.columns]
    merged = merged.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} identifier/text columns")
    
    # Drop object columns (can't be used directly in ML)
    object_cols = merged.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        merged = merged.drop(columns=object_cols)
        print(f"  Dropped {len(object_cols)} remaining object columns: {object_cols[:5]}...")
    
    # Drop columns with >50% missing
    missing_threshold = 0.5
    missing_pct = merged.isna().mean()
    high_missing = missing_pct[missing_pct > missing_threshold].index.tolist()
    if high_missing:
        merged = merged.drop(columns=high_missing)
        print(f"  Dropped {len(high_missing)} columns with >{missing_threshold:.0%} missing")
    
    # Move label to first column
    if "label" in merged.columns:
        cols = ["label"] + [c for c in merged.columns if c != "label"]
        merged = merged[cols]
    
    print(f"\n  Final shape: {merged.shape}")
    print(f"  Features: {merged.shape[1] - 1}")
    print("  Label distribution:")
    label_counts = merged["label"].value_counts()
    for label, count in label_counts.items():
        print(f"    {label}: {count} ({count/len(merged):.1%})")
    
    # === STEP 6: Save ===
    print("\n--- Step 6: Saving ---")
    
    with open(cache_path, "wb") as f:
        pickle.dump(merged, f)
    print(f"  Saved to {cache_path}")
    
    # Also save to results for easy access
    results_path = RESULTS_DIR / "merged_df.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(merged, f)
    print(f"  Also saved to {results_path}")
    
    return merged


def get_feature_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of features in the merged dataset.
    
    Returns:
        DataFrame with feature statistics
    """
    summary = []
    
    for col in merged_df.columns:
        if col == "label":
            continue
            
        info = {
            "feature": col,
            "dtype": str(merged_df[col].dtype),
            "non_null": merged_df[col].notna().sum(),
            "null_pct": merged_df[col].isna().mean(),
            "unique": merged_df[col].nunique(),
        }
        
        if merged_df[col].dtype in ["float64", "int64", "float32", "int32"]:
            info["mean"] = merged_df[col].mean()
            info["std"] = merged_df[col].std()
            info["min"] = merged_df[col].min()
            info["max"] = merged_df[col].max()
        
        summary.append(info)
    
    return pd.DataFrame(summary)
