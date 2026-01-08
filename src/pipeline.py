"""Main pipeline for fetching and processing all data.

Matches the logic from notebooks:
- 01_1-data_retrieval.ipynb
- 01_2-rdkit_fingerprints.ipynb
- 01_3-additional_disease_features.ipynb
- 02-data_preparation.ipynb
"""

import argparse
import pandas as pd

from .data_fetchers import (
    fetch_mesh_ids,
    fetch_drug_indications,
    fetch_drugs,
    fetch_mechanisms_and_targets,
    fetch_disease_info,
    fetch_clinical_trials,
    load_raw_data,
    save_raw_data,
)
from .data_processors import (
    process_drugs_data,
    process_diseases_data,
    process_trials_data,
    process_indications_data,
    save_processed_data,
    load_processed_data,
)
from .embeddings import compute_drug_disease_similarities
from .fingerprints import generate_fingerprints
from .disease_features import generate_extended_disease_features
from .pathway_mapping import map_targets_to_pathways, generate_pathway_embeddings


def run_data_fetching(force: bool = False):
    """
    Run all data fetching steps.
    
    Args:
        force: If True, refetch all data even if cached
    """
    print("\n" + "="*60)
    print("STEP 1: FETCHING RAW DATA")
    print("="*60)
    
    # 1. Fetch MeSH IDs
    print("\n[1/6] Fetching MeSH IDs...")
    mesh_ids = fetch_mesh_ids(force=force)
    
    # 2. Fetch drug indications
    print("\n[2/6] Fetching drug indications...")
    indications_df = fetch_drug_indications(mesh_ids, force=force)
    chembl_ids = pd.unique(indications_df["molecule_chembl_id"]).tolist()
    
    # 3. Fetch drugs
    print("\n[3/6] Fetching drugs...")
    drugs_df = fetch_drugs(chembl_ids, force=force)
    
    # 4. Fetch mechanisms and targets
    print("\n[4/6] Fetching mechanisms and targets...")
    mechanism_df, targets_df = fetch_mechanisms_and_targets(chembl_ids, force=force)
    
    # 5. Fetch disease information
    print("\n[5/6] Fetching disease information...")
    efo_ids = indications_df["efo_id"].unique().tolist()
    diseases_df = fetch_disease_info(efo_ids, force=force)
    
    # 6. Fetch clinical trials
    print("\n[6/6] Fetching clinical trials...")
    nct_refs = []
    for _, indication in indications_df.iterrows():
        for ref in indication["indication_refs"]:
            if ref["ref_type"] == "ClinicalTrials":
                nct_refs.append(ref["ref_id"].split(","))
    
    all_nct_refs = pd.unique([nct_ref for nct_ref_list in nct_refs for nct_ref in nct_ref_list])
    trials_df = fetch_clinical_trials(all_nct_refs, force=force)
    
    print("\n✓ Raw data fetching complete!")
    return {
        "mesh_ids": mesh_ids,
        "indications_df": indications_df,
        "drugs_df": drugs_df,
        "mechanism_df": mechanism_df,
        "targets_df": targets_df,
        "diseases_df": diseases_df,
        "trials_df": trials_df,
    }


def run_data_processing(raw_data: dict = None, force: bool = False):
    """
    Run all data processing steps.
    
    Matches notebooks 01_1, 01_2, 01_3, and 02 logic.
    
    Args:
        raw_data: Dictionary of raw dataframes (if None, loads from cache)
        force: If True, reprocess even if processed data exists
    """
    print("\n" + "="*60)
    print("STEP 2: PROCESSING DATA")
    print("="*60)
    
    # Load raw data if not provided
    if raw_data is None:
        print("\nLoading raw data from cache...")
        raw_data = {
            "drugs_df": load_raw_data("drugs_df.pkl"),
            "mechanism_df": load_raw_data("mechanism_df.pkl"),
            "diseases_df": load_raw_data("diseases_df.pkl"),
            "trials_df": load_raw_data("trials_df.pkl"),
            "indications_df": load_raw_data("indications_df.pkl"),
        }
    
    # 1. Process drugs (matching notebook 01_1)
    print("\n[1/8] Processing drugs...")
    final_drugs_df = process_drugs_data(
        raw_data["drugs_df"],
        raw_data["mechanism_df"]
    )
    save_raw_data(final_drugs_df, "drugs_df.pkl")
    
    # 2. Process diseases (matching notebook 01_1)
    print("\n[2/8] Processing diseases...")
    final_diseases_df = process_diseases_data(raw_data["diseases_df"])
    save_raw_data(final_diseases_df, "diseases_df.pkl")
    
    # 3. Map targets to pathways (matching notebook 01_1)
    print("\n[3/8] Mapping targets to Reactome pathways...")
    final_drugs_df, final_diseases_df = map_targets_to_pathways(
        final_drugs_df,
        final_diseases_df,
        force=force
    )
    
    # 4. Generate pathway embeddings
    print("\n[4/8] Generating pathway embeddings...")
    final_drugs_df, final_diseases_df = generate_pathway_embeddings(
        final_drugs_df,
        final_diseases_df,
        force=force
    )
    save_raw_data(final_drugs_df, "drugs_df.pkl")
    save_raw_data(final_diseases_df, "diseases_df.pkl")
    
    # 5. Generate molecular fingerprints (matching notebook 01_2)
    print("\n[5/8] Generating molecular fingerprints...")
    fingerprints_df = generate_fingerprints(
        raw_data["drugs_df"],  # Use raw drugs for SMILES
        use_morgan=True,
        n_components=100,
        use_only_safe_drugs=True,
        force=force
    )
    save_raw_data(fingerprints_df, "fingerprints_df.pkl")
    
    # 6. Generate extended disease features (matching notebook 01_3)
    print("\n[6/8] Generating extended disease features...")
    extended_diseases_df = generate_extended_disease_features(
        final_diseases_df,
        raw_data["indications_df"],
        force=force
    )
    save_raw_data(extended_diseases_df, "extended_diseases_df.pkl")
    
    # 7. Process trials (matching notebook 02)
    print("\n[7/8] Processing trials...")
    final_trials_df = process_trials_data(raw_data["trials_df"])
    save_processed_data(final_trials_df, "trials_df.pkl")
    
    # 8. Process indications with overall_success (matching notebook 02)
    print("\n[8/8] Processing indications...")
    final_indications_df, dates_df = process_indications_data(
        raw_data["indications_df"],
        final_trials_df,
        final_drugs_df
    )
    save_processed_data(final_indications_df, "indications_df.pkl")
    save_processed_data(dates_df, "dates_df.pkl")
    
    # Also save drugs and diseases to processed directory
    save_processed_data(final_drugs_df, "drugs_df.pkl")
    save_processed_data(extended_diseases_df, "diseases_df.pkl")
    
    print("\n✓ Data processing complete!")
    return {
        "final_drugs_df": final_drugs_df,
        "final_diseases_df": extended_diseases_df,
        "final_trials_df": final_trials_df,
        "final_indications_df": final_indications_df,
        "fingerprints_df": fingerprints_df,
        "dates_df": dates_df,
    }


def run_embeddings(processed_data: dict = None, force: bool = False):
    """
    Generate embeddings and compute similarities.
    
    Matches notebook 01_1-data_retrieval.ipynb logic.
    
    Args:
        processed_data: Dictionary of processed dataframes (if None, loads from cache)
        force: If True, recompute even if embeddings exist
    """
    print("\n" + "="*60)
    print("STEP 3: GENERATING EMBEDDINGS")
    print("="*60)
    
    # Load processed data if not provided
    if processed_data is None:
        print("\nLoading processed data from cache...")
        processed_data = {
            "final_drugs_df": load_processed_data("drugs_df.pkl"),
            "final_diseases_df": load_processed_data("diseases_df.pkl"),
        }
    
    print("\nComputing drug-disease name similarities...")
    embeddings_df = compute_drug_disease_similarities(
        processed_data["final_drugs_df"],
        processed_data["final_diseases_df"],
        force=force
    )
    save_raw_data(embeddings_df, "embeddings_df.pkl")
    
    print("\n✓ Embeddings generation complete!")
    return embeddings_df


def run_full_pipeline(force_fetch: bool = False, force_process: bool = False):
    """
    Run the complete data pipeline.
    
    Matches the full flow from notebooks 01_1 through 02.
    
    Args:
        force_fetch: If True, refetch all raw data
        force_process: If True, reprocess all data
    """
    print("\n" + "="*80)
    print(" "*20 + "DALAS DATA PIPELINE")
    print("="*80)
    
    # Step 1: Fetch raw data
    raw_data = run_data_fetching(force=force_fetch)
    
    # Step 2: Process data (includes fingerprints, pathways, disease features)
    processed_data = run_data_processing(raw_data=raw_data, force=force_process)
    
    # Step 3: Generate embeddings
    _ = run_embeddings(processed_data=processed_data, force=force_process)
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    print("\nGenerated datasets:")
    print("  Raw data (data/01-result/):")
    print("    - mesh_ids.pkl")
    print("    - indications_df.pkl")
    print("    - drugs_df.pkl (with targets and pathways)")
    print("    - mechanism_df.pkl")
    print("    - targets_df.pkl")
    print("    - diseases_df.pkl (with disease_targets and pathways)")
    print("    - trials_df.pkl")
    print("    - fingerprints_df.pkl")
    print("    - embeddings_df.pkl")
    print("    - extended_diseases_df.pkl")
    print("\n  Processed data (data/02-result/):")
    print("    - drugs_df.pkl")
    print("    - diseases_df.pkl")
    print("    - trials_df.pkl")
    print("    - indications_df.pkl (with overall_success)")
    print("    - dates_df.pkl")
    print("\n")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="DALAS Data Pipeline - Fetch and process autoimmune disease drug data"
    )
    parser.add_argument(
        "--step",
        choices=["fetch", "process", "embeddings", "all"],
        default="all",
        help="Which step(s) to run"
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force re-fetching of raw data"
    )
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Force re-processing of data"
    )
    
    args = parser.parse_args()
    
    if args.step == "all":
        run_full_pipeline(
            force_fetch=args.force_fetch,
            force_process=args.force_process
        )
    elif args.step == "fetch":
        run_data_fetching(force=args.force_fetch)
    elif args.step == "process":
        run_data_processing(force=args.force_process)
    elif args.step == "embeddings":
        run_embeddings(force=args.force_process)


if __name__ == "__main__":
    main()
