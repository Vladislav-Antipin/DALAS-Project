"""Main pipeline for fetching and processing all data."""

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
    
    # 1. Process drugs
    print("\n[1/4] Processing drugs...")
    final_drugs_df = process_drugs_data(
        raw_data["drugs_df"],
        raw_data["mechanism_df"]
    )
    save_processed_data(final_drugs_df, "final_drugs_df.pkl")
    
    # 2. Process diseases
    print("\n[2/4] Processing diseases...")
    final_diseases_df = process_diseases_data(raw_data["diseases_df"])
    save_processed_data(final_diseases_df, "final_diseases_df.pkl")
    
    # 3. Process trials
    print("\n[3/4] Processing trials...")
    final_trials_df = process_trials_data(raw_data["trials_df"])
    save_processed_data(final_trials_df, "final_trials_df.pkl")
    
    # 4. Process indications
    print("\n[4/4] Processing indications...")
    final_indications_df = process_indications_data(
        raw_data["indications_df"],
        final_trials_df
    )
    save_processed_data(final_indications_df, "final_indications_df.pkl")
    
    print("\n✓ Data processing complete!")
    return {
        "final_drugs_df": final_drugs_df,
        "final_diseases_df": final_diseases_df,
        "final_trials_df": final_trials_df,
        "final_indications_df": final_indications_df,
    }


def run_embeddings(processed_data: dict = None, force: bool = False):
    """
    Generate embeddings and compute similarities.
    
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
            "final_drugs_df": load_processed_data("final_drugs_df.pkl"),
            "final_diseases_df": load_processed_data("final_diseases_df.pkl"),
        }
    
    print("\nComputing drug-disease name similarities...")
    embeddings_df = compute_drug_disease_similarities(
        processed_data["final_drugs_df"],
        processed_data["final_diseases_df"],
        force=force
    )
    save_processed_data(embeddings_df, "embeddings_df.pkl")
    
    print("\n✓ Embeddings generation complete!")
    return embeddings_df


def run_full_pipeline(force_fetch: bool = False, force_process: bool = False):
    """
    Run the complete data pipeline.
    
    Args:
        force_fetch: If True, refetch all raw data
        force_process: If True, reprocess all data
    """
    print("\n" + "="*80)
    print(" "*20 + "DALAS DATA PIPELINE")
    print("="*80)
    
    # Step 1: Fetch raw data
    raw_data = run_data_fetching(force=force_fetch)
    
    # Step 2: Process data
    processed_data = run_data_processing(raw_data=raw_data, force=force_process)
    
    # Step 3: Generate embeddings
    _ = run_embeddings(processed_data=processed_data, force=force_process)  # Saved to disk
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    print("\nGenerated datasets:")
    print("  Raw data (data/raw/):")
    print("    - mesh_ids.pkl")
    print("    - indications_df.pkl")
    print("    - drugs_df.pkl")
    print("    - mechanism_df.pkl")
    print("    - targets_df.pkl")
    print("    - diseases_df.pkl")
    print("    - trials_df.pkl")
    print("\n  Processed data (data/processed/):")
    print("    - final_drugs_df.pkl")
    print("    - final_diseases_df.pkl")
    print("    - final_trials_df.pkl")
    print("    - final_indications_df.pkl")
    print("    - embeddings_df.pkl")
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
