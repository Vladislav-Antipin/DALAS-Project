"""Functions for processing and cleaning raw data."""

import pickle
from typing import Any, Dict, List

import pandas as pd

from .config import PROCESSED_DATA_DIR


def save_processed_data(obj: Any, filename: str) -> None:
    """Save object as pickle in processed data directory."""
    filepath = PROCESSED_DATA_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {filepath}")


def load_processed_data(filename: str) -> Any:
    """Load pickle from processed data directory."""
    filepath = PROCESSED_DATA_DIR / filename
    if filepath.exists():
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None


def process_drugs_data(drugs_df: pd.DataFrame, mechanism_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean drug data.
    
    Args:
        drugs_df: Raw drugs DataFrame
        mechanism_df: Raw mechanisms DataFrame with targets
        
    Returns:
        Processed drugs DataFrame
    """
    print("Processing drug data...")
    
    # Filter: available and not withdrawn (matching notebook logic with NA handling)
    # (availability_type > 0) | isna() AND (~withdrawn_flag | isna())
    final_drugs_df = (
        drugs_df[
            ((drugs_df['availability_type'] > 0) | drugs_df['availability_type'].isna())
            & (~drugs_df['withdrawn_flag'] | drugs_df['withdrawn_flag'].isna())
        ]
        .drop(columns=[
            'atc_classifications', 'availability_type',
            'cross_references', 'molecule_hierarchy',
            'molecule_synonyms', 'polymer_flag',
            'usan_stem', 'usan_substem'
        ])
    )
    
    # Process biotherapeutic flag
    final_drugs_df["biotherapeutic"] = final_drugs_df["biotherapeutic"].notnull().astype(int)
    
    # Process chirality
    chirality_dict = {2: "achiral", 1: "single_enantiomer", 0: "mixture", -1: None}
    final_drugs_df["chirality"] = final_drugs_df["chirality"].apply(lambda x: chirality_dict[x])
    
    # Unpack molecule properties and structures
    final_drugs_df = pd.concat([
        final_drugs_df,
        pd.DataFrame([
            d if d is not None else {}
            for d in final_drugs_df["molecule_properties"].to_list()
        ]),
        pd.DataFrame([
            d if d is not None else {}
            for d in final_drugs_df["molecule_structures"].to_list()
        ])
    ], axis=1).drop(columns=["molecule_properties", "molecule_structures", "molfile"])
    
    # Normalize drug names
    final_drugs_df["pref_name"] = final_drugs_df["pref_name"].str.lower()
    
    # Reorder columns
    first_columns = ['molecule_chembl_id', 'pref_name']
    final_drugs_df = final_drugs_df[
        first_columns + [c for c in final_drugs_df.columns if c not in first_columns]
    ]
    
    # Rename for consistency
    final_drugs_df.rename(columns={
        "molecule_chembl_id": "drug_id",
        "pref_name": "drug_name"
    }, inplace=True)
    
    # Add targets dictionary {action_type: [targets]}
    final_drugs_df["targets"] = None
    for i, drug in final_drugs_df.iterrows():
        drug_targets = mechanism_df.loc[
            mechanism_df["molecule_chembl_id"] == drug["drug_id"],
            ["action_type", "uniprot_ids"]
        ]
        if drug_targets.shape[0] > 0:
            targets_dict = {}
            for _, r in drug_targets.iterrows():
                action_type = r["action_type"]
                if action_type not in targets_dict:
                    targets_dict[action_type] = []
                targets_dict[action_type] += r["uniprot_ids"]
            final_drugs_df.at[i, "targets"] = targets_dict
    
    print(f"Processed {len(final_drugs_df)} drugs")
    return final_drugs_df


def process_diseases_data(diseases_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean disease data.
    
    Args:
        diseases_df: Raw diseases DataFrame (with 'id', 'name', 'disease_targets', etc.)
        
    Returns:
        Processed diseases DataFrame with 'disease_id' column
    """
    print("Processing disease data...")
    
    final_diseases_df = diseases_df.copy()
    
    # Rename 'id' to 'disease_id' for consistency (matches notebook)
    if 'id' in final_diseases_df.columns:
        final_diseases_df.rename(columns={"id": "disease_id"}, inplace=True)
    
    print(f"Processed {len(final_diseases_df)} diseases")
    return final_diseases_df


def process_trials_data(trials_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean clinical trials data.
    
    Extracts objective structural features from nested ClinicalTrials.gov data.
    Does NOT attempt to classify trial success/failure based on p-values,
    as this is methodologically problematic for drug repurposing analysis.
    
    Args:
        trials_df: Raw trials DataFrame from ClinicalTrials.gov API
        
    Returns:
        Processed trials DataFrame with flattened structural features
    """
    print("Processing clinical trials data...")
    
    # Drop annotation and document sections (not useful for analysis)
    cols_to_drop = [c for c in ["annotationSection", "documentSection"] if c in trials_df.columns]
    final_trials_df = trials_df.drop(columns=cols_to_drop)
    
    # Initialize new columns for extracted features
    final_trials_df["nct_id"] = None
    final_trials_df["status"] = None
    final_trials_df["phase"] = None
    final_trials_df["start_date"] = None
    final_trials_df["end_date"] = None
    final_trials_df["why_stopped"] = None
    final_trials_df["enrollment"] = None
    
    # Extract key information from nested structures
    for i, study in final_trials_df.iterrows():
        protocol = study["protocolSection"]
        
        # Basic identification
        final_trials_df.loc[i, "nct_id"] = protocol["identificationModule"]["nctId"]
        
        # Status information
        status_module = protocol["statusModule"]
        final_trials_df.loc[i, "status"] = status_module["overallStatus"]
        final_trials_df.loc[i, "why_stopped"] = status_module.get("whyStopped")
        
        # Dates
        start_date_struct = status_module.get("startDateStruct", {})
        final_trials_df.loc[i, "start_date"] = start_date_struct.get("date")
        
        completion_date_struct = status_module.get("completionDateStruct", {})
        final_trials_df.loc[i, "end_date"] = completion_date_struct.get("date")
        
        # Design information
        design_module = protocol.get("designModule", {})
        
        # Phase (extract max phase number)
        phases = design_module.get("phases", [])
        phase_nums = []
        for phase in phases:
            if phase != "NA" and phase:
                # Handle formats like "PHASE3", "PHASE2", "EARLY_PHASE1"
                digits = ''.join(c for c in phase if c.isdigit())
                if digits:
                    phase_nums.append(int(digits))
        if phase_nums:
            final_trials_df.loc[i, "phase"] = max(phase_nums)
        
        # Enrollment (sample size)
        enrollment_info = design_module.get("enrollmentInfo", {})
        final_trials_df.loc[i, "enrollment"] = enrollment_info.get("count")
    
    # Drop nested structures (raw data preserved in trials_df if needed)
    cols_to_drop = [c for c in ["derivedSection", "protocolSection", "resultsSection"] 
                    if c in final_trials_df.columns]
    final_trials_df = final_trials_df.drop(columns=cols_to_drop)
    
    # Reorder columns - objective features first
    first_columns = [
        'nct_id', 'phase', 'status', 'enrollment',
        'hasResults', 'start_date', 'end_date', 'why_stopped'
    ]
    existing_first = [c for c in first_columns if c in final_trials_df.columns]
    other_cols = [c for c in final_trials_df.columns if c not in first_columns]
    final_trials_df = final_trials_df[existing_first + other_cols]
    
    print(f"Processed {len(final_trials_df)} trials")
    print(f"  Status breakdown: {final_trials_df['status'].value_counts().to_dict()}")
    
    return final_trials_df


def process_indications_data(
    indications_df: pd.DataFrame,
    trials_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process and clean drug-disease indications data.
    
    Extracts NCT IDs from indication references and links to trial metadata.
    Does NOT classify trial success/failure - provides objective trial counts instead.
    
    Args:
        indications_df: Raw indications DataFrame
        trials_df: Processed trials DataFrame (with nct_id, phase, status, etc.)
        
    Returns:
        Processed indications DataFrame with trial linkage
    """
    print("Processing indications data...")
    
    # Drop unnecessary columns
    cols_to_drop = [c for c in ["drugind_id", "mesh_heading", "parent_molecule_chembl_id"] 
                    if c in indications_df.columns]
    final_indications_df = indications_df.drop(columns=cols_to_drop)
    
    # Extract NCT IDs from indication references
    final_indications_df["nct_ids"] = None
    for i, indication in indications_df.iterrows():
        for ref in indication["indication_refs"]:
            if ref["ref_type"] == "ClinicalTrials":
                final_indications_df.at[i, "nct_ids"] = ref["ref_id"].split(",")
    
    # Add objective trial summary (counts, not classifications)
    final_indications_df["n_trials"] = 0
    final_indications_df["max_trial_phase"] = None
    final_indications_df["has_completed_trial"] = False
    
    for i, indication in final_indications_df.iterrows():
        if indication["nct_ids"]:
            # Get trial info for this indication's NCT IDs
            matched_trials = trials_df[trials_df["nct_id"].isin(indication["nct_ids"])]
            
            if len(matched_trials) > 0:
                final_indications_df.loc[i, "n_trials"] = len(matched_trials)
                
                # Max phase reached
                phases = matched_trials["phase"].dropna()
                if len(phases) > 0:
                    final_indications_df.loc[i, "max_trial_phase"] = int(phases.max())
                
                # Has any completed trial
                final_indications_df.loc[i, "has_completed_trial"] = \
                    (matched_trials["status"] == "COMPLETED").any()
    
    # Rename for consistency
    final_indications_df.rename(columns={"molecule_chembl_id": "drug_id"}, inplace=True)
    
    # Reorder columns
    first_columns = [
        'drug_id', 'efo_term', 'efo_id', 'mesh_id',
        'max_phase_for_ind', 'n_trials', 'max_trial_phase', 'has_completed_trial'
    ]
    existing_first = [c for c in first_columns if c in final_indications_df.columns]
    other_cols = [c for c in final_indications_df.columns if c not in first_columns]
    final_indications_df = final_indications_df[existing_first + other_cols]
    
    print(f"Processed {len(final_indications_df)} indications")
    print(f"  Indications with trials: {(final_indications_df['n_trials'] > 0).sum()}")
    
    return final_indications_df


def add_pathway_annotations(
    drugs_df: pd.DataFrame,
    diseases_df: pd.DataFrame,
    reactome_map: Dict[str, List[str]]
) -> tuple:
    """
    Add Reactome pathway annotations to drugs and diseases.
    
    Maps each drug's targets and each disease's targets to their
    corresponding Reactome pathways using the provided mapping.
    
    Args:
        drugs_df: Processed drugs DataFrame with 'targets' column
        diseases_df: Processed diseases DataFrame with 'disease_targets' column
        reactome_map: Dictionary mapping UniProt IDs to Reactome pathway IDs
        
    Returns:
        Tuple of (updated_drugs_df, updated_diseases_df) with pathway columns added
    """
    print("Adding pathway annotations...")
    
    # Add drug_pathways column
    drugs_df = drugs_df.copy()
    drugs_df["drug_pathways"] = drugs_df["targets"].apply(
        lambda targets: [
            react 
            for up_list in targets.values() 
            for up in up_list 
            if up in reactome_map 
            for react in reactome_map[up]
        ] if targets else []
    )
    
    # Add disease_pathways column
    diseases_df = diseases_df.copy()
    diseases_df["disease_pathways"] = diseases_df["disease_targets"].apply(
        lambda targets: [
            react 
            for up in targets 
            if up in reactome_map 
            for react in reactome_map[up]
        ] if targets else []
    )
    
    drug_with_pathways = (drugs_df["drug_pathways"].apply(len) > 0).sum()
    disease_with_pathways = (diseases_df["disease_pathways"].apply(len) > 0).sum()
    
    print(f"  Drugs with pathways: {drug_with_pathways}/{len(drugs_df)}")
    print(f"  Diseases with pathways: {disease_with_pathways}/{len(diseases_df)}")
    
    return drugs_df, diseases_df
