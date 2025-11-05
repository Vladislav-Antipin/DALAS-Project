"""Functions for processing and cleaning raw data."""

import pickle
from typing import Any

import pandas as pd
import numpy as np

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
    
    # Filter: available and not withdrawn
    final_drugs_df = (
        drugs_df[
            (drugs_df['availability_type'] > 0)
            & ~drugs_df['withdrawn_flag']
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
        diseases_df: Raw diseases DataFrame
        
    Returns:
        Processed diseases DataFrame
    """
    print("Processing disease data...")
    
    # TODO: Add more processing as needed
    final_diseases_df = diseases_df.copy()
    
    print(f"Processed {len(final_diseases_df)} diseases")
    return final_diseases_df


def process_trials_data(trials_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean clinical trials data, including success classification.
    
    Args:
        trials_df: Raw trials DataFrame
        
    Returns:
        Processed trials DataFrame with success metrics
    """
    print("Processing clinical trials data...")
    
    # Drop annotation and document sections
    final_trials_df = trials_df.drop(columns=["annotationSection", "documentSection"])
    
    # Initialize new columns
    final_trials_df["nct_id"] = None
    final_trials_df["status"] = None
    final_trials_df["phase"] = None
    final_trials_df["success"] = None
    final_trials_df["median_p_value"] = None
    final_trials_df["p_value_list"] = None
    
    # Extract key information from nested structures
    for i, study in final_trials_df.iterrows():
        protocol = study["protocolSection"]
        
        # Basic info
        final_trials_df.loc[i, "nct_id"] = protocol["identificationModule"]["nctId"]
        final_trials_df.loc[i, "status"] = protocol["statusModule"]["overallStatus"]
        
        # Dates
        start_date_struct = protocol["statusModule"].get("startDateStruct", {})
        final_trials_df.loc[i, "start_date"] = start_date_struct.get("date")
        
        completion_date_struct = protocol["statusModule"].get("completionDateStruct", {})
        final_trials_df.loc[i, "end_date"] = completion_date_struct.get("date")
        
        final_trials_df.loc[i, "why_stopped"] = protocol["statusModule"].get("whyStopped")
        
        # Phase
        phases = protocol["designModule"].get("phases", [])
        phases = [int(phase[-1]) for phase in phases if phase != "NA"]
        if phases:
            final_trials_df.loc[i, "phase"] = np.max(phases)
        
        # P-values from results
        if study["hasResults"]:
            measures = study["resultsSection"]["outcomeMeasuresModule"]["outcomeMeasures"]
            p_values = [
                measure["analyses"][0]["pValue"]
                for measure in measures
                if "analyses" in measure
                if "pValue" in measure["analyses"][0]
            ]
            # Extract numeric values from p-value strings
            p_values = [
                float(''.join([ch for ch in p if ch.isdigit() or ch == "."]))
                for p in p_values
            ]
            if len(p_values) > 0:
                final_trials_df.at[i, "p_value_list"] = p_values
    
    # Drop nested structures
    final_trials_df = final_trials_df.drop(columns=["derivedSection", "protocolSection"])
    
    # Reorder columns
    first_columns = [
        'nct_id', 'success', "median_p_value", 'phase', 'status',
        'hasResults', 'why_stopped'
    ]
    final_trials_df = final_trials_df[
        first_columns + [c for c in final_trials_df.columns if c not in first_columns]
    ]
    
    # Classify success based on p-values
    for i, study in final_trials_df.iterrows():
        if study["p_value_list"]:
            p_vals = np.array(study["p_value_list"])
            medp = np.median(p_vals)
            prop05 = (p_vals < 0.05).sum() / len(p_vals)
            conflicting = (np.min(p_vals) < 0.05) and ((p_vals > 0.2).sum() / len(p_vals) > 0.5)
            
            final_trials_df.loc[i, "median_p_value"] = medp
            
            if medp <= 0.05 or (medp <= 0.1 and prop05 >= 0.5):
                final_trials_df.loc[i, "success"] = "success"
            elif 0.05 < medp <= 0.20 or (0.10 < medp <= 0.50 and 0.1 <= prop05 < 0.5) or conflicting:
                final_trials_df.loc[i, "success"] = "unknown"
            elif medp > 0.2 and prop05 < 0.1:
                final_trials_df.loc[i, "success"] = "fail"
    
    print(f"Processed {len(final_trials_df)} trials")
    return final_trials_df


def process_indications_data(
    indications_df: pd.DataFrame,
    trials_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process and clean drug-disease indications data with trial evidence.
    
    Args:
        indications_df: Raw indications DataFrame
        trials_df: Processed trials DataFrame
        
    Returns:
        Processed indications DataFrame
    """
    print("Processing indications data...")
    
    # Drop unnecessary columns
    final_indications_df = indications_df.drop(columns=[
        "drugind_id", "mesh_heading", "parent_molecule_chembl_id"
    ])
    
    # Extract NCT IDs from indication references
    final_indications_df["nct_ids"] = None
    for i, indication in indications_df.iterrows():
        for ref in indication["indication_refs"]:
            if ref["ref_type"] == "ClinicalTrials":
                final_indications_df.at[i, "nct_ids"] = ref["ref_id"].split(",")
    
    # Add trial evidence
    final_indications_df["nct_evidence"] = None
    
    for i, indication in final_indications_df.iterrows():
        if indication["nct_ids"]:
            success_list = []
            for nct_id in indication["nct_ids"]:
                success = trials_df.loc[trials_df["nct_id"] == nct_id, "success"].values
                phase = trials_df.loc[trials_df["nct_id"] == nct_id, "phase"].values
                
                if len(success) > 0 and success[0] is not None:
                    phase_str = str(int(phase[0])) if len(phase) > 0 and not pd.isna(phase[0]) else ""
                    success_list.append(success[0] + phase_str)
            
            final_indications_df.at[i, "nct_evidence"] = success_list
    
    # Rename for consistency
    final_indications_df.rename(columns={"molecule_chembl_id": "drug_id"}, inplace=True)
    
    # Reorder columns
    first_columns = [
        'drug_id', 'efo_term', 'efo_id', 'mesh_id',
        'max_phase_for_ind', 'nct_evidence'
    ]
    final_indications_df = final_indications_df[
        first_columns + [c for c in final_indications_df.columns if c not in first_columns]
    ]
    
    print(f"Processed {len(final_indications_df)} indications")
    return final_indications_df
