"""Functions for processing and cleaning raw data."""

import pickle
from typing import Any, Tuple

import pandas as pd
import numpy as np

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR


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


def save_raw_data(obj: Any, filename: str) -> None:
    """Save object as pickle in raw data directory."""
    filepath = RAW_DATA_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {filepath}")


def process_drugs_data(drugs_df: pd.DataFrame, mechanism_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean drug data.
    
    Args:
        drugs_df: Raw drugs DataFrame (or already processed from notebook)
        mechanism_df: Raw mechanisms DataFrame with targets
        
    Returns:
        Processed drugs DataFrame
    """
    print("Processing drug data...")
    
    # Check if data is already processed (from notebook output)
    if 'drug_id' in drugs_df.columns and 'drug_name' in drugs_df.columns:
        print("  Data appears to be already processed, adding targets...")
        final_drugs_df = drugs_df.copy()
        
        # Add targets if not present
        if 'targets' not in final_drugs_df.columns:
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
    
    # Process raw data from ChEMBL API
    # Filter: available and not withdrawn
    cols_to_drop = [c for c in [
        'atc_classifications', 'availability_type',
        'cross_references', 'molecule_hierarchy',
        'molecule_synonyms', 'polymer_flag',
        'usan_stem', 'usan_substem'
    ] if c in drugs_df.columns]
    
    # Handle availability_type filter
    if 'availability_type' in drugs_df.columns:
        filter_mask = (
            (drugs_df['availability_type'] > 0) | drugs_df['availability_type'].isna()
        ) & (~drugs_df['withdrawn_flag'] | drugs_df['withdrawn_flag'].isna())
    else:
        filter_mask = ~drugs_df.get('withdrawn_flag', pd.Series([False] * len(drugs_df)))
    
    final_drugs_df = drugs_df[filter_mask].drop(columns=cols_to_drop, errors='ignore')
    
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
    
    Matches notebook 02-data_preparation.ipynb logic with first_date and last_date.
    
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
    final_trials_df["first_date"] = None
    final_trials_df["last_date"] = None
    
    # Extract key information from nested structures
    for i, study in final_trials_df.iterrows():
        protocol = study["protocolSection"]
        status_module = protocol["statusModule"]
        
        # Basic info
        final_trials_df.loc[i, "nct_id"] = protocol["identificationModule"]["nctId"]
        final_trials_df.loc[i, "status"] = status_module["overallStatus"]
        
        # Dates - find first and last dates from all date fields
        first_date = "9999"  # largest by default
        last_date = "0000"   # smallest by default
        
        for key, val in status_module.items():
            if "date" in key.lower():
                # if dictionary with the "date" field
                if isinstance(val, dict) and "date" in val:
                    first_date = min(first_date, val["date"])
                    last_date = max(last_date, val["date"])
                # if itself a date string
                elif isinstance(val, str):
                    first_date = min(first_date, val)
                    last_date = max(last_date, val)
        
        final_trials_df.loc[i, "first_date"] = first_date
        final_trials_df.loc[i, "last_date"] = last_date
        
        # End date
        completion_date_struct = status_module.get("completionDateStruct", {"date": None})
        final_trials_df.loc[i, "end_date"] = completion_date_struct.get("date")
        
        final_trials_df.loc[i, "why_stopped"] = status_module.get("whyStopped")
        
        # Phase
        if "designModule" in protocol:
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
    
    # Classify success based on p-values (matching notebook logic)
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
    
    # Drop duplicates
    final_trials_df = final_trials_df.drop_duplicates(subset='nct_id')
    
    print(f"Processed {len(final_trials_df)} trials")
    return final_trials_df


def process_indications_data(
    indications_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    drugs_df: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process and clean drug-disease indications data with trial evidence.
    
    Matches notebook 02-data_preparation.ipynb logic with overall_success
    determination and dates_df generation.
    
    Args:
        indications_df: Raw indications DataFrame
        trials_df: Processed trials DataFrame
        drugs_df: Processed drugs DataFrame (for first_approval dates)
        
    Returns:
        Tuple of (processed indications DataFrame, dates DataFrame)
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
    
    # Add trial evidence and dates
    final_indications_df["nct_evidence"] = None
    final_indications_df["first_trial_date"] = "9999"
    final_indications_df["last_trial_date"] = "0000"
    
    for i, indication in final_indications_df.iterrows():
        if indication["nct_ids"]:
            success_list = []
            first_trial_date = "9999"
            last_trial_date = "0000"
            
            for nct_id in set(indication["nct_ids"]):
                success = trials_df.loc[trials_df["nct_id"] == nct_id, "success"].values
                phase = trials_df.loc[trials_df["nct_id"] == nct_id, "phase"].values
                
                if len(success) > 0 and success[0] is not None:
                    phase_str = str(int(phase[0])) if len(phase) > 0 and not pd.isna(phase[0]) else "nan"
                    success_list.append(success[0] + phase_str)
                
                # Get dates
                first_date = trials_df.loc[trials_df["nct_id"] == nct_id, "first_date"]
                if not first_date.empty:
                    first_date = first_date.values[0]
                    if isinstance(first_date, str) and first_date:
                        first_trial_date = min(first_trial_date, first_date)
                
                last_date = trials_df.loc[trials_df["nct_id"] == nct_id, "last_date"]
                if not last_date.empty:
                    last_date = last_date.values[0]
                    if isinstance(last_date, str) and last_date:
                        last_trial_date = max(last_trial_date, last_date)
            
            if first_trial_date != "9999":
                final_indications_df.loc[i, "first_trial_date"] = first_trial_date
            if last_trial_date != "0000":
                final_indications_df.loc[i, "last_trial_date"] = last_trial_date
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
    
    # Add disease_id column
    final_indications_df["disease_id"] = final_indications_df["efo_id"]
    
    # Drop duplicates
    final_indications_df = final_indications_df.drop_duplicates(subset=['disease_id', 'drug_id'])
    
    # Determine overall_success (matching notebook 02 logic)
    final_indications_df["overall_success"] = None
    
    for i, row in final_indications_df.iterrows():
        # If phase 4 is passed, it's definitely a success
        if final_indications_df.loc[i, "max_phase_for_ind"] == "4.0":
            final_indications_df.at[i, "overall_success"] = True
        
        # If there's evidence about trials for this combination
        elif final_indications_df.loc[i, "nct_evidence"]:
            all_results = final_indications_df.loc[i, "nct_evidence"]
            # Filter out results with "None" in them
            valid_results = [r for r in all_results if "None" not in r and "nan" not in r]
            
            if valid_results:
                highest_phase = max([int(result[-1]) for result in valid_results])
                latest_results = np.array([
                    result[:-1] for result in valid_results
                    if int(result[-1]) == highest_phase
                ])
                nb_success = np.sum(latest_results == "success")
                nb_fail = np.sum(latest_results == "fail")
                
                if highest_phase >= 3 and nb_success > 2 * nb_fail:
                    final_indications_df.at[i, "overall_success"] = True
                elif nb_fail > 2 * nb_success:
                    final_indications_df.at[i, "overall_success"] = False
        
        # If still not decided - check if abandoned
        if final_indications_df.at[i, "overall_success"] is None:
            last_trial_date = final_indications_df.loc[i, "last_trial_date"]
            if last_trial_date < "2020" or last_trial_date == "0000":
                final_indications_df.at[i, "overall_success"] = False
    
    final_indications_df["overall_success"] = final_indications_df["overall_success"].astype("boolean")
    
    # Reorder columns with overall_success
    first_columns = ["drug_id", "disease_id", "overall_success", "nct_evidence", "max_phase_for_ind"]
    final_indications_df = final_indications_df[
        first_columns + [c for c in final_indications_df.columns if c not in first_columns]
    ]
    
    # Generate dates_df (matching notebook 02 logic)
    dates_df = final_indications_df.groupby(
        ["drug_id", "disease_id"], as_index=False
    )["first_trial_date"].min()
    
    # Merge with drug first_approval if available
    if drugs_df is not None and "first_approval" in drugs_df.columns:
        drugs_df_copy = drugs_df.copy()
        drugs_df_copy["first_approval"] = drugs_df_copy["first_approval"].apply(
            lambda x: str(int(x)) if pd.notna(x) else "9999"
        )
        dates_df = dates_df.merge(
            drugs_df_copy[["drug_id", "first_approval"]], 
            how="left", 
            on="drug_id"
        ).fillna("9999")
        
        # If no trial date, use first approval date
        for i, row in dates_df.iterrows():
            if dates_df.loc[i, "first_trial_date"] == "9999" and dates_df.loc[i, "first_approval"] != "9999":
                dates_df.loc[i, "first_trial_date"] = dates_df.loc[i, "first_approval"]
        
        dates_df = dates_df.drop(columns=["first_approval"])
    
    # Replace absent dates with "0000" (will take only newest for test)
    dates_df["first_trial_date"] = dates_df["first_trial_date"].apply(
        lambda x: "0000" if x == "9999" else x
    )
    
    print(f"Processed {len(final_indications_df)} indications")
    return final_indications_df, dates_df
