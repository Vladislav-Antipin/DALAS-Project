"""Utility functions for loading processed data."""

import pandas as pd
from typing import Dict, List

from .data_processors import load_processed_data
from .data_fetchers import load_raw_data


def load_all_processed() -> Dict[str, pd.DataFrame]:
    """
    Load all processed dataframes.
    
    Returns:
        Dictionary with all processed dataframes
    """
    return {
        "drugs": load_processed_data("final_drugs_df.pkl"),
        "diseases": load_processed_data("final_diseases_df.pkl"),
        "trials": load_processed_data("final_trials_df.pkl"),
        "indications": load_processed_data("final_indications_df.pkl"),
        "embeddings": load_processed_data("embeddings_df.pkl"),
    }


def load_all_raw() -> Dict[str, pd.DataFrame]:
    """
    Load all raw dataframes.
    
    Returns:
        Dictionary with all raw dataframes
    """
    return {
        "mesh_ids": load_raw_data("mesh_ids.pkl"),
        "indications": load_raw_data("indications_df.pkl"),
        "drugs": load_raw_data("drugs_df.pkl"),
        "mechanism": load_raw_data("mechanism_df.pkl"),
        "targets": load_raw_data("targets_df.pkl"),
        "diseases": load_raw_data("diseases_df.pkl"),
        "trials": load_raw_data("trials_df.pkl"),
        "reactome_map": load_raw_data("reactome_map.pkl"),
    }


def load_drugs() -> pd.DataFrame:
    """Load processed drugs dataframe."""
    return load_processed_data("final_drugs_df.pkl")


def load_diseases() -> pd.DataFrame:
    """Load processed diseases dataframe."""
    return load_processed_data("final_diseases_df.pkl")


def load_trials() -> pd.DataFrame:
    """Load processed trials dataframe."""
    return load_processed_data("final_trials_df.pkl")


def load_indications() -> pd.DataFrame:
    """Load processed indications dataframe."""
    return load_processed_data("final_indications_df.pkl")


def load_embeddings() -> pd.DataFrame:
    """Load embeddings dataframe."""
    return load_processed_data("embeddings_df.pkl")


def load_reactome_map() -> Dict[str, List[str]]:
    """Load Reactome pathway mapping (UniProt ID -> [Reactome pathway IDs])."""
    result = load_raw_data("reactome_map.pkl")
    return result if result else {}
