"""Utility functions for loading processed data.

Matches the data structure from notebooks 01_1 through 02.
"""

import pandas as pd
from typing import Dict

from .data_processors import load_processed_data
from .data_fetchers import load_raw_data


def load_all_processed() -> Dict[str, pd.DataFrame]:
    """
    Load all processed dataframes from data/02-result/.
    
    Returns:
        Dictionary with all processed dataframes
    """
    return {
        "drugs": load_processed_data("drugs_df.pkl"),
        "diseases": load_processed_data("diseases_df.pkl"),
        "trials": load_processed_data("trials_df.pkl"),
        "indications": load_processed_data("indications_df.pkl"),
        "dates": load_processed_data("dates_df.pkl"),
    }


def load_all_raw() -> Dict[str, pd.DataFrame]:
    """
    Load all raw dataframes from data/01-result/.
    
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
        "embeddings": load_raw_data("embeddings_df.pkl"),
        "fingerprints": load_raw_data("fingerprints_df.pkl"),
        "extended_diseases": load_raw_data("extended_diseases_df.pkl"),
    }


def load_drugs() -> pd.DataFrame:
    """Load processed drugs dataframe."""
    return load_processed_data("drugs_df.pkl")


def load_diseases() -> pd.DataFrame:
    """Load processed diseases dataframe."""
    return load_processed_data("diseases_df.pkl")


def load_trials() -> pd.DataFrame:
    """Load processed trials dataframe."""
    return load_processed_data("trials_df.pkl")


def load_indications() -> pd.DataFrame:
    """Load processed indications dataframe."""
    return load_processed_data("indications_df.pkl")


def load_dates() -> pd.DataFrame:
    """Load dates dataframe."""
    return load_processed_data("dates_df.pkl")


def load_embeddings() -> pd.DataFrame:
    """Load embeddings dataframe from raw data."""
    return load_raw_data("embeddings_df.pkl")


def load_fingerprints() -> pd.DataFrame:
    """Load fingerprints dataframe from raw data."""
    return load_raw_data("fingerprints_df.pkl")


def load_extended_diseases() -> pd.DataFrame:
    """Load extended diseases dataframe from raw data."""
    return load_raw_data("extended_diseases_df.pkl")
