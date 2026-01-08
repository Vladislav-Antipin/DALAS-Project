"""Functions for mapping drug and disease targets to Reactome pathways.

Matches notebook 01_1-data_retrieval.ipynb logic for pathway extraction.
"""

import pickle
from typing import Any, List, Dict

import pandas as pd
import numpy as np
from bioservices import UniProt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .config import RAW_DATA_DIR


def save_pathway_data(obj: Any, filename: str) -> None:
    """Save object as pickle in raw data directory."""
    filepath = RAW_DATA_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {filepath}")


def load_pathway_data(filename: str) -> Any:
    """Load pickle from raw data directory."""
    filepath = RAW_DATA_DIR / filename
    if filepath.exists():
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None


def map_targets_to_pathways(
    drugs_df: pd.DataFrame,
    diseases_df: pd.DataFrame,
    force: bool = False
) -> tuple:
    """
    Map drug and disease targets to Reactome pathways.
    
    Matches notebook 01_1-data_retrieval.ipynb logic.
    
    Args:
        drugs_df: Processed drugs DataFrame with 'targets' column
        diseases_df: Processed diseases DataFrame with 'disease_targets' column
        force: If True, remap even if cached data exists
        
    Returns:
        Tuple of (updated drugs_df, updated diseases_df) with pathway columns
    """
    drugs_filename = "drugs_with_pathways_df.pkl"
    diseases_filename = "diseases_with_pathways_df.pkl"
    
    if not force:
        cached_drugs = load_pathway_data(drugs_filename)
        cached_diseases = load_pathway_data(diseases_filename)
        if cached_drugs is not None and cached_diseases is not None:
            print("Loaded cached pathway mappings")
            return cached_drugs, cached_diseases
    
    print("Mapping targets to Reactome pathways...")
    
    # Extract all unique drug targets
    drug_targets = np.unique([
        uniprot
        for targets in drugs_df["targets"]
        if targets
        for uniprots in targets.values()
        for uniprot in uniprots
    ])
    
    # Extract all unique disease targets
    disease_targets = np.unique([
        uniprot
        for uniprots in diseases_df["disease_targets"]
        for uniprot in uniprots
    ])
    
    print(f"Found {len(drug_targets)} unique drug targets")
    print(f"Found {len(disease_targets)} unique disease targets")
    
    # Map drug targets to Reactome
    up = UniProt()
    
    print("Mapping drug targets to Reactome...")
    drug_react_map = []
    for i in range(0, len(drug_targets), 20):
        res = up.mapping(fr="UniProtKB_AC-ID", to="Reactome", query=drug_targets[i:i+20].tolist())
        if res and "results" in res:
            drug_react_map += res["results"]
    
    print("Mapping disease targets to Reactome...")
    disease_react_map = []
    for i in range(0, len(disease_targets), 20):
        res = up.mapping(fr="UniProtKB_AC-ID", to="Reactome", query=disease_targets[i:i+20].tolist())
        if res and "results" in res:
            disease_react_map += res["results"]
    
    # Combine and deduplicate
    drug_react_df = pd.DataFrame(drug_react_map)
    disease_react_df = pd.DataFrame(disease_react_map)
    react_df = pd.concat([drug_react_df, disease_react_df], axis=0).drop_duplicates()
    
    # Create mapping dictionary
    up_react_map = {}
    for _, react in react_df.iterrows():
        if react["from"] not in up_react_map:
            up_react_map[react["from"]] = [react["to"]]
        else:
            up_react_map[react["from"]].append(react["to"])
    
    # Add pathway lists to drugs
    drugs_df = drugs_df.copy()
    drugs_df["drug_pathways"] = drugs_df["targets"].apply(
        lambda targets: [
            react
            for up_list in targets.values()
            for up in up_list
            if up in up_react_map
            for react in up_react_map[up]
        ] if targets else []
    )
    
    # Add pathway lists to diseases
    diseases_df = diseases_df.copy()
    diseases_df["disease_pathways"] = diseases_df["disease_targets"].apply(
        lambda targets: [
            react
            for up in targets
            if up in up_react_map
            for react in up_react_map[up]
        ] if targets else []
    )
    
    save_pathway_data(drugs_df, drugs_filename)
    save_pathway_data(diseases_df, diseases_filename)
    
    print(f"Mapped pathways for {len(drugs_df)} drugs and {len(diseases_df)} diseases")
    
    return drugs_df, diseases_df


def generate_pathway_embeddings(
    drugs_df: pd.DataFrame,
    diseases_df: pd.DataFrame,
    n_components: int = 50,
    force: bool = False
) -> tuple:
    """
    Generate TF-IDF embeddings from pathway lists for drugs and diseases.
    
    Matches notebook 02-data_preparation.ipynb logic.
    
    Args:
        drugs_df: Drugs DataFrame with 'drug_pathways' column
        diseases_df: Diseases DataFrame with 'disease_pathways' column
        n_components: Number of SVD components for dimensionality reduction
        force: If True, regenerate even if cached data exists
        
    Returns:
        Tuple of (updated drugs_df, updated diseases_df) with pathway embedding columns
    """
    drugs_filename = "drugs_with_path_embeddings_df.pkl"
    diseases_filename = "diseases_with_path_embeddings_df.pkl"
    
    if not force:
        cached_drugs = load_pathway_data(drugs_filename)
        cached_diseases = load_pathway_data(diseases_filename)
        if cached_drugs is not None and cached_diseases is not None:
            print("Loaded cached pathway embeddings")
            return cached_drugs, cached_diseases
    
    print("Generating pathway TF-IDF embeddings...")
    
    drugs_df = drugs_df.copy()
    diseases_df = diseases_df.copy()
    
    # Convert pathway lists to strings for TF-IDF
    drugs_df["drug_path_str"] = drugs_df["drug_pathways"].apply(
        lambda x: " ".join(x) if x else ""
    )
    diseases_df["disease_path_str"] = diseases_df["disease_pathways"].apply(
        lambda x: " ".join(x) if x else ""
    )
    
    # Generate TF-IDF for drugs
    drug_vectorizer = TfidfVectorizer()
    drug_path_docs = drugs_df["drug_path_str"].tolist()
    
    if any(doc for doc in drug_path_docs):
        drug_tfidf = drug_vectorizer.fit_transform(drug_path_docs)
        
        # Reduce dimensionality
        n_drug_components = min(n_components, drug_tfidf.shape[1] - 1)
        if n_drug_components > 0:
            drug_svd = TruncatedSVD(n_components=n_drug_components)
            drug_embeddings = drug_svd.fit_transform(drug_tfidf)
            
            drug_emb_df = pd.DataFrame(
                drug_embeddings,
                columns=[f"drug_path_{i}" for i in range(drug_embeddings.shape[1])]
            )
            drugs_df = pd.concat([
                drugs_df.reset_index(drop=True),
                drug_emb_df
            ], axis=1)
    
    # Generate TF-IDF for diseases
    disease_vectorizer = TfidfVectorizer()
    disease_path_docs = diseases_df["disease_path_str"].tolist()
    
    if any(doc for doc in disease_path_docs):
        disease_tfidf = disease_vectorizer.fit_transform(disease_path_docs)
        
        # Reduce dimensionality
        n_disease_components = min(n_components, disease_tfidf.shape[1] - 1)
        if n_disease_components > 0:
            disease_svd = TruncatedSVD(n_components=n_disease_components)
            disease_embeddings = disease_svd.fit_transform(disease_tfidf)
            
            disease_emb_df = pd.DataFrame(
                disease_embeddings,
                columns=[f"disease_path_{i}" for i in range(disease_embeddings.shape[1])]
            )
            diseases_df = pd.concat([
                diseases_df.reset_index(drop=True),
                disease_emb_df
            ], axis=1)
    
    save_pathway_data(drugs_df, drugs_filename)
    save_pathway_data(diseases_df, diseases_filename)
    
    print("Generated pathway embeddings")
    
    return drugs_df, diseases_df


def compute_pathway_similarity(drug_pathways: List[str], disease_pathways: List[str]) -> float:
    """
    Compute Jaccard similarity between drug and disease pathway lists.
    
    Matches notebook 03-EDA.ipynb logic.
    
    Args:
        drug_pathways: List of Reactome pathway IDs for a drug
        disease_pathways: List of Reactome pathway IDs for a disease
        
    Returns:
        Jaccard similarity score (0-1)
    """
    if not drug_pathways or not disease_pathways:
        return 0.0
    
    drug_set = set(drug_pathways)
    disease_set = set(disease_pathways)
    
    intersection = len(drug_set & disease_set)
    union = len(drug_set | disease_set)
    
    return intersection / union if union > 0 else 0.0
