"""Functions for generating additional disease features from MeSH ontology.

Matches notebook 01_3-additional_disease_features.ipynb logic.
"""

import pickle
from typing import Any
from urllib.parse import quote

import pandas as pd
import numpy as np
import requests
from lxml import etree
from sklearn.preprocessing import MultiLabelBinarizer

from .config import MESH_URL, RAW_DATA_DIR


def save_extended_data(obj: Any, filename: str) -> None:
    """Save object as pickle in raw data directory."""
    filepath = RAW_DATA_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {filepath}")


def load_extended_data(filename: str) -> Any:
    """Load pickle from raw data directory."""
    filepath = RAW_DATA_DIR / filename
    if filepath.exists():
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None


def fetch_mesh_ontology_features(force: bool = False) -> pd.DataFrame:
    """
    Fetch MeSH ontology features including tree numbers and categories.
    
    Matches notebook 01_3-additional_disease_features.ipynb logic.
    
    Args:
        force: If True, refetch even if cached data exists
        
    Returns:
        DataFrame with MeSH ontology features
    """
    filename = "mesh_ontology_df.pkl"
    
    if not force:
        cached = load_extended_data(filename)
        if cached is not None:
            print(f"Loaded cached MeSH ontology: {len(cached)} entries")
            return cached
    
    print("Fetching MeSH ontology features...")
    
    # Query for disease tree numbers
    mesh_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
PREFIX mesh2025: <http://id.nlm.nih.gov/mesh/2025/>
PREFIX mesh2024: <http://id.nlm.nih.gov/mesh/2024/>
PREFIX mesh2023: <http://id.nlm.nih.gov/mesh/2023/>

SELECT ?dis ?tn (GROUP_CONCAT(DISTINCT ?other_tn; separator=",") AS ?all_tn_list)
FROM <http://id.nlm.nih.gov/mesh>
WHERE {  
  ?root rdfs:label "Immune System Diseases"@en .
  ?root meshv:treeNumber ?tn_root .
  ?dis meshv:treeNumber ?tn .
  FILTER(STRSTARTS(STR(?tn),STR(?tn_root))) .
  ?dis meshv:treeNumber ?other_tn .
}
GROUP BY ?dis ?tn
"""
    
    response = requests.get(f"{MESH_URL}?query={quote(mesh_query)}")
    
    if not response.ok:
        response.raise_for_status()
    
    root = etree.fromstring(response.text.encode())
    ns = {"sr": "http://www.w3.org/2005/sparql-results#"}
    
    mesh_ids = root.xpath("//sr:binding[@name=\"dis\"]/sr:uri/text()", namespaces=ns)
    tree_nb = root.xpath("//sr:result/sr:binding[@name=\"tn\"]/sr:uri/text()", namespaces=ns)
    all_tree_nb = root.xpath("//sr:result/sr:binding[@name=\"all_tn_list\"]/sr:literal/text()", namespaces=ns)
    
    # Extract IDs from URIs
    mesh_ids = [id.split("/")[-1] for id in mesh_ids]
    tree_nb = [tn.split("/")[-1] for tn in tree_nb]
    all_tree_nb = [[tn.split("/")[-1] for tn in tns.split(",")] for tns in all_tree_nb]
    
    diseases_df = pd.DataFrame({
        "mesh_id": mesh_ids,
        "tree_number": tree_nb,
        "all_tree_numbers": all_tree_nb
    })
    
    # Calculate ontology depth
    diseases_df["ontology_depth"] = diseases_df["tree_number"].str.split(".").str.len()
    
    # Calculate descendant counts
    descendant_counts = {
        tn: diseases_df["tree_number"].str.startswith(tn).sum()
        for tn in diseases_df["tree_number"]
    }
    diseases_df["nb_descendants"] = diseases_df["tree_number"].map(descendant_counts)
    
    save_extended_data(diseases_df, filename)
    print(f"Fetched ontology features for {len(diseases_df)} diseases")
    
    return diseases_df


def fetch_disease_categories(force: bool = False) -> dict:
    """
    Fetch disease category mappings from MeSH.
    
    Args:
        force: If True, refetch even if cached data exists
        
    Returns:
        Dictionary mapping tree number prefixes to category names
    """
    filename = "disease_categories.pkl"
    
    if not force:
        cached = load_extended_data(filename)
        if cached is not None:
            print(f"Loaded cached disease categories: {len(cached)} categories")
            return cached
    
    print("Fetching disease categories...")
    
    mesh_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
PREFIX mesh2025: <http://id.nlm.nih.gov/mesh/2025/>
PREFIX mesh2024: <http://id.nlm.nih.gov/mesh/2024/>
PREFIX mesh2023: <http://id.nlm.nih.gov/mesh/2023/>

SELECT ?tn ?lab
FROM <http://id.nlm.nih.gov/mesh>
WHERE {  
?cat rdfs:label ?lab .
?cat meshv:treeNumber ?tn .
 FILTER(STRSTARTS(STR(?tn), "http://id.nlm.nih.gov/mesh/C") && STRLEN(STR(?tn)) < 31 ).
}
"""
    
    response = requests.get(f"{MESH_URL}?query={quote(mesh_query)}")
    
    if not response.ok:
        response.raise_for_status()
    
    root = etree.fromstring(response.text.encode())
    ns = {"sr": "http://www.w3.org/2005/sparql-results#"}
    
    category_ids = root.xpath("//sr:binding[@name=\"tn\"]/sr:uri/text()", namespaces=ns)
    category_names = root.xpath("//sr:binding[@name=\"lab\"]/sr:literal/text()", namespaces=ns)
    
    category_ids = [id.split("/")[-1] for id in category_ids]
    category_names = [
        name.lower().replace(" ", "_").replace(",", "")
        for name in category_names
    ]
    
    categories = dict(zip(category_ids, category_names))
    
    save_extended_data(categories, filename)
    print(f"Fetched {len(categories)} disease categories")
    
    return categories


def generate_extended_disease_features(
    diseases_df: pd.DataFrame,
    indications_df: pd.DataFrame,
    force: bool = False
) -> pd.DataFrame:
    """
    Generate extended disease features including ontology and prevalence proxies.
    
    Matches notebook 01_3-additional_disease_features.ipynb logic.
    
    Args:
        diseases_df: Raw diseases DataFrame from Open Targets
        indications_df: Raw indications DataFrame
        force: If True, regenerate even if cached data exists
        
    Returns:
        Extended diseases DataFrame with additional features
    """
    filename = "extended_diseases_df.pkl"
    
    if not force:
        cached = load_extended_data(filename)
        if cached is not None:
            print(f"Loaded cached extended diseases: {len(cached)} diseases")
            return cached
    
    print("Generating extended disease features...")
    
    # Get MeSH ontology features
    mesh_df = fetch_mesh_ontology_features(force=force)
    
    # Get disease categories
    categories = fetch_disease_categories(force=force)
    
    # Assign categories to diseases
    mesh_df["categories"] = [
        [categories.get(tn.split(".")[0], "not_disease") for tn in tns]
        for tns in mesh_df["all_tree_numbers"]
    ]
    
    # One-hot encode categories
    mlb = MultiLabelBinarizer()
    encoded_cat = pd.DataFrame(
        mlb.fit_transform(mesh_df["categories"]),
        columns=[f"cat_{c}" for c in mlb.classes_]
    )
    
    mesh_df = pd.concat([
        mesh_df.drop(columns=["tree_number", "all_tree_numbers", "categories"]),
        encoded_cat
    ], axis=1)
    
    # Map MeSH IDs to EFO IDs
    indications_copy = indications_df.copy()
    indications_copy["efo_id"] = indications_copy["efo_id"].str.replace(":", "_")
    mesh2efo = dict(zip(indications_copy["mesh_id"], indications_copy["efo_id"]))
    mesh_df["disease_id"] = mesh_df["mesh_id"].apply(lambda id: mesh2efo.get(id, ""))
    
    # Calculate prevalence proxy (number of indications per disease)
    nb_indications = {
        mesh: (indications_copy["efo_id"] == mesh2efo.get(mesh, "")).sum()
        for mesh in mesh_df["mesh_id"]
    }
    mesh_df["nb_indications"] = mesh_df["mesh_id"].map(nb_indications)
    
    # Drop mesh_id column (we have disease_id now)
    mesh_df = mesh_df.drop(columns=["mesh_id"])
    
    # Merge with original diseases DataFrame
    extended_df = diseases_df.merge(mesh_df, on="disease_id", how="left")
    
    # Drop duplicates that may arise from ID mappings
    extended_df = extended_df.drop_duplicates(subset="disease_id")
    
    save_extended_data(extended_df, filename)
    print(f"Generated extended features for {len(extended_df)} diseases")
    
    return extended_df
