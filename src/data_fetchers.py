"""Functions for fetching data from various APIs and sources."""

import time
import pickle
import json
import warnings
from typing import List, Any, Dict
from urllib.parse import quote

import pandas as pd
import numpy as np
import requests
from lxml import etree
from chembl_webresource_client.new_client import new_client
from bioservices import UniProt

from .config import (
    MESH_SPARQL_URL, OT_URL, NCT_URL, RAW_DATA_DIR,
    NB_TOP_TARGETS
)


def save_raw_data(obj: Any, filename: str) -> None:
    """Save object as pickle in raw data directory."""
    filepath = RAW_DATA_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {filepath}")


def load_raw_data(filename: str) -> Any:
    """Load pickle from raw data directory."""
    filepath = RAW_DATA_DIR / filename
    if filepath.exists():
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None


def fetch_mesh_ids(force: bool = False) -> List[str]:
    """
    Extract all MeSH IDs for immune system diseases using SPARQL query.
    
    This uses the MeSH RDF SPARQL endpoint to query for all diseases
    under "Immune System Diseases" in the MeSH hierarchy.
    
    Args:
        force: If True, refetch even if cached data exists
        
    Returns:
        List of MeSH IDs (e.g., ['D001327', 'D001528', ...])
    """
    filename = "mesh_ids.pkl"
    
    if not force:
        cached = load_raw_data(filename)
        if cached is not None:
            print(f"Loaded cached MeSH IDs: {len(cached)} IDs")
            return cached
    
    print("Fetching MeSH IDs via SPARQL query...")
    
    # SPARQL query to get all diseases under "Immune System Diseases"
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

    SELECT ?dis
    FROM <http://id.nlm.nih.gov/mesh>
    WHERE {  
      ?root rdfs:label "Immune System Diseases"@en .
      ?root meshv:treeNumber ?tn_root .
      ?dis meshv:treeNumber ?tn .
      FILTER(STRSTARTS(STR(?tn),STR(?tn_root))) .
    }
    """
    
    response = requests.get(f"{MESH_SPARQL_URL}?query={quote(mesh_query)}")
    
    if not response.ok:
        response.raise_for_status()
    
    # Parse XML response
    root = etree.fromstring(response.text.encode())
    ns = {"sr": "http://www.w3.org/2005/sparql-results#"}
    mesh_uris = root.xpath("//sr:uri/text()", namespaces=ns)
    
    # Extract just the MeSH ID from the URI (e.g., "D001327" from "http://id.nlm.nih.gov/mesh/D001327")
    mesh_ids = [mesh_uri.split("/")[-1] for mesh_uri in mesh_uris]
    
    # Remove duplicates while preserving order
    mesh_ids = list(dict.fromkeys(mesh_ids))
    
    save_raw_data(mesh_ids, filename)
    print(f"Fetched {len(mesh_ids)} MeSH IDs")
    return mesh_ids


def fetch_drug_indications(mesh_ids: List[str], force: bool = False) -> pd.DataFrame:
    """
    Fetch drug indications from ChEMBL for given MeSH IDs.
    
    Args:
        mesh_ids: List of MeSH IDs
        force: If True, refetch even if cached data exists
        
    Returns:
        DataFrame with drug indications
    """
    filename = "indications_df.pkl"
    
    if not force:
        cached = load_raw_data(filename)
        if cached is not None:
            print(f"Loaded cached indications: {len(cached)} records")
            return cached
    
    print("Fetching drug indications from ChEMBL...")
    drug_indication = new_client.drug_indication
    indications_df = pd.DataFrame(drug_indication.filter(mesh_id__in=mesh_ids))
    
    save_raw_data(indications_df, filename)
    print(f"Fetched {len(indications_df)} indications")
    return indications_df


def fetch_drugs(chembl_ids: List[str], force: bool = False) -> pd.DataFrame:
    """
    Fetch drug information from ChEMBL.
    
    Args:
        chembl_ids: List of ChEMBL molecule IDs
        force: If True, refetch even if cached data exists
        
    Returns:
        DataFrame with drug information
    """
    filename = "drugs_df.pkl"
    
    if not force:
        cached = load_raw_data(filename)
        if cached is not None:
            print(f"Loaded cached drugs: {len(cached)} drugs")
            return cached
    
    print("Fetching drug data from ChEMBL...")
    molecule = new_client.molecule
    drugs_df = pd.DataFrame(molecule.filter(molecule_chembl_id__in=chembl_ids))
    
    save_raw_data(drugs_df, filename)
    print(f"Fetched {len(drugs_df)} drugs")
    return drugs_df


def fetch_mechanisms_and_targets(chembl_ids: List[str], force: bool = False) -> tuple:
    """
    Fetch drug mechanisms and target information from ChEMBL.
    
    Args:
        chembl_ids: List of ChEMBL molecule IDs
        force: If True, refetch even if cached data exists
        
    Returns:
        Tuple of (mechanism_df, targets_df)
    """
    mech_filename = "mechanism_df.pkl"
    targets_filename = "targets_df.pkl"
    
    if not force:
        cached_mech = load_raw_data(mech_filename)
        cached_targets = load_raw_data(targets_filename)
        if cached_mech is not None and cached_targets is not None:
            print("Loaded cached mechanisms and targets")
            return cached_mech, cached_targets
    
    print("Fetching mechanisms and targets from ChEMBL...")
    
    # Fetch mechanisms
    mechanism = new_client.mechanism
    mechanism_df = pd.DataFrame(
        mechanism.filter(molecule_chembl_id__in=chembl_ids).only([
            "molecule_chembl_id",
            "action_type",
            "disease_efficacy",
            "mechanism_of_action",
            "target_chembl_id"
        ])
    )
    
    # Fetch targets
    target_ids = mechanism_df["target_chembl_id"].unique().tolist()
    target = new_client.target
    targets_df = pd.DataFrame(
        target.filter(target_chembl_id__in=target_ids).only([
            "target_chembl_id",
            "target_type",
            "target_components"
        ])
    )
    
    # Filter protein targets
    targets_df = targets_df[
        targets_df["target_type"].str.contains("PROTEIN") &
        (targets_df["target_components"].apply(len) > 0)
    ]
    
    # Extract UniProt IDs
    targets_df["uniprot_ids"] = targets_df["target_components"].apply(
        lambda comps: [
            ref["xref_id"]
            for c in comps
            for ref in c.get("target_component_xrefs", [])
            if ref.get("xref_src_db") == "UniProt"
        ]
    )
    
    # Map to primary UniProt IDs
    print("Mapping UniProt IDs...")
    up = UniProt()
    all_uniprot_ids = list({x for ids in targets_df["uniprot_ids"] for x in ids})
    
    if all_uniprot_ids:
        up_id_map = up.mapping(fr="UniProtKB", to="UniProtKB", query=all_uniprot_ids)
        primary_up_ids = list({
            result["to"]["primaryAccession"] 
            for result in up_id_map["results"]
        })
        
        targets_df["uniprot_ids"] = targets_df["uniprot_ids"].apply(
            lambda up_ids: [up_id for up_id in up_ids if up_id in primary_up_ids]
        )
    
    # Merge mechanism with targets
    mechanism_df = mechanism_df.merge(targets_df, on="target_chembl_id")
    
    save_raw_data(mechanism_df, mech_filename)
    save_raw_data(targets_df, targets_filename)
    print(f"Fetched {len(mechanism_df)} mechanisms and {len(targets_df)} targets")
    
    return mechanism_df, targets_df


def fetch_disease_info(efo_ids: List[str], force: bool = False) -> pd.DataFrame:
    """
    Fetch disease information and associated targets from Open Targets.
    
    Uses the notebook's approach: extracts UniProt IDs directly from proteinIds
    with source 'uniprot_swissprot' (no Ensembl mapping needed).
    
    Args:
        efo_ids: List of EFO IDs (can contain None/NaN values)
        force: If True, refetch even if cached data exists
        
    Returns:
        DataFrame with disease information including disease_targets
    """
    filename = "diseases_df.pkl"
    
    if not force:
        cached = load_raw_data(filename)
        if cached is not None:
            print(f"Loaded cached diseases: {len(cached)} diseases")
            return cached
    
    print("Fetching disease data from Open Targets...")
    
    # Clean EFO IDs - handle NaN/None values
    clean_efo_ids = [eid for eid in efo_ids if eid and pd.notna(eid)]
    
    ot_targets_info = []
    total = len(clean_efo_ids)
    
    for idx, efo_id in enumerate(clean_efo_ids):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{total} diseases")
        
        query_string = """
        query disease($efoId: String!, $size: Int!){
            disease(efoId: $efoId) {
                id
                name
                description
                associatedTargets(page: {index: 0, size: $size}, orderByScore: "score desc"){
                    rows{
                        target {
                            id
                            proteinIds{
                                id
                                source
                            }
                        }
                        score
                    }
                }
                phenotypes {
                    rows {
                        phenotypeHPO {
                            id
                            name
                            description
                        }
                    }
                }
            }
        }
        """
        
        response = requests.post(
            OT_URL,
            json={
                "query": query_string,
                "variables": {
                    "efoId": efo_id.replace(":", "_"),
                    "size": NB_TOP_TARGETS
                }
            }
        )
        
        if response.status_code == 200:
            data = json.loads(response.text)["data"]["disease"]
            if data:
                ot_targets_info.append(data)
        else:
            warnings.warn(f"Failed to fetch disease {efo_id}: {response.status_code}")
    
    diseases_df = pd.DataFrame([elt for elt in ot_targets_info if elt is not None])
    
    # Extract disease targets directly from proteinIds (notebook approach)
    # Uses uniprot_swissprot source - no Ensembl mapping needed
    diseases_df["disease_targets"] = diseases_df["associatedTargets"].apply(
        lambda result: [
            uniprot["id"]
            for row in result["rows"]
            for uniprot in row["target"]["proteinIds"]
            if uniprot["source"] == 'uniprot_swissprot'
        ]
    )
    
    save_raw_data(diseases_df, filename)
    print(f"Fetched {len(diseases_df)} diseases")
    return diseases_df


def fetch_clinical_trials(nct_ids: List[str], force: bool = False) -> pd.DataFrame:
    """
    Fetch clinical trial information from ClinicalTrials.gov.
    
    Args:
        nct_ids: List of NCT IDs
        force: If True, refetch even if cached data exists
        
    Returns:
        DataFrame with clinical trial information
    """
    filename = "trials_df.pkl"
    
    if not force:
        cached = load_raw_data(filename)
        if cached is not None:
            print(f"Loaded cached trials: {len(cached)} trials")
            return cached
    
    print("Fetching clinical trials from ClinicalTrials.gov...")
    
    nct_info = []
    total = len(nct_ids)
    
    for i in range(0, total, 3):
        if i % 150 == 0:
            print(f"Progress: {i}/{total} trials")
        
        batch = nct_ids[i:i+3]
        nct_filter = '0%7C'.join(batch)
        
        response = requests.get(f"{NCT_URL}?filter.ids={nct_filter}&format=json")
        time.sleep(0.1)  # Respect rate limit (10 req/sec)
        
        if response.status_code == 200:
            nct_info.append(response.json()["studies"])
        else:
            warnings.warn(f"Failed to fetch trials batch at index {i}: {response.status_code}")
    
    trials_df = pd.DataFrame([ct for cts in nct_info for ct in cts])
    
    save_raw_data(trials_df, filename)
    print(f"Fetched {len(trials_df)} trials")
    return trials_df


def fetch_reactome_pathways(
    drugs_df: pd.DataFrame,
    diseases_df: pd.DataFrame,
    force: bool = False
) -> Dict[str, List[str]]:
    """
    Fetch Reactome pathway mappings for drug and disease targets.
    
    Maps UniProt IDs to Reactome pathways using the UniProt ID mapping service.
    This is used to find pathway overlaps between drugs and diseases.
    
    Args:
        drugs_df: DataFrame with 'targets' column containing {action_type: [uniprot_ids]}
        diseases_df: DataFrame with 'disease_targets' column containing [uniprot_ids]
        force: If True, refetch even if cached data exists
        
    Returns:
        Dictionary mapping UniProt IDs to lists of Reactome pathway IDs
    """
    filename = "reactome_map.pkl"
    
    if not force:
        cached = load_raw_data(filename)
        if cached is not None:
            print(f"Loaded cached Reactome mappings: {len(cached)} proteins")
            return cached
    
    print("Fetching Reactome pathway mappings...")
    
    # Collect all unique UniProt IDs from drugs
    drug_targets = np.unique([
        uniprot 
        for targets in drugs_df["targets"] 
        if targets
        for uniprots in targets.values() 
        for uniprot in uniprots
    ])
    
    # Collect all unique UniProt IDs from diseases
    disease_targets = np.unique([
        uniprot
        for uniprots in diseases_df["disease_targets"]
        for uniprot in uniprots
    ])
    
    print(f"  Drug targets: {len(drug_targets)}")
    print(f"  Disease targets: {len(disease_targets)}")
    
    up = UniProt()
    
    # Map drug targets to Reactome (in batches of 20)
    print("  Mapping drug targets to Reactome...")
    drug_react_map = []
    for i in range(0, len(drug_targets), 20):
        res = up.mapping(fr="UniProtKB_AC-ID", to="Reactome", query=drug_targets[i:i+20].tolist())
        if res and "results" in res:
            drug_react_map += res["results"]
    
    # Map disease targets to Reactome (in batches of 20)
    print("  Mapping disease targets to Reactome...")
    disease_react_map = []
    for i in range(0, len(disease_targets), 20):
        res = up.mapping(fr="UniProtKB_AC-ID", to="Reactome", query=disease_targets[i:i+20].tolist())
        if res and "results" in res:
            disease_react_map += res["results"]
    
    # Combine and deduplicate
    drug_react_df = pd.DataFrame(drug_react_map) if drug_react_map else pd.DataFrame(columns=["from", "to"])
    disease_react_df = pd.DataFrame(disease_react_map) if disease_react_map else pd.DataFrame(columns=["from", "to"])
    react_df = pd.concat([drug_react_df, disease_react_df], axis=0).drop_duplicates()
    
    # Build mapping dictionary: UniProt ID -> [Reactome pathway IDs]
    up_react_map: Dict[str, List[str]] = {}
    for _, react in react_df.iterrows():
        uniprot_id = react["from"]
        if uniprot_id not in up_react_map:
            up_react_map[uniprot_id] = [react["to"]]
        else:
            up_react_map[uniprot_id].append(react["to"])
    
    save_raw_data(up_react_map, filename)
    print(f"Mapped {len(up_react_map)} proteins to Reactome pathways")
    return up_react_map
