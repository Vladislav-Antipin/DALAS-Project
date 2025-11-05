"""Functions for fetching data from various APIs and sources."""

import time
import pickle
import json
import warnings
from typing import List, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from chembl_webresource_client.new_client import new_client
from bioservices import UniProt

from .config import (
    MESH_URL, OT_URL, NCT_URL, RAW_DATA_DIR,
    NB_TOP_TARGETS, NB_EVIDENCES
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
    Extract all MeSH IDs for autoimmune diseases.
    
    Args:
        force: If True, refetch even if cached data exists
        
    Returns:
        List of MeSH IDs
    """
    filename = "mesh_ids.pkl"
    
    if not force:
        cached = load_raw_data(filename)
        if cached is not None:
            print(f"Loaded cached MeSH IDs: {len(cached)} IDs")
            return cached
    
    print("Fetching MeSH IDs from NCBI...")
    response = requests.get(MESH_URL)
    
    if response.status_code != 200:
        raise ConnectionError(f"Failed to retrieve MeSH data: {response.status_code}")
    
    soup = BeautifulSoup(response.text, "html.parser")
    autoimm_ul = soup.find("span", string="Autoimmune Diseases").find_all_next("ul")
    
    mesh_ids = []
    for disease_a in autoimm_ul[8].find_all("a"):
        disease_url = disease_a.get('href')
        response = requests.get(f'https://www.ncbi.nlm.nih.gov{disease_url}')
        
        if response.status_code == 200:
            disease_soup = BeautifulSoup(response.text, "html.parser")
            mesh_id_element = disease_soup.find(
                "p", string=lambda text: text and text.startswith("MeSH Unique ID:")
            )
            if mesh_id_element:
                mesh_ids.append(mesh_id_element.text.split()[-1])
        else:
            warnings.warn(f"Failed to retrieve {disease_url}", RuntimeWarning)
    
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
    
    Args:
        efo_ids: List of EFO IDs
        force: If True, refetch even if cached data exists
        
    Returns:
        DataFrame with disease information
    """
    filename = "diseases_df.pkl"
    
    if not force:
        cached = load_raw_data(filename)
        if cached is not None:
            print(f"Loaded cached diseases: {len(cached)} diseases")
            return cached
    
    print("Fetching disease data from Open Targets...")
    
    ot_targets_info = []
    for efo_id in efo_ids:
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
    
    diseases_df = pd.DataFrame(ot_targets_info)
    
    # Extract Ensembl IDs
    diseases_df["ensembl_ids"] = diseases_df["associatedTargets"].apply(
        lambda result: [row["target"]["id"] for row in result["rows"]]
    )
    
    # Map Ensembl to UniProt
    print("Mapping Ensembl IDs to UniProt...")
    up = UniProt()
    all_ensembl_ids = list({x for ids in diseases_df["ensembl_ids"] for x in ids})
    
    if all_ensembl_ids:
        ensembl_id_map = up.mapping(fr="Ensembl", to="UniProtKB", query=all_ensembl_ids)
        uniprot_map = {
            result["from"]: result["to"]["primaryAccession"]
            for result in ensembl_id_map["results"]
        }
        
        diseases_df["uniprot_ids"] = diseases_df["ensembl_ids"].apply(
            lambda ensembl_ids: [
                uniprot_map[eid] for eid in ensembl_ids if eid in uniprot_map
            ]
        )
    
    # Fetch target evidence
    print("Fetching target evidence...")
    diseases_df["target_evidence"] = [[] for _ in range(len(diseases_df))]
    
    for i, disease in diseases_df.iterrows():
        query_string = """
        query disease($efoId: String!, $ensemblIds : [String!]!, $size: Int!){
            disease(efoId: $efoId) {
                evidences(ensemblIds: $ensemblIds, size: $size){ 
                    count
                    rows{
                        target{
                            id
                        }
                        score
                        datatypeId
                        variantEffect
                        targetModulation
                        targetRole
                        directionOnTrait
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
                    "efoId": disease["id"],
                    "ensemblIds": disease["ensembl_ids"],
                    "size": NB_EVIDENCES
                }
            }
        )
        
        if response.status_code == 200:
            evidence_data = json.loads(response.text)["data"]["disease"]["evidences"]["rows"]
            diseases_df.at[i, "target_evidence"] = evidence_data
        else:
            warnings.warn(f"Failed to fetch evidence for {disease['id']}")
    
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
