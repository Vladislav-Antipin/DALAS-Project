import requests
from bs4 import BeautifulSoup
from chembl_webresource_client.new_client import new_client
import pandas as pd
import warnings
import pickle
import os

DATA_DIR = "data"
MESH_URL = "https://www.ncbi.nlm.nih.gov/mesh?Db=mesh&Cmd=DetailsSearch&Term=%22Autoimmune+Diseases%22%5BMeSH+Terms%5D"

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def save_pickle(obj, filename):
    with open(os.path.join(DATA_DIR, filename), "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def get_mesh_ids(force=False):
    cached = load_pickle("mesh_ids.pkl")
    if cached is not None and not force:
        return cached
    
    print("Scraping MeSH IDs...")
    response = requests.get(MESH_URL)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to retrieve data from {MESH_URL}, status code: {response.status_code}")
    
    soup = BeautifulSoup(response.text, "html.parser")
    autoimm_ul = soup.find("span", string="Autoimmune Diseases").find_all_next("ul")

    mesh_ids = []
    for disease_a in autoimm_ul[8].find_all("a"):
        disease_url = disease_a.get('href')
        response = requests.get(f'https://www.ncbi.nlm.nih.gov{disease_url}')
        if response.status_code == 200:
            html_content = response.text
            disease_soup = BeautifulSoup(html_content, "html.parser")
            mesh_ids.append(
                disease_soup.find("p", string=lambda text: text and text.startswith("MeSH Unique ID:")).text.split()[-1]
            )
        else:
            warnings.warn(f"Failed to retrieve page for {disease_url}", RuntimeWarning)
    save_pickle(mesh_ids, "mesh_ids.pkl")
    return mesh_ids

def get_autoimmune_indications(mesh_ids, force=False):
    cached = load_pickle("indications.pkl")
    if cached is not None and not force:
        return cached
    
    print("Loading drug indications...")
    drug_indication = new_client.drug_indication
    autoimmune_ind = drug_indication.filter(mesh_id__in=mesh_ids)
    indications_df = pd.DataFrame(autoimmune_ind)
    chembl_ids = pd.unique(indications_df["molecule_chembl_id"]).tolist()

    save_pickle((indications_df, chembl_ids), "indications.pkl")
    return indications_df, chembl_ids

def get_autoimmune_drugs(chembl_ids, force=False):
    cached = load_pickle("drugs.pkl")
    if cached is not None and not force:
        return cached
    
    print("Loading drug information")
    molecule = new_client.molecule
    autoimmune_drugs = molecule.filter(molecule_chembl_id__in = chembl_ids, max_phase__gte = 3)
    drugs_df = pd.DataFrame(autoimmune_drugs)
    save_pickle(drugs_df, "drugs.pkl")
    return drugs_df

def main(step="all", force=False):
    ensure_data_dir()
    
    mesh_ids = None
    chembl_ids = None

    if step in ["mesh", "all"]:
        mesh_ids = get_mesh_ids(force)
        print(f"Mesh IDs retrieved: {len(mesh_ids)}")

    if step in ["indications", "all"]:
        if mesh_ids is None:
            mesh_ids = get_mesh_ids()  # load from cache if not already
        indications_df, chembl_ids = get_autoimmune_indications(mesh_ids, force)
        print(f"Indications retrieved: {len(indications_df)}")

    if step in ["drugs", "all"]:
        if chembl_ids is None:
            _, chembl_ids = get_autoimmune_indications(mesh_ids)
        drugs_df = get_autoimmune_drugs(chembl_ids, force)
        print(f"Drugs retrieved: {len(drugs_df)}")

    print("Pipeline step completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["mesh", "indications", "drugs", "all"], default="all",
                        help="Which step of the pipeline to run")
    parser.add_argument("--force", action="store_true", help="Force re-download and overwrite cache")
    args = parser.parse_args()
    main(step=args.step, force=args.force)


    