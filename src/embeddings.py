"""Functions for generating embeddings and computing similarities.

Matches notebook 01_1-data_retrieval.ipynb logic.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import ST_MODEL


def compute_drug_disease_similarities(
    drugs_df: pd.DataFrame,
    diseases_df: pd.DataFrame,
    model_name: str = ST_MODEL,
    force: bool = False
) -> pd.DataFrame:
    """
    Compute semantic similarity between drug and disease names using embeddings.
    
    Matches notebook 01_1-data_retrieval.ipynb logic.
    
    Args:
        drugs_df: DataFrame with drug information (must have 'drug_name' column)
        diseases_df: DataFrame with disease information (must have 'name' or 'disease_name' column)
        model_name: Name of the sentence transformer model to use
        force: If True, recompute even if embeddings exist
        
    Returns:
        DataFrame with drug-disease pairs and their name similarity scores
    """
    print(f"Computing drug-disease name similarities using {model_name}...")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Normalize names - handle both column naming conventions
    drug_names = drugs_df["drug_name"].str.lower()
    
    # Handle different disease name column names
    if "disease_name" in diseases_df.columns:
        disease_names = diseases_df["disease_name"].str.lower()
    elif "name" in diseases_df.columns:
        disease_names = diseases_df["name"].str.lower()
    else:
        raise ValueError("diseases_df must have 'disease_name' or 'name' column")
    
    # Handle disease ID column
    if "disease_id" in diseases_df.columns:
        disease_id_col = "disease_id"
    elif "id" in diseases_df.columns:
        disease_id_col = "id"
    else:
        raise ValueError("diseases_df must have 'disease_id' or 'id' column")
    
    # Compute embeddings
    print("Encoding drug names...")
    drug_embeddings = model.encode(drug_names.tolist())
    
    print("Encoding disease names...")
    disease_embeddings = model.encode(disease_names.tolist())
    
    # Compute cosine similarities
    print("Computing similarities...")
    similarities = model.similarity(drug_embeddings, disease_embeddings)
    
    # Create result dataframe (matching notebook structure)
    name_similarities = []
    for j, (disease_idx, disease) in enumerate(diseases_df.iterrows()):
        for i, (drug_idx, drug) in enumerate(drugs_df.iterrows()):
            disease_id = disease[disease_id_col]
            disease_name_val = disease["disease_name"] if "disease_name" in diseases_df.columns else disease["name"]
            
            name_similarities.append({
                "disease_id": disease_id,
                "drug_id": drug["drug_id"],
                "disease_name": disease_name_val.lower(),
                "drug_name": drug["drug_name"].lower(),
                "name_similarity": float(similarities[i, j])
            })
    
    embeddings_df = pd.DataFrame(name_similarities)
    
    print(f"Computed {len(embeddings_df)} drug-disease similarity scores")
    return embeddings_df


def generate_text_embeddings(
    texts: list,
    model_name: str = ST_MODEL
) -> list:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model_name: Name of the sentence transformer model to use
        
    Returns:
        List of embedding vectors
    """
    print(f"Generating embeddings for {len(texts)} texts...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings
