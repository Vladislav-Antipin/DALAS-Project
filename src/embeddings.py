"""Functions for generating embeddings and computing similarities."""

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
    
    Args:
        drugs_df: DataFrame with drug information (must have 'drug_name' column)
        diseases_df: DataFrame with disease information (must have 'name' column)
        model_name: Name of the sentence transformer model to use
        force: If True, recompute even if embeddings exist
        
    Returns:
        DataFrame with drug-disease pairs and their name similarity scores
    """
    print(f"Computing drug-disease name similarities using {model_name}...")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Normalize names
    drug_names = drugs_df["drug_name"].str.lower()
    disease_names = diseases_df["name"].str.lower()
    
    # Compute embeddings
    print("Encoding drug names...")
    drug_embeddings = model.encode(drug_names)
    
    print("Encoding disease names...")
    disease_embeddings = model.encode(disease_names)
    
    # Compute cosine similarities
    print("Computing similarities...")
    similarities = model.similarity(drug_embeddings, disease_embeddings)
    
    # Create result dataframe
    name_similarities = []
    for j, disease in diseases_df.iterrows():
        for i, drug in drugs_df.iterrows():
            name_similarities.append({
                "disease_id": disease["id"],
                "drug_id": drug["drug_id"],
                "disease_name": disease["name"].lower(),
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
