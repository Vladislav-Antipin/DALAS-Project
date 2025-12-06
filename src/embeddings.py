"""Functions for generating embeddings and computing similarities."""

import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import ST_MODEL


def compute_drug_disease_similarities(
    drugs_df: pd.DataFrame,
    diseases_df: pd.DataFrame,
    model_name: str = ST_MODEL,
    force: bool = False,
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
    
    # Normalize and validate names
    # SentenceTransformer expects a list of strings, so we need to make sure there
    # are no NaNs/None/floats that would propagate into tokenization.
    drugs_clean = (
        drugs_df.loc[
            drugs_df["drug_name"].notna() & (drugs_df["drug_name"].astype(str).str.strip() != "")
        ]
        .copy()
    )
    diseases_clean = (
        diseases_df.loc[
            diseases_df["name"].notna() & (diseases_df["name"].astype(str).str.strip() != "")
        ]
        .copy()
    )

    if drugs_clean.empty:
        raise ValueError("No valid drug names available for embeddings.")
    if diseases_clean.empty:
        raise ValueError("No valid disease names available for embeddings.")

    # Reset indices to avoid KeyError when the model iterates by index positions.
    drugs_clean.reset_index(drop=True, inplace=True)
    diseases_clean.reset_index(drop=True, inplace=True)

    drug_names = drugs_clean["drug_name"].astype(str).str.lower().tolist()
    disease_names = diseases_clean["name"].astype(str).str.lower().tolist()
    
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
    for j, disease in diseases_clean.iterrows():
        for i, drug in drugs_clean.iterrows():
            # Use disease_id if available, else fall back to id
            disease_id = disease.get("disease_id", disease.get("id", ""))
            name_similarities.append({
                "disease_id": disease_id,
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
