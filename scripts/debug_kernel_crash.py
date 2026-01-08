#!/usr/bin/env python3
"""
Debug script to test if the notebook code works outside Jupyter.
Run this from terminal: python scripts/debug_kernel_crash.py
"""

import sys
import traceback

def step(name):
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print('='*60)

try:
    step("1. Basic imports")
    import pickle
    import numpy as np
    import pandas as pd
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")

    step("2. PyTorch import")
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    step("3. Load saved data")
    with open("data/03-result/drugs_df.pkl", "rb") as f:
        drugs_df = pickle.load(f)
    with open("data/03-result/diseases_df.pkl", "rb") as f:
        diseases_df = pickle.load(f)
    with open("data/03-result/merged_df.pkl", "rb") as f:
        merged_df = pickle.load(f)
    print(f"drugs_df: {drugs_df.shape}")
    print(f"diseases_df: {diseases_df.shape}")
    print(f"merged_df: {merged_df.shape}")

    step("4. Load saved models (RUN_TUNING=False path)")
    MODEL_PATH = "results/models/time_split-"
    with open(MODEL_PATH + "summary_df.pkl", "rb") as f:
        summary_df = pickle.load(f)
    with open(MODEL_PATH + "models.pkl", "rb") as f:
        models = pickle.load(f)
    print(f"summary_df:\n{summary_df}")
    print(f"Models loaded: {list(models.keys())}")

    step("5. Prepare drug features (the problematic cell)")
    drug_features = drugs_df.drop(columns=drugs_df.filter(regex="path").columns)
    drug_features = drug_features.select_dtypes(include=["number", "bool"])
    print(f"drug_features shape: {drug_features.shape}")
    print(f"dtypes:\n{drug_features.dtypes.value_counts()}")
    
    step("6. Convert to tensor")
    # Fill NaN first
    drug_features_clean = drug_features.fillna(0)
    arr = drug_features_clean.values.astype(np.float32)
    print(f"Numpy array dtype: {arr.dtype}, shape: {arr.shape}")
    
    drug_tensor = torch.from_numpy(arr)
    print(f"Tensor created: {drug_tensor.shape}, dtype: {drug_tensor.dtype}")

    step("7. Disease features")
    disease_features = diseases_df.drop(columns=diseases_df.filter(regex="path").columns)
    disease_features = disease_features.select_dtypes(include=["number", "bool"])
    disease_features_clean = disease_features.fillna(0)
    disease_tensor = torch.from_numpy(disease_features_clean.values.astype(np.float32))
    print(f"Disease tensor: {disease_tensor.shape}")

    step("8. Pathway extraction")
    drug_paths = drugs_df["drug_pathways"].explode()
    disease_paths = diseases_df["disease_pathways"].explode()
    all_pathways = np.array(list(set(drug_paths) | set(disease_paths)), dtype=str)
    print(f"Total unique pathways: {len(all_pathways)}")

    step("9. PyTorch Geometric import")
    from torch_geometric.data import HeteroData
    print("torch_geometric imported successfully")
    
    data = HeteroData()
    data["drug"].x = drug_tensor
    data["disease"].x = disease_tensor
    print(f"HeteroData created: {data}")

    print("\n" + "="*60)
    print("SUCCESS: All steps completed without crash!")
    print("="*60)

except Exception as e:
    print(f"\n{'!'*60}")
    print(f"FAILED at step above")
    print(f"Error: {type(e).__name__}: {e}")
    print(f"{'!'*60}")
    traceback.print_exc()
    sys.exit(1)
