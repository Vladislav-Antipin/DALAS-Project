#!/usr/bin/env python3
"""
Run the GNN section from 04-model_training.ipynb as a standalone script.
This bypasses Jupyter kernel issues.

Usage: python scripts/run_gnn_section.py
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns

# Paths
FIG_PATH = "results/figures/04-time_split-"
MODEL_PATH = "results/models/time_split-"

print("Loading data...")
with open("data/03-result/drugs_df.pkl", "rb") as f:
    drugs_df = pickle.load(f)
with open("data/03-result/diseases_df.pkl", "rb") as f:
    diseases_df = pickle.load(f)
with open("data/03-result/merged_df.pkl", "rb") as f:
    merged_df = pickle.load(f)
with open("data/02-result/dates_df.pkl", "rb") as f:
    dates_df = pickle.load(f)

print(f"drugs_df: {drugs_df.shape}")
print(f"diseases_df: {diseases_df.shape}")
print(f"merged_df: {merged_df.shape}")

# Time split
SPLIT_BY_TIME = True
TRAIN_SIZE = 0.85

if SPLIT_BY_TIME:
    merged_df = (merged_df
        .merge(dates_df[["drug_id", "disease_id", "first_trial_date"]], 
               how="left", on=["drug_id", "disease_id"])
        .sort_values(by="first_trial_date")
        .reset_index(drop=True)
    )
    print(f"Split date: {merged_df['first_trial_date'][int(merged_df.shape[0]*0.9)]}")
    merged_df = merged_df.drop(columns=["first_trial_date"])

# Prepare features
print("\nPreparing features...")
drug_features = drugs_df.drop(columns=drugs_df.filter(regex="path").columns)
drug_features = drug_features.select_dtypes(include=["number", "bool"])
drug_features = torch.tensor(drug_features.fillna(0).values.astype(np.float32), dtype=torch.float)

disease_features = diseases_df.drop(columns=diseases_df.filter(regex="path").columns)
disease_features = disease_features.select_dtypes(include=["number", "bool"])
disease_features = torch.tensor(disease_features.fillna(0).values.astype(np.float32), dtype=torch.float)

print(f"Drug features: {drug_features.shape}")
print(f"Disease features: {disease_features.shape}")

# Pathways
all_pathways = np.array(
    list(set(drugs_df["drug_pathways"].explode()) | set(diseases_df["disease_pathways"].explode())), 
    dtype=str
)
print(f"Unique pathways: {len(all_pathways)}")

# Create mappings - use enumerate to get sequential indices
path2id = {p: i for i, p in enumerate(all_pathways)}
drug2id = {drug_id: i for i, drug_id in enumerate(drugs_df["drug_id"].tolist())}
disease2id = {disease_id: i for i, disease_id in enumerate(diseases_df["disease_id"].tolist())}

print(f"drug2id range: 0-{max(drug2id.values())}")
print(f"disease2id range: 0-{max(disease2id.values())}")

# Build edges
print("\nBuilding graph edges...")
drug_pathway_edges = []
for _, row in drugs_df.iterrows():
    for path in row["drug_pathways"]:
        drug_pathway_edges.append([drug2id[row["drug_id"]], path2id[path]])
drug_pathway_edges = torch.tensor(drug_pathway_edges, dtype=torch.long).t().contiguous()

disease_pathway_edges = []
for _, row in diseases_df.iterrows():
    for path in row["disease_pathways"]:
        disease_pathway_edges.append([disease2id[row["disease_id"]], path2id[path]])
disease_pathway_edges = torch.tensor(disease_pathway_edges, dtype=torch.long).t().contiguous()

# Drug-disease edges with labels
n_samples = len(merged_df)
n_train = int(n_samples * TRAIN_SIZE)
TRAIN_IDX, TEST_IDX = range(n_train), range(n_train, n_samples)

drug_disease_edges = []
labels = []
for _, row in merged_df.iterrows():
    if row["drug_id"] in drug2id and row["disease_id"] in disease2id:
        drug_disease_edges.append([drug2id[row["drug_id"]], disease2id[row["disease_id"]]])
        labels.append(int(row["success"]))

drug_disease_edges = torch.tensor(drug_disease_edges, dtype=torch.long).t().contiguous()
labels = torch.tensor(labels, dtype=torch.long)

print(f"Drug-pathway edges: {drug_pathway_edges.shape}")
print(f"Disease-pathway edges: {disease_pathway_edges.shape}")
print(f"Drug-disease edges: {drug_disease_edges.shape}")

# Create HeteroData
print("\nCreating HeteroData...")
num_pathways = len(all_pathways)

data = HeteroData()
data["drug"].x = drug_features
data["disease"].x = disease_features
data["pathway"].x = torch.ones((num_pathways, 1))

data['drug', 'interacts', 'pathway'].edge_index = drug_pathway_edges
data['disease', 'associated_with', 'pathway'].edge_index = disease_pathway_edges
data['drug', 'tryed_against', 'disease'].edge_index = drug_disease_edges
data['drug', 'tryed_against', 'disease'].edge_label = labels

# Add reverse edges so all node types get updated during message passing
data['pathway', 'rev_interacts', 'drug'].edge_index = drug_pathway_edges.flip(0)
data['pathway', 'rev_associated_with', 'disease'].edge_index = disease_pathway_edges.flip(0)
data['disease', 'rev_tryed_against', 'drug'].edge_index = drug_disease_edges.flip(0)

print(data)

# Define GNN model
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def forward(self, x_drug, x_disease, edge_label_index):
        edge_feat_drug = x_drug[edge_label_index[0]]
        edge_feat_disease = x_disease[edge_label_index[1]]
        return (edge_feat_drug * edge_feat_disease).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.drug_lin = torch.nn.Linear(drug_features.shape[1], hidden_channels)
        self.disease_lin = torch.nn.Linear(disease_features.shape[1], hidden_channels)
        self.pathway_lin = torch.nn.Linear(1, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=metadata)
        self.classifier = Classifier()

    def forward(self, data):
        x_dict = {
            "drug": self.drug_lin(data["drug"].x),
            "disease": self.disease_lin(data["disease"].x),
            "pathway": self.pathway_lin(data["pathway"].x),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["drug"],
            x_dict["disease"],
            data['drug', 'tryed_against', 'disease'].edge_index
        )
        return pred

# Train
print("\nTraining GNN...")
model = Model(hidden_channels=64, metadata=data.metadata())
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Create train/test masks
train_mask = torch.zeros(len(labels), dtype=torch.bool)
train_mask[:n_train] = True
test_mask = ~train_mask

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    pred = model(data)
    
    train_pred = pred[train_mask]
    train_labels = labels[train_mask].float()
    loss = F.binary_cross_entropy_with_logits(train_pred, train_labels)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = (pred[test_mask] > 0).long()
            test_labels = labels[test_mask]
            acc = accuracy_score(test_labels, test_pred)
            f1 = f1_score(test_labels, test_pred, zero_division=0)
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Test Acc={acc:.4f}, F1={f1:.4f}")

# Final evaluation
print("\nFinal evaluation...")
model.eval()
with torch.no_grad():
    pred = model(data)
    test_pred = (pred[test_mask] > 0).long().numpy()
    test_labels = labels[test_mask].numpy()
    
    acc = accuracy_score(test_labels, test_pred)
    f1 = f1_score(test_labels, test_pred, zero_division=0)
    print(f"Final Test Accuracy: {acc:.4f}")
    print(f"Final Test F1: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('GNN Confusion Matrix')
    plt.savefig(FIG_PATH + "gnn-confusion_matrix.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {FIG_PATH}gnn-confusion_matrix.png")

print("\nDone!")
