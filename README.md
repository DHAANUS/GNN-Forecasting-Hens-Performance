# Hens Mobility Network Forecasting with GNN

This repository contains a **Graph Neural Network (GNN)** model built with **PyTorch Geometric** to analyze hens’ mobility and contact network data.  
The goal is to **forecast laying rate and mortality rate** based on mobility patterns, network interactions, and edge attributes.

---

## 📖 Project Overview
- **Data**: Mobility and contact networks of hens across different pens and time periods.
- **Nodes**: Individual hens.
- **Edges**: Pairwise interactions / mobility-based contacts between hens.
- **Edge Attributes**: Interaction strength, frequency, or temporal features.
- **Targets**: 
  - **Laying rate**
  - **Mortality rate**

We implement a **custom edge-aware Graph Attention Network (GAT)** to capture both node and edge features for improved prediction accuracy.

---

## Key Components
- `ManualEdgeAttentionConv` → custom GAT layer with edge-conditioned attention.  
- `ManualEdgeGATRegressor` → two-layer GNN for regression forecasting.  
- `EarlyStopping` → utility to prevent overfitting during training.  

---

## Training Workflow
1. Preprocess mobility/contact network data into `PyG Data` objects.  
2. Split into train / validation sets.  
3. Train the `ManualEdgeGATRegressor` with early stopping.  
4. Evaluate on unseen networks.  

Example (simplified):
```python
model = ManualEdgeGATRegressor(
    in_channels=features_dim,
    hidden_channels=64,
    edge_dim=edge_features_dim,
    dropout=0.3
)

out = model(x, edge_index, edge_attr)
