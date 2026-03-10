import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pickle
import torch
from model.tc_gat import TCGAT

# Load snapshots
with open("data/processed/snapshots.pkl", "rb") as f:
    snapshots = pickle.load(f)

# Dummy graph object (scaled demo)
class Graph:
    def __init__(self):
        self.x = torch.randn(25, 10)
        self.edge_index = torch.randint(0, 25, (2, 60))
        self.edge_type = torch.randint(0, 2, (60,))

model = TCGAT()
model.eval()

print("\nAPT Detection Scores per Snapshot:\n")

for i in range(5):
    graph_seq = [Graph() for _ in range(4)]
    score = model(graph_seq).item()
    print(f"Snapshot {i+1}: APT Score = {score:.4f}")
