import sys
import os
import pandas as pd
import torch

# --------------------------------------------------
# Force project root
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# Normal imports
# --------------------------------------------------
from training.trainer import train
from model.tc_gat import TCGAT


# --------------------------------------------------
# Dummy graph for review-stage training
# --------------------------------------------------
class DummyGraph:
    def __init__(self, x):
        self.x = x


def build_dummy_graphs(csv_path, num_windows=10):
    df = pd.read_csv(csv_path)

    graphs = []
    labels = []

    for i in range(num_windows):
        x = torch.randn(20, 16)  # simulate node features
        graphs.append(DummyGraph(x))
        labels.append(df["label"].iloc[min(i, len(df) - 1)])

    return graphs, torch.tensor(labels)


def main():
    csv_path = os.path.join(
        PROJECT_ROOT, "data", "raw", "logs.csv"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError("logs.csv not found. Dataset missing.")

    print("📄 Loading APT logs from CSV...")
    graphs, labels = build_dummy_graphs(csv_path)

    print("🔥 Training TC-GAT with stage consistency loss...")
    model = TCGAT()
    train(model, graphs, labels)


if __name__ == "__main__":
    main()