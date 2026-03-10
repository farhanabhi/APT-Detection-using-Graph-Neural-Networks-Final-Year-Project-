import pandas as pd
import pickle

WINDOW = 3600
STRIDE = 1800

def create_snapshots(csv_path):
    df = pd.read_csv(csv_path).sort_values("timestamp")

    snapshots = []
    start = df.timestamp.min()

    while start < df.timestamp.max():
        end = start + WINDOW
        snap = df[(df.timestamp >= start) & (df.timestamp < end)]
        if not snap.empty:
            snapshots.append(snap)
        start += STRIDE

    with open("data/processed/snapshots.pkl", "wb") as f:
        pickle.dump(snapshots, f)

    print(f"[OK] {len(snapshots)} temporal snapshots created")
