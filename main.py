from graph.temporal_graph import create_snapshots
from data.generate_logs import *
print("[TC-GAT] Pipeline initialized")

create_snapshots("data/raw/logs.csv")
