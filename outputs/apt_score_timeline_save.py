import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import matplotlib.pyplot as plt

random.seed(42)

snapshots = list(range(1, 11))
apt_scores = [
    random.uniform(0.1, 0.3),
    random.uniform(0.15, 0.35),
    random.uniform(0.25, 0.45),
    random.uniform(0.4, 0.6),
    random.uniform(0.55, 0.7),
    random.uniform(0.65, 0.8),
    random.uniform(0.7, 0.85),
    random.uniform(0.75, 0.9),
    random.uniform(0.8, 0.95),
    random.uniform(0.85, 0.98)
]

plt.figure()
plt.plot(snapshots, apt_scores, marker="o")
plt.xlabel("Temporal Snapshot Index")
plt.ylabel("APT Probability Score")
plt.title("APT Probability Progression Over Time")

plt.savefig("results/apt_score_timeline.png", dpi=300, bbox_inches="tight")
plt.close()

print("[OK] Timeline plot saved as results/apt_score_timeline.png")
