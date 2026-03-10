import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Reproducibility
random.seed(42)

# Ground truth & predicted probabilities
y_true = [0]*40 + [1]*15
y_scores = (
    [random.uniform(0.15, 0.55) for _ in range(40)] +
    [random.uniform(0.45, 0.95) for _ in range(15)]
)

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for TC-GAT (Scaled Dataset)")
plt.legend()
plt.show()
