import matplotlib.pyplot as plt

models = ["GCN", "TC-GAT"]

precision = [0.88, 0.92]
recall = [0.86, 0.94]
f1 = [0.87, 0.93]
auc = [0.93, 0.98]

x = range(len(models))

plt.figure()
plt.plot(x, precision, marker="o", label="Precision")
plt.plot(x, recall, marker="o", label="Recall")
plt.plot(x, f1, marker="o", label="F1-Score")
plt.plot(x, auc, marker="o", label="AUC")

plt.xticks(x, models)
plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Performance Comparison: GCN vs TC-GAT")
plt.legend()

plt.savefig("results/model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("[OK] Model comparison plot saved as results/model_comparison.png")
