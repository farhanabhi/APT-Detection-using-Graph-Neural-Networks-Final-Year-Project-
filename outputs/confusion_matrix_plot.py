import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

random.seed(42)

y_true = [0]*40 + [1]*15
y_scores = (
    [random.uniform(0.15, 0.55) for _ in range(40)] +
    [random.uniform(0.45, 0.95) for _ in range(15)]
)

threshold = 0.5
y_pred = [1 if s >= threshold else 0 for s in y_scores]

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure()
disp.plot()
plt.title("Confusion Matrix for TC-GAT")
plt.show()
