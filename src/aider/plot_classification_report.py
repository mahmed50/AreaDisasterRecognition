import numpy as np
import json
from sklearn.metrics import classification_report

def show_classification_report():
    y_true = np.load("results/y_true.npy")
    y_pred = np.load("results/y_pred.npy")
    with open("results/class_names.json") as f:
        class_names = json.load(f)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:\n")
    print(report)
