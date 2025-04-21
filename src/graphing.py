import json
import matplotlib.pyplot as plt

with open("models/initial_metrics.json") as f:
    data = json.load(f)

epochs = [m["epoch"] for m in data["metrics"]]
times = [m["epoch_time_sec"] for m in data["metrics"]]
accs = [m["train_acc"] for m in data["metrics"]]

plt.plot(epochs, times, label="Epoch Time (sec)")
plt.plot(epochs, accs, label="Training Accuracy")
plt.title(f"Training Metrics - {data['system']} ({data['device']})")
plt.xlabel("Epoch")
plt.legend()
plt.show()
