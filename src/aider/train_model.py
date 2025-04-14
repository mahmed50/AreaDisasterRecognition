import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import time
import platform
import json

# Configuration
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "AIDER")
)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 2
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "aider_resnet18.pt")
)

# Data transforms
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Load pretrained model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)


results = {
    "system": platform.node(),
    "device": str(DEVICE),
    "model": "resnet18",
    "epochs": EPOCHS,
    "metrics": [],
}

for epoch in range(EPOCHS):
    epoch_start = time.time()

    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_dataset)
    epoch_end = time.time()

    results["metrics"].append(
        {
            "epoch": epoch + 1,
            "train_acc": acc,
            "train_loss": running_loss,
            "epoch_time_sec": epoch_end - epoch_start,
        }
    )

    print(f"Epoch {epoch + 1}: Loss={running_loss:.4f}, Accuracy={acc:.4f}")

# Optional: save after all training
metrics_path = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"⏱️ Metrics saved to {metrics_path}")

# Measure validation accuracy
val_start = time.time()
# validation loop ...
val_time = time.time() - val_start

results["val_acc"] = val_acc
results["val_time_sec"] = val_time

# Save benchmark
with open(
    os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json"), "w"
) as f:
    json.dump(results, f, indent=2)

print("⏱️ Training metrics saved.")

# Evaluation
model.eval()
correct = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()

val_acc = correct / len(val_dataset)
print(f"Validation Accuracy: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
