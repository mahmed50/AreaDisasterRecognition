import os
import random
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# Settings
NUM_CLASSES = 4
SAMPLE_SIZE = 500
SELECTED_INDICES = [0, 9, 49, 98]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "initial_model.pt")
)
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "AIDER", "val")
)

# Transforms
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Load dataset
val_dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
class_names = val_dataset.classes

# Sample 500 random images
random.seed(42)
subset_indices = random.sample(range(len(val_dataset)), SAMPLE_SIZE)
subset = Subset(val_dataset, subset_indices)
val_loader = DataLoader(subset, batch_size=1)

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Inference
correct = 0
results = []

with torch.no_grad():
    for idx, (images, labels) in enumerate(val_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()

        results.append(
            {
                "index": idx,
                "expected": class_names[labels.item()],
                "predicted": class_names[predicted.item()],
            }
        )

accuracy = correct / SAMPLE_SIZE
print(f"✅ Accuracy on 500 samples: {accuracy:.4f}")

# Log specific percentile predictions
for idx in SELECTED_INDICES:
    res = results[idx]
    print(f"[{idx:>3}] Expected: {res['expected']:10} | Predicted: {res['predicted']}")

