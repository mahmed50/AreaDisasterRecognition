import torch
from torchvision import transforms, models
from PIL import Image
import os
import argparse

# Paths
model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "distributed_vm.pt")
)

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to input image")
args = parser.parse_args()

image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.image))

num_classes = 4  # change if needed
class_names = ["Earthquake", "Fire", "Flood", "Normal"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image and preprocess
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
img = Image.open(image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Predict
with torch.no_grad():
    output = model(input_tensor)
    predicted = output.argmax(1).item()

print(f"\nImage: {os.path.basename(image_path)}")
print(f"Predicted Class: {class_names[predicted]}\n")
