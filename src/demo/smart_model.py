import torch
from torchvision import transforms, models
from PIL import Image
import os

# Paths
model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "old_VM1_model.pt")
)
image_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "./", "earthquake.png")
)

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

print(f"Image: {os.path.basename(image_path)}")
print(f"Predicted Class: {class_names[predicted]}")
