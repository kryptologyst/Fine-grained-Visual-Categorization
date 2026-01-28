"""Example notebook for fine-grained visual categorization."""

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Import our modules
from src.models.fine_grained_models import create_model
from src.data.dataset import get_transforms
from src.utils.utils import get_device, set_seed

# Set random seed for reproducibility
set_seed(42)

# Get device
device = get_device("auto")
print(f"Using device: {device}")

# Create a simple model for demonstration
model = create_model("resnet50", num_classes=5, attention_type="cbam")
model = model.to(device)
model.eval()

# Get transforms
transform = get_transforms("test", augmentation="standard")

# Create a dummy image for demonstration
dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

# Preprocess image
input_tensor = transform(dummy_image).unsqueeze(0).to(device)

# Make prediction
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = F.softmax(outputs, dim=1)
    predicted_class = torch.argmax(outputs, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.3f}")

# Display image
plt.figure(figsize=(8, 6))
plt.imshow(dummy_image)
plt.title(f"Predicted: Class {predicted_class} (Confidence: {confidence:.3f})")
plt.axis('off')
plt.show()

# Show top-5 predictions
top5_probs, top5_indices = torch.topk(probabilities[0], 5)
print("\nTop-5 Predictions:")
for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
    print(f"{i+1}. Class {idx.item()}: {prob.item():.3f}")

print("\nModel Information:")
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2:.2f} MB")
