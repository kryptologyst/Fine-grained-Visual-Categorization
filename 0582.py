"""
Project 582: Fine-grained Visual Categorization - Modernized Implementation

This is a modernized version of the original fine-grained visual categorization project.
The original basic implementation has been replaced with a comprehensive framework.

For the full implementation, please refer to the src/ directory and use the scripts:

1. Training: python scripts/train.py --model resnet50 --attention cbam
2. Demo: python demo/gradio_demo.py
3. Example: python notebooks/example_usage.py

Key improvements:
- Advanced attention mechanisms (CBAM, SE, ECA)
- Multiple model architectures (ResNet, ViT, Multi-scale)
- Modern training pipeline with mixed precision
- Comprehensive evaluation metrics
- Interactive demos (Gradio/Streamlit)
- Production-ready code structure
- Type hints and documentation
- Comprehensive testing
"""

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.fine_grained_models import create_model
from src.data.dataset import get_transforms
from src.utils.utils import get_device, set_seed


def main():
    """Main function demonstrating the modernized fine-grained categorization."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device (auto-detect CUDA/MPS/CPU)
    device = get_device("auto")
    print(f"Using device: {device}")
    
    # Create model with attention mechanism
    print("Creating ResNet50 model with CBAM attention...")
    model = create_model(
        model_name="resnet50",
        num_classes=5,  # Example with 5 classes
        attention_type="cbam",
        pretrained=True
    )
    model = model.to(device)
    model.eval()
    
    # Get preprocessing transforms
    transform = get_transforms("test", augmentation="standard")
    
    # Create a sample image (in practice, you would load a real image)
    print("Creating sample image...")
    sample_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    
    # Preprocess image
    input_tensor = transform(sample_image).unsqueeze(0).to(device)
    
    # Make prediction
    print("Making prediction...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Display results
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.3f}")
    
    # Show top-5 predictions
    top5_probs, top5_indices = torch.topk(probabilities[0], 5)
    print("\nTop-5 Predictions:")
    for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
        print(f"{i+1}. Class {idx.item()}: {prob.item():.3f}")
    
    # Display image with prediction
    plt.figure(figsize=(10, 6))
    plt.imshow(sample_image)
    plt.title(f"Fine-grained Classification\nPredicted: Class {predicted_class} (Confidence: {confidence:.3f})")
    plt.axis('off')
    plt.show()
    
    # Model information
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    print(f"\nModel Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    print("\n" + "="*60)
    print("This is a demonstration of the modernized framework.")
    print("For full functionality, please use:")
    print("1. Training: python scripts/train.py --model resnet50 --attention cbam")
    print("2. Demo: python demo/gradio_demo.py")
    print("3. Example: python notebooks/example_usage.py")
    print("="*60)


if __name__ == "__main__":
    main()
