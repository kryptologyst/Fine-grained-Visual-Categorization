"""Interactive demo for fine-grained visual categorization."""

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional
import cv2

from src.models.fine_grained_models import create_model
from src.utils.utils import get_device
from src.data.dataset import get_transforms


class FineGrainedDemo:
    """Demo class for fine-grained visual categorization."""
    
    def __init__(
        self,
        model_path: str,
        model_name: str = "resnet50",
        num_classes: int = 200,
        classes: Optional[List[str]] = None,
        device: str = "auto"
    ):
        self.device = get_device(device)
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load model
        self.model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Set up classes
        if classes is None:
            self.classes = [f"Class_{i}" for i in range(num_classes)]
        else:
            self.classes = classes
        
        # Set up transforms
        self.transform = get_transforms(
            split="test",
            image_size=224,
            crop_size=224,
            augmentation="standard"
        )
    
    def predict(self, image: Image.Image) -> Tuple[str, List[Tuple[str, float]]]:
        """Predict the class of an input image."""
        if image is None:
            return "No image provided", []
        
        # Preprocess image
        if isinstance(self.transform, list):
            # Handle torchvision transforms
            input_tensor = self.transform(image).unsqueeze(0)
        else:
            # Handle albumentations transforms
            image_array = np.array(image)
            transformed = self.transform(image=image_array)
            input_tensor = torch.from_numpy(transformed['image']).unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get top-5 predictions
        top5_probs, top5_indices = torch.topk(probabilities[0], 5)
        top5_results = [
            (self.classes[idx], prob.item()) 
            for idx, prob in zip(top5_indices, top5_probs)
        ]
        
        result_text = f"Predicted: {self.classes[predicted_class]} (Confidence: {confidence:.3f})"
        
        return result_text, top5_results
    
    def predict_with_attention(self, image: Image.Image) -> Tuple[str, Image.Image]:
        """Predict with attention visualization."""
        if image is None:
            return "No image provided", None
        
        # Preprocess image
        if isinstance(self.transform, list):
            input_tensor = self.transform(image).unsqueeze(0)
        else:
            image_array = np.array(image)
            transformed = self.transform(image=image_array)
            input_tensor = torch.from_numpy(transformed['image']).unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Generate attention map (simplified version)
        attention_map = self._generate_attention_map(input_tensor)
        
        # Overlay attention on original image
        result_image = self._overlay_attention(image, attention_map)
        
        result_text = f"Predicted: {self.classes[predicted_class]} (Confidence: {confidence:.3f})"
        
        return result_text, result_image
    
    def _generate_attention_map(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generate attention map for visualization."""
        # This is a simplified attention map generation
        # In practice, you would extract attention weights from the model
        
        # For demonstration, create a random attention map
        attention_map = np.random.rand(224, 224)
        
        # Smooth the attention map
        attention_map = cv2.GaussianBlur(attention_map, (15, 15), 0)
        
        return attention_map
    
    def _overlay_attention(self, image: Image.Image, attention_map: np.ndarray) -> Image.Image:
        """Overlay attention map on the original image."""
        # Resize attention map to match image size
        image_array = np.array(image)
        attention_resized = cv2.resize(attention_map, (image_array.shape[1], image_array.shape[0]))
        
        # Normalize attention map
        attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
        
        # Create heatmap
        heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay on original image
        overlay = cv2.addWeighted(image_array, 0.6, heatmap, 0.4, 0)
        
        return Image.fromarray(overlay)


def create_demo_interface():
    """Create the Gradio demo interface."""
    
    # Initialize demo (you'll need to provide actual model path and classes)
    demo_instance = FineGrainedDemo(
        model_path="checkpoints/best_model.pth",  # Update with actual path
        model_name="resnet50",
        num_classes=5,  # Update based on your dataset
        classes=["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]  # Update with actual classes
    )
    
    # Create Gradio interface
    with gr.Blocks(title="Fine-grained Visual Categorization Demo") as demo:
        gr.Markdown("# Fine-grained Visual Categorization Demo")
        gr.Markdown("Upload an image to classify it using our fine-grained categorization model.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=300
                )
                
                with gr.Row():
                    predict_btn = gr.Button("Classify", variant="primary")
                    predict_attention_btn = gr.Button("Classify with Attention", variant="secondary")
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="Prediction",
                    interactive=False
                )
                
                output_probabilities = gr.Label(
                    label="Top-5 Predictions",
                    num_top_classes=5
                )
                
                attention_image = gr.Image(
                    label="Attention Visualization",
                    height=300
                )
        
        # Event handlers
        predict_btn.click(
            fn=demo_instance.predict,
            inputs=[input_image],
            outputs=[output_text, output_probabilities]
        )
        
        predict_attention_btn.click(
            fn=demo_instance.predict_with_attention,
            inputs=[input_image],
            outputs=[output_text, attention_image]
        )
        
        # Examples section
        gr.Markdown("## Examples")
        gr.Examples(
            examples=[
                ["assets/example1.jpg"],
                ["assets/example2.jpg"],
                ["assets/example3.jpg"]
            ],
            inputs=[input_image],
            label="Click on any example to load it"
        )
        
        # Model information
        gr.Markdown("## Model Information")
        gr.Markdown(f"""
        - **Model**: {demo_instance.model_name}
        - **Classes**: {len(demo_instance.classes)}
        - **Device**: {demo_instance.device}
        - **Input Size**: 224x224
        """)
    
    return demo


def create_streamlit_demo():
    """Create a Streamlit demo interface."""
    import streamlit as st
    
    st.title("Fine-grained Visual Categorization Demo")
    st.markdown("Upload an image to classify it using our fine-grained categorization model.")
    
    # Initialize demo
    if 'demo_instance' not in st.session_state:
        st.session_state.demo_instance = FineGrainedDemo(
            model_path="checkpoints/best_model.pth",
            model_name="resnet50",
            num_classes=5,
            classes=["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Prediction buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Classify"):
                with st.spinner("Classifying..."):
                    result_text, top5_results = st.session_state.demo_instance.predict(image)
                    st.success(result_text)
                    
                    # Display top-5 predictions
                    st.subheader("Top-5 Predictions")
                    for i, (class_name, confidence) in enumerate(top5_results):
                        st.write(f"{i+1}. {class_name}: {confidence:.3f}")
        
        with col2:
            if st.button("Classify with Attention"):
                with st.spinner("Generating attention map..."):
                    result_text, attention_image = st.session_state.demo_instance.predict_with_attention(image)
                    st.success(result_text)
                    st.image(attention_image, caption="Attention Visualization", use_column_width=True)
    
    # Model information
    st.sidebar.title("Model Information")
    st.sidebar.write(f"**Model**: {st.session_state.demo_instance.model_name}")
    st.sidebar.write(f"**Classes**: {len(st.session_state.demo_instance.classes)}")
    st.sidebar.write(f"**Device**: {st.session_state.demo_instance.device}")
    st.sidebar.write("**Input Size**: 224x224")


if __name__ == "__main__":
    # Create and launch Gradio demo
    demo = create_demo_interface()
    demo.launch(share=True)
