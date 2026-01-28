"""Test suite for fine-grained visual categorization."""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.fine_grained_models import create_model, FineGrainedResNet, CBAM, SEBlock, ECABlock
from src.data.dataset import FineGrainedDataset, get_transforms, create_dataloaders
from src.utils.utils import set_seed, get_device, count_parameters
from src.train.trainer import get_loss_function, get_optimizer, get_scheduler


class TestModels:
    """Test model functionality."""
    
    def test_create_model(self):
        """Test model creation."""
        model = create_model("resnet50", num_classes=10)
        assert isinstance(model, FineGrainedResNet)
        assert count_parameters(model) > 0
    
    def test_resnet_forward(self):
        """Test ResNet forward pass."""
        model = create_model("resnet50", num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_attention_modules(self):
        """Test attention modules."""
        x = torch.randn(2, 2048, 7, 7)
        
        # Test CBAM
        cbam = CBAM(2048)
        output = cbam(x)
        assert output.shape == x.shape
        
        # Test SE Block
        se = SEBlock(2048)
        output = se(x)
        assert output.shape == x.shape
        
        # Test ECA Block
        eca = ECABlock(2048)
        output = eca(x)
        assert output.shape == x.shape


class TestData:
    """Test data functionality."""
    
    def test_transforms(self):
        """Test data transforms."""
        # Test torchvision transforms
        transform = get_transforms("train", augmentation="standard")
        assert transform is not None
        
        # Test albumentations transforms
        transform = get_transforms("train", augmentation="albumentations")
        assert transform is not None
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a toy dataset
            dataset = FineGrainedDataset(
                data_dir=temp_dir,
                split="train",
                transform=get_transforms("train")
            )
            
            # Should create toy dataset automatically
            assert len(dataset) > 0
            assert len(dataset.classes) > 0
    
    def test_dataloader_creation(self):
        """Test dataloader creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, val_loader, classes = create_dataloaders(
                data_dir=temp_dir,
                batch_size=2,
                num_workers=0  # Use 0 for testing
            )
            
            assert len(classes) > 0
            assert len(train_loader) > 0
            assert len(val_loader) > 0
            
            # Test data loading
            for images, targets in train_loader:
                assert images.shape[0] <= 2
                assert images.shape[1:] == (3, 224, 224)
                assert targets.shape[0] <= 2
                break


class TestTraining:
    """Test training functionality."""
    
    def test_loss_functions(self):
        """Test loss functions."""
        # Test cross entropy
        criterion = get_loss_function("cross_entropy")
        assert criterion is not None
        
        # Test focal loss
        criterion = get_loss_function("focal")
        assert criterion is not None
        
        # Test label smoothing
        criterion = get_loss_function("label_smoothing", smoothing=0.1)
        assert criterion is not None
    
    def test_optimizers(self):
        """Test optimizers."""
        model = create_model("resnet50", num_classes=10)
        
        # Test SGD
        optimizer = get_optimizer(model, "sgd", 0.001, 1e-4)
        assert optimizer is not None
        
        # Test Adam
        optimizer = get_optimizer(model, "adam", 0.001, 1e-4)
        assert optimizer is not None
        
        # Test AdamW
        optimizer = get_optimizer(model, "adamw", 0.001, 1e-4)
        assert optimizer is not None
    
    def test_schedulers(self):
        """Test schedulers."""
        model = create_model("resnet50", num_classes=10)
        optimizer = get_optimizer(model, "adam", 0.001, 1e-4)
        
        # Test cosine scheduler
        scheduler = get_scheduler(optimizer, "cosine", T_max=10)
        assert scheduler is not None
        
        # Test step scheduler
        scheduler = get_scheduler(optimizer, "step", step_size=10)
        assert scheduler is not None
        
        # Test plateau scheduler
        scheduler = get_scheduler(optimizer, "plateau")
        assert scheduler is not None


class TestUtils:
    """Test utility functions."""
    
    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42)
        # This should not raise an exception
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device("auto")
        assert device is not None
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_parameter_counting(self):
        """Test parameter counting."""
        model = create_model("resnet50", num_classes=10)
        param_count = count_parameters(model)
        assert param_count > 0
        assert isinstance(param_count, int)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training_step(self):
        """Test a single training step."""
        # Create model
        model = create_model("resnet50", num_classes=5)
        criterion = get_loss_function("cross_entropy")
        optimizer = get_optimizer(model, "adam", 0.001, 1e-4)
        
        # Create dummy data
        images = torch.randn(2, 3, 224, 224)
        targets = torch.randint(0, 5, (2,))
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check outputs
        assert outputs.shape == (2, 5)
        assert loss.item() > 0
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        model = create_model("resnet50", num_classes=10)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            
            # Load model
            new_model = create_model("resnet50", num_classes=10)
            new_model.load_state_dict(torch.load(f.name))
            
            # Test that models produce same output
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output1 = model(x)
                output2 = new_model(x)
                assert torch.allclose(output1, output2)
            
            # Clean up
            os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__])
